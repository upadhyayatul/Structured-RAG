"""Hybrid retrieval: dense (Qdrant) + BM25 fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import snowballstemmer
from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from upsc_rag.chunking.structured import extract_entities

_RRF_K = 60  # constant from the RRF paper (Cormack et al. 2009)

# Shared tokenizer for BM25: lowercase, split on non-alphanumerics, then stem so
# morphological variants collapse to one token (appointed/appoints/appointment ->
# appoint). This closes the lexical gap that caused phrasing-sensitive retrieval.
_STEMMER = snowballstemmer.stemmer("english")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, and Porter-stem each one."""
    return _STEMMER.stemWords(_TOKEN_RE.findall(text.lower()))


def load_chunks_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


class HybridRetriever:
    """Dense + BM25 hybrid retriever with RRF score fusion and parent-text expansion."""

    def __init__(self, cfg: dict[str, Any], chunks_path: Path) -> None:
        retrieval_cfg = cfg.get("retrieval", {})
        indexing_cfg = cfg.get("indexing", {})

        self._collection: str = indexing_cfg["collection_name"]
        self._embedding_model: str = indexing_cfg["embedding_model"]
        self._default_top_k: int = retrieval_cfg.get("top_k", 30)
        self._default_rerank_top_k: int = retrieval_cfg.get("rerank_top_k", 8)

        rewrite_cfg = retrieval_cfg.get("rewrite", {})
        self._rewrite_enabled: bool = bool(rewrite_cfg.get("enabled", False))
        self._rewrite_num_variants: int = rewrite_cfg.get("num_variants", 3)
        self._rewrite_model: str = rewrite_cfg.get("model", "gpt-4o-mini")
        # Gate: skip the rewrite LLM call + variant embeds when the original query's
        # top dense cosine score already clears this bar (the first pass is "good
        # enough"). Only weak first passes pay for query expansion.
        self._rewrite_score_threshold: float = rewrite_cfg.get("score_threshold", 0.5)

        # Graph-RAG expansion: after fusion, walk the section<->article graph to pull
        # in cross-referenced sections that share rare Articles with the top hits.
        graph_cfg = retrieval_cfg.get("graph", {})
        self._graph_enabled: bool = bool(graph_cfg.get("enabled", False))
        self._graph_seed_sections: int = graph_cfg.get("seed_sections", 5)
        self._graph_max_neighbors: int = graph_cfg.get("max_neighbors", 3)
        self._graph_rrf_weight: float = graph_cfg.get("rrf_weight", 1.0)
        self._graph = None

        # Chapter-level article catalog: maps chapter_num -> {"Article 124", ...} parsed
        # from each chapter's "Articles ... at a Glance" table. Used to attribute a
        # chapter's governing articles to its sections (whose prose names the topic but
        # not the article number). Built below once chunks are loaded.
        catalog_cfg = retrieval_cfg.get("catalog", {})
        self._catalog_enabled: bool = bool(catalog_cfg.get("enabled", False))
        self._chapter_articles: dict[int, set[str]] = {}

        self._qdrant = QdrantClient(url=indexing_cfg["qdrant_url"])
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Load all chunks once; split into child corpus (BM25) and parent text map
        all_chunks = list(load_chunks_jsonl(chunks_path))
        self._chunks: list[dict[str, Any]] = [
            c for c in all_chunks if c.get("content_type") == "child"
        ]
        self._parent_texts: dict[str, str] = {
            c["id"]: c["text"] for c in all_chunks if c.get("content_type") == "parent"
        }

        tokenized_corpus = [tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        # Fast lookup: chunk id → index in self._chunks
        self._chunk_index: dict[str, int] = {c["id"]: i for i, c in enumerate(self._chunks)}

        # Map each parent section id → its child chunk ids, so a graph-expanded
        # section can be represented in the fused candidate pool by a real child.
        self._section_children: dict[str, list[str]] = defaultdict(list)
        for c in self._chunks:
            pid = c.get("parent_id")
            if pid:
                self._section_children[pid].append(c["id"])

        if self._graph_enabled:
            from upsc_rag.indexing.graph_store import load_graph

            graph_path = chunks_path.parent / "graph.pkl"
            if graph_path.exists():
                self._graph = load_graph(graph_path)
            else:
                # Graph not built for this book — disable rather than fail.
                self._graph_enabled = False

        if self._catalog_enabled:
            from upsc_rag.enrichment.articles_catalog import build_chapter_article_map

            self._chapter_articles = {
                ch: set(arts)
                for ch, arts in build_chapter_article_map(all_chunks).items()
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return rerank_top_k results ranked by RRF, with parent text for generation.

        When query rewriting is enabled, retrieves for each query variant and fuses
        all dense+BM25 rank lists together (multi-query RRF) to improve recall.
        """
        top_k = top_k if top_k is not None else self._default_top_k
        rerank_top_k = rerank_top_k if rerank_top_k is not None else self._default_rerank_top_k

        # First pass: original query only (one embed). Cheap, and often enough.
        dense_ranks, payloads, top_score = self._dense_search(query, top_k)
        rank_lists: list[dict[str, int]] = [dense_ranks, self._bm25_search(query, top_k)]
        qdrant_payloads: dict[str, dict] = dict(payloads)

        # Gate: only expand with rewrite variants when the first pass looks weak.
        # A strong top dense score means we already found the right section, so we
        # skip the rewrite LLM call + extra embeds entirely.
        if self._rewrite_enabled and top_score < self._rewrite_score_threshold:
            variants = [v for v in self._expand_queries(query) if v != query]
            if variants:
                # Each variant needs its own OpenAI embedding round-trip; fan them
                # out across a thread pool so the I/O-bound work overlaps.
                with ThreadPoolExecutor(max_workers=len(variants)) as pool:
                    for d_ranks, b_ranks, pls in pool.map(
                        lambda q: self._search_one(q, top_k), variants
                    ):
                        rank_lists.append(d_ranks)
                        rank_lists.append(b_ranks)
                        qdrant_payloads.update(pls)

        fused = self._rrf_fuse(rank_lists)

        # Graph-RAG: pull in cross-referenced sections sharing rare Articles with the
        # top hits, fusing them into the scores so a strong signal can reach top-k.
        if self._graph is not None:
            self._graph_expand(fused, qdrant_payloads)

        top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:rerank_top_k]

        return [
            self._build_result(cid, fused[cid], qdrant_payloads)
            for cid in top_ids
        ]

    def _expand_queries(self, query: str) -> list[str]:
        """Return [query] or, if rewriting is enabled, the original plus LLM variants."""
        if not self._rewrite_enabled:
            return [query]
        from upsc_rag.retrieval.rewrite import rewrite_query

        return rewrite_query(
            query, self._openai, model=self._rewrite_model, num_variants=self._rewrite_num_variants
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parent_of(self, chunk_id: str, qdrant_payloads: dict[str, dict]) -> str | None:
        """The parent section id of a fused chunk (from payload or in-memory chunk)."""
        if chunk_id in qdrant_payloads:
            return qdrant_payloads[chunk_id].get("parent_id")
        idx = self._chunk_index.get(chunk_id)
        if idx is not None:
            return self._chunks[idx].get("parent_id")
        return None

    def _graph_expand(self, fused: dict[str, float], qdrant_payloads: dict[str, dict]) -> None:
        """Mutate ``fused`` in place: add graph-related sections as new candidates.

        Seeds are the parent sections of the current top fused chunks. Each neighbour
        section returned by the graph is represented by one of its child chunks and
        given an RRF contribution (scaled by ``graph.rrf_weight``) keyed off its graph
        rank, so the strongest cross-references can climb into the final top-k.
        """
        from upsc_rag.indexing.graph_store import expand_sections

        ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        seeds: list[str] = []
        seen: set[str] = set()
        for cid in ranked[: self._graph_seed_sections]:
            pid = self._parent_of(cid, qdrant_payloads)
            if pid and pid not in seen:
                seen.add(pid)
                seeds.append(pid)

        neighbours = expand_sections(
            self._graph, seeds, max_neighbors=self._graph_max_neighbors
        )
        for rank, section_id in enumerate(neighbours):
            children = self._section_children.get(section_id)
            if not children:
                continue
            rep = children[0]
            fused[rep] = fused.get(rep, 0.0) + self._graph_rrf_weight * _rrf_score(rank)

    def _search_one(
        self, query: str, top_k: int
    ) -> tuple[dict[str, int], dict[str, int], dict[str, dict]]:
        """Run dense + BM25 for a single query variant (called concurrently per variant)."""
        dense_ranks, payloads, _ = self._dense_search(query, top_k)
        bm25_ranks = self._bm25_search(query, top_k)
        return dense_ranks, bm25_ranks, payloads

    def _dense_search(
        self, query: str, top_k: int
    ) -> tuple[dict[str, int], dict[str, dict], float]:
        query_vector = self._embed_query(query)
        response = self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
        ranks = {hit.payload["chunk_id"]: rank for rank, hit in enumerate(hits)}
        payloads = {hit.payload["chunk_id"]: hit.payload for hit in hits}
        top_score = hits[0].score if hits else 0.0
        return ranks, payloads, top_score

    def _bm25_search(self, query: str, top_k: int) -> dict[str, int]:
        scores = self._bm25.get_scores(tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return {self._chunks[i]["id"]: rank for rank, i in enumerate(top_indices)}

    @staticmethod
    def _rrf_fuse(rank_lists: list[dict[str, int]]) -> dict[str, float]:
        """Reciprocal Rank Fusion across any number of rank lists (dense, BM25, per-variant)."""
        fused: dict[str, float] = {}
        for ranks in rank_lists:
            for cid, rank in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + _rrf_score(rank)
        return fused

    def _build_result(
        self,
        chunk_id: str,
        rrf_score: float,
        qdrant_payloads: dict[str, dict],
    ) -> dict[str, Any]:
        # Prefer Qdrant payload (has parent_text); fall back to in-memory chunk for BM25-only hits
        if chunk_id in qdrant_payloads:
            p = qdrant_payloads[chunk_id]
            text = p.get("parent_text") or p.get("text", "")
        else:
            idx = self._chunk_index[chunk_id]
            chunk = self._chunks[idx]
            parent_id = chunk.get("parent_id")
            text = (
                self._parent_texts.get(parent_id, "") if parent_id else chunk.get("text", "")
            )
            p = chunk  # field names match except chunk_id vs id (handled below)

        # Articles named in the returned text (parent section), not the embedded child
        # span — the child often omits the reference the parent carries.
        entities = set(extract_entities(text))
        # Plus the chapter's catalog articles: Laxmikanth's prose names the topic but
        # the article number lives only in the chapter's "at a Glance" table.
        if self._chapter_articles:
            entities |= self._chapter_articles.get(p.get("chapter_num"), set())

        return {
            "chunk_id": chunk_id,
            "text": text,
            "section_path": p.get("section_path", []),
            "chapter_title": p.get("chapter_title", ""),
            "chapter_num": p.get("chapter_num"),
            "part": p.get("part"),
            "page_start": p.get("page_start"),
            "page_end": p.get("page_end"),
            "entities": sorted(entities),
            "rrf_score": round(rrf_score, 6),
        }

    def _embed_query(self, query: str) -> list[float]:
        response = self._openai.embeddings.create(input=[query], model=self._embedding_model)
        return response.data[0].embedding
