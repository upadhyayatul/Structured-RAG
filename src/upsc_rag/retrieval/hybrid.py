"""Hybrid retrieval: dense (Qdrant) + BM25 fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import snowballstemmer
from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

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

        queries = self._expand_queries(query)

        # Each query variant needs an OpenAI embedding round-trip; running them
        # sequentially stacks the network latency. Fan out across a thread pool so
        # the (I/O-bound) embed + Qdrant + BM25 work for all variants overlaps.
        if len(queries) == 1:
            per_query = [self._search_one(queries[0], top_k)]
        else:
            with ThreadPoolExecutor(max_workers=len(queries)) as pool:
                per_query = list(pool.map(lambda q: self._search_one(q, top_k), queries))

        rank_lists: list[dict[str, int]] = []
        qdrant_payloads: dict[str, dict] = {}
        for dense_ranks, bm25_ranks, payloads in per_query:
            rank_lists.append(dense_ranks)
            rank_lists.append(bm25_ranks)
            qdrant_payloads.update(payloads)

        fused = self._rrf_fuse(rank_lists)
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

    def _search_one(
        self, query: str, top_k: int
    ) -> tuple[dict[str, int], dict[str, int], dict[str, dict]]:
        """Run dense + BM25 for a single query variant (called concurrently per variant)."""
        dense_ranks, payloads = self._dense_search(query, top_k)
        bm25_ranks = self._bm25_search(query, top_k)
        return dense_ranks, bm25_ranks, payloads

    def _dense_search(
        self, query: str, top_k: int
    ) -> tuple[dict[str, int], dict[str, dict]]:
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
        return ranks, payloads

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

        return {
            "chunk_id": chunk_id,
            "text": text,
            "section_path": p.get("section_path", []),
            "chapter_title": p.get("chapter_title", ""),
            "chapter_num": p.get("chapter_num"),
            "part": p.get("part"),
            "page_start": p.get("page_start"),
            "page_end": p.get("page_end"),
            "entities": p.get("entities", []),
            "rrf_score": round(rrf_score, 6),
        }

    def _embed_query(self, query: str) -> list[float]:
        response = self._openai.embeddings.create(input=[query], model=self._embedding_model)
        return response.data[0].embedding
