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

from upsc_rag.chunking.structured import extract_entities
from upsc_rag.llm.clients import get_openai_client
from upsc_rag.observability import NOOP_CONTEXT, trace_manager

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
        # Off-topic floor: a first pass below this is junk/out-of-scope and will be
        # rejected by the caller's relevance gate — so don't waste a rewrite on it.
        self._relevance_floor: float = retrieval_cfg.get("relevance_floor", 0.0)

        # Section-level article catalog: maps parent_id -> {"Article 124", ...}, matched
        # by subject similarity from each chapter's "Articles ... at a Glance" table.
        # Used to attribute a section's governing articles (whose prose names the topic
        # but not the article number). Built below once chunks are loaded.
        catalog_cfg = retrieval_cfg.get("catalog", {})
        self._catalog_enabled: bool = bool(catalog_cfg.get("enabled", False))
        self._catalog_threshold: float = catalog_cfg.get("score_threshold", 0.30)
        self._catalog_max_articles: int = catalog_cfg.get("max_articles", 4)
        self._catalog_min_articles: int = catalog_cfg.get("min_articles", 1)
        self._section_articles: dict[str, set[str]] = {}      # parent_id -> articles

        # Cross-encoder reranker: re-score (query, full-section-text) jointly over a
        # widened deduped candidate pool, then truncate to rerank_top_k. Fixes the
        # bi-encoder's sibling-section confusion (right chapter, wrong section first).
        rerank_cfg = retrieval_cfg.get("rerank", {})
        self._rerank_enabled: bool = bool(rerank_cfg.get("enabled", False))
        self._rerank_model: str = rerank_cfg.get("model", "ms-marco-MiniLM-L-12-v2")
        self._rerank_pool: int = rerank_cfg.get("candidate_pool", 25)
        self._rerank_max_chars: int = rerank_cfg.get("max_chars", 2000)
        self._rerank_weight: float = rerank_cfg.get("weight", 0.7)
        self._reranker = None  # constructed lazily on first use
        if self._rerank_enabled:
            from upsc_rag.retrieval.rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(
                model_name=self._rerank_model, max_chars=self._rerank_max_chars
            )

        self._qdrant = QdrantClient(url=indexing_cfg["qdrant_url"])
        # Embeddings stay on the direct OpenAI endpoint (dimension-locked to Qdrant,
        # deliberately NOT routed through the LiteLLM gateway).
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # Query rewrite is a chat call, so it goes through the gateway when enabled.
        self._chat_client = get_openai_client()

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

        if self._catalog_enabled:
            self._build_catalog_attribution(all_chunks, chunks_path)

    # ------------------------------------------------------------------
    # Catalog attribution (section-precise)
    # ------------------------------------------------------------------

    def _build_catalog_attribution(
        self, all_chunks: list[dict[str, Any]], chunks_path: Path
    ) -> None:
        """Populate the section-precise article map.

        Matches each section to only its governing article(s) by subject similarity;
        the result is cached next to chunks.jsonl so the one-off embedding pass runs
        only when chunks or params change.
        """
        from upsc_rag.enrichment.articles_catalog import build_section_article_map

        cache_path = chunks_path.parent / "section_articles.json"
        meta = {
            "mode": "embedding",
            "model": self._embedding_model,
            "threshold": self._catalog_threshold,
            "max_articles": self._catalog_max_articles,
            "min_articles": self._catalog_min_articles,
            "chunks_mtime": chunks_path.stat().st_mtime,
        }
        cached = self._load_section_cache(cache_path, meta)
        if cached is not None:
            self._section_articles = cached
            return

        from upsc_rag.indexing.embedder import embed_texts

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return embed_texts(texts, model=self._embedding_model, batch_size=200)

        section_map = build_section_article_map(
            all_chunks,
            embed_fn,
            score_threshold=self._catalog_threshold,
            max_articles=self._catalog_max_articles,
            min_articles=self._catalog_min_articles,
        )
        self._section_articles = {pid: set(arts) for pid, arts in section_map.items()}
        self._write_section_cache(cache_path, meta, section_map)

    @staticmethod
    def _load_section_cache(
        path: Path, meta: dict[str, Any]
    ) -> dict[str, set[str]] | None:
        """Return the cached section->articles map iff its meta matches, else None."""
        if not path.exists():
            return None
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if blob.get("meta") != meta:
            return None
        return {pid: set(arts) for pid, arts in blob.get("map", {}).items()}

    @staticmethod
    def _write_section_cache(
        path: Path, meta: dict[str, Any], section_map: dict[str, list[str]]
    ) -> None:
        payload = {"meta": meta, "map": section_map}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        session_id: str | None = None,
        parent: Any = None,
    ) -> list[dict[str, Any]]:
        """Return rerank_top_k results ranked by RRF, with parent text for generation.

        When query rewriting is enabled, retrieves for each query variant and fuses
        all dense+BM25 rank lists together (multi-query RRF) to improve recall.

        ``session_id`` (optional) groups this trace with the answer trace for the same
        question under one Langfuse Session. Pass ``parent`` to nest this as a child
        span under a request-level root trace instead of emitting its own root trace.
        """
        top_k = top_k if top_k is not None else self._default_top_k
        rerank_top_k = rerank_top_k if rerank_top_k is not None else self._default_rerank_top_k

        with trace_manager.start(
            "retrieve",
            parent=parent,
            input={"query": query, "top_k": top_k, "rerank_top_k": rerank_top_k},
            session_id=session_id,
        ) as trace:

            # First pass: original query only (one embed). Cheap, and often enough.
            with trace.span("first_pass") as fp:
                dense_ranks, payloads, top_score = self._dense_search(query, top_k, obs=fp)
                rank_lists: list[dict[str, int]] = [dense_ranks, self._bm25_search(query, top_k, obs=fp)]
                qdrant_payloads: dict[str, dict] = dict(payloads)

            # Gate: expand with rewrite variants only when the first pass is weak BUT
            # still plausibly on-topic. A score below the relevance floor is off-topic
            # and will be rejected downstream, so skip the costly rewrite there.
            #
            # The lower bound is load-bearing, and NOT (as it looks) a mere cost saving.
            # Rewriting a sub-floor query and letting its variants raise the score would
            # break the off-topic gate: the rewrite LLM is instructed to emit *polity*
            # queries, so for "best recipe for pasta carbonara" it duly produces
            # polity-flavoured variants that match real sections (measured: first pass
            # 0.09 -> 0.43, sailing past the 0.30 floor). The gate would then be asking
            # "can the rewriter find anything in the book?" — to which the answer is
            # always yes — instead of "is the user's question in the book?".
            # Abbreviations that sink a query below the floor ("from where was the FD
            # taken?") are resolved upstream in generation/condense.py instead.
            rewrite_fired = False
            if (
                self._rewrite_enabled
                and self._relevance_floor <= top_score < self._rewrite_score_threshold
            ):
                with trace.span("rewrite", input={"top_score": top_score}) as rw:
                    variants = [v for v in self._expand_queries(query, obs=rw) if v != query]
                if variants:
                    rewrite_fired = True
                    with trace.span("variant_search", input={"num_variants": len(variants)}) as vs:
                        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
                            for d_ranks, b_ranks, pls in pool.map(
                                lambda q: self._search_one(q, top_k, obs=vs), variants
                            ):
                                rank_lists.append(d_ranks)
                                rank_lists.append(b_ranks)
                                qdrant_payloads.update(pls)

            with trace.span("rrf_fusion", input={"num_lists": len(rank_lists)}):
                fused = self._rrf_fuse(rank_lists)

            # Dedupe to DISTINCT parent sections: children of one section share a
            # parent_id and otherwise flood top-k (e.g. 7 'Notes and References' child
            # chunks crowding out the section that actually answers the query). Keep the
            # highest-fused child per parent — the parent-text expansion returns the same
            # section text regardless of which child won.
            #
            # When reranking is on we keep a WIDER deduped pool (candidate_pool) so a
            # section RRF buried at e.g. #20 can still be promoted by the cross-encoder;
            # otherwise we stop at rerank_top_k (RRF order is final).
            limit = self._rerank_pool if (self._reranker is not None) else rerank_top_k
            ranked = sorted(fused, key=lambda cid: fused[cid], reverse=True)
            top_ids: list[str] = []
            seen_parents: set[str] = set()
            for cid in ranked:
                pid = self._parent_of(cid, qdrant_payloads) or cid
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)
                top_ids.append(cid)
                if len(top_ids) >= limit:
                    break
            results = [self._build_result(cid, fused[cid], qdrant_payloads) for cid in top_ids]

            # Cross-encoder rerank: re-score (query, full section text) jointly and
            # reorder, then truncate to rerank_top_k. RRF (bi-encoder) can't separate
            # sibling sections; the cross-encoder reads both texts together and can.
            if self._reranker is not None and results:
                with trace.span("rerank", input={"candidates": len(results)}) as rr:
                    results = self._rerank_results(query, results, rerank_top_k, obs=rr)
            else:
                results = results[:rerank_top_k]

            # Expose the original query's top dense-cosine score on every result so
            # callers can apply a relevance floor (off-topic gate) without re-embedding.
            for r in results:
                r["dense_top_score"] = round(top_score, 6)

            trace.end(output={
                "top_score": round(top_score, 4),
                "rewrite_fired": rewrite_fired,
                "results_returned": len(results),
            })
            return results

    def _expand_queries(self, query: str, obs: Any = NOOP_CONTEXT) -> list[str]:
        """Return [query] or, if rewriting is enabled, the original plus LLM variants."""
        if not self._rewrite_enabled:
            return [query]
        from upsc_rag.retrieval.rewrite import rewrite_query

        return rewrite_query(
            query, self._chat_client, model=self._rewrite_model,
            num_variants=self._rewrite_num_variants, obs=obs,
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

    def _search_one(
        self, query: str, top_k: int, obs: Any = NOOP_CONTEXT
    ) -> tuple[dict[str, int], dict[str, int], dict[str, dict]]:
        """Run dense + BM25 for a single query variant (called concurrently per variant).

        The variant's own dense top score is deliberately dropped: the relevance gate
        must judge the USER's query, not the best score a rewrite could manufacture
        (see the rewrite gate in ``retrieve``).
        """
        dense_ranks, payloads, _ = self._dense_search(query, top_k, obs=obs)
        bm25_ranks = self._bm25_search(query, top_k, obs=obs)
        return dense_ranks, bm25_ranks, payloads

    def _dense_search(
        self, query: str, top_k: int, obs: Any = NOOP_CONTEXT
    ) -> tuple[dict[str, int], dict[str, dict], float]:
        # Embedding is a billable model call — trace it as a generation so Langfuse
        # records its token usage and computes cost (like the answer generation).
        embed = obs.generation("embed_query", model=self._embedding_model, input=query)
        with embed:
            query_vector, usage = self._embed_query(query)
            embed.end(usage=usage)
        with obs.span("qdrant_search", input={"top_k": top_k}) as s:
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
            s.end(output={"hits": len(hits), "top_score": round(top_score, 4)})
        return ranks, payloads, top_score

    def _bm25_search(self, query: str, top_k: int, obs: Any = NOOP_CONTEXT) -> dict[str, int]:
        with obs.span("bm25_search", input={"top_k": top_k}):
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
        # Plus the section's catalog articles: Laxmikanth's prose names the topic but
        # the article number lives only in the chapter's "at a Glance" table.
        if self._section_articles:
            entities |= self._section_articles.get(p.get("parent_id"), set())

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

    def _rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        rerank_top_k: int,
        obs: Any = NOOP_CONTEXT,
    ) -> list[dict[str, Any]]:
        """Blend the cross-encoder score with the RRF score, reorder, truncate.

        Scores each result's full section ``text`` against the query with the
        cross-encoder, then combines it with the incoming RRF score as
        ``weight*rerank + (1-weight)*rrf`` after min-max normalizing BOTH over the
        candidate pool. Blending (rather than replacing the order) keeps the fusion
        signal, so the cross-encoder refines ranking without catastrophically
        demoting a strong RRF hit or dropping it out of top-k. Attaches
        ``rerank_score`` and ``blend_score`` to each result; returns top
        ``rerank_top_k``. On any reranker failure, falls back to RRF order.
        """
        candidates = [(r["chunk_id"], r.get("text", "")) for r in results]
        try:
            scored = self._reranker.rerank(query, candidates)
        except Exception as exc:  # pragma: no cover — degrade gracefully, keep RRF order
            obs.end(output={"error": str(exc), "fell_back": True})
            return results[:rerank_top_k]

        rerank_by_id = {cid: score for cid, score in scored}
        rrf_by_id = {r["chunk_id"]: r.get("rrf_score", 0.0) for r in results}
        norm_rerank = self._minmax(rerank_by_id)
        norm_rrf = self._minmax(rrf_by_id)

        w = self._rerank_weight
        for r in results:
            cid = r["chunk_id"]
            r["rerank_score"] = round(float(rerank_by_id.get(cid, 0.0)), 6)
            r["blend_score"] = round(
                w * norm_rerank.get(cid, 0.0) + (1.0 - w) * norm_rrf.get(cid, 0.0), 6
            )
        reordered = sorted(results, key=lambda r: r["blend_score"], reverse=True)
        obs.end(output={"reranked": len(reordered), "kept": min(rerank_top_k, len(reordered))})
        return reordered[:rerank_top_k]

    @staticmethod
    def _minmax(scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalize a {id: score} map to [0, 1]; flat input maps to all 1.0."""
        if not scores:
            return {}
        vals = scores.values()
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {k: 1.0 for k in scores}
        span = hi - lo
        return {k: (v - lo) / span for k, v in scores.items()}

    def _embed_query(self, query: str) -> tuple[list[float], dict[str, int]]:
        """Return the query embedding plus token usage (for cost tracing)."""
        response = self._openai.embeddings.create(input=[query], model=self._embedding_model)
        usage = {"input": response.usage.prompt_tokens, "total": response.usage.total_tokens}
        return response.data[0].embedding, usage
