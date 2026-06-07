"""Hybrid retrieval: dense (Qdrant) + BM25 fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

_RRF_K = 60  # constant from the RRF paper (Cormack et al. 2009)


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

        tokenized_corpus = [c["text"].lower().split() for c in self._chunks]
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
        """Return rerank_top_k results ranked by RRF, with parent text for generation."""
        top_k = top_k if top_k is not None else self._default_top_k
        rerank_top_k = rerank_top_k if rerank_top_k is not None else self._default_rerank_top_k

        dense_ranks, qdrant_payloads = self._dense_search(query, top_k)
        bm25_ranks = self._bm25_search(query, top_k)

        fused = self._rrf_fuse(dense_ranks, bm25_ranks)
        top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:rerank_top_k]

        return [
            self._build_result(cid, fused[cid], qdrant_payloads)
            for cid in top_ids
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return {self._chunks[i]["id"]: rank for rank, i in enumerate(top_indices)}

    @staticmethod
    def _rrf_fuse(
        dense_ranks: dict[str, int], bm25_ranks: dict[str, int]
    ) -> dict[str, float]:
        all_ids = set(dense_ranks) | set(bm25_ranks)
        fused: dict[str, float] = {}
        for cid in all_ids:
            score = 0.0
            if cid in dense_ranks:
                score += _rrf_score(dense_ranks[cid])
            if cid in bm25_ranks:
                score += _rrf_score(bm25_ranks[cid])
            fused[cid] = score
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
