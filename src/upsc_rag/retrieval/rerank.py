"""Cross-encoder reranking via FlashRank (ONNX runtime, CPU, no torch).

The hybrid retriever fuses a *bi-encoder* (Qdrant dense) with BM25 and orders by
RRF. A bi-encoder embeds the query and each passage *separately*, so two sibling
sections in the same chapter — near-identical in vector space — are hard to tell
apart, and the wrong one often ranks first. A *cross-encoder* reads the question
and the full section text *jointly* and emits one relevance score, which separates
those lookalikes. We use it as a second stage: re-score a widened candidate pool of
deduped sections, then keep the top rerank_top_k.

FlashRank runs the same MS-MARCO cross-encoder weights as sentence-transformers but
through onnxruntime, so there is no torch dependency — important for this project's
fragile cloned venv. The model (~22MB) is downloaded and cached on first use.
"""
from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    """Lazy FlashRank wrapper: re-scores (query, passage) pairs jointly.

    The FlashRank model is loaded on first ``rerank()`` call, not at construction,
    so importing/constructing a retriever with rerank *disabled* pays nothing, and
    the one-off model load (~3-4s) happens only when reranking actually runs.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", max_chars: int = 2000) -> None:
        self._model_name = model_name
        self._max_chars = max_chars
        self._ranker: Any = None  # built lazily

    def _ensure_ranker(self) -> Any:
        if self._ranker is None:
            from flashrank import Ranker

            self._ranker = Ranker(model_name=self._model_name)
        return self._ranker

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """Return ``[(id, score), ...]`` sorted by cross-encoder relevance, descending.

        ``candidates`` is ``[(id, text), ...]``. Texts are truncated to ``max_chars``
        (the model only reads ~512 tokens anyway) to keep latency flat on long
        parent sections. An empty candidate list returns ``[]``.
        """
        if not candidates:
            return []

        from flashrank import RerankRequest

        passages = [
            {"id": cid, "text": (text or "")[: self._max_chars]}
            for cid, text in candidates
        ]
        ranked = self._ensure_ranker().rerank(RerankRequest(query=query, passages=passages))
        # FlashRank returns dicts sorted by descending score.
        return [(str(item["id"]), float(item["score"])) for item in ranked]
