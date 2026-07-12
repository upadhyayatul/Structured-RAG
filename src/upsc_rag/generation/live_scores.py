"""Attach cheap generation-quality scores to a live ``ask`` trace.

Reuses the reference-light signals from ``eval/generation.py`` (citation validity,
embedding-cosine groundedness, answer-relevance) plus free retrieval signals, and pushes
them as Langfuse scores on the request's root trace so the **Scores** tab is populated per
question. One extra ``text-embedding-3-small`` call per answer; entirely best-effort — any
failure is swallowed so scoring never breaks a response.

Off by default is a config choice: gated on ``cfg["observability"]["scores"]["enabled"]``.
"""
from __future__ import annotations

from typing import Any

from upsc_rag.eval.generation import score_answer


def record_answer_scores(
    root: Any,
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    """Compute cheap signals for (query, answer, sources) and score ``root``.

    ``root`` is the unified ``ask`` trace context (its ``.score()`` writes trace-level
    scores). No-op when scoring is disabled or ``root`` is falsy (e.g. graph/agentic
    paths that don't open a root trace).
    """
    scores_cfg = (cfg.get("observability", {}) or {}).get("scores", {}) or {}
    if not root or not scores_cfg.get("enabled", False):
        return

    try:
        with root.span("score", input={"num_sources": len(sources)}):
            # gold_articles is empty live (no labeled gold at inference) -> article_recall
            # comes back None and is skipped below.
            s = score_answer(
                query,
                answer,
                sources,
                gold_articles=[],
                embed_model=scores_cfg.get("embed_model", "text-embedding-3-small"),
                ground_threshold=scores_cfg.get("ground_threshold", 0.65),
            )

            # Deterministic / retrieval signals (always present).
            root.score("cited_fraction", float(s.cited_fraction))
            root.score("citation_valid", 0.0 if s.has_invalid_marker else 1.0)
            root.score("num_sources", float(len(sources)))
            root.score("num_sentences", float(s.num_sentences))
            if sources and sources[0].get("dense_top_score") is not None:
                root.score("retrieval_top_score", float(sources[0]["dense_top_score"]))

            # Embedding-derived signals (None when the answer had no scoreable sentences).
            if s.grounded_fraction is not None:
                root.score("grounded_fraction", float(s.grounded_fraction))
            if s.mean_support is not None:
                root.score("mean_support", float(s.mean_support))
            if s.answer_relevance is not None:
                root.score("answer_relevance", float(s.answer_relevance))
    except Exception:
        # Scoring is best-effort observability — never let it break the answer.
        pass
