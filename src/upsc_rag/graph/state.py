"""Shared state for the LangGraph ``ask`` pipeline.

``AskState`` is the single dict that flows through every node. Nodes read the input
fields (query + retrieval knobs) and progressively fill the output fields (route,
results, answer, sources, usage). It mirrors exactly what the FastAPI path passes
around so the two pipelines stay byte-comparable.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

# How a query was routed. Set by the smalltalk node (smalltalk|answer) and the gate
# node (off_topic|answer); the conditional edges branch on it.
Route = Literal["smalltalk", "off_topic", "answer"]


class AskState(TypedDict, total=False):
    # --- inputs ---
    query: str
    top_k: int | None
    rerank_top_k: int | None
    session_id: str | None

    # --- routing / outputs ---
    route: Route
    results: list[dict[str, Any]]      # raw HybridRetriever results (for generation)
    answer: str
    sources: list[dict[str, Any]]      # deduped, renumbered citable sources
    usage: dict[str, Any]              # token counts + estimated cost (generation only)
