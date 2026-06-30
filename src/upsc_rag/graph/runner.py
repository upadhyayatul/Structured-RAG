"""Run helpers bridging the compiled graph to the API's request/response shapes.

- ``run_ask`` drives the full graph for the non-streaming ``/ask`` endpoint.
- ``prepare_stream`` runs only the routing + retrieval prefix (smalltalk → retrieve →
  gate) so the API can then stream generation tokens via the existing
  ``generate_answer_stream`` — preserving the NDJSON contract the frontend depends on.
  It reuses the SAME node factories as the graph, so the prefix logic stays single-sourced.
"""
from __future__ import annotations

from typing import Any

from upsc_rag.graph.nodes import make_gate_node, make_retrieve_node, make_smalltalk_node
from upsc_rag.graph.state import AskState


def run_ask(
    graph: Any,
    query: str,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
    session_id: str | None = None,
) -> AskState:
    """Invoke the compiled graph and return the final state (answer + sources + usage)."""
    initial: AskState = {
        "query": query,
        "top_k": top_k,
        "rerank_top_k": rerank_top_k,
        "session_id": session_id,
    }
    return graph.invoke(initial)


def prepare_stream(
    retriever: Any,
    cfg: dict[str, Any],
    query: str,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
    session_id: str | None = None,
) -> AskState:
    """Run the smalltalk → retrieve → gate prefix; leave token generation to the caller.

    Returns the partial ``AskState``. The caller inspects ``route``:
    ``"smalltalk"``/``"off_topic"`` → emit the canned ``answer`` (no LLM); ``"answer"``
    → stream tokens from ``results`` via ``generate_answer_stream``.
    """
    state: AskState = {
        "query": query,
        "top_k": top_k,
        "rerank_top_k": rerank_top_k,
        "session_id": session_id,
    }

    state.update(make_smalltalk_node()(state))
    if state["route"] == "smalltalk":
        return state

    state.update(make_retrieve_node(retriever)(state))
    state.update(make_gate_node(cfg)(state))
    return state
