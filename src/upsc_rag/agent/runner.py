"""Run helpers bridging the compiled agentic graph to the API's request/response shapes.

- ``run_ask`` drives the full graph for the non-streaming ``/ask`` endpoint.
- ``prepare_stream`` runs everything EXCEPT synthesis (smalltalk → domain gate → the
  agent ⇄ tools loop), returning the gathered textbook/web results so the API can stream
  the final synthesis via ``generate_agentic_answer_stream`` — preserving the NDJSON
  contract (only the final answer streams; the tool-use turns do not). It reuses the same
  node factories as the graph, so the loop logic stays single-sourced.
"""
from __future__ import annotations

from typing import Any

from upsc_rag.agent.nodes import (
    make_agent_node,
    make_domain_gate_node,
    make_should_continue,
    make_tools_node,
)
from upsc_rag.agent.state import AgentState
from upsc_rag.graph.nodes import make_smalltalk_node
from upsc_rag.retrieval.hybrid import HybridRetriever


def _initial_state(
    query: str,
    top_k: int | None,
    rerank_top_k: int | None,
    session_id: str | None,
    history: list[dict[str, Any]] | None,
) -> AgentState:
    return {
        "query": query,
        "history": history,
        "top_k": top_k,
        "rerank_top_k": rerank_top_k,
        "session_id": session_id,
        "messages": [],
        "iterations": 0,
        "contexts": [],
        "web": [],
    }


def run_ask(
    graph: Any,
    query: str,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
    session_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AgentState:
    """Invoke the compiled agentic graph; return the final state (answer + sources + usage)."""
    return graph.invoke(
        _initial_state(query, top_k, rerank_top_k, session_id, history)
    )


def prepare_stream(
    retriever: HybridRetriever,
    cfg: dict[str, Any],
    query: str,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
    session_id: str | None = None,
    history: list[dict[str, Any]] | None = None,
) -> AgentState:
    """Run smalltalk → domain gate → agent⇄tools loop; leave synthesis to the caller.

    Returns the partial ``AgentState``. The caller inspects ``route``:
    ``"smalltalk"``/``"off_topic"`` → emit the canned ``answer`` (no LLM synthesis);
    ``"answer"`` → stream tokens over ``contexts`` + ``web`` via
    ``generate_agentic_answer_stream``. Mirrors the compiled graph's control flow but
    drives the nodes directly so token streaming stays outside the graph.
    """
    state = _initial_state(query, top_k, rerank_top_k, session_id, history)

    state.update(make_smalltalk_node()(state))
    if state.get("route") == "smalltalk":
        return state

    state.update(make_domain_gate_node(cfg)(state))
    if state.get("route") == "off_topic":
        return state

    agent_node = make_agent_node(cfg)
    tools_node = make_tools_node(retriever, cfg)
    should_continue = make_should_continue(cfg)

    while True:
        upd = agent_node(state)
        state["messages"] = state["messages"] + upd["messages"]
        state["iterations"] = state["iterations"] + upd.get("iterations", 0)
        if should_continue(state) != "tools":
            break
        tupd = tools_node(state)
        state["messages"] = state["messages"] + tupd["messages"]
        state["contexts"] = state["contexts"] + tupd.get("contexts", [])
        state["web"] = state["web"] + tupd.get("web", [])

    return state
