"""LangGraph nodes — thin wrappers over the existing pipeline functions.

Each node is built by a small factory that captures the retriever / config, then
returns a ``(state) -> partial_state`` callable. The nodes contain NO retrieval,
ranking, or generation logic of their own — they call ``HybridRetriever.retrieve``,
``generate_answer``, and ``generation/router`` exactly as the FastAPI path does, so
behavior (and the eval numbers) stay identical. The graph only orchestrates them.
"""
from __future__ import annotations

from typing import Any, Callable

from upsc_rag.generation.answer import generate_answer
from upsc_rag.generation.router import (
    OUT_OF_SCOPE_REPLY,
    is_off_topic,
    smalltalk_reply,
)
from upsc_rag.generation.sources import build_source_dicts
from upsc_rag.graph.state import AskState
from upsc_rag.llm.clients import langchain_backend_enabled
from upsc_rag.retrieval.hybrid import HybridRetriever

Node = Callable[[AskState], dict[str, Any]]


def make_smalltalk_node() -> Node:
    """Gate 1: short-circuit pure greetings/chit-chat with a canned reply (no retrieval)."""

    def smalltalk_node(state: AskState) -> dict[str, Any]:
        canned = smalltalk_reply(state["query"])
        if canned is not None:
            return {"route": "smalltalk", "answer": canned, "sources": []}
        return {"route": "answer"}

    return smalltalk_node


def make_retrieve_node(retriever: HybridRetriever) -> Node:
    """Run hybrid retrieval; stash raw results for the gate + generate nodes."""

    def retrieve_node(state: AskState) -> dict[str, Any]:
        results = retriever.retrieve(
            state["query"],
            top_k=state.get("top_k"),
            rerank_top_k=state.get("rerank_top_k"),
            session_id=state.get("session_id"),
        )
        return {"results": results}

    return retrieve_node


def make_gate_node(cfg: dict[str, Any]) -> Node:
    """Gate 2: real question but no sufficiently-relevant source — return out-of-scope."""
    floor = cfg.get("retrieval", {}).get("relevance_floor", 0.0)

    def gate_node(state: AskState) -> dict[str, Any]:
        results = state.get("results") or []
        if is_off_topic(results, floor):
            return {"route": "off_topic", "answer": OUT_OF_SCOPE_REPLY, "sources": []}
        return {"route": "answer"}

    return gate_node


def make_generate_node(cfg: dict[str, Any]) -> Node:
    """Generate the grounded, cited answer and build the deduped source list."""

    def generate_node(state: AskState) -> dict[str, Any]:
        results = state.get("results") or []
        usage_sink: dict[str, Any] = {}
        if langchain_backend_enabled():
            answer = _generate_langchain(state["query"], results, cfg, usage_sink)
        else:
            answer = generate_answer(
                state["query"],
                results,
                cfg,
                session_id=state.get("session_id"),
                usage_sink=usage_sink,
            )
        return {
            "answer": answer,
            "sources": build_source_dicts(results),
            "usage": usage_sink,
        }

    return generate_node


def _generate_langchain(
    query: str,
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
    usage_sink: dict[str, Any],
) -> str:
    """ChatOpenAI variant of generate_answer (portability seam, off by default).

    Reuses the exact same system + user prompts as the OpenAI-SDK path so the answer is
    equivalent; fills ``usage_sink`` from LangChain's usage metadata.
    """
    from upsc_rag.generation.answer import _SYSTEM_PROMPT, build_answer_prompt, estimate_cost
    from upsc_rag.llm.clients import get_chat_model

    model = get_chat_model(cfg)
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", build_answer_prompt(query, results)),
    ]
    resp = model.invoke(messages)
    meta = getattr(resp, "usage_metadata", None) or {}
    in_tok = meta.get("input_tokens", 0)
    out_tok = meta.get("output_tokens", 0)
    model_name = cfg.get("generation", {}).get("model", "gpt-4o-mini")
    usage_sink["input_tokens"] = in_tok
    usage_sink["output_tokens"] = out_tok
    usage_sink["cost_usd"] = estimate_cost(model_name, in_tok, out_tok)
    return resp.content or ""
