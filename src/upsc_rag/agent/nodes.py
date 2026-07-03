"""Nodes for the agentic ask pipeline.

Node factories capture the retriever / config and return ``(state) -> partial_state``
callables, mirroring ``graph/nodes.py``. Reused work lives in the existing modules:
``HybridRetriever.retrieve`` (textbook tool), ``retrieval/web.web_search`` (web tool),
``generation/answer.generate_agentic_answer`` (synthesis), and the smalltalk gate from
``generation/router``. The nodes add only the tool-calling orchestration.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from openai import OpenAI

from upsc_rag.agent.state import AgentState
from upsc_rag.agent.tools import (
    TOOL_SCHEMAS,
    format_textbook_digest,
    format_web_digest,
)
from upsc_rag.generation.answer import _history_messages, generate_agentic_answer
from upsc_rag.generation.router import OUT_OF_SCOPE_REPLY
from upsc_rag.generation.sources import build_agentic_sources
from upsc_rag.retrieval.hybrid import HybridRetriever
from upsc_rag.retrieval.web import web_search

Node = Callable[[AgentState], dict[str, Any]]

# The agent loop's job is only to GATHER sources with the tools — the synthesis node
# writes the final grounded, cited answer. So the agent is told NOT to answer here.
_AGENT_LOOP_SYSTEM = (
    "You are the research step of a UPSC Indian Polity assistant. Your ONLY job is to "
    "gather the sources needed to answer the user's question using the available tools. "
    "You do NOT write the final answer.\n"
    "- Call `SearchTextbook` for settled, foundational constitutional facts (this is the "
    "primary source).\n"
    "- Call `WebSearch` for information the 2011 textbook cannot contain: recent "
    "amendments, post-2011 judgments, current office-holders, and recent events.\n"
    "- You may call tools more than once (e.g. textbook first, then the web to check for "
    "recent developments), and you may call both.\n"
    "When you have gathered enough to answer, reply with the single word DONE and make no "
    "further tool calls."
)

_DOMAIN_GATE_SYSTEM = (
    "You are a scope filter for an Indian Polity study assistant that can also search the "
    "web for current affairs. Decide whether the user's message could PLAUSIBLY relate to "
    "Indian government, politics, the Constitution, law, the judiciary, Parliament, "
    "elections, public administration, constitutional or statutory bodies, policy, or "
    "current Indian political/legal affairs.\n"
    "Answer 'no' ONLY when the message is CLEARLY about an unrelated topic — e.g. cooking, "
    "sports, movies, celebrities, coding, math, personal chit-chat, or general science "
    "trivia with no governmental angle. If the message is ambiguous, awkwardly phrased, "
    "incomplete, or could reasonably concern Indian polity/governance/law (even loosely, "
    "e.g. judges, rights, officials, amendments, courts), answer 'yes'. When in doubt, "
    "answer 'yes'.\n"
    "Reply with exactly one word: 'yes' or 'no'."
)


def _to_lc_messages(history: list[dict[str, Any]] | None, turns: int) -> list[BaseMessage]:
    """Convert windowed {role, content} history turns into LangChain messages."""
    out: list[BaseMessage] = []
    for m in _history_messages(history, turns):
        if m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
        else:
            out.append(HumanMessage(content=m["content"]))
    return out


def make_domain_gate_node(cfg: dict[str, Any]) -> Node:
    """Hard gate: only Indian-polity questions may reach the tools (esp. web search).

    A cheap yes/no LLM classifier. Non-polity → canned out-of-scope reply (END). Polity →
    seed the agent's message list (system + windowed history + question) and enter the loop.
    Fails OPEN (treats as polity) on any classifier error so an API blip can't block a real
    question; the agent prompt + tool descriptions keep the tools polity-scoped as a backstop.
    """
    agent_cfg = cfg.get("agent", {})
    model = agent_cfg.get("domain_gate_model", "gpt-4.1-nano")
    turns = cfg.get("conversation", {}).get("history_turns", 3)

    def domain_gate_node(state: AgentState) -> dict[str, Any]:
        if not _is_polity(state["query"], model, state.get("session_id")):
            return {"route": "off_topic", "answer": OUT_OF_SCOPE_REPLY, "sources": []}
        messages: list[BaseMessage] = [SystemMessage(content=_AGENT_LOOP_SYSTEM)]
        messages += _to_lc_messages(state.get("history"), turns)
        messages.append(HumanMessage(content=state["query"]))
        return {"route": "answer", "messages": messages}

    return domain_gate_node


def _is_polity(query: str, model: str, session_id: str | None) -> bool:
    """Cheap LLM classifier: is ``query`` about Indian polity? Fails open (True) on error."""
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _DOMAIN_GATE_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        return (resp.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception:
        return True


def make_agent_node(cfg: dict[str, Any]) -> Node:
    """The reasoning step: an LLM with both tools bound decides what to fetch next."""
    agent_cfg = cfg.get("agent", {})
    model_name = agent_cfg.get("model", "gpt-4o-mini")
    temperature = agent_cfg.get("temperature", 0.0)

    def agent_node(state: AgentState) -> dict[str, Any]:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY"),
        ).bind_tools(TOOL_SCHEMAS)
        resp = llm.invoke(state["messages"])
        return {"messages": [resp], "iterations": 1}

    return agent_node


def make_tools_node(retriever: HybridRetriever, cfg: dict[str, Any]) -> Node:
    """Execute the tool calls on the latest AI message; record results into state.

    Custom (not the prebuilt ToolNode) because we need the structured results in
    ``state.contexts`` / ``state.web`` for the synthesis step, not just text back to the LLM.
    """
    web_cfg = cfg.get("web_search", {})
    max_results = web_cfg.get("max_results", 5)
    region = web_cfg.get("region", "in-en")

    def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        out_messages: list[BaseMessage] = []
        contexts: list[dict[str, Any]] = []
        web: list[dict[str, Any]] = []

        for call in tool_calls:
            name = call.get("name", "")
            query = (call.get("args") or {}).get("query", "") or state["query"]
            call_id = call.get("id", "")
            if name == "SearchTextbook":
                recs = retriever.retrieve(
                    query,
                    top_k=state.get("top_k"),
                    rerank_top_k=state.get("rerank_top_k"),
                    session_id=state.get("session_id"),
                )
                contexts.extend(recs)
                digest = format_textbook_digest(recs)
            elif name == "WebSearch":
                recs = web_search(
                    query,
                    max_results=max_results,
                    region=region,
                    session_id=state.get("session_id"),
                )
                web.extend(recs)
                digest = format_web_digest(recs)
            else:
                digest = f"Unknown tool: {name}"
            out_messages.append(ToolMessage(content=digest, tool_call_id=call_id))

        return {"messages": out_messages, "contexts": contexts, "web": web}

    return tools_node


def make_synthesize_node(cfg: dict[str, Any]) -> Node:
    """Write the final grounded, cited answer over the gathered textbook + web sources."""

    def synthesize_node(state: AgentState) -> dict[str, Any]:
        contexts = state.get("contexts") or []
        web = state.get("web") or []
        usage_sink: dict[str, Any] = {}
        answer = generate_agentic_answer(
            state["query"],
            contexts,
            web,
            cfg,
            session_id=state.get("session_id"),
            usage_sink=usage_sink,
            history=state.get("history"),
        )
        return {
            "answer": answer,
            "sources": build_agentic_sources(contexts, web),
            "usage": usage_sink,
        }

    return synthesize_node


def make_should_continue(cfg: dict[str, Any]) -> Callable[[AgentState], str]:
    """Conditional-edge router: loop to tools if the agent asked for a tool and we're
    under the iteration cap; otherwise proceed to synthesis."""
    max_iter = cfg.get("agent", {}).get("max_iterations", 4)

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tool = bool(getattr(last, "tool_calls", None))
        if wants_tool and state.get("iterations", 0) < max_iter:
            return "tools"
        return "synthesize"

    return should_continue
