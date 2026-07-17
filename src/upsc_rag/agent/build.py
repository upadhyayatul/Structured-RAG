"""Compile the agentic ask graph — a tool-calling ReAct loop with a polity gate.

    START → smalltalk ─(smalltalk)→ END
                      └(answer)→ domain_gate ─(off_topic)→ END
                                             └(answer)→ agent ⇄ tools ─→ synthesize → END


The agent ⇄ tools cycle is the ReAct loop: the agent (LLM with tools bound) decides
whether to fetch more, the tools node executes and records results, and control returns
to the agent until it stops requesting tools (or the iteration cap trips) — then the
synthesize node writes the final grounded answer. Compiled once per (retriever, cfg).
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from upsc_rag.agent.nodes import (
    make_agent_node,
    make_domain_gate_node,
    make_should_continue,
    make_synthesize_node,
    make_tools_node,
)
from upsc_rag.agent.state import AgentState
from upsc_rag.graph.nodes import make_smalltalk_node  # reuse the regex smalltalk gate
from upsc_rag.retrieval.hybrid import HybridRetriever


def build_agentic_graph(retriever: HybridRetriever, cfg: dict[str, Any]):
    """Wire and compile the agentic pipeline graph for one (retriever, cfg) pair."""
    g = StateGraph(AgentState)
    g.add_node("smalltalk", make_smalltalk_node())
    g.add_node("domain_gate", make_domain_gate_node(cfg))
    g.add_node("agent", make_agent_node(cfg))
    g.add_node("tools", make_tools_node(retriever, cfg))
    g.add_node("synthesize", make_synthesize_node(cfg))

    g.add_edge(START, "smalltalk")
    g.add_conditional_edges(
        "smalltalk",
        lambda s: s["route"],
        {"smalltalk": END, "answer": "domain_gate"},
    )
    g.add_conditional_edges(
        "domain_gate",
        lambda s: s["route"],
        {"off_topic": END, "answer": "agent"},
    )
    g.add_conditional_edges(
        "agent",
        make_should_continue(cfg),
        {"tools": "tools", "synthesize": "synthesize"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("synthesize", END)
    return g.compile()
