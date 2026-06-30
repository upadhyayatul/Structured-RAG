"""Compile the ``ask`` LangGraph: smalltalk gate → retrieve → relevance gate → generate.

    START → smalltalk ─(smalltalk)→ END
                      └(answer)──→ retrieve → gate ─(off_topic)→ END
                                                  └(answer)────→ generate → END

The retriever and config are captured at build time, so the compiled graph is reused
across requests (built once in the API lifespan, like the direct path's retriever).
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from upsc_rag.graph.nodes import (
    make_gate_node,
    make_generate_node,
    make_retrieve_node,
    make_smalltalk_node,
)
from upsc_rag.graph.state import AskState
from upsc_rag.retrieval.hybrid import HybridRetriever


def build_ask_graph(retriever: HybridRetriever, cfg: dict[str, Any]):
    """Wire and compile the ask pipeline graph for one (retriever, cfg) pair."""
    g = StateGraph(AskState)
    g.add_node("smalltalk", make_smalltalk_node())
    g.add_node("retrieve", make_retrieve_node(retriever))
    g.add_node("gate", make_gate_node(cfg))
    g.add_node("generate", make_generate_node(cfg))

    g.add_edge(START, "smalltalk")
    # Smalltalk short-circuits straight to END; everything else goes to retrieval.
    g.add_conditional_edges(
        "smalltalk",
        lambda s: s["route"],
        {"smalltalk": END, "answer": "retrieve"},
    )
    g.add_edge("retrieve", "gate")
    # Off-topic short-circuits to END; relevant questions proceed to generation.
    g.add_conditional_edges(
        "gate",
        lambda s: s["route"],
        {"off_topic": END, "answer": "generate"},
    )
    g.add_edge("generate", END)
    return g.compile()
