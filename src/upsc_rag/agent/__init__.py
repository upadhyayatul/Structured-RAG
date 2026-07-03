"""Agentic (tool-calling) ask pipeline — a ReAct loop over textbook + web search.

Parallel to the linear ``graph`` package. Selected via ``UPSC_RAG_PIPELINE=agentic``.
A tool-calling LLM decides, per polity question, whether to search the Laxmikanth
textbook, the web (for post-2011 / current info), or both, then a synthesis step writes
a grounded, cited answer that may draw on either.
"""
from upsc_rag.agent.build import build_agentic_graph
from upsc_rag.agent.runner import prepare_stream, run_ask
from upsc_rag.agent.state import AgentState

__all__ = ["build_agentic_graph", "run_ask", "prepare_stream", "AgentState"]
