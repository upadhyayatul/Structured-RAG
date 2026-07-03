"""Shared state for the agentic (tool-calling) ask pipeline.

Unlike the linear ``graph`` pipeline's single-pass ``AskState`` (whose nodes overwrite
their keys), the agent runs a ReAct loop that must *accumulate*: the message list grows
each turn, and every tool call appends more textbook/web results. So the loop fields use
reducers — ``add_messages`` for the chat history and ``operator.add`` for the result
lists and the iteration counter.

Flow: smalltalk gate → polity-domain gate → (agent ⇄ tools)* → synthesize.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

# smalltalk → END (canned), off_topic → END (out-of-scope), answer → enter the agent loop.
Route = Literal["smalltalk", "off_topic", "answer"]


class AgentState(TypedDict, total=False):
    # --- inputs ---
    query: str
    history: list[dict[str, Any]] | None  # prior {role, content} turns
    top_k: int | None
    rerank_top_k: int | None
    session_id: str | None

    # --- agent loop (accumulating) ---
    messages: Annotated[list, add_messages]        # LC messages: system/human/AI/tool
    iterations: Annotated[int, operator.add]       # agent turns taken (loop guard)
    contexts: Annotated[list[dict[str, Any]], operator.add]  # textbook results gathered
    web: Annotated[list[dict[str, Any]], operator.add]       # web results gathered

    # --- routing / outputs ---
    route: Route
    answer: str
    sources: list[dict[str, Any]]  # combined book + web citable sources
    usage: dict[str, Any]          # token counts + estimated cost (synthesis only)
