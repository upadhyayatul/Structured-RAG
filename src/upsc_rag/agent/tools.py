"""Tool schemas + digest formatters for the tool-calling agent.

The two tools are declared as pydantic schemas (name = class name, description =
docstring, args = fields) and bound to the LLM via ``bind_tools`` in ``agent/nodes.py``.
The LLM only *decides* which tool to call with what query; the actual work is done by
the custom tools node, which dispatches by tool name to ``HybridRetriever.retrieve`` and
``retrieval/web.web_search`` and records the structured results into graph state. Keeping
execution in the node (not in the schema classes) lets the results flow into state
instead of only back to the LLM as text.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchTextbook(BaseModel):
    """Search M. Laxmikanth's 'Indian Polity' (6th ed.) textbook for SETTLED, foundational
    Indian constitutional and political facts: constitutional provisions and Articles;
    the structure, powers, and functions of Parliament, the judiciary, the executive, and
    the states; federalism; emergency provisions; constitutional and statutory bodies; and
    established procedures. Use this FIRST for any standard, timeless polity concept."""

    query: str = Field(description="A focused search query for the textbook.")


class WebSearch(BaseModel):
    """Search the web for the LATEST or CURRENT Indian-polity information that a 2011
    textbook cannot contain: recent constitutional amendments (e.g. the 103rd, 105th,
    106th), post-2011 Supreme Court judgments (e.g. the 2017 Puttaswamy privacy ruling),
    the current holders of constitutional offices, and recent political events. Use ONLY
    for questions about Indian polity, government, or the Constitution."""

    query: str = Field(description="A focused web search query about Indian polity.")


# Bound to the LLM (order fixed) and dispatched by class name in the tools node.
TOOL_SCHEMAS = [SearchTextbook, WebSearch]


def format_textbook_digest(records: list[dict[str, Any]], max_chars: int = 500) -> str:
    """Compact digest of textbook results for the agent loop (full text goes to synthesis).

    Kept short so multi-round tool use doesn't blow the context — the synthesis step
    receives the full parent-section text via graph state, not this digest.
    """
    if not records:
        return "No textbook sections found for that query."
    lines: list[str] = []
    for i, r in enumerate(records, start=1):
        title = " > ".join(r.get("section_path") or []) or r.get("chapter_title", "Unknown")
        text = (r.get("text") or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"{i}. {title}: {text}")
    return "\n".join(lines)


def format_web_digest(records: list[dict[str, Any]]) -> str:
    """Compact digest of web results (title + snippet + url) for the agent loop."""
    if not records:
        return "No web results found for that query."
    lines: list[str] = []
    for i, r in enumerate(records, start=1):
        lines.append(
            f"{i}. {r.get('title', '')} — {r.get('snippet', '')} ({r.get('url', '')})"
        )
    return "\n".join(lines)
