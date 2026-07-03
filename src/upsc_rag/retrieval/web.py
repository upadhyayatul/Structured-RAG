"""DuckDuckGo web search — fetch the LATEST Indian-polity info the 2011 textbook can't cover.

Free, no API key (``ddgs`` package). Returns snippet-level results as
``{title, url, snippet}`` dicts. Any failure (import error, network error, rate
limit) returns ``[]`` so a search error can never break a request — the same
never-crash discipline as ``retrieval/rewrite.py``.

Used by the agentic pipeline's ``web_search`` tool (see ``agent/tools.py``); the
tool-calling LLM decides when to invoke it, gated to polity topics upstream.
"""
from __future__ import annotations

from typing import Any

from upsc_rag.observability import trace_manager


def web_search(
    query: str,
    max_results: int = 5,
    region: str = "in-en",
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return up to ``max_results`` web results for ``query`` as ``{title, url, snippet}``.

    Never raises: on any failure returns ``[]`` (retrieval/generation must proceed
    even if the web is unreachable).
    """
    try:
        from ddgs import DDGS
    except Exception:
        return []

    try:
        with trace_manager.trace(
            "web_search", input={"query": query}, session_id=session_id
        ) as trace:
            with DDGS() as ddgs:
                raw = ddgs.text(query, region=region, max_results=max_results) or []
            results = [
                {
                    "title": (r.get("title") or "").strip(),
                    "url": (r.get("href") or r.get("url") or "").strip(),
                    "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                }
                for r in raw
                if (r.get("href") or r.get("url"))
            ]
            trace.end(output={"num_results": len(results)})
        return results
    except Exception:
        return []
