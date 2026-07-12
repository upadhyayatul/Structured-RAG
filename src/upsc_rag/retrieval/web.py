"""DuckDuckGo web search — fetch the LATEST Indian-polity info the 2011 textbook can't cover.

Free, no API key (``ddgs`` package). Returns snippet-level results as
``{title, url, snippet}`` dicts. Any failure (import error, network error, rate
limit) returns ``[]`` so a search error can never break a request — the same
never-crash discipline as ``retrieval/rewrite.py``.

Used by the agentic pipeline's ``web_search`` tool (see ``agent/tools.py``); the
tool-calling LLM decides when to invoke it, gated to polity topics upstream.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from upsc_rag.observability import trace_manager


def web_search_multi(
    queries: list[str],
    max_results: int = 5,
    region: str = "in-en",
    session_id: str | None = None,
    parent: Any = None,
) -> list[dict[str, Any]]:
    """Search several phrasings in parallel and merge them, round-robin, deduped by URL.

    A search engine is far more literal than a vector index: sent a user's question
    verbatim ("who is the current CJI and UPSC chairmain") it returns SEO filler, because
    the abbreviation is opaque, the typo is real, and two questions are mashed into one.
    Callers therefore pass keyword-y rewrites (``retrieval/rewrite.py``), which also split
    a compound question into one query per part.

    Round-robin (rather than concatenating) is what guarantees EVERY sub-question is
    represented once the merged list is truncated to ``max_results`` — otherwise the first
    query's results would fill the budget and the second half of the question would go
    unanswered.
    """
    if not queries:
        return []
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        per_query = list(
            pool.map(
                lambda q: web_search(q, max_results, region, session_id, parent), queries
            )
        )

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank in range(max_results):
        for results in per_query:
            if rank >= len(results):
                continue
            url = results[rank].get("url", "")
            if url and url not in seen:
                seen.add(url)
                merged.append(results[rank])
    return merged[:max_results]


def web_search(
    query: str,
    max_results: int = 5,
    region: str = "in-en",
    session_id: str | None = None,
    parent: Any = None,
) -> list[dict[str, Any]]:
    """Return up to ``max_results`` web results for ``query`` as ``{title, url, snippet}``.

    Never raises: on any failure returns ``[]`` (retrieval/generation must proceed
    even if the web is unreachable). ``parent`` nests the search under a request-level
    root trace, so a web fallback is visible inside the question's single ``ask`` tree.
    """
    try:
        from ddgs import DDGS
    except Exception:
        return []

    try:
        with trace_manager.start(
            "web_search", parent=parent, input={"query": query}, session_id=session_id
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
