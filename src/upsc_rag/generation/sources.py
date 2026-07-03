"""Shared source-list builder: dedupe retrieval results into citable sources.

Both the FastAPI path (``api/app.py``) and the LangGraph path (``graph/``) format the
final source list the same way — deduped by section + page span, renumbered from 1 — so
that helper lives here once and is reused by both. Returns plain dicts; callers wrap them
in their own response model (e.g. the API's ``Source`` pydantic type).
"""
from __future__ import annotations

from typing import Any


def dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the full result dicts deduped by section_path + page span, order preserved.

    Parent-text expansion can map several child chunks onto the same parent section;
    this collapses them while keeping the full dict (text, entities, ...) so callers
    that need the source *content* (e.g. the prompt builder) stay aligned with the
    numbering ``build_source_dicts`` produces from the same list.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in results:
        path = r.get("section_path") or []
        key = f"{' > '.join(path)}|{r.get('page_start')}-{r.get('page_end')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_source_dicts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe retrieval results by section_path + page span and renumber from 1.

    Each source is tagged ``type: "book"`` so the response model can carry both
    textbook and web sources in one list (the agentic path adds web sources).
    """
    sources: list[dict[str, Any]] = []
    for r in dedupe_results(results):
        sources.append(
            {
                "n": len(sources) + 1,
                "type": "book",
                "section_path": r.get("section_path") or [],
                "chapter_title": r.get("chapter_title", ""),
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
            }
        )
    return sources


def build_web_source_dicts(
    web: list[dict[str, Any]], start_n: int = 0
) -> list[dict[str, Any]]:
    """Turn web results into citable sources, deduped by URL, numbered from ``start_n + 1``.

    Each source is tagged ``type: "web"`` and carries title/url/snippet instead of the
    section/page fields a book source has.
    """
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for w in web:
        url = w.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append(
            {
                "n": start_n + len(sources) + 1,
                "type": "web",
                "title": w.get("title", ""),
                "url": url,
                "snippet": w.get("snippet", ""),
            }
        )
    return sources


def build_agentic_sources(
    contexts: list[dict[str, Any]], web: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Combined, consistently-numbered source list: textbook sources first, then web.

    Numbering matches ``build_agentic_prompt`` (answer.py), which formats the same
    deduped textbook results first and appends the same URL-deduped web results.
    """
    book = build_source_dicts(contexts)
    return book + build_web_source_dicts(web, start_n=len(book))
