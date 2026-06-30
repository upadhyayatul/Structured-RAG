"""Shared source-list builder: dedupe retrieval results into citable sources.

Both the FastAPI path (``api/app.py``) and the LangGraph path (``graph/``) format the
final source list the same way — deduped by section + page span, renumbered from 1 — so
that helper lives here once and is reused by both. Returns plain dicts; callers wrap them
in their own response model (e.g. the API's ``Source`` pydantic type).
"""
from __future__ import annotations

from typing import Any


def build_source_dicts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe retrieval results by section_path + page span and renumber from 1.

    Parent-text expansion can map several child chunks onto the same parent section,
    producing duplicate sources; this collapses them, preserving first-seen order.
    """
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for r in results:
        path = r.get("section_path") or []
        key = f"{' > '.join(path)}|{r.get('page_start')}-{r.get('page_end')}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "n": len(sources) + 1,
                "section_path": path,
                "chapter_title": r.get("chapter_title", ""),
                "page_start": r.get("page_start"),
                "page_end": r.get("page_end"),
            }
        )
    return sources
