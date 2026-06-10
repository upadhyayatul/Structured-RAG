"""Parse Laxmikanth's 'Articles Related to <topic> at a Glance' tables.

Every chapter ends with a catalog table mapping each Constitutional Article to its
subject matter, e.g.::

    Table 26.2 Articles Related to Supreme Court at a Glance
    Article No.    Subject Matter
    124.           Establishment and Constitution of Supreme Court
    125.           Salaries, etc., of Judges
    ...

PDF extraction strips the word "Article", leaving bare ``124.`` lines that the
entity regex (which looks for "Article 124") never matches. As a result the
article number is absent from the chapter's prose sections AND from these catalogs
as far as entity extraction is concerned — so a question like "how is an SC judge
appointed?" can never surface "Article 124".

This module recovers the catalog as a ``chapter_num -> {"Article 124": subject}``
map, letting a chapter's governing articles be attributed back to its sections.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# "Articles Related to Supreme Court at a Glance" (case-insensitive, flexible spacing)
_HEADER_RE = re.compile(r"Articles?\s+Related\s+to\s+.*?\bat\s+a\s+Glance", re.I)
# A catalog row's article cell: a number (optionally with a letter suffix) alone on
# its line, followed by a period — e.g. "124.", "124A.", "131A.".
_NUM_RE = re.compile(r"^\s*(\d{1,3}[A-Z]?)\.\s*$")
# Lines that mark the end of the catalog block (a following table or notes section).
_STOP_RE = re.compile(r"^\s*(Table\s+\d|Notes?\s+and\s+References)\b", re.I)
_COL_HEADERS = {"article no.", "subject matter", "subject-matter"}


def parse_catalog(text: str) -> dict[str, str]:
    """Return ``{"Article 124": "Establishment and Constitution ...", ...}`` for one table.

    Parses the first 'Articles Related to ... at a Glance' block found in ``text``.
    Returns an empty dict if no such table is present.
    """
    lines = text.splitlines()
    start: int | None = None
    for i, ln in enumerate(lines):
        if _HEADER_RE.search(ln):
            start = i + 1
            break
    if start is None:
        return {}

    out: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal cur, buf
        if cur:
            subj = " ".join(s.strip() for s in buf if s.strip())
            if subj:
                out[f"Article {cur}"] = subj
        cur, buf = None, []

    for ln in lines[start:]:
        if _STOP_RE.match(ln):
            flush()
            break
        m = _NUM_RE.match(ln)
        if m:
            flush()
            cur = m.group(1)
            continue
        if cur is not None and ln.strip().lower() not in _COL_HEADERS:
            buf.append(ln)
    flush()
    return out


def build_chapter_article_map(
    chunks: Iterable[dict[str, Any]]
) -> dict[int, dict[str, str]]:
    """Scan parent chunks for catalog tables, grouped by ``chapter_num``.

    Returns ``{chapter_num: {"Article 124": subject, ...}}``. A chapter whose catalog
    is split across chunks has its entries merged.
    """
    out: dict[int, dict[str, str]] = {}
    for c in chunks:
        if c.get("content_type") != "parent":
            continue
        text = c.get("text", "")
        if not _HEADER_RE.search(text):
            continue
        chapter = c.get("chapter_num")
        if chapter is None:
            continue
        parsed = parse_catalog(text)
        if parsed:
            out.setdefault(chapter, {}).update(parsed)
    return out
