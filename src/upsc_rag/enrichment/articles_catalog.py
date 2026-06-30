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

import math
import re
from typing import Any, Callable, Iterable

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


# ----------------------------------------------------------------------
# Section-level attribution (subject-matter matching)
# ----------------------------------------------------------------------
# build_chapter_article_map attributes a chapter's WHOLE article set to every one
# of its sections — coarse, and it floods generation with articles a section never
# governs. The functions below refine that: each catalog article carries a SUBJECT
# string ("Establishment and Constitution of Supreme Court"), so we embed every
# subject and every section once and attach an article to a section only when their
# vectors are close. The terse subjects rarely share tokens with section titles
# (Article 124's subject vs the "Composition and Appointment" section), so the match
# is semantic — a lexical overlap test would drop exactly the articles we care about.


def _section_doc(chunk: dict[str, Any]) -> str:
    """The text we embed to represent a section: its path plus a prose prefix."""
    path = " > ".join(chunk.get("section_path") or [])
    title = chunk.get("chapter_title", "")
    body = (chunk.get("text") or "")[:1200]
    return f"{title}. {path}. {body}".strip()


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cosine_normed(a: list[float], b: list[float]) -> float:
    """Dot product of two already-L2-normalized vectors."""
    return sum(x * y for x, y in zip(a, b))


def build_section_article_map(
    chunks: Iterable[dict[str, Any]],
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    score_threshold: float = 0.30,
    max_articles: int = 4,
    min_articles: int = 1,
) -> dict[str, list[str]]:
    """Match each parent section to its governing article(s) by subject-matter similarity.

    Returns ``{parent_section_id: ["Article 124", ...]}``. For every chapter that has
    an 'Articles ... at a Glance' catalog, embeds each article's subject and each of
    the chapter's parent sections, then attaches to a section every article whose
    subject vector clears ``score_threshold`` (capped at ``max_articles``). To protect
    recall, at least ``min_articles`` top-scoring article(s) are always attached so a
    section that owns a catalog is never left with none.

    ``embed_fn`` must return L2-comparable vectors in input order (e.g. a closure over
    ``indexing.embedder.embed_texts`` bound to the configured model).
    """
    chunk_list = list(chunks)
    chapter_articles = build_chapter_article_map(chunk_list)
    if not chapter_articles:
        return {}

    parents_by_chapter: dict[int, list[dict[str, Any]]] = {}
    for c in chunk_list:
        if c.get("content_type") != "parent":
            continue
        ch = c.get("chapter_num")
        if ch in chapter_articles:
            parents_by_chapter.setdefault(ch, []).append(c)

    # Collect every text to embed (sections + subjects) into one batched call, then
    # slice the returned vectors back out by recorded offsets.
    texts: list[str] = []
    section_spans: dict[int, tuple[int, int]] = {}   # chapter -> (start, end) in texts
    subject_spans: dict[int, tuple[int, int, list[str]]] = {}  # chapter -> (start, end, articles)

    for ch, parents in parents_by_chapter.items():
        start = len(texts)
        texts.extend(_section_doc(p) for p in parents)
        section_spans[ch] = (start, len(texts))

        articles = list(chapter_articles[ch].items())  # [(article, subject), ...]
        start = len(texts)
        texts.extend(subject for _, subject in articles)
        subject_spans[ch] = (start, len(texts), [a for a, _ in articles])

    if not texts:
        return {}

    vectors = [_norm(v) for v in embed_fn(texts)]

    out: dict[str, list[str]] = {}
    for ch, parents in parents_by_chapter.items():
        s0, s1 = section_spans[ch]
        b0, b1, articles = subject_spans[ch]
        subject_vecs = vectors[b0:b1]

        for offset, parent in enumerate(parents):
            sec_vec = vectors[s0 + offset]
            scored = sorted(
                ((_cosine_normed(sec_vec, subject_vecs[i]), articles[i]) for i in range(len(articles))),
                key=lambda t: t[0],
                reverse=True,
            )
            kept = [art for score, art in scored if score >= score_threshold][:max_articles]
            if len(kept) < min_articles:
                kept = [art for _, art in scored[:min_articles]]
            out[parent["id"]] = kept

    return out
