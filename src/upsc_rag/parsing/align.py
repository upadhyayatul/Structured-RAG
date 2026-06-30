"""Align TOC nodes to body pages by fuzzy heading search, then extract per-section text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Any

import fitz
from upsc_rag.parsing.toc import TocNode
from upsc_rag.parsing.pdf import iter_pages


def _normalize(text: str) -> str:
    """Lowercase and map every non-alphanumeric run to a single space.

    Robust to the punctuation/encoding differences between TOC titles and body
    headings (curly apostrophes, em/en dashes, trailing periods on roman numerals),
    which an earlier whitespace-only normaliser tripped over.
    """
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def _caps_headings(page_text: str) -> list[str]:
    """Normalised text of each ALL-CAPS heading line on a page.

    Laxmikanth renders section headings in full caps (PROCEDURE FOR AMENDMENT),
    which is a far stronger anchor than 'title appears somewhere on the page' — the
    latter matches stray mentions and desyncs the whole forward scan.
    """
    out: list[str] = []
    for ln in page_text.splitlines():
        ls = ln.strip()
        if len(ls) >= 3 and ls.isupper() and any(c.isalpha() for c in ls):
            out.append(_normalize(ls))
    return out


def align_toc_with_body(
    doc: fitz.Document,
    toc_nodes: list[TocNode],
    start_page: int,
    end_page: int,
) -> None:
    """Assign ``page_start`` to every chapter (L2) and section (L3) TOC node.

    Two anchored, document-order passes (monotonic, so the same heading text in two
    chapters — e.g. 'Composition and Appointment' in both Supreme Court and High
    Court — resolves to the correct occurrence):

      A. Chapters: match the chapter title (sans leading number) as a standalone body
         line. This yields per-chapter page windows that bound section search and
         prevent a missing/odd section heading from overshooting into a later chapter.
      B. Sections: within each chapter window, match the section's ALL-CAPS heading
         line; sections whose heading wraps/differs fall back to a substring search
         *inside the window only* (safe — can't cross a chapter boundary).

    Modifies ``toc_nodes`` in place.
    """
    flat: list[TocNode] = []

    def _flatten(nodes: list[TocNode]) -> None:
        for n in nodes:
            flat.append(n)
            _flatten(n.children)

    _flatten(toc_nodes)

    # Precompute body page text and caps headings once (1-based page numbers).
    page_text: dict[int, str] = {}
    page_caps: dict[int, list[str]] = {}
    page_raw: dict[int, list[str]] = {}
    for page_num, text in iter_pages(doc, start=start_page, end=end_page):
        page_text[page_num] = text
        page_caps[page_num] = _caps_headings(text)
        page_raw[page_num] = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pages = sorted(page_text)

    chapters = [n for n in flat if n.level == 2]

    # A chapter heading equals the chapter title, optionally followed by a parenthetical
    # subtitle ('NITI Aayog (National Institution for Transforming India)') and possibly
    # WRAPPED across up to 3 physical lines ('Special Officer for Linguistic' / 'Minorities').
    # We join 1–3 consecutive lines, strip a trailing (...), and require EQUALITY — precise
    # enough that it never matches a sentence merely starting with a short title ('President'),
    # and finding the wrapped heading at its real page prevents a stray later mention from
    # mis-anchoring the chapter and cascading the monotonic scan.
    _paren = re.compile(r'\s*\([^)]*\)\s*$')

    def _page_has_heading(raw_lines: list[str], title: str) -> bool:
        for i in range(len(raw_lines)):
            for k in (1, 2, 3):
                if i + k > len(raw_lines):
                    break
                joined = " ".join(raw_lines[i:i + k])
                if _normalize(_paren.sub('', joined)) == title:
                    return True
        return False

    # --- Pass A: anchor chapters by their (number-stripped) title as a body heading ---
    cur = start_page
    for ch in chapters:
        title = _normalize(re.sub(r'^\d+\s+', '', ch.title))
        if len(title) < 4:
            continue
        for p in pages:
            if p < cur:
                continue
            if _page_has_heading(page_raw[p], title):
                ch.page_start = p
                cur = p
                break

    # --- Pass B: anchor each chapter's sections within its page window ---
    for ci, ch in enumerate(chapters):
        if ch.page_start is None:
            continue
        win_start = ch.page_start
        win_end = end_page
        for nxt in chapters[ci + 1:]:
            if nxt.page_start is not None:
                win_end = nxt.page_start - 1
                break

        sections = [c for c in ch.children if c.level == 3]

        # B1: monotonic caps-heading match inside the window.
        cur = win_start
        for sec in sections:
            nt = _normalize(sec.title)
            if not nt:
                continue
            for p in range(cur, win_end + 1):
                caps = page_caps.get(p, [])
                if nt in caps or (len(caps) >= 2 and nt in _normalize(" ".join(caps))):
                    sec.page_start = p
                    cur = p
                    break

        # B2: windowed substring fallback for sections whose heading didn't match
        # (long headings that wrap across lines, or TOC titles split mid-phrase).
        for si, sec in enumerate(sections):
            if sec.page_start is not None:
                continue
            lo = win_start
            for j in range(si - 1, -1, -1):
                if sections[j].page_start is not None:
                    lo = sections[j].page_start
                    break
            hi = win_end
            for j in range(si + 1, len(sections)):
                if sections[j].page_start is not None:
                    hi = sections[j].page_start
                    break
            nt = _normalize(sec.title)
            sec.page_start = lo  # safe default: stays within the chapter window
            for p in range(lo, hi + 1):
                if nt and nt in _normalize(page_text.get(p, "")):
                    sec.page_start = p
                    break


def fill_page_end(
    toc_nodes: list[TocNode],
    content_end_page: int,
) -> None:
    """
    Compute ``page_end`` for every node whose ``page_start`` is set.

    Each node's section runs up to and including the page before the next
    node's ``page_start``, or up to *content_end_page* for the final node.
    Modifies *toc_nodes* in-place.
    """
    flat_nodes: list[TocNode] = []

    def _flatten(nodes: list[TocNode]) -> None:
        for n in nodes:
            flat_nodes.append(n)
            _flatten(n.children)

    _flatten(toc_nodes)

    for i, node in enumerate(flat_nodes):
        if node.page_start is None:
            continue
        next_start = content_end_page
        for j in range(i + 1, len(flat_nodes)):
            if flat_nodes[j].page_start is not None:
                next_start = flat_nodes[j].page_start
                break
        # page_end is inclusive: the last page belonging to this section.
        # next_start is where the NEXT section begins, so this section ends one page before.
        node.page_end = max(node.page_start, next_start - 1)


def extract_sections(
    doc: fitz.Document,
    toc_nodes: list[TocNode],
    end_page: int
) -> Iterator[dict[str, Any]]:
    """
    Extracts text for each level-3 section using its aligned page range.

    Uses page-based extraction (page_start..page_end inclusive) rather than
    searching for headings in concatenated body text, which is fragile when
    TOC titles differ from body heading formatting.
    """
    @dataclass
    class FlatSection:
        node: TocNode
        part: str | None
        chapter_num: int | None
        chapter_title: str | None
        section_path: list[str]

    flat_sections: list[FlatSection] = []

    def _flatten_with_meta(
        nodes: list[TocNode],
        current_part: str | None,
        current_chapter_num: int | None,
        current_chapter_title: str | None,
        current_path: list[str],
    ) -> None:
        for n in nodes:
            part = current_part
            ch_num = current_chapter_num
            ch_title = current_chapter_title

            if n.level == 1:
                part = n.title
            elif n.level == 2:
                match = re.match(r'^(\d+)\s+(.+)$', n.title)
                if match:
                    ch_num = int(match.group(1))
                    ch_title = match.group(2)
                else:
                    ch_title = n.title

            path = current_path + [n.title]
            flat_sections.append(FlatSection(
                node=n,
                part=part,
                chapter_num=ch_num,
                chapter_title=ch_title,
                section_path=path,
            ))
            _flatten_with_meta(n.children, part, ch_num, ch_title, path)

    _flatten_with_meta(toc_nodes, None, None, None, [])

    for i, fsec in enumerate(flat_sections):
        node = fsec.node

        # Only extract leaf-level section nodes; PART (L1) and chapter (L2) are structural.
        if node.level != 3:
            continue

        if node.page_start is None or node.page_end is None:
            continue

        # Extract text page by page across the section's range (inclusive).
        text_parts: list[str] = []
        for p in range(node.page_start, node.page_end + 1):
            if p <= doc.page_count:
                text_parts.append(doc[p - 1].get_text("text"))

        text = "\n".join(text_parts).strip()
        if not text:
            continue

        yield {
            "section_id": f"sec_{i:04d}",
            "text": text,
            "part": fsec.part,
            "chapter_num": fsec.chapter_num,
            "chapter_title": fsec.chapter_title,
            "section_path": fsec.section_path,
            "page_start": node.page_start,
            "page_end": node.page_end,
        }
