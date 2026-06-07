"""Parse the printed Contents pages into a TocNode tree (PART → Chapter → Section)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from upsc_rag.parsing.pdf import extract_page_text, open_pdf


@dataclass
class TocNode:
    """One node in the TOC tree; page_start/page_end are filled later by align.py."""

    title: str
    level: int
    children: list[TocNode] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


_PART_RE = re.compile(r"^PART[- ]([IVXLC]+)\s*$", re.I)
# Cap at 3 digits so 4-digit years like "2019" are not mistaken for chapter numbers.
_CHAPTER_RE = re.compile(r"^(\d{1,3})\s+(.+)$")

# Words that cannot naturally end a complete phrase; their presence signals a wrapped line.
_HANGING_ENDINGS = frozenset({
    "a", "an", "the", "of", "in", "and", "or", "to", "for", "with", "on", "by", "from",
})


def _heading_level(line: str) -> int | None:
    """Classify a stripped TOC line as PART (1), chapter (2), or section (3); None otherwise."""
    stripped = line.strip()
    if not stripped:
        return None
    if _PART_RE.match(stripped):
        return 1
    if _CHAPTER_RE.match(stripped):
        return 2
    # Level 3: starts with uppercase OR a leading digit (handles "73rd Amendment Act of 1992" etc.)
    if (stripped[0].isupper() or stripped[0].isdigit()) and len(stripped) < 80:
        return 3
    return None


def _is_continuation(title: str) -> bool:
    """True if title was cut mid-phrase (ends with a hanging word or an unclosed parenthesis)."""
    s = title.rstrip()
    if not s:
        return False
    last_word = s.rsplit(None, 1)[-1].lower().strip(".,;:")
    if last_word in _HANGING_ENDINGS:
        return True
    if s.count("(") > s.count(")"):
        return True
    return False


def parse_table_of_contents(
    pdf_path: Path,
    toc_start_page: int,
    toc_end_page: int,
) -> list[TocNode]:
    """
    Parse the book Contents pages into a shallow tree (PART → Chapter → Section).

    Page boundaries are filled in a later pass once body headings are aligned.
    """
    doc = open_pdf(pdf_path)
    try:
        lines: list[str] = []
        for page in range(toc_start_page, toc_end_page + 1):
            lines.extend(extract_page_text(doc, page).splitlines())

        roots: list[TocNode] = []
        stack: list[TocNode] = []

        pending_chapter_num = None
        found_part = False
        last_node: TocNode | None = None

        for raw in lines:
            line = raw.strip()
            if not line or line.lower() == "contents":
                continue

            # Stop at back-matter appendix listings — not chapter content.
            if line == "Appendices":
                break

            if line.isdigit():
                pending_chapter_num = line
                continue

            level = _heading_level(line)

            if pending_chapter_num is not None and level == 3:
                line = f"{pending_chapter_num} {line}"
                level = 2

            pending_chapter_num = None

            if level is None:
                continue

            if level == 1:
                found_part = True

            if not found_part:
                continue

            # Merge continuation lines into the previous node (e.g. "...of the" + "Constitution").
            if last_node is not None and _is_continuation(last_node.title):
                last_node.title += " " + line
                continue

            # The first level-3 line directly under a PART node is the PART's subtitle, not a section.
            if level == 3 and stack and stack[-1].level == 1 and not stack[-1].children:
                continue

            node = TocNode(title=line, level=level)
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
            last_node = node

        return roots
    finally:
        doc.close()
