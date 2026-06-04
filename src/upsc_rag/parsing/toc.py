from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from upsc_rag.parsing.pdf import extract_page_text, open_pdf


@dataclass
class TocNode:
    title: str
    level: int
    children: list[TocNode] = field(default_factory=list)
    page_start: int | None = None


_PART_RE = re.compile(r"^PART[- ]([IVXLC]+)\s*$", re.I)
_CHAPTER_RE = re.compile(r"^(\d+)\s+(.+)$")


def _heading_level(line: str) -> int | None:
    stripped = line.strip()
    if not stripped:
        return None
    if _PART_RE.match(stripped):
        return 1
    if _CHAPTER_RE.match(stripped):
        return 2
    if stripped[0].isupper() and len(stripped) < 80:
        return 3
    return None


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
        
        for raw in lines:
            line = raw.strip()
            if not line or line.lower() == "contents":
                continue
                
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

            node = TocNode(title=line, level=level)
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)

        return roots
    finally:
        doc.close()
