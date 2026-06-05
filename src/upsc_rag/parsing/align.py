"""
Aligns parsed document structures.
Responsible for mapping and aligning different structural elements,
such as matching table of contents entries with physical pages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Any

import fitz
from upsc_rag.parsing.toc import TocNode
from upsc_rag.parsing.pdf import iter_pages


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def align_toc_with_body(
    doc: fitz.Document,
    toc_nodes: list[TocNode],
    start_page: int,
    end_page: int
) -> None:
    """
    Finds the start page for each TOC node by searching the PDF body text.
    Modifies toc_nodes in-place.
    """
    flat_nodes: list[TocNode] = []

    def _flatten(nodes: list[TocNode]) -> None:
        for n in nodes:
            flat_nodes.append(n)
            _flatten(n.children)

    _flatten(toc_nodes)

    node_idx = 0
    total_nodes = len(flat_nodes)

    for page_num, text in iter_pages(doc, start=start_page, end=end_page):
        if node_idx >= total_nodes:
            break

        norm_text = _normalize(text)

        while node_idx < total_nodes:
            found_ahead = -1
            # Look ahead up to 5 nodes to recover from slight text mismatches
            for offset in range(min(5, total_nodes - node_idx)):
                node = flat_nodes[node_idx + offset]
                norm_title = _normalize(node.title)

                if norm_title in norm_text:
                    found_ahead = offset
                    break

            if found_ahead != -1:
                # Found a node on this page; assign skipped nodes to this page as fallback.
                for i in range(found_ahead + 1):
                    if flat_nodes[node_idx + i].page_start is None:
                        flat_nodes[node_idx + i].page_start = page_num
                node_idx += found_ahead + 1
            else:
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
