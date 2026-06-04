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
                # We found a node on this page!
                # If we skipped some nodes, assign them to this page as a fallback.
                for i in range(found_ahead + 1):
                    if flat_nodes[node_idx + i].page_start is None:
                        flat_nodes[node_idx + i].page_start = page_num
                node_idx += found_ahead + 1
            else:
                # No more nodes found on this page
                break


def extract_sections(
    doc: fitz.Document,
    toc_nodes: list[TocNode],
    end_page: int
) -> Iterator[dict[str, Any]]:
    """
    Extracts text for each section in the TOC.
    Yields dicts ready to be chunked.
    """
    # Flatten with path metadata
    @dataclass
    class FlatSection:
        node: TocNode
        part: str | None
        chapter_num: int | None
        chapter_title: str | None
        section_path: list[str]

    flat_sections: list[FlatSection] = []
    
    def _flatten_with_meta(nodes: list[TocNode], current_part: str | None, current_chapter_num: int | None, current_chapter_title: str | None, current_path: list[str]) -> None:
        for n in nodes:
            part = current_part
            ch_num = current_chapter_num
            ch_title = current_chapter_title
            
            if n.level == 1:
                part = n.title
            elif n.level == 2:
                # Extract number if possible
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
                section_path=path
            ))
            
            _flatten_with_meta(n.children, part, ch_num, ch_title, path)
            
    _flatten_with_meta(toc_nodes, None, None, None, [])
    
    first_page = min((f.node.page_start for f in flat_sections if f.node.page_start), default=1)
    
    full_text = ""
    for p in range(first_page, end_page + 1):
        if p <= doc.page_count:
            full_text += doc[p - 1].get_text("text") + "\n"
            
    node_offsets = []
    search_idx = 0
    
    for fsec in flat_sections:
        if not fsec.node.page_start:
            node_offsets.append(search_idx)
            continue
            
        words = fsec.node.title.strip().split()
        if not words:
            node_offsets.append(search_idx)
            continue
            
        pattern = r'\s+'.join(re.escape(w) for w in words)
        title_re = re.compile(pattern, re.IGNORECASE)
        
        match = title_re.search(full_text, search_idx)
        if match:
            # We don't advance search_idx past the start of the title, 
            # so the title is included in the extracted text.
            offset = match.start()
            node_offsets.append(offset)
            search_idx = offset + 1
        else:
            node_offsets.append(search_idx)
            
    node_offsets.append(len(full_text))
    
    for i, fsec in enumerate(flat_sections):
        start_page = fsec.node.page_start
        if not start_page:
            continue
            
        end_page_meta = end_page
        for j in range(i + 1, len(flat_sections)):
            if flat_sections[j].node.page_start:
                end_page_meta = flat_sections[j].node.page_start
                break
                
        text_chunk = full_text[node_offsets[i]:node_offsets[i+1]].strip()
        
        if not text_chunk:
            continue
            
        yield {
            "section_id": f"sec_{i:04d}",
            "text": text_chunk,
            "part": fsec.part,
            "chapter_num": fsec.chapter_num,
            "chapter_title": fsec.chapter_title,
            "section_path": fsec.section_path,
            "page_start": start_page,
            "page_end": end_page_meta,
        }
