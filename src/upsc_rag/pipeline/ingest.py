"""
Defines the data ingestion pipeline.
Coordinates the end-to-end process of loading documents, parsing,
chunking, enriching with metadata, and indexing them into the store.
"""
from __future__ import annotations
from upsc_rag.parsing.toc import TocNode

import argparse
from pathlib import Path

from upsc_rag.config import get_settings, load_book_config, load_runtime_config
from upsc_rag.indexing.store import save_chunks_jsonl
import dataclasses
import json

from upsc_rag.parsing.pdf import open_pdf
from upsc_rag.parsing.toc import parse_table_of_contents
from upsc_rag.parsing.align import align_toc_with_body, extract_sections, fill_page_end
from upsc_rag.chunking.structured import chunk_section_text


def run_ingest(book_id: str, output_dir: Path | None = None) -> Path:
    """
    Run the ingest pipeline for a configured book.

    Stages (incremental): validate PDF → TOC → chunk → write JSONL.
    """
    settings = get_settings()
    runtime = load_runtime_config(book_id)
    book = load_book_config(book_id)
    pdf_path = book.resolved_pdf_path(settings)

    out = output_dir or settings.resolve(settings.processed_dir) / book_id
    out.mkdir(parents=True, exist_ok=True)

    doc = open_pdf(pdf_path)
    try:
        page_count = doc.page_count
        manifest = {
            "book_id": book_id,
            "title": book.title,
            "pdf_path": str(pdf_path),
            "page_count": page_count,
        }
        manifest_path = out / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    finally:
        doc.close()

    toc_start_page = runtime.get("parsing", {}).get("toc_start_page")
    toc_end_page = runtime.get("parsing", {}).get("toc_end_page")
    
    if toc_start_page is not None and toc_end_page is not None:
        content_start_page = runtime.get("parsing", {}).get("content_start_page")
        content_end_page = runtime.get("parsing", {}).get("content_end_page")
        
        all_chunks = []
        if content_start_page and content_end_page:
            toc_path = out / "toc.json"
            if toc_path.exists():
                print(f"Loading existing TOC from {toc_path} (skipping TOC parsing)")
                toc_data = json.loads(toc_path.read_text(encoding="utf-8"))
                def dict_to_toc_nodes(data_list) -> list[TocNode]:
                    nodes = []
                    for d in data_list:
                        children = dict_to_toc_nodes(d.get("children", []))
                        nodes.append(TocNode(
                            title=d["title"],
                            level=d["level"],
                            children=children,
                            page_start=d.get("page_start"),
                        page_end=d.get("page_end")
                        ))
                    return nodes
                toc_nodes = dict_to_toc_nodes(toc_data)
            else:
                print("Parsing TOC from PDF pages...")
                toc_nodes = parse_table_of_contents(pdf_path, toc_start_page, toc_end_page)
            
            align_doc = open_pdf(pdf_path)
            try:
                align_toc_with_body(align_doc, toc_nodes, content_start_page, content_end_page)
                fill_page_end(toc_nodes, content_end_page)
                
                toc_dict = [dataclasses.asdict(node) for node in toc_nodes]
                toc_path.write_text(json.dumps(toc_dict, indent=2), encoding="utf-8")
                
                sections = list(extract_sections(align_doc, toc_nodes, content_end_page))
            finally:
                align_doc.close()
                
            for sec in sections:
                chunks_iter = chunk_section_text(
                    text=sec["text"],
                    book_id=book_id,
                    section_id=sec["section_id"],
                    metadata={
                        "part": sec["part"],
                        "chapter_num": sec["chapter_num"],
                        "chapter_title": sec["chapter_title"],
                        "section_path": sec["section_path"],
                        "page_start": sec["page_start"],
                        "page_end": sec["page_end"],
                    }
                )
                for chunk in chunks_iter:
                    all_chunks.append(chunk.to_dict())

        chunks_path = out / "chunks.jsonl"
        save_chunks_jsonl(all_chunks, chunks_path)
        return chunks_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a UPSC textbook into structured chunks")
    parser.add_argument("--book", default="laxmikanth_6", help="Book id from config/books/")
    parser.add_argument("--output", type=Path, default=None, help="Override processed output dir")
    args = parser.parse_args()
    path = run_ingest(args.book, args.output)
    print(f"Wrote {path}")
