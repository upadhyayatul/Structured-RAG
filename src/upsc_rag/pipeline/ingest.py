from __future__ import annotations

import argparse
from pathlib import Path

from upsc_rag.config import get_settings, load_book_config, load_runtime_config
from upsc_rag.indexing.store import save_chunks_jsonl
from upsc_rag.parsing.pdf import open_pdf


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
            __import__("json").dumps(manifest, indent=2),
            encoding="utf-8",
        )
    finally:
        doc.close()

    # Placeholder until TOC + section chunking is wired
    chunks_path = out / "chunks.jsonl"
    save_chunks_jsonl([], chunks_path)
    return chunks_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a UPSC textbook into structured chunks")
    parser.add_argument("--book", default="laxmikanth_6", help="Book id from config/books/")
    parser.add_argument("--output", type=Path, default=None, help="Override processed output dir")
    args = parser.parse_args()
    path = run_ingest(args.book, args.output)
    print(f"Wrote {path}")
