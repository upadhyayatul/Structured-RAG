"""Calibrate / verify PDF page ranges from config/books/<book_id>.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from upsc_rag.config import PROJECT_ROOT, load_runtime_config  # noqa: E402
from upsc_rag.parsing.pdf import extract_page_text, open_pdf  # noqa: E402
from upsc_rag.parsing.toc import parse_table_of_contents  # noqa: E402


def _preview(text: str, max_lines: int = 18) -> None:
    """Print up to max_lines stripped lines of text, each truncated to 100 chars."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:max_lines]:
        print(f"  {line[:100]}")


def main() -> None:
    """Print sampled page text and TOC parse summary to validate config page ranges."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="laxmikanth_6")
    args = parser.parse_args()

    cfg = load_runtime_config(args.book)
    parsing = cfg["parsing"]
    pdf_path = PROJECT_ROOT / cfg["book"]["pdf_path"]

    required = (
        "toc_start_page",
        "toc_end_page",
        "content_start_page",
        "content_end_page",
    )
    missing = [k for k in required if parsing.get(k) is None]
    if missing:
        raise SystemExit(f"Missing parsing keys in config: {missing}")

    cs = parsing["toc_start_page"]
    ce = parsing["toc_end_page"]
    body_start = parsing["content_start_page"]
    body_end = parsing["content_end_page"]
    ch7 = parsing.get("sanity_chapter_7_start_page")

    doc = open_pdf(pdf_path)
    try:
        print(f"book: {args.book}")
        print(f"pdf: {pdf_path}")
        print(f"page_count: {doc.page_count}\n")

        print(f"=== Contents (pages {cs}-{ce}) ===")
        _preview(extract_page_text(doc, cs))
        print("  ...")
        _preview(extract_page_text(doc, ce), max_lines=10)

        roots = parse_table_of_contents(pdf_path, cs, ce)
        parts = [n.title for n in roots if n.level == 1]
        print(f"\nTOC parse: {len(roots)} top-level nodes, PART headings: {parts[:5]} ...")

        print(f"\n=== Ch. 1 body (page {body_start}) ===")
        _preview(extract_page_text(doc, body_start))

        if ch7:
            print(f"\n=== Ch. 7 sanity (page {ch7}) ===")
            _preview(extract_page_text(doc, ch7))

        print(f"\n=== Body end (page {body_end}) ===")
        _preview(extract_page_text(doc, body_end), max_lines=10)

        next_page = body_end + 1
        if next_page <= doc.page_count:
            head = extract_page_text(doc, next_page).splitlines()
            head = next((l.strip() for l in head if l.strip()), "")
            print(f"\nPage after body ({next_page}): {head[:80]}")
    finally:
        doc.close()


if __name__ == "__main__":
    main()
