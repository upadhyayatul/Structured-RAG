"""PyMuPDF wrapper for raw text extraction from the book PDF."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import fitz


def open_pdf(path: Path) -> fitz.Document:
    """Open a PDF and return the fitz.Document; raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(path)
    return fitz.open(path)


def extract_page_text(doc: fitz.Document, page_num: int) -> str:
    """Return plain text for a 1-based page number."""
    page = doc[page_num - 1]
    return page.get_text("text")


def iter_pages(
    doc: fitz.Document,
    start: int = 1,
    end: int | None = None,
) -> Iterator[tuple[int, str]]:
    """Yield (page_num, text) for each page in [start, end] inclusive."""
    last = end or doc.page_count
    for num in range(start, last + 1):
        yield num, extract_page_text(doc, num)
