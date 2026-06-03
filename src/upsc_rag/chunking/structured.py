from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class ChunkRecord:
    id: str
    text: str
    book_id: str
    part: str | None = None
    chapter_num: int | None = None
    chapter_title: str | None = None
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    content_type: str = "body"
    parent_id: str | None = None
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ARTICLE_RE = re.compile(r"\bArticle\s+\d+[A-Z]?\b", re.I)


def extract_entities(text: str) -> list[str]:
    return sorted({m.group(0) for m in _ARTICLE_RE.finditer(text)})


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_section_text(
    text: str,
    *,
    book_id: str,
    section_id: str,
    max_tokens: int = 600,
    overlap_tokens: int = 80,
    metadata: dict[str, Any] | None = None,
) -> Iterator[ChunkRecord]:
    """Split section text into overlapping chunks without crossing paragraph boundaries."""
    meta = metadata or {}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return

    buffer: list[str] = []
    buffer_tokens = 0
    chunk_index = 0

    def flush() -> ChunkRecord | None:
        nonlocal chunk_index, buffer, buffer_tokens
        if not buffer:
            return None
        body = "\n\n".join(buffer)
        chunk_id = f"{section_id}_{chunk_index:03d}"
        digest = hashlib.sha256(body.encode()).hexdigest()[:12]
        record = ChunkRecord(
            id=f"{chunk_id}_{digest}",
            text=body,
            book_id=book_id,
            entities=extract_entities(body),
            **{k: v for k, v in meta.items() if k in ChunkRecord.__dataclass_fields__},
        )
        chunk_index += 1
        return record

    for para in paragraphs:
        para_tokens = _token_estimate(para)
        if buffer and buffer_tokens + para_tokens > max_tokens:
            record = flush()
            if record:
                yield record
            if overlap_tokens > 0 and buffer:
                overlap: list[str] = []
                overlap_count = 0
                for p in reversed(buffer):
                    overlap.insert(0, p)
                    overlap_count += _token_estimate(p)
                    if overlap_count >= overlap_tokens:
                        break
                buffer = overlap
                buffer_tokens = sum(_token_estimate(p) for p in buffer)
            else:
                buffer = []
                buffer_tokens = 0

        buffer.append(para)
        buffer_tokens += para_tokens

    record = flush()
    if record:
        yield record
