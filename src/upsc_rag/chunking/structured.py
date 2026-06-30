"""Split section text into a parent ChunkRecord plus overlapping child ChunkRecords for indexing."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class ChunkRecord:
    """
    One indexable text unit.

    ``content_type="parent"`` stores the full section text (not embedded).
    ``content_type="child"`` stores an overlapping ~600-token split (embedded into Qdrant).
    """

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
        """Serialize to a plain dict for JSONL writing."""
        return asdict(self)


_ARTICLE_RE = re.compile(r"\bArticle\s+\d+[A-Z]?\b", re.I)


def extract_entities(text: str) -> list[str]:
    """Return a sorted, deduplicated list of 'Article NNN' references found in text.

    Internal whitespace is normalized to a single space so PDF line breaks
    ('Article\\n312') collapse to the canonical 'Article 312' — otherwise the same
    reference fails to match across chunks, the gold set, and graph nodes.
    """
    refs = {re.sub(r"\s+", " ", m.group(0)) for m in _ARTICLE_RE.finditer(text)}
    return sorted(refs)


def _token_estimate(text: str) -> int:
    """Cheap token estimate: len / 4, avoiding a full tokenizer dependency."""
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
    """
    Yield one parent chunk (full section) followed by overlapping child chunks.

    Splits at paragraph boundaries so no paragraph is ever cut mid-sentence.
    The parent is not embedded; children are embedded and point back to the parent
    via parent_id for context expansion at retrieval time.
    """
    meta = metadata or {}
    meta_filtered = {k: v for k, v in meta.items() if k in ChunkRecord.__dataclass_fields__}
    
    parent_id = section_id
    parent_meta = dict(meta_filtered)
    parent_meta["content_type"] = "parent"
    parent_meta["parent_id"] = None
    
    yield ChunkRecord(
        id=parent_id,
        text=text,
        book_id=book_id,
        entities=extract_entities(text),
        **parent_meta,
    )

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return

    buffer: list[str] = []
    buffer_tokens = 0
    chunk_index = 0
    
    child_meta = dict(meta_filtered)
    child_meta["content_type"] = "child"
    child_meta["parent_id"] = parent_id

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
            **child_meta,
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
