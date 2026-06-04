"""
Manages the indexing and storage of document chunks.
Includes vector store integrations and keyword-based
index handling for efficient retrieval operations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def save_chunks_jsonl(chunks: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1
    return count
