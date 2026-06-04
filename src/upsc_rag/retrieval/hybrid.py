"""
Implements hybrid retrieval strategies.
Combines different retrieval methods, such as dense vector search
and sparse keyword search, to improve retrieval accuracy and recall.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def load_chunks_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
