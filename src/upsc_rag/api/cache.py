"""Exact-match question→answer cache (sqlite, direct path only).

Keyed on the normalized condensed query, so caps/whitespace/trailing-punctuation
variants hit the same row. Web-fallback answers are never stored (they go stale);
textbook answers don't, so there is no TTL. Delete the db file to clear the cache.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_cache (
    key TEXT PRIMARY KEY,
    query TEXT,
    answer TEXT,
    sources TEXT,
    created_at TEXT DEFAULT (datetime('now'))
)
"""


def _normalize(query: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    return " ".join(query.lower().split()).rstrip(" ?.!")


class AnswerCache:
    """Persistent exact-match cache for generated answers + their sources."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    # ponytail: one connection per call — dodges FastAPI threadpool thread-safety;
    # pool the connection if volume ever matters.
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def get(self, query: str) -> tuple[str, list[dict[str, Any]]] | None:
        """Return (answer, sources) for a previously stored question, else None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT answer, sources FROM qa_cache WHERE key = ?",
                (_normalize(query),),
            ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def put(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        """Store (overwrite) the answer + sources for this question."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO qa_cache (key, query, answer, sources) "
                "VALUES (?, ?, ?, ?)",
                (_normalize(query), query, answer, json.dumps(sources)),
            )
