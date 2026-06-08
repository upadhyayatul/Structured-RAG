"""Add syllabus tags and other book-level metadata to chunks before indexing."""
from __future__ import annotations

from typing import Any


def enrich_chunk(chunk: dict[str, Any], runtime_config: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of chunk with additional metadata fields applied.

    Currently stamps syllabus_tags=[GS2_Polity] if not already set.
    Extend here when more tag categories (GS1, GS3, etc.) are needed.
    """
    _ = runtime_config
    enriched = dict(chunk)
    enriched.setdefault("syllabus_tags", ["GS2_Polity"])
    return enriched
