from __future__ import annotations

from typing import Any


def enrich_chunk(chunk: dict[str, Any], runtime_config: dict[str, Any]) -> dict[str, Any]:
    """Apply book-level tags and normalizations before indexing."""
    _ = runtime_config
    enriched = dict(chunk)
    enriched.setdefault("syllabus_tags", ["GS2_Polity"])
    return enriched
