"""Qdrant collection management: create the vector index and upsert child-chunk points."""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministically convert a string chunk ID to a UUID string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dim: int,
) -> None:
    """Create the collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
        )
        print(f"Created Qdrant collection '{collection_name}' (dim={embedding_dim})")
    else:
        print(f"Collection '{collection_name}' already exists — upserting into it")


def upsert_points(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    parent_map: dict[str, str],
    batch_size: int = 256,
) -> int:
    """
    Upsert (chunk, vector) pairs into Qdrant in batches; return total points written.

    parent_map[parent_id] = parent_text is stored in the payload so retrieval can
    expand a child hit to its full section context without a second lookup.
    """
    total = 0
    points: list[models.PointStruct] = []

    for chunk, vector in zip(chunks, vectors):
        payload: dict[str, Any] = {
            "chunk_id": chunk["id"],
            "parent_id": chunk.get("parent_id"),
            "parent_text": parent_map.get(chunk.get("parent_id", ""), ""),
            "text": chunk["text"],
            "book_id": chunk.get("book_id"),
            "part": chunk.get("part"),
            "chapter_num": chunk.get("chapter_num"),
            "chapter_title": chunk.get("chapter_title"),
            "section_path": chunk.get("section_path", []),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "entities": chunk.get("entities", []),
        }
        points.append(
            models.PointStruct(
                id=_chunk_id_to_uuid(chunk["id"]),
                vector=vector,
                payload=payload,
            )
        )

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            total += len(points)
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)

    return total
