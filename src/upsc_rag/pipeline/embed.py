"""Embed child chunks from chunks.jsonl and upsert into Qdrant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.indexing.embedder import embed_texts
from upsc_rag.indexing.qdrant_store import ensure_collection, upsert_points


def run_embed(book_id: str, chunks_path: Path | None = None) -> None:
    """Embed child chunks from chunks.jsonl and upsert them into the Qdrant collection."""
    settings = get_settings()
    runtime = load_runtime_config(book_id)
    idx_cfg = runtime.get("indexing", {})

    collection_name: str = idx_cfg["collection_name"]
    qdrant_url: str = idx_cfg.get("qdrant_url", "http://localhost:6333")
    embedding_model: str = idx_cfg.get("embedding_model", "text-embedding-3-small")
    embedding_dim: int = int(idx_cfg.get("embedding_dim", 1536))
    batch_size: int = int(idx_cfg.get("embed_batch_size", 100))

    resolved_chunks = chunks_path or (
        settings.resolve(settings.processed_dir) / book_id / "chunks.jsonl"
    )

    if not resolved_chunks.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {resolved_chunks}. Run ingest first.")

    print(f"Loading chunks from {resolved_chunks}...")
    all_chunks: list[dict] = []
    with resolved_chunks.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_chunks.append(json.loads(line))

    parent_map: dict[str, str] = {
        c["id"]: c["text"] for c in all_chunks if c.get("content_type") == "parent"
    }
    child_chunks = [c for c in all_chunks if c.get("content_type") == "child"]

    print(f"  {len(all_chunks)} total chunks -> {len(parent_map)} parents, {len(child_chunks)} children to embed")

    print(f"Embedding {len(child_chunks)} child chunks with {embedding_model} (batch={batch_size})...")
    texts = [c["text"] for c in child_chunks]
    vectors = embed_texts(texts, model=embedding_model, batch_size=batch_size)
    print(f"  Done — {len(vectors)} vectors produced")

    client = QdrantClient(url=qdrant_url)
    ensure_collection(client, collection_name, embedding_dim)

    print(f"Upserting into Qdrant collection '{collection_name}'…")
    total = upsert_points(
        client=client,
        collection_name=collection_name,
        chunks=child_chunks,
        vectors=vectors,
        parent_map=parent_map,
    )
    print(f"Done — {total} points upserted into '{collection_name}'")


def main() -> None:
    """CLI entry point: parse --book / --chunks args and run run_embed."""
    parser = argparse.ArgumentParser(description="Embed chunks and upsert into Qdrant")
    parser.add_argument("--book", default="laxmikanth_6", help="Book id from config/books/")
    parser.add_argument("--chunks", type=Path, default=None, help="Override path to chunks.jsonl")
    args = parser.parse_args()
    run_embed(args.book, args.chunks)
