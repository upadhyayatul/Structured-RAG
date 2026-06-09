"""CLI: build the section<->article knowledge graph from chunks.jsonl and persist it."""
from __future__ import annotations

import argparse
from pathlib import Path

from upsc_rag.config import get_settings
from upsc_rag.indexing.graph_store import build_graph, save_graph
from upsc_rag.retrieval.hybrid import load_chunks_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the UPSC-RAG knowledge graph")
    parser.add_argument("--book", default="laxmikanth_6")
    args = parser.parse_args()

    settings = get_settings()
    book_dir = settings.resolve(settings.processed_dir) / args.book
    chunks_path = book_dir / "chunks.jsonl"
    graph_path = book_dir / "graph.pkl"

    print(f"Reading chunks from {chunks_path}…")
    graph = build_graph(load_chunks_jsonl(chunks_path))

    sections = [n for n, d in graph.nodes(data=True) if d.get("kind") == "section"]
    articles = [n for n, d in graph.nodes(data=True) if d.get("kind") == "article"]
    print(f"Graph: {len(sections)} section nodes, {len(articles)} article nodes, "
          f"{graph.number_of_edges()} MENTIONS edges")

    save_graph(graph, graph_path)
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
