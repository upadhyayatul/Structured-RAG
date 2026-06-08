"""CLI: run a hybrid retrieval query and print ranked results."""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    """Build HybridRetriever from config, run the query, and print ranked results."""
    parser = argparse.ArgumentParser(description="Hybrid retrieval over the UPSC-RAG index")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--book", default="laxmikanth_6")
    parser.add_argument("--top-k", type=int, default=None, help="Dense+BM25 candidate pool size")
    parser.add_argument("--rerank", type=int, default=None, help="Final results to return")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_runtime_config(args.book)
    chunks_path = settings.resolve(settings.processed_dir) / args.book / "chunks.jsonl"

    print(f"Building BM25 index from {chunks_path}…")
    retriever = HybridRetriever(cfg, chunks_path)

    print(f"\nQuery: {args.query!r}\n")
    results = retriever.retrieve(args.query, top_k=args.top_k, rerank_top_k=args.rerank)

    for i, r in enumerate(results, start=1):
        path = " > ".join(r.get("section_path") or [r.get("chapter_title", "")])
        pages = f"p.{r['page_start']}–{r['page_end']}" if r.get("page_start") else ""
        print(f"[{i}] (rrf={r['rrf_score']}) {path}  {pages}")
        print(f"     {r['text'][:200].strip()}…")
        print()


if __name__ == "__main__":
    main()
