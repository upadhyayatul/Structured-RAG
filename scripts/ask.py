"""CLI: retrieve sources for a question and generate a grounded, cited answer."""
from __future__ import annotations

import argparse

from dotenv import load_dotenv
load_dotenv()

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.generation.answer import generate_answer
from upsc_rag.generation.router import OUT_OF_SCOPE_REPLY, is_off_topic, smalltalk_reply
from upsc_rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    """Retrieve via HybridRetriever, generate an answer with the LLM, and print both."""
    parser = argparse.ArgumentParser(description="Ask a grounded question over the UPSC-RAG index")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--book", default="laxmikanth_6")
    parser.add_argument("--top-k", type=int, default=None, help="Dense+BM25 candidate pool size")
    parser.add_argument("--rerank", type=int, default=None, help="Sources to pass to the LLM")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_runtime_config(args.book)
    chunks_path = settings.resolve(settings.processed_dir) / args.book / "chunks.jsonl"

    print(f"Building BM25 index from {chunks_path}…")
    retriever = HybridRetriever(cfg, chunks_path)

    print(f"\nQuery: {args.query!r}\n")
    # Gate 1: pure greeting / chit-chat — reply without retrieving or generating.
    canned = smalltalk_reply(args.query)
    if canned is not None:
        print("=" * 70)
        print(canned)
        print("=" * 70)
        return

    results = retriever.retrieve(args.query, top_k=args.top_k, rerank_top_k=args.rerank)

    # Gate 2: real question, but no relevant source in the book — skip generation.
    floor = cfg.get("retrieval", {}).get("relevance_floor", 0.0)
    if is_off_topic(results, floor):
        print("=" * 70)
        print(OUT_OF_SCOPE_REPLY)
        print("=" * 70)
        return

    print("Generating answer…\n")
    answer = generate_answer(args.query, results, cfg)

    print("=" * 70)
    print(answer)
    print("=" * 70)
    print("\nSources:")
    for i, r in enumerate(results, start=1):
        path = " > ".join(r.get("section_path") or [r.get("chapter_title", "")])
        pages = f"p.{r['page_start']}–{r['page_end']}" if r.get("page_start") else ""
        print(f"  [{i}] {path}  {pages}")


if __name__ == "__main__":
    main()
