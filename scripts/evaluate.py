"""CLI: evaluate retrieval quality (hit@k, MRR, article-recall) over the gold set."""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.eval.harness import evaluate, load_gold
from upsc_rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UPSC-RAG retrieval quality")
    parser.add_argument("--book", default="laxmikanth_6")
    parser.add_argument("--rerank", type=int, default=None, help="Top-k results to score")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable query rewriting for this run")
    parser.add_argument("--gold", default=None, help="Path to gold jsonl (default data/eval/<book>.jsonl)")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_runtime_config(args.book)
    if args.no_rewrite:
        cfg = {**cfg, "retrieval": {**cfg.get("retrieval", {}),
                                    "rewrite": {**cfg.get("retrieval", {}).get("rewrite", {}), "enabled": False}}}

    processed = settings.resolve(settings.processed_dir) / args.book
    gold_path = Path(args.gold) if args.gold else settings.resolve(Path("data/eval")) / f"{args.book}.jsonl"

    gold = load_gold(gold_path)
    print(f"Loaded {len(gold)} gold questions from {gold_path}")
    print(f"Rewrite: {'OFF' if args.no_rewrite else 'ON'}\n")

    retriever = HybridRetriever(cfg, processed / "chunks.jsonl")
    report = evaluate(retriever, gold, rerank_top_k=args.rerank)

    for q in report["per_question"]:
        rank = f"#{q.rank}" if q.rank else "MISS"
        art = "" if q.article_found is None else ("  art:OK" if q.article_found else "  art:miss")
        print(f"  [{rank:>4}]{art}  {q.question}")

    print()
    print(f"  hit@k          : {report['hit_at_k']:.2%}  ({report['n']} questions)")
    print(f"  MRR            : {report['mrr']:.3f}")
    ar = report["article_recall"]
    print(f"  article_recall : {ar:.2%}" if ar is not None else "  article_recall : n/a")


if __name__ == "__main__":
    main()
