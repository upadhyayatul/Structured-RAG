"""CLI: evaluate GENERATION quality with 3 cheap signals (no LLM judge).

For each gold question: retrieve -> generate an answer -> score it on
  * citation       — gold-Article recall in the answer + source-marker validity
  * groundedness   — fraction of answer sentences supported by the sources (embedding cosine)
  * answer_relevance — cosine(answer, question)

Runs the generation LLM once per question (the only real cost), so use --limit to
sample while iterating. Retrieval flags mirror scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.eval.generation import aggregate, score_answer
from upsc_rag.eval.harness import load_gold
from upsc_rag.generation.answer import generate_answer
from upsc_rag.generation.router import is_off_topic, smalltalk_reply
from upsc_rag.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UPSC-RAG generation quality (cheap signals)")
    parser.add_argument("--book", default="laxmikanth_6")
    parser.add_argument("--rerank", type=int, default=None, help="Sources to pass to the LLM")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N gold questions (cost control)")
    parser.add_argument("--gold", default=None, help="Path to gold jsonl (default data/eval/<book>.jsonl)")
    parser.add_argument("--gen-model", default=None, help="Override the generation model for this run only (production config untouched)")
    parser.add_argument("--embed-model", default=None, help="Embedding model for groundedness/relevance")
    parser.add_argument("--ground-threshold", type=float, default=None, help="Cosine cutoff for a 'supported' sentence")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_runtime_config(args.book)
    if args.gen_model:  # eval-only override; never written back to config/default.yaml
        cfg.setdefault("generation", {})["model"] = args.gen_model
    eval_cfg = cfg.get("eval", {}).get("generation", {})
    embed_model = args.embed_model or eval_cfg.get("embed_model", "text-embedding-3-small")
    ground_threshold = args.ground_threshold if args.ground_threshold is not None else eval_cfg.get("ground_threshold", 0.5)

    processed = settings.resolve(settings.processed_dir) / args.book
    gold_path = Path(args.gold) if args.gold else settings.resolve(Path("data/eval")) / f"{args.book}.jsonl"

    gold = load_gold(gold_path)
    if args.limit:
        gold = gold[: args.limit]
    print(f"Loaded {len(gold)} gold questions from {gold_path}")
    print(f"Generator: {cfg.get('generation', {}).get('model')}  |  embed model: {embed_model}  |  ground_threshold: {ground_threshold}\n")

    retriever = HybridRetriever(cfg, processed / "chunks.jsonl")
    floor = cfg.get("retrieval", {}).get("relevance_floor", 0.0)

    scores = []
    for g in gold:
        # Mirror the ask.py gates: skip smalltalk / off-topic so we only score real answers.
        if smalltalk_reply(g.question) is not None:
            print(f"  [skip greeting]  {g.question}")
            continue
        results = retriever.retrieve(g.question, rerank_top_k=args.rerank)
        if is_off_topic(results, floor):
            print(f"  [skip off-topic]  {g.question}")
            continue

        answer = generate_answer(g.question, results, cfg)
        s = score_answer(
            g.question, answer, results, g.articles,
            embed_model=embed_model, ground_threshold=ground_threshold,
        )
        scores.append(s)

        ar = "  -" if s.article_recall is None else f"art {s.article_recall:.0%}"
        gf = "  -" if s.grounded_fraction is None else f"grnd {s.grounded_fraction:.0%}"
        rel = "  -" if s.answer_relevance is None else f"rel {s.answer_relevance:.2f}"
        bad = "  !marker" if s.has_invalid_marker else ""
        print(f"  [{ar}  {gf}  cite {s.cited_fraction:.0%}  {rel}]{bad}  {g.question}")

    report = aggregate(scores)
    print()
    print(f"  questions scored      : {report.n}")
    ar = report.article_recall
    print(f"  article_recall (ans)  : {ar:.2%}" if ar is not None else "  article_recall (ans)  : n/a")
    print(f"  cited_fraction        : {report.cited_fraction:.2%}  (share of the supplied sources cited; <100% is fine — focused answers use a subset)")
    print(f"  uncited_answer_rate   : {report.uncited_answer_rate:.2%}  (answers with NO citation at all — the real citation gap)")
    print(f"  invalid_marker_rate   : {report.invalid_marker_rate:.2%}")
    gf = report.grounded_fraction
    print(f"  grounded_fraction     : {gf:.2%}" if gf is not None else "  grounded_fraction     : n/a")
    ms = report.mean_support
    print(f"  mean_support (cosine) : {ms:.3f}" if ms is not None else "  mean_support (cosine) : n/a")
    rel = report.answer_relevance
    print(f"  answer_relevance      : {rel:.3f}" if rel is not None else "  answer_relevance      : n/a")


if __name__ == "__main__":
    main()
