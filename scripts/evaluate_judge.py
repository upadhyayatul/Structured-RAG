"""CLI: LLM-judge generation quality AND compare it to the 3 cheap signals.

For each gold question: retrieve -> generate an answer -> score it BOTH ways on the
same answer:
  * cheap signals (eval/generation.py)   — citation / groundedness / answer-relevance
  * LLM judge (eval/judge.py, gpt-5-mini) — faithfulness / completeness /
                                            exam_appropriateness / citation_quality (1-5)

Then it prints a side-by-side aggregate and a Pearson-correlation table between the
judge criteria and the cheap proxies that should track them — so we learn whether the
cheap gates are trustworthy stand-ins for the (slower) judge.

Cost: 1 gpt-4o-mini generation + 1 small embed (cheap signals) + 1 gpt-5-mini judge call
per question (~a few cents for the full 30q set). Requires OPENAI_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so any unicode in output can't crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.eval.generation import aggregate, score_answer
from upsc_rag.eval.harness import load_gold
from upsc_rag.eval.judge import (
    CRITERIA,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    aggregate_judge,
    judge_answer,
)
from upsc_rag.generation.answer import generate_answer
from upsc_rag.generation.router import is_off_topic, smalltalk_reply
from upsc_rag.retrieval.hybrid import HybridRetriever


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length series; None if undefined (constant series)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def _fmt_r(r: float | None) -> str:
    return " n/a " if r is None else f"{r:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge generation quality + compare to cheap signals")
    parser.add_argument("--book", default="laxmikanth_6")
    parser.add_argument("--rerank", type=int, default=None, help="Sources to pass to the LLM")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N gold questions (cost / rate-limit control)")
    parser.add_argument("--gold", default=None, help="Path to gold jsonl (default data/eval/<book>.jsonl)")
    parser.add_argument("--judge-model", default=None, help="OpenAI judge model (default from config)")
    parser.add_argument("--interval", type=float, default=None, help="Seconds to sleep between judge calls (free-tier TPM pacing; default from config)")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_runtime_config(args.book)
    gen_eval_cfg = cfg.get("eval", {}).get("generation", {})
    judge_cfg = cfg.get("eval", {}).get("judge", {})
    embed_model = gen_eval_cfg.get("embed_model", "text-embedding-3-small")
    ground_threshold = gen_eval_cfg.get("ground_threshold", 0.5)

    judge_model = args.judge_model or judge_cfg.get("model", DEFAULT_JUDGE_MODEL)
    base_url = judge_cfg.get("base_url", DEFAULT_JUDGE_BASE_URL)
    api_key_env = judge_cfg.get("api_key_env", "OPENAI_API_KEY")
    temperature = judge_cfg.get("temperature", 0.0)
    reasoning_effort = judge_cfg.get("reasoning_effort", "low")
    max_source_chars = judge_cfg.get("max_source_chars", 1500)
    # Free-tier pacing: sleep between judged questions so the per-minute token budget refills.
    request_interval = args.interval if args.interval is not None else judge_cfg.get("request_interval_sec", 8)

    import os
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(
            f"{api_key_env} is not set. Add it to .env (key from https://platform.openai.com/api-keys)."
        )
    # One OpenAI-pointed client reused across the run.
    judge_client = OpenAI(base_url=base_url, api_key=api_key)

    processed = settings.resolve(settings.processed_dir) / args.book
    gold_path = Path(args.gold) if args.gold else settings.resolve(Path("data/eval")) / f"{args.book}.jsonl"

    gold = load_gold(gold_path)
    if args.limit:
        gold = gold[: args.limit]
    print(f"Loaded {len(gold)} gold questions from {gold_path}")
    print(f"Judge: {judge_model} @ {base_url}  |  Generator: {cfg.get('generation', {}).get('model')}")
    print(f"Cheap-signal embed: {embed_model}  |  ground_threshold: {ground_threshold}\n")

    retriever = HybridRetriever(cfg, processed / "chunks.jsonl")
    floor = cfg.get("retrieval", {}).get("relevance_floor", 0.0)

    cheap_scores = []
    judge_scores = []
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
        cheap = score_answer(
            g.question, answer, results, g.articles,
            embed_model=embed_model, ground_threshold=ground_threshold,
        )
        try:
            js = judge_answer(
                g.question, answer, results,
                model=judge_model, base_url=base_url, api_key=api_key,
                temperature=temperature, reasoning_effort=reasoning_effort,
                max_source_chars=max_source_chars, client=judge_client,
            )
        except RateLimitError as exc:
            # If OpenAI rate-limits (e.g. low-tier ITPM/RPM), stop gracefully and summarize
            # whatever was scored rather than losing the whole run.
            print(f"\n  [rate limit hit — stopping early after {len(judge_scores)} scored] {exc}")
            break
        cheap_scores.append(cheap)
        judge_scores.append(js)

        gf = "  - " if cheap.grounded_fraction is None else f"{cheap.grounded_fraction:.0%}"
        print(
            f"  judge F{js.faithfulness} C{js.completeness} E{js.exam_appropriateness} "
            f"Cite{js.citation_quality} (ov {js.overall:.2f}) | "
            f"cheap grnd {gf} cite {cheap.cited_fraction:.0%} | {g.question}"
        )
        # Pace requests so the next judge call doesn't collide with the free-tier TPM window.
        if request_interval:
            time.sleep(request_interval)

    if not judge_scores:
        print("\nNo answers scored (all gated out).")
        return

    creport = aggregate(cheap_scores)
    jreport = aggregate_judge(judge_scores)

    # ---- Aggregate side-by-side -------------------------------------------------
    print("\n=== JUDGE (1-5, and /5 normalized) ===")
    for c in CRITERIA:
        val = getattr(jreport, c)
        print(f"  {c:22s}: {val:.2f}   ({val / 5:.0%})")
    print(f"  {'overall':22s}: {jreport.overall:.2f}   ({jreport.overall / 5:.0%})")

    print("\n=== CHEAP SIGNALS (for comparison) ===")
    ar = creport.article_recall
    print(f"  article_recall (ans)  : {ar:.2%}" if ar is not None else "  article_recall (ans)  : n/a")
    print(f"  cited_fraction        : {creport.cited_fraction:.2%}")
    print(f"  uncited_answer_rate   : {creport.uncited_answer_rate:.2%}")
    print(f"  invalid_marker_rate   : {creport.invalid_marker_rate:.2%}")
    gf = creport.grounded_fraction
    print(f"  grounded_fraction     : {gf:.2%}" if gf is not None else "  grounded_fraction     : n/a")
    ms = creport.mean_support
    print(f"  mean_support (cosine) : {ms:.3f}" if ms is not None else "  mean_support (cosine) : n/a")
    rel = creport.answer_relevance
    print(f"  answer_relevance      : {rel:.3f}" if rel is not None else "  answer_relevance      : n/a")

    # ---- Correlation: does each cheap proxy track its judge criterion? ----------
    # Only questions where the cheap signal is defined enter each pair.
    def _series(judge_key: str, cheap_attr: str) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for j, c in zip(judge_scores, cheap_scores):
            cv = getattr(c, cheap_attr)
            if cv is None:
                continue
            xs.append(float(getattr(j, judge_key)))
            ys.append(float(cv))
        return xs, ys

    pairs = [
        ("faithfulness", "grounded_fraction", "judge faithfulness <-> grounded_fraction"),
        ("faithfulness", "mean_support", "judge faithfulness <-> mean_support"),
        ("citation_quality", "cited_fraction", "judge citation_quality <-> cited_fraction"),
        ("completeness", "answer_relevance", "judge completeness <-> answer_relevance"),
    ]
    print("\n=== CORRELATION (Pearson r, judge vs cheap proxy) ===")
    print("  r->+1 means the cheap signal tracks the judge; ~0 means it doesn't.")
    for jkey, cattr, label in pairs:
        xs, ys = _series(jkey, cattr)
        print(f"  {label:44s}: r = {_fmt_r(_pearson(xs, ys))}  (n={len(xs)})")

    # ---- Biggest disagreements (judge/5 vs the paired cheap fraction) -----------
    # Where the judge and the groundedness proxy most disagree on faithfulness — the
    # rows worth a human look, and the tell for whether a cheap gate misleads.
    disagreements = []
    for j, c in zip(judge_scores, cheap_scores):
        if c.grounded_fraction is None:
            continue
        gap = abs(j.faithfulness / 5 - c.grounded_fraction)
        disagreements.append((gap, j, c))
    disagreements.sort(key=lambda t: t[0], reverse=True)
    if disagreements:
        print("\n=== TOP FAITHFULNESS DISAGREEMENTS (judge/5 vs grounded_fraction) ===")
        for gap, j, c in disagreements[:3]:
            print(
                f"  d={gap:.2f}  judge {j.faithfulness}/5 ({j.faithfulness/5:.0%}) vs "
                f"grounded {c.grounded_fraction:.0%}  | {j.question}"
            )
            reason = j.rationales.get("faithfulness", "")
            if reason:
                print(f"         judge: {reason}")


if __name__ == "__main__":
    main()
