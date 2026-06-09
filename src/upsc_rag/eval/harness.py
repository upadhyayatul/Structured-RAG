"""Retrieval-quality evaluation: hit@k, MRR, and article-recall over a labeled gold set.

The gold set (``data/eval/<book>.jsonl``) labels each question with the section it
*should* retrieve (``section_contains``: substrings that must all appear in the
matched section_path) and, optionally, the governing ``articles``. The harness runs
the retriever per question and scores:

  * hit@k          — fraction of questions with a relevant section in the top-k
  * MRR            — mean reciprocal rank of the first relevant section
  * article_recall — of questions with labeled articles, fraction whose expected
                     article appears in any top-k result's entities

It's deliberately retrieval-only (no LLM judge): cheap, deterministic, and exactly
the signal needed to tune retrieval.rewrite.score_threshold and to decide whether
graph expansion helps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldQuestion:
    question: str
    section_contains: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)


def load_gold(path: Path) -> list[GoldQuestion]:
    out: list[GoldQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                GoldQuestion(
                    question=d["question"],
                    section_contains=[s.lower() for s in d.get("section_contains", [])],
                    articles=d.get("articles", []),
                )
            )
    return out


def _section_relevant(result: dict[str, Any], section_contains: list[str]) -> bool:
    """True if every required substring appears in the result's section_path."""
    if not section_contains:
        return False
    path = " > ".join(result.get("section_path") or []).lower()
    return all(sub in path for sub in section_contains)


@dataclass
class QuestionResult:
    question: str
    rank: int | None          # 1-based rank of first relevant result, None if missed
    article_found: bool | None  # None if no articles labeled


def evaluate(retriever: Any, gold: list[GoldQuestion], rerank_top_k: int | None = None) -> dict[str, Any]:
    """Run the retriever over every gold question and return aggregate + per-question metrics."""
    per_question: list[QuestionResult] = []

    for g in gold:
        results = retriever.retrieve(g.question, rerank_top_k=rerank_top_k)

        rank: int | None = None
        for i, r in enumerate(results, start=1):
            if _section_relevant(r, g.section_contains):
                rank = i
                break

        article_found: bool | None = None
        if g.articles:
            retrieved_articles = {a for r in results for a in (r.get("entities") or [])}
            article_found = any(a in retrieved_articles for a in g.articles)

        per_question.append(QuestionResult(g.question, rank, article_found))

    n = len(per_question)
    hits = sum(1 for q in per_question if q.rank is not None)
    mrr = sum((1.0 / q.rank) for q in per_question if q.rank) / n if n else 0.0

    with_articles = [q for q in per_question if q.article_found is not None]
    article_recall = (
        sum(1 for q in with_articles if q.article_found) / len(with_articles)
        if with_articles else None
    )

    return {
        "n": n,
        "hit_at_k": hits / n if n else 0.0,
        "mrr": mrr,
        "article_recall": article_recall,
        "per_question": per_question,
    }
