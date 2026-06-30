"""Reference-light generation-quality signals: cheap, no LLM judge.

Three signals per (question, answer, sources):

  1. citation        — deterministic, no API call. ``article_recall`` (gold Articles
                       named in the answer) plus source-marker validity (``[n]`` markers
                       that point at a real source, and what fraction of sources are cited).
  2. groundedness    — embedding cosine. Fraction of answer sentences whose best cosine
                       to any source text clears ``ground_threshold`` (a hallucination proxy),
                       plus the mean best-cosine over sentences.
  3. answer_relevance — embedding cosine between the whole answer and the question.

Signals 2-3 reuse the OpenAI embedder (``text-embedding-3-small`` by default) — one
batched call per answer, far cheaper than an LLM judge. Treat these as an always-affordable
gate; reserve an LLM judge for the nuanced rubric (completeness, exam-appropriateness).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from upsc_rag.chunking.structured import extract_entities
from upsc_rag.indexing.embedder import embed_texts

# Markdown noise to strip before splitting an answer into sentences.
_MD_NOISE_RE = re.compile(r"(\*\*|__|`|^#{1,6}\s*|^[-*]\s+|\[\d+\])", re.M)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SOURCE_MARKER_RE = re.compile(r"\[(\d+)\]")
# Sentences shorter than this (after stripping) are headings/fragments — drop from groundedness.
_MIN_SENT_CHARS = 25


def _split_sentences(answer: str) -> list[str]:
    """Strip markdown and split an answer into scoreable sentences."""
    clean = _MD_NOISE_RE.sub("", answer)
    parts = (s.strip() for s in _SENT_SPLIT_RE.split(clean))
    return [s for s in parts if len(s) >= _MIN_SENT_CHARS]


def _source_windows(text: str) -> list[str]:
    """Split a source section into sentence-level windows for groundedness matching.

    Sources arrive as whole parent sections (often ~2000 chars). Embedding each as a
    single vector dilutes support: a short, paraphrased answer sentence washes out
    against the big section vector, so faithful procedural steps score as 'unsupported'.
    Matching against sentence windows restores the signal. Falls back to the whole text
    when it has no sentence-sized pieces.
    """
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text)]
    windows = [s for s in parts if len(s) >= _MIN_SENT_CHARS]
    return windows or ([text.strip()] if text.strip() else [])


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def citation_signals(answer: str, gold_articles: list[str], num_sources: int) -> dict[str, Any]:
    """Deterministic citation checks — no API call.

    * ``article_recall``     — fraction of labeled gold Articles named in the answer
                               (None when the gold question has no labeled articles).
    * ``cited_fraction``     — fraction of the supplied sources referenced via a ``[n]`` marker.
    * ``has_invalid_marker`` — True if the answer cites a ``[n]`` outside 1..num_sources.
    """
    named = set(extract_entities(answer))
    article_recall: float | None = None
    if gold_articles:
        hit = sum(1 for a in gold_articles if a in named)
        article_recall = hit / len(gold_articles)

    markers = {int(m) for m in _SOURCE_MARKER_RE.findall(answer)}
    valid = {m for m in markers if 1 <= m <= num_sources}
    invalid = markers - valid
    cited_fraction = len(valid) / num_sources if num_sources else 0.0

    return {
        "article_recall": article_recall,
        "cited_fraction": cited_fraction,
        "has_invalid_marker": bool(invalid),
    }


@dataclass
class AnswerScore:
    question: str
    article_recall: float | None
    cited_fraction: float
    has_invalid_marker: bool
    grounded_fraction: float | None  # fraction of sentences clearing ground_threshold
    mean_support: float | None       # mean best-cosine over sentences
    answer_relevance: float | None   # cosine(answer, question)
    num_sentences: int = 0


def score_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    gold_articles: list[str],
    *,
    embed_model: str = "text-embedding-3-small",
    ground_threshold: float = 0.5,
) -> AnswerScore:
    """Compute all three cheap signals for one (question, answer, sources) triple.

    Embeds the question, every answer sentence, and every source text in a single
    batched call, then derives groundedness + answer-relevance from cosines.
    """
    cite = citation_signals(answer, gold_articles, len(sources))
    sentences = _split_sentences(answer)
    # Match against source SENTENCE WINDOWS, not whole sections — see _source_windows.
    windows: list[str] = []
    for s in sources:
        windows.extend(_source_windows(s.get("text", "")))

    grounded_fraction: float | None = None
    mean_support: float | None = None
    answer_relevance: float | None = None

    if sentences and windows:
        # One batched embed call: [question, *sentences, *source_windows].
        payload = [question, *sentences, *windows]
        vecs = embed_texts(payload, model=embed_model)
        q_vec = vecs[0]
        sent_vecs = vecs[1 : 1 + len(sentences)]
        src_vecs = vecs[1 + len(sentences) :]

        best = [max(_cosine(sv, cv) for cv in src_vecs) for sv in sent_vecs]
        grounded_fraction = sum(1 for b in best if b >= ground_threshold) / len(best)
        mean_support = sum(best) / len(best)

        # Answer-relevance: cosine of the mean answer-sentence vector to the question.
        dim = len(sent_vecs[0])
        centroid = [sum(v[i] for v in sent_vecs) / len(sent_vecs) for i in range(dim)]
        answer_relevance = _cosine(centroid, q_vec)

    return AnswerScore(
        question=question,
        article_recall=cite["article_recall"],
        cited_fraction=cite["cited_fraction"],
        has_invalid_marker=cite["has_invalid_marker"],
        grounded_fraction=grounded_fraction,
        mean_support=mean_support,
        answer_relevance=answer_relevance,
        num_sentences=len(sentences),
    )


@dataclass
class GenerationReport:
    n: int
    article_recall: float | None
    cited_fraction: float
    uncited_answer_rate: float   # fraction of answers with NO [n] marker at all — the real citation gap
    invalid_marker_rate: float
    grounded_fraction: float | None
    mean_support: float | None
    answer_relevance: float | None
    per_question: list[AnswerScore] = field(default_factory=list)


def _avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def aggregate(scores: list[AnswerScore]) -> GenerationReport:
    """Aggregate per-question signals into mean metrics over the run."""
    n = len(scores)
    return GenerationReport(
        n=n,
        article_recall=_avg([s.article_recall for s in scores]),
        cited_fraction=sum(s.cited_fraction for s in scores) / n if n else 0.0,
        uncited_answer_rate=sum(1 for s in scores if s.cited_fraction == 0.0) / n if n else 0.0,
        invalid_marker_rate=sum(1 for s in scores if s.has_invalid_marker) / n if n else 0.0,
        grounded_fraction=_avg([s.grounded_fraction for s in scores]),
        mean_support=_avg([s.mean_support for s in scores]),
        answer_relevance=_avg([s.answer_relevance for s in scores]),
        per_question=scores,
    )
