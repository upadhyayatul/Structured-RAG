"""LLM-as-judge for generation quality — a rubric the cheap signals can't cover.

The cheap signals (``eval/generation.py``) are an always-affordable gate: citation
validity, an embedding-cosine groundedness proxy, and answer-relevance. They cannot
score the *nuanced* things a UPSC examiner cares about — is the answer **complete**,
is it **exam-appropriate** in structure and depth — and their faithfulness signal is
only a cosine heuristic. This module adds a rubric LLM judge to score those directly
and to cross-check the cheap proxies.

Judge model: ``gpt-5-mini`` on **OpenAI**. It is a stronger, *different* model from the
``gpt-4o-mini`` generator, so it won't rubber-stamp the generator's own style — same-provider
self-preference is reduced (though not eliminated the way a cross-provider judge would).
gpt-5 is a reasoning model: it rejects ``temperature != 1``, so the judge call omits
``temperature`` for gpt-5/o-series models and instead sets ``reasoning_effort``.

One chat call per (question, answer, sources) triple, JSON-mode output. The full 30q run
costs only a few cents.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

DEFAULT_JUDGE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"


def _is_reasoning_model(model: str) -> bool:
    """True for OpenAI reasoning models (gpt-5 / o-series) that reject ``temperature != 1``.

    These take ``reasoning_effort`` instead of a temperature; a plain ``temperature=0``
    returns a 400. Match by name prefix so any gpt-5.x / o1 / o3 / o4 variant is covered.
    """
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))

# The four rubric criteria, scored 1..5. Order is fixed — the report/comparison code
# and the JSON contract both rely on these keys.
CRITERIA = ("faithfulness", "completeness", "exam_appropriateness", "citation_quality")

_SYSTEM_PROMPT = (
    "You are a strict, fair examiner grading answers for UPSC (Indian Civil Services) "
    "Polity preparation. You are given a QUESTION, the numbered SOURCES the answer was "
    "supposed to use, and the ANSWER under review. Grade the ANSWER on four criteria, "
    "each on an INTEGER scale from 1 (very poor) to 5 (excellent):\n\n"
    "1. faithfulness — is every claim supported by the SOURCES only? Penalise anything "
    "asserted that the sources do not contain, even if it is true in the real world.\n"
    "2. completeness — does it cover the key points an ideal UPSC answer to this question "
    "needs, given what the sources make available?\n"
    "3. exam_appropriateness — is it well structured, correctly framed, and at the right "
    "depth/precision for the exam (governing Articles named, clear organisation)?\n"
    "4. citation_quality — are the bracketed [n] source markers present on claims and do "
    "they point at sources that actually support those claims?\n\n"
    "Be discriminating: reserve 5 for genuinely excellent, use the low end when warranted. "
    'Respond ONLY with a JSON object of this exact shape (no prose outside the json):\n'
    '{"faithfulness": {"score": <1-5>, "reason": "<one short sentence>"}, '
    '"completeness": {"score": <1-5>, "reason": "..."}, '
    '"exam_appropriateness": {"score": <1-5>, "reason": "..."}, '
    '"citation_quality": {"score": <1-5>, "reason": "..."}}'
)


@dataclass
class JudgeScore:
    question: str
    faithfulness: int
    completeness: int
    exam_appropriateness: int
    citation_quality: int
    rationales: dict[str, str] = field(default_factory=dict)
    raw: str = ""  # the model's raw reply, kept for debugging a bad parse

    @property
    def overall(self) -> float:
        """Mean of the four 1-5 criteria scores."""
        return sum(getattr(self, c) for c in CRITERIA) / len(CRITERIA)


def _build_user_prompt(
    question: str, answer: str, sources: list[dict[str, Any]], max_source_chars: int = 1000
) -> str:
    """Render the numbered sources + answer into the judge's user turn.

    Each source's text is truncated to ``max_source_chars`` to keep the request modest —
    8 full parent sections (~2k chars each) make for a large prompt. Truncation leaves
    enough of each section to grade faithfulness/citation against without bloating latency.
    """
    blocks = []
    for i, s in enumerate(sources, start=1):
        title = " > ".join(s.get("section_path") or []) or s.get("chapter_title", "Unknown")
        text = s.get("text", "")
        if max_source_chars and len(text) > max_source_chars:
            text = text[:max_source_chars].rstrip() + " …[truncated]"
        blocks.append(f"[{i}] {title}\n{text}")
    sources_block = "\n\n".join(blocks) if blocks else "(no sources)"
    return (
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{sources_block}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Grade the ANSWER and return the json object described in the instructions."
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def _parse_scores(raw: str) -> tuple[dict[str, int], dict[str, str]]:
    """Parse the judge's JSON reply into {criterion: score} + {criterion: reason}.

    Tries a direct ``json.loads`` first; falls back to grabbing the outermost ``{...}``
    if the model wrapped the object in stray prose. Clamps scores to 1..5.
    """
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            raise ValueError(f"judge returned no JSON object: {raw!r}")
        data = json.loads(m.group(0))

    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for c in CRITERIA:
        entry = data.get(c)
        if isinstance(entry, dict):
            val = entry.get("score")
            reasons[c] = str(entry.get("reason", ""))
        else:  # tolerate a bare number, e.g. {"faithfulness": 4}
            val = entry
            reasons[c] = ""
        if val is None:
            raise ValueError(f"judge omitted criterion {c!r}: {raw!r}")
        scores[c] = max(1, min(5, int(round(float(val)))))
    return scores, reasons


def judge_answer(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    gold_articles: list[str] | None = None,  # accepted for a uniform call signature; unused
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_JUDGE_BASE_URL,
    api_key: str | None = None,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    max_source_chars: int = 1500,
    client: OpenAI | None = None,
) -> JudgeScore:
    """Score one answer on the 4-criterion rubric via the OpenAI-hosted judge (gpt-5-mini).

    Makes a single JSON-mode chat call. For reasoning models (gpt-5 / o-series) it omits
    ``temperature`` (they reject anything but the default) and passes ``reasoning_effort``;
    for other models it passes ``temperature`` as usual. Retries up to 4x: a longer wait on
    rate-limit errors and a short backoff on other transient errors. Pass ``client`` to reuse
    one ``OpenAI`` instance across a run; ``max_source_chars`` truncates each source.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — add it to .env to use the LLM judge "
            "(key from https://platform.openai.com/api-keys)."
        )
    client = client or OpenAI(base_url=base_url, api_key=api_key)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(question, answer, sources, max_source_chars)},
    ]
    # Reasoning models (gpt-5/o-series) reject temperature!=1; give them reasoning_effort
    # instead. Non-reasoning models keep the deterministic temperature.
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if _is_reasoning_model(model):
        request_kwargs["reasoning_effort"] = reasoning_effort
    else:
        request_kwargs["temperature"] = temperature

    raw = ""
    attempts = 4
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(**request_kwargs)
            raw = resp.choices[0].message.content or ""
            scores, reasons = _parse_scores(raw)
            return JudgeScore(
                question=question,
                faithfulness=scores["faithfulness"],
                completeness=scores["completeness"],
                exam_appropriateness=scores["exam_appropriateness"],
                citation_quality=scores["citation_quality"],
                rationales=reasons,
                raw=raw,
            )
        except Exception as exc:  # transient API error or an unparseable reply
            if attempt == attempts - 1:
                raise
            # Rate-limit (429/413) waits out the provider's per-minute window;
            # other transient errors just get a short exponential backoff.
            is_rate_limit = "rate_limit" in str(exc) or "429" in str(exc) or "413" in str(exc)
            wait = 30 if is_rate_limit else 2 ** attempt
            print(f"  Judge attempt {attempt + 1} failed ({exc}), retrying in {wait}s…")
            time.sleep(wait)

    raise RuntimeError(f"judge failed after retries; last reply: {raw!r}")


@dataclass
class JudgeReport:
    n: int
    faithfulness: float | None
    completeness: float | None
    exam_appropriateness: float | None
    citation_quality: float | None
    overall: float | None
    per_question: list[JudgeScore] = field(default_factory=list)


def aggregate_judge(scores: list[JudgeScore]) -> JudgeReport:
    """Average each 1-5 criterion (and the overall) across the run."""
    n = len(scores)
    mean = (lambda key: sum(getattr(s, key) for s in scores) / n if n else None)
    return JudgeReport(
        n=n,
        faithfulness=mean("faithfulness"),
        completeness=mean("completeness"),
        exam_appropriateness=mean("exam_appropriateness"),
        citation_quality=mean("citation_quality"),
        overall=(sum(s.overall for s in scores) / n if n else None),
        per_question=scores,
    )
