"""LLM query rewriting: expand a user question into canonical retrieval variants.

Closes the vocabulary gap that stemming can't — synonyms ("selected" -> "appointed")
and abbreviations ("SC" -> "Supreme Court") — by asking a small LLM to rephrase the
question into the terminology the textbook actually uses. The variants are used for
multi-query retrieval (retrieve for each, then fuse), improving recall.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from upsc_rag.observability import NOOP_CONTEXT

_REWRITE_SYSTEM = (
    "You rewrite a user's question into search queries for an Indian Polity "
    "(M. Laxmikanth) textbook index. Expand abbreviations (SC -> Supreme Court, "
    "HC -> High Court, FR -> Fundamental Rights, FD -> Fundamental Duties, "
    "DPSP -> Directive Principles, PM -> Prime Minister, CJI -> Chief Justice of "
    "India), and replace colloquial "
    "verbs with the book's formal terminology (selected/chosen/nominated -> "
    "appointed/appointment; fired/sacked -> removed/removal). Produce concise, "
    "keyword-rich variants that capture the same intent. "
    'Respond ONLY as JSON: {"queries": ["...", "..."]}.'
)

# Process-level cache: rewriting is deterministic-ish (temperature 0) and questions
# repeat, so we avoid paying the LLM call twice for the same query.
_CACHE: dict[tuple[str, str, int], list[str]] = {}


def _call_llm(
    query: str, client: OpenAI, model: str, num_variants: int, obs: Any = NOOP_CONTEXT
) -> list[str]:
    """Ask the LLM for rephrasings; return [] on any failure (retrieval must not break)."""
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": f"Question: {query}\nGive up to {num_variants} variants."},
    ]
    # Billable LLM call — trace as a generation so its token usage and cost show up.
    gen = obs.generation("rewrite_llm", model=model, input=messages,
                         model_parameters={"temperature": 0.0})
    try:
        with gen:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = resp.choices[0].message.content or "{}"
            u = resp.usage
            gen.end(
                output=content,
                usage={"input": u.prompt_tokens, "output": u.completion_tokens, "total": u.total_tokens},
            )
        data: dict[str, Any] = json.loads(content)
        variants = data.get("queries", [])
        return [v for v in variants if isinstance(v, str) and v.strip()]
    except Exception:
        return []


def rewrite_query(
    query: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    num_variants: int = 3,
    obs: Any = NOOP_CONTEXT,
) -> list[str]:
    """Return the original query plus up to `num_variants` LLM rephrasings (original first, deduped)."""
    key = (query, model, num_variants)
    if key not in _CACHE:
        # Only a cache miss makes the (billable, traced) LLM call.
        _CACHE[key] = _call_llm(query, client, model, num_variants, obs=obs)

    out = [query]
    seen = {query.strip().lower()}
    for v in _CACHE[key]:
        norm = v.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(v.strip())
    return out[: num_variants + 1]
