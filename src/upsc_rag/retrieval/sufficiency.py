"""Do the retrieved textbook sections actually answer the question?

Relevance and sufficiency are different things, and the relevance floor
(``generation/router.py``) only measures the former. "Why does the President appoint the
PM — what did the Constitution makers intend?" retrieves the *right* chapter with a high
cosine, yet Laxmikanth's Article 75 sections say nothing about the framers' intent, so the
answer degenerates to "the sources do not specify". Constituent-Assembly rationale,
post-2011 amendments and current office-holders are all gaps the 2011 textbook cannot fill.

This module asks a cheap LLM whether the retrieved sections contain enough to answer. A
"no" is what triggers the web fallback in ``api/app.py``; the web results are then merged
with the book sections by the existing agentic prompt.

Fails OPEN (``True`` = sufficient): if the classifier errors, we degrade to today's
textbook-only behavior rather than spraying web searches. Same never-crash discipline as
``retrieval/rewrite.py`` and ``retrieval/web.py``.
"""
from __future__ import annotations

from typing import Any

from upsc_rag.llm.clients import get_openai_client
from upsc_rag.observability import trace_manager

_SUFFICIENCY_SYSTEM = (
    "You check whether an excerpt from an Indian Polity textbook contains enough "
    "information to answer a student's question.\n"
    "Answer 'yes' if the sources state the facts the question asks for — even partially, "
    "as long as a substantive answer can be grounded in them.\n"
    "Answer 'no' when the sources are merely ON THE TOPIC but do not contain what was "
    "asked. In particular answer 'no' when the question asks for something the sources "
    "never state: the intent/rationale of the Constitution makers or Constituent Assembly "
    "debates, events or amendments after 2011, who currently holds an office, or specific "
    "figures/dates/judgments absent from the text.\n"
    "Reply with exactly one word: yes or no."
)


def _format_sources(results: list[dict[str, Any]], max_chars: int) -> str:
    """Render the retrieved sections (heading + head of the text) for the classifier."""
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        path = " > ".join(r.get("section_path") or []) or r.get("chapter_title") or "?"
        text = (r.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{i}] {path}\n{text[:max_chars]}")
    return "\n\n".join(lines)


def sources_answer_question(
    query: str,
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
    session_id: str | None = None,
    parent: Any = None,
) -> bool:
    """True when ``results`` contain enough to answer ``query`` (so no web fallback).

    Returns True unchanged (no LLM call) when the fallback is disabled in config or there
    is nothing to judge.
    """
    fb_cfg = (cfg.get("retrieval", {}) or {}).get("web_fallback", {}) or {}
    if not fb_cfg.get("enabled", False) or not results:
        return True

    model = fb_cfg.get("model", "gpt-4.1-nano")
    messages = [
        {"role": "system", "content": _SUFFICIENCY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\nSources:\n"
                f"{_format_sources(results, fb_cfg.get('max_source_chars', 600))}\n\n"
                "Do these sources contain enough to answer the question? (yes/no)"
            ),
        },
    ]
    try:
        with trace_manager.start(
            "sufficiency", parent=parent, input={"query": query, "num_sources": len(results)},
            session_id=session_id,
        ) as trace:
            gen = trace.generation(
                "sufficiency_llm", model=model, input=messages,
                model_parameters={"temperature": 0.0},
            )
            with gen:
                resp = get_openai_client().chat.completions.create(
                    model=model, temperature=0.0, max_tokens=4, messages=messages,
                )
                out = (resp.choices[0].message.content or "").strip().lower()
                gen.end(output=out, usage=_usage(resp.usage))
            sufficient = not out.startswith("no")
            trace.end(output={"sufficient": sufficient, "raw": out})
        return sufficient
    except Exception:
        # Classifier unavailable -> assume the book suffices (today's behavior).
        return True


def _usage(usage: Any) -> dict[str, int]:
    """Token counts in Langfuse's shape; empty when the provider omitted usage."""
    if usage is None:
        return {}
    return {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "total": usage.total_tokens,
    }
