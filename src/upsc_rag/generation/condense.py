"""History-aware query condensing: rewrite a follow-up into a standalone search query.

Retrieval embeds the query STRING alone, so a follow-up like "why was it created"
(no subject) retrieves the wrong sections. Given the recent conversation, a small LLM
rewrites it into a self-contained question ("Why was NITI Aayog created?") BEFORE
retrieval runs. Generation still receives the raw history separately (see answer.py);
this step only fixes the retrieval query.
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from upsc_rag.generation.answer import _usage_dict
from upsc_rag.llm.clients import get_openai_client
from upsc_rag.observability import trace_manager

_CONDENSE_SYSTEM = (
    "You rewrite a user's latest question into a single standalone question by "
    "resolving references (it / they / this / that / he / she / that one) using the "
    "conversation so far, and carrying over the specific entity/topic and any "
    "constraints established in earlier turns.\n"
    "Rules:\n"
    "- Output the SHORTEST self-contained question that preserves the original intent.\n"
    "- Substitute the referenced entity in place of the pronoun and change NOTHING "
    "else about the wording.\n"
    "- Do NOT answer it. Do NOT add explanations, context, framing, sources, or extra "
    "qualifiers (e.g. never append phrases like 'in the context of ...' or 'according "
    "to the textbook').\n"
    "- If the latest question is already self-contained, return it unchanged.\n"
    "Output ONLY the rewritten question, as one line of plain text."
)


def _format_history(history: list[dict[str, Any]]) -> str:
    """Render prior turns as a compact transcript for the condense prompt."""
    lines: list[str] = []
    for turn in history:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        who = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {content}")
    return "\n".join(lines)


def condense_query(
    query: str,
    history: list[dict[str, Any]] | None,
    client: OpenAI | None = None,
    model: str = "gpt-4.1-nano",
    session_id: str | None = None,
) -> str:
    """Return a standalone version of ``query`` given the prior conversation turns.

    No history -> returns ``query`` unchanged with NO LLM call (first-turn no-op).
    On any failure -> returns ``query`` unchanged (retrieval must never break).
    """
    transcript = _format_history(history or [])
    if not transcript:
        return query

    client = client or get_openai_client()
    messages = [
        {"role": "system", "content": _CONDENSE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{transcript}\n\n"
                f"Latest question: {query}\n\nStandalone question:"
            ),
        },
    ]
    try:
        with trace_manager.trace(
            "condense", input={"query": query}, session_id=session_id
        ) as trace:
            gen = trace.generation(
                "condense_llm",
                model=model,
                input=messages,
                model_parameters={"temperature": 0.0},
            )
            with gen:
                resp = client.chat.completions.create(
                    model=model, temperature=0.0, messages=messages
                )
                out = (resp.choices[0].message.content or "").strip()
                gen.end(output=out, usage=_usage_dict(resp.usage))
            trace.end(output={"standalone": out})
        return out or query
    except Exception:
        # Any failure (API error, bad response) must not break retrieval.
        return query
