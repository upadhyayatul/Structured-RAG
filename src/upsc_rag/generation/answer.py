"""Prompt builder and LLM call for grounded, source-cited answers from retrieved chunks."""
from __future__ import annotations

import os
from typing import Any, Iterator

from openai import OpenAI

_SYSTEM_PROMPT = (
    "You are a precise assistant for UPSC Indian Polity preparation. "
    "Answer strictly from the provided sources, cite the source numbers you "
    "use (e.g. [1], [3]), and never invent facts. If the sources do not "
    "contain the answer, say so plainly."
)


def build_answer_prompt(query: str, contexts: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered-source prompt ready for an LLM call.

    Each context dict must contain 'text'; 'section_path', 'chapter_title', and
    'page_start' are used for the citation line. Pass the output directly as the
    user turn to Claude or OpenAI chat completions.
    """
    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        title = " > ".join(ctx.get("section_path") or []) or ctx.get("chapter_title", "Unknown")
        pages = ctx.get("page_start")
        cite = f"{title} (p. {pages})" if pages else title
        blocks.append(f"[{i}] {cite}\n{ctx.get('text', '')}")
    context_block = "\n\n".join(blocks)
    return (
        "Answer using only the sources below. Cite source numbers. "
        "If the answer is not in the sources, say so.\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Answer:"
    )


def generate_answer(
    query: str,
    contexts: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: OpenAI | None = None,
) -> str:
    """
    Build the grounded prompt and call the LLM to produce a cited answer.

    Reads model/temperature/max_tokens from cfg["generation"]. Pass `client`
    to reuse an existing OpenAI instance; otherwise one is built from
    OPENAI_API_KEY in the environment.
    """
    gen_cfg = cfg.get("generation", {})
    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=gen_cfg.get("model", "gpt-4o-mini"),
        temperature=gen_cfg.get("temperature", 0.2),
        max_tokens=gen_cfg.get("max_tokens", 1024),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_answer_prompt(query, contexts)},
        ],
    )
    return response.choices[0].message.content or ""


def generate_answer_stream(
    query: str,
    contexts: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: OpenAI | None = None,
) -> Iterator[str]:
    """Yield the answer text incrementally as the LLM streams tokens."""
    gen_cfg = cfg.get("generation", {})
    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    stream = client.chat.completions.create(
        model=gen_cfg.get("model", "gpt-4o-mini"),
        temperature=gen_cfg.get("temperature", 0.2),
        max_tokens=gen_cfg.get("max_tokens", 1024),
        stream=True,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_answer_prompt(query, contexts)},
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
