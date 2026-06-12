"""Prompt builder and LLM call for grounded, source-cited answers from retrieved chunks."""
from __future__ import annotations

import os
from typing import Any, Iterator

from openai import OpenAI

from upsc_rag.observability import trace_manager

_SYSTEM_PROMPT = (
    "You are a precise assistant for UPSC Indian Polity preparation. "
    "Answer strictly from the provided sources, cite the source numbers you "
    "use (e.g. [1], [3]), and never invent facts. If the sources do not "
    "contain the answer, say so plainly.\n\n"
    "State the governing Constitutional Article(s) explicitly — they are listed "
    "with each source under 'Articles:'. Open with a one-sentence direct answer "
    "that names the relevant Article(s) in **bold**. Then use Markdown structure: "
    "short `##` headings to group ideas, bullet points, and **bold** the key "
    "operative terms. End with notable exceptions or conditions if the sources "
    "mention any."
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
        header = f"[{i}] {cite}"
        ents = ", ".join(e for e in (ctx.get("entities") or []) if e)
        if ents:
            header += f" — Articles: {ents}"
        blocks.append(f"{header}\n{ctx.get('text', '')}")
    context_block = "\n\n".join(blocks)
    return (
        "Answer using only the sources below. Cite source numbers. "
        "Name the relevant Constitutional Article(s) explicitly — each source "
        "lists its Articles in the header. If the answer is not in the sources, "
        "say so.\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Answer:"
    )


def generate_answer(
    query: str,
    contexts: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: OpenAI | None = None,
    session_id: str | None = None,
) -> str:
    """
    Build the grounded prompt and call the LLM to produce a cited answer.

    Reads model/temperature/max_tokens from cfg["generation"]. Pass `client`
    to reuse an existing OpenAI instance; otherwise one is built from
    OPENAI_API_KEY in the environment.
    """
    gen_cfg = cfg.get("generation", {})
    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = gen_cfg.get("model", "gpt-4o-mini")
    prompt = build_answer_prompt(query, contexts)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    with trace_manager.trace(
        "answer",
        input={"query": query, "num_sources": len(contexts)},
        session_id=session_id,
    ) as trace:
        gen = trace.generation(
            "llm",
            model=model,
            input=messages,
            model_parameters={
                "temperature": gen_cfg.get("temperature", 0.2),
                "max_tokens": gen_cfg.get("max_tokens", 1024),
            },
        )
        with gen:
            response = client.chat.completions.create(
                model=model,
                temperature=gen_cfg.get("temperature", 0.2),
                max_tokens=gen_cfg.get("max_tokens", 1024),
                messages=messages,
            )
            text = response.choices[0].message.content or ""
            gen.end(output=text, usage=_usage_dict(response.usage))
        trace.end(output={"answer_chars": len(text)})
        return text


def generate_answer_stream(
    query: str,
    contexts: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: OpenAI | None = None,
    session_id: str | None = None,
) -> Iterator[str]:
    """Yield the answer text incrementally as the LLM streams tokens."""
    gen_cfg = cfg.get("generation", {})
    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = gen_cfg.get("model", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": build_answer_prompt(query, contexts)},
    ]

    with trace_manager.trace(
        "answer",
        input={"query": query, "num_sources": len(contexts)},
        session_id=session_id,
    ) as trace:
        gen = trace.generation(
            "llm",
            model=model,
            input=messages,
            model_parameters={
                "temperature": gen_cfg.get("temperature", 0.2),
                "max_tokens": gen_cfg.get("max_tokens", 1024),
            },
        )
        with gen:
            stream = client.chat.completions.create(
                model=model,
                temperature=gen_cfg.get("temperature", 0.2),
                max_tokens=gen_cfg.get("max_tokens", 1024),
                stream=True,
                # Ask OpenAI for a final usage chunk so we can report token counts.
                stream_options={"include_usage": True},
                messages=messages,
            )
            parts: list[str] = []
            usage: Any = None
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield delta
            text = "".join(parts)
            gen.end(output=text, usage=_usage_dict(usage))
        trace.end(output={"answer_chars": len(text)})


def _usage_dict(usage: Any) -> dict[str, int] | None:
    """Map an OpenAI usage object to Langfuse's token fields, or None if absent."""
    if usage is None:
        return None
    return {
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "total": usage.total_tokens,
    }
