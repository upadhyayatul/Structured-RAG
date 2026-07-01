"""Prompt builder and LLM call for grounded, source-cited answers from retrieved chunks."""
from __future__ import annotations

import os
from typing import Any, Iterator

from openai import OpenAI

from upsc_rag.observability import trace_manager

# Approx USD per 1M tokens, by model (OpenAI list prices — update if they change).
# Used to show a rough per-answer cost in the UI; embeddings/rewrite are tiny next
# to generation, so the displayed figure is the answer-generation cost.
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Approximate USD cost for a model call from its token counts (0 if unpriced)."""
    p = _PRICING.get(model)
    if not p:
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def _fill_usage_sink(sink: dict[str, Any] | None, model: str, usage: Any) -> None:
    """Populate ``sink`` (if given) with token counts + estimated cost for one call."""
    if sink is None or usage is None:
        return
    sink["input_tokens"] = usage.prompt_tokens
    sink["output_tokens"] = usage.completion_tokens
    sink["cost_usd"] = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)


_SYSTEM_PROMPT = (
    "You are a precise assistant for UPSC Indian Polity preparation. The sources "
    "provided below are your ENTIRE universe of knowledge for this question.\n\n"
    "GROUND EVERY CLAIM — do not use outside or prior knowledge:\n"
    "- Use only facts written in the sources. Do NOT add names, numbers, dates, "
    "case law, judgments, committee findings, Article numbers, or other specifics "
    "that are not written in a source, even if you are confident they are true — the "
    "textbook may be dated or incomplete, and any unverifiable addition is treated "
    "as an error.\n"
    "- Do NOT sharpen or elaborate beyond what a source states. If a source is "
    "general, keep it general — do not turn it into a specific rule, count, or named "
    "procedure from your own knowledge (e.g. do not expand 'consultation with the "
    "judges' into 'a collegium of the four seniormost judges'). Stay at the level of "
    "detail the sources use.\n"
    "- Before writing that the sources do not cover something, check ALL the sources "
    "below — say so only if none of them contain it. If the sources cover the "
    "question only partially, answer the part they cover, note what they omit, and "
    "stop — never fill the gap from memory. When coverage is thin, qualify "
    "('according to the sources...') rather than asserting.\n\n"
    "CITE ACCURATELY — this is mandatory:\n"
    "- End every factual sentence, bullet, and procedural step with the bracketed "
    "number(s) of the source(s) that actually state that specific claim, e.g. "
    "'... is appointed by the President [1].' or '... after due inquiry [2][4].'\n"
    "- A citation must point to a source that genuinely contains the fact. NEVER "
    "cite a source merely because it is on a related topic. If no source supports a "
    "sentence, do not write that sentence. Use only the source numbers supplied.\n\n"
    "State the governing Constitutional Article(s) explicitly — they are listed with "
    "each source under 'Articles:'; name an Article only when a source ties it to the "
    "claim. Open with a one-sentence direct answer that names the relevant Article(s) "
    "in **bold** (with its source citation). Then use Markdown structure: short `##` "
    "headings to group ideas, bullet points, and **bold** the key operative terms. "
    "End with notable exceptions or conditions if the sources mention any."
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
        "Answer using only the sources below — do not use any outside knowledge. End "
        "every sentence and bullet with the bracketed number(s) of the source(s) that "
        "actually state it, e.g. [1] or [2][3]; never cite a source that does not "
        "support the claim. Name a Constitutional Article only when a source ties it "
        "to the claim (each source lists its Articles in the header). If the sources "
        "do not cover part or all of the question, say so plainly instead of filling "
        "the gap from memory.\n\n"
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
    usage_sink: dict[str, Any] | None = None,
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
            _fill_usage_sink(usage_sink, model, response.usage)
        trace.end(output={"answer_chars": len(text)})
        return text


def generate_answer_stream(
    query: str,
    contexts: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: OpenAI | None = None,
    session_id: str | None = None,
    usage_sink: dict[str, Any] | None = None,
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
            _fill_usage_sink(usage_sink, model, usage)
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
