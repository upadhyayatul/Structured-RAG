"""Prompt builder and LLM call for grounded, source-cited answers from retrieved chunks."""
from __future__ import annotations

from typing import Any, Iterator

from openai import OpenAI

from upsc_rag.llm.clients import get_openai_client
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
    "End with notable exceptions or conditions if the sources mention any.\n\n"
    "Earlier conversation turns may be provided before the current question — use them "
    "ONLY to understand what the current question refers to (e.g. to resolve pronouns "
    "like 'it' or 'they'). Still answer only the current question, grounded solely in "
    "the sources supplied for it, and cite those sources as instructed above."
)


# Agentic path: sources may come from the textbook AND/OR the web (see agent/). Unlike
# _SYSTEM_PROMPT (book-only, no outside knowledge), this permits grounding in web
# sources for current/post-2011 info the 2011 textbook can't contain — still strictly
# from the supplied sources, never model memory, and still fully cited.
_AGENTIC_SYSTEM_PROMPT = (
    "You are a precise assistant for UPSC Indian Polity preparation. You are given "
    "numbered sources from two kinds of place: the M. Laxmikanth 'Indian Polity' "
    "TEXTBOOK (settled constitutional facts, but 6th ed. — dated to 2011) and the WEB "
    "(latest/current information the textbook may predate). These sources are your "
    "ENTIRE universe of knowledge for this question.\n\n"
    "GROUND EVERY CLAIM in the sources — do not use outside or prior knowledge. Do NOT "
    "add names, numbers, dates, case law, judgments, or Article numbers that are not "
    "written in a source, even if you are confident they are true.\n"
    "- Prefer the TEXTBOOK for settled constitutional provisions, structure, and "
    "procedures.\n"
    "- Use the WEB sources for recent constitutional amendments, post-2011 Supreme "
    "Court judgments, current office-holders, and recent events; when a claim is "
    "current or recent, say so and cite the web source.\n"
    "- If the textbook and the web conflict (e.g. the book is out of date), follow the "
    "web for the current position and note that the textbook predates it.\n"
    "- If neither the textbook nor the web sources cover the question, say so plainly "
    "instead of filling the gap from memory.\n\n"
    "CITE ACCURATELY — this is mandatory: end every factual sentence, bullet, and "
    "procedural step with the bracketed number(s) of the source(s) that actually state "
    "that specific claim, e.g. '... [1].' or '... [2][4].'. Never cite a source that "
    "does not support the claim. Use only the source numbers supplied.\n\n"
    "Open with a one-sentence direct answer; name the governing Constitutional "
    "Article(s) in **bold** when a source ties one to the claim. Then use short `##` "
    "headings, bullet points, and **bold** the key operative terms. Earlier "
    "conversation turns may precede the question — use them ONLY to resolve references; "
    "still answer only the current question, grounded solely in the supplied sources."
)


def build_agentic_prompt(
    query: str,
    contexts: list[dict[str, Any]],
    web: list[dict[str, Any]],
) -> str:
    """Format textbook + web sources into one numbered-source prompt for synthesis.

    Textbook sources come first (deduped like ``build_source_dicts``), then web sources
    (deduped by URL), continuing the same numbering — so the ``[n]`` markers the model
    emits line up with ``build_agentic_sources`` (sources.py).
    """
    from upsc_rag.generation.sources import dedupe_results

    blocks: list[str] = []
    n = 0
    for ctx in dedupe_results(contexts):
        n += 1
        title = " > ".join(ctx.get("section_path") or []) or ctx.get("chapter_title", "Unknown")
        pages = ctx.get("page_start")
        cite = f"{title} (p. {pages})" if pages else title
        header = f"[{n}] TEXTBOOK — {cite}"
        ents = ", ".join(e for e in (ctx.get("entities") or []) if e)
        if ents:
            header += f" — Articles: {ents}"
        blocks.append(f"{header}\n{ctx.get('text', '')}")

    seen_urls: set[str] = set()
    for w in web:
        url = w.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        n += 1
        blocks.append(f"[{n}] WEB — {w.get('title', '')} ({url})\n{w.get('snippet', '')}")

    context_block = "\n\n".join(blocks) or "(no sources found)"
    return (
        "Answer using only the sources below — do not use any outside knowledge. Prefer "
        "the TEXTBOOK for settled provisions and the WEB sources for latest/recent "
        "information. End every sentence and bullet with the bracketed number(s) of the "
        "source(s) that actually state it, e.g. [1] or [2][3]; never cite a source that "
        "does not support the claim. If the sources do not cover part or all of the "
        "question, say so plainly instead of filling the gap from memory.\n\n"
        f"Question: {query}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Answer:"
    )


def generate_agentic_answer(
    query: str,
    contexts: list[dict[str, Any]],
    web: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: OpenAI | None = None,
    session_id: str | None = None,
    usage_sink: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Synthesize a grounded, cited answer over combined textbook + web sources."""
    gen_cfg = cfg.get("generation", {})
    client = client or get_openai_client()
    model = gen_cfg.get("model", "gpt-4o-mini")
    hist = _history_messages(history, cfg.get("conversation", {}).get("history_turns", 3))
    messages = [
        {"role": "system", "content": _AGENTIC_SYSTEM_PROMPT},
        *hist,
        {"role": "user", "content": build_agentic_prompt(query, contexts, web)},
    ]

    with trace_manager.trace(
        "agentic_answer",
        input={"query": query, "num_book": len(contexts), "num_web": len(web)},
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


def generate_agentic_answer_stream(
    query: str,
    contexts: list[dict[str, Any]],
    web: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: OpenAI | None = None,
    session_id: str | None = None,
    usage_sink: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Stream the synthesized answer over combined textbook + web sources."""
    gen_cfg = cfg.get("generation", {})
    client = client or get_openai_client()
    model = gen_cfg.get("model", "gpt-4o-mini")
    hist = _history_messages(history, cfg.get("conversation", {}).get("history_turns", 3))
    messages = [
        {"role": "system", "content": _AGENTIC_SYSTEM_PROMPT},
        *hist,
        {"role": "user", "content": build_agentic_prompt(query, contexts, web)},
    ]

    with trace_manager.trace(
        "agentic_answer",
        input={"query": query, "num_book": len(contexts), "num_web": len(web)},
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


def _history_messages(
    history: list[dict[str, Any]] | None, history_turns: int
) -> list[dict[str, str]]:
    """Windowed prior turns as chat messages: last ``history_turns`` exchanges (2N msgs)."""
    if not history:
        return []
    turns = [
        {"role": t["role"], "content": (t.get("content") or "").strip()}
        for t in history
        if t.get("role") in ("user", "assistant") and (t.get("content") or "").strip()
    ]
    return turns[-history_turns * 2:] if history_turns > 0 else turns


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
    history: list[dict[str, Any]] | None = None,
) -> str:
    """
    Build the grounded prompt and call the LLM to produce a cited answer.

    Reads model/temperature/max_tokens from cfg["generation"]. Pass `client`
    to reuse an existing OpenAI instance; otherwise one is built from
    OPENAI_API_KEY in the environment. `history` (prior {role, content} turns) is
    windowed via cfg["conversation"] and inserted before the current question so
    the model can resolve follow-up references.
    """
    gen_cfg = cfg.get("generation", {})
    client = client or get_openai_client()
    model = gen_cfg.get("model", "gpt-4o-mini")
    prompt = build_answer_prompt(query, contexts)
    hist = _history_messages(history, cfg.get("conversation", {}).get("history_turns", 3))
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *hist,
        {"role": "user", "content": prompt},
    ]

    with trace_manager.trace(
        "answer",
        input={"query": query, "num_sources": len(contexts), "num_history": len(hist)},
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
    history: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Yield the answer text incrementally as the LLM streams tokens.

    `history` (prior {role, content} turns) is windowed via cfg["conversation"] and
    inserted before the current question so the model can resolve follow-up references.
    """
    gen_cfg = cfg.get("generation", {})
    client = client or get_openai_client()
    model = gen_cfg.get("model", "gpt-4o-mini")
    hist = _history_messages(history, cfg.get("conversation", {}).get("history_turns", 3))
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *hist,
        {"role": "user", "content": build_answer_prompt(query, contexts)},
    ]

    with trace_manager.trace(
        "answer",
        input={"query": query, "num_sources": len(contexts), "num_history": len(hist)},
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
