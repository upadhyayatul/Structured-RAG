"""Prompt builder for grounded, source-cited answers from retrieved chunks."""
from __future__ import annotations

from typing import Any


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
