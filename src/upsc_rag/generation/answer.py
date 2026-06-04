"""
Handles answer generation using Large Language Models (LLMs).
Provides functionalities to synthesize answers from retrieved
context documents to satisfy user queries.
"""
from __future__ import annotations

from typing import Any


def build_answer_prompt(query: str, contexts: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a grounded-answer prompt."""
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
