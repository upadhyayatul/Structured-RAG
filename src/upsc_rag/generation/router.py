"""Pre-retrieval query gating: short-circuit greetings/chit-chat and off-topic asks.

Two cheap gates keep the RAG pipeline from answering questions it shouldn't:

1. ``smalltalk_reply`` — a regex that matches a query that is *entirely* a greeting
   or pleasantry ("hi", "how are you", "thanks") and returns a canned reply. Runs
   before retrieval, so these cost zero embeddings / LLM tokens.
2. ``is_off_topic`` — a relevance floor on the top dense-cosine score from retrieval.
   A real question unrelated to the book (e.g. "what's the weather") retrieves only
   weak matches; below the floor we return ``OUT_OF_SCOPE_REPLY`` instead of letting
   the LLM hallucinate over irrelevant sources.
"""
from __future__ import annotations

import re
from typing import Any

# Anchored with ^...$ so only a query that is *wholly* smalltalk matches — a real
# question that merely contains "the"/"is"/"hi" (e.g. "What is the role of the
# President?") will never be misrouted here. First matching pattern wins.
_SMALLTALK: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^\s*(hi+|hey+|hello+|yo|hii+|namaste|greetings)\s*[!.?]*\s*$", re.I),
        "Hi! I'm your UPSC Indian Polity assistant. Ask me anything about the "
        "Constitution, Parliament, fundamental rights, the judiciary, and more.",
    ),
    (
        re.compile(
            r"^\s*(good\s+(morning|afternoon|evening|night))\s*[!.?]*\s*$", re.I
        ),
        "Hello! Ready when you are — what topic in Indian Polity can I help you with?",
    ),
    (
        re.compile(
            r"^\s*(how\s+(are\s+you|are\s+u|r\s+u|do\s+you\s+do)|how'?s\s+it\s+going"
            r"|what'?s\s+up|sup|wassup)\s*[!.?]*\s*$",
            re.I,
        ),
        "Doing well, thanks for asking! What would you like to know about Indian "
        "Polity?",
    ),
    (
        re.compile(
            r"^\s*(thanks?|thank\s+you|thx|ty|much\s+appreciated|great|nice|cool|ok|okay)"
            r"\s*[!.?]*\s*$",
            re.I,
        ),
        "You're welcome! Happy to help with any other polity questions.",
    ),
    (
        re.compile(
            r"^\s*(bye|goodbye|good\s+bye|see\s+you|see\s+ya|cya|take\s+care)\s*[!.?]*\s*$",
            re.I,
        ),
        "All the best with your preparation — come back anytime!",
    ),
    (
        re.compile(
            r"^\s*(who\s+are\s+you|what\s+(are|can)\s+you\s+do|what\s+is\s+this"
            r"|help|what\s+can\s+i\s+ask)\s*[!.?]*\s*$",
            re.I,
        ),
        "I'm a study assistant for M. Laxmikanth's *Indian Polity*. Ask me about any "
        "topic in the Indian Constitution and I'll answer with cited sources from the "
        "book — try \"What are the fundamental rights?\" or \"How is the President elected?\"",
    ),
]

# Returned when a genuine question has no good match in the book (off-topic / out
# of the book's scope). Keeps the bot from inventing an answer over weak sources.
OUT_OF_SCOPE_REPLY = (
    "I couldn't find anything about that in *Indian Polity* by M. Laxmikanth. "
    "I can only answer questions covered by the book — topics like the Constitution, "
    "Parliament, the judiciary, fundamental rights, federalism, and constitutional "
    "bodies. Could you rephrase your question around one of those?"
)


def smalltalk_reply(query: str) -> str | None:
    """Return a canned reply if ``query`` is purely smalltalk, else ``None``.

    A ``None`` result means "not smalltalk — proceed to retrieval + generation".
    """
    for pattern, reply in _SMALLTALK:
        if pattern.match(query):
            return reply
    return None


def is_off_topic(results: list[dict[str, Any]], floor: float) -> bool:
    """True when retrieval found no sufficiently-relevant source for the query.

    Uses the pass-level top dense-cosine score that ``HybridRetriever.retrieve``
    attaches to each result as ``dense_top_score``. Empty results also count as
    off-topic. A ``floor`` of 0 disables the gate.
    """
    if floor <= 0:
        return False
    if not results:
        return True
    return results[0].get("dense_top_score", 0.0) < floor
