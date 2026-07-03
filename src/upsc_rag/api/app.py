"""FastAPI service: exposes the hybrid-retrieval + generation pipeline over HTTP.

Run with:
    "P:/ML-AI/git repo/Structured-RAG/.venv/Scripts/python.exe" -m uvicorn \
        upsc_rag.api.app:app --reload --port 8000

The retriever (BM25 index + Qdrant client) is built once at startup and reused
across requests. The Next.js frontend POSTs to /ask and renders {answer, sources}.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Iterator, Literal

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.generation.answer import (
    generate_agentic_answer_stream,
    generate_answer,
    generate_answer_stream,
)
from upsc_rag.generation.condense import condense_query
from upsc_rag.generation.router import OUT_OF_SCOPE_REPLY, is_off_topic, smalltalk_reply
from upsc_rag.generation.sources import build_agentic_sources, build_source_dicts
from upsc_rag.retrieval.hybrid import HybridRetriever

# Default book served by the API; override with UPSC_RAG_BOOK env var.
BOOK_ID = os.environ.get("UPSC_RAG_BOOK", "laxmikanth_6")

# Orchestration backend (UPSC_RAG_PIPELINE): "graph" routes through the linear LangGraph
# pipeline; "agentic" routes through the tool-calling agent (textbook + web search);
# anything else uses the direct path (default). All three are built once in lifespan.
_PIPELINE = os.environ.get("UPSC_RAG_PIPELINE", "").lower()
USE_GRAPH = _PIPELINE == "graph"
USE_AGENTIC = _PIPELINE == "agentic"

# Populated at startup, reused across requests.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the retriever + config once when the server boots."""
    settings = get_settings()
    cfg = load_runtime_config(BOOK_ID)
    chunks_path = settings.resolve(settings.processed_dir) / BOOK_ID / "chunks.jsonl"
    _state["cfg"] = cfg
    retriever = HybridRetriever(cfg, chunks_path)
    _state["retriever"] = retriever
    # Build the selected orchestration graph once (reused per request).
    if USE_GRAPH:
        from upsc_rag.graph import build_ask_graph

        _state["graph"] = build_ask_graph(retriever, cfg)
    elif USE_AGENTIC:
        from upsc_rag.agent import build_agentic_graph

        _state["graph"] = build_agentic_graph(retriever, cfg)
    yield
    _state.clear()


app = FastAPI(title="UPSC-RAG API", lifespan=lifespan)

# Allow the Next.js dev server (and prod origin) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "UPSC_RAG_CORS_ORIGINS", "http://localhost:3000"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question")
    history: list[Turn] | None = Field(
        default=None,
        description="Prior conversation turns (recent first-to-last) for follow-up resolution",
    )
    top_k: int | None = Field(default=None, description="Dense+BM25 candidate pool size")
    rerank_top_k: int | None = Field(default=None, description="Sources passed to the LLM")
    session_id: str | None = Field(
        default=None,
        description="Conversation id; groups this question's traces in Langfuse Sessions",
    )


def _history_and_search_query(req: "AskRequest", cfg: dict[str, Any]) -> tuple[
    list[dict[str, Any]] | None, str
]:
    """Resolve conversation history + the (possibly condensed) query used for retrieval.

    Returns ``(history_dicts, search_query)``. When conversation is enabled and history
    is present, the follow-up is condensed into a standalone search query; otherwise the
    raw query is used and history is None. The raw ``req.query`` is always what the answer
    LLM sees (with history) — condensing only affects retrieval.
    """
    conv_cfg = cfg.get("conversation", {})
    if not (conv_cfg.get("enabled", True) and req.history):
        return None, req.query
    history = [t.model_dump() for t in req.history]
    search_query = condense_query(
        req.query,
        history,
        model=conv_cfg.get("condense_model", "gpt-4.1-nano"),
        session_id=req.session_id,
    )
    return history, search_query


class Source(BaseModel):
    """A citable source — either a textbook section (``type="book"``) or a web result
    (``type="web"``). Book fields (section_path/chapter_title/pages) and web fields
    (title/url/snippet) are both optional so the one model carries both kinds; the
    agentic path returns a mix, the direct/graph paths return only book sources."""

    n: int
    type: str = "book"
    # book source
    section_path: list[str] | None = None
    chapter_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    # web source
    title: str | None = None
    url: str | None = None
    snippet: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _build_sources(results: list[dict[str, Any]]) -> list[Source]:
    """Turn retrieval results into Source objects, deduped by section+pages, renumbered."""
    return [Source(**s) for s in build_source_dicts(results)]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check; reports whether the retriever is loaded."""
    return {"status": "ok" if "retriever" in _state else "loading", "book": BOOK_ID}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Retrieve sources for the question and generate a grounded, cited answer."""
    retriever: HybridRetriever | None = _state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready")

    # Graph / agentic mode: the compiled pipeline runs the gates + retrieve + generate.
    if USE_GRAPH or USE_AGENTIC:
        if USE_AGENTIC:
            from upsc_rag.agent import run_ask
        else:
            from upsc_rag.graph import run_ask

        history = [t.model_dump() for t in req.history] if req.history else None
        final = run_ask(
            _state["graph"], req.query, top_k=req.top_k,
            rerank_top_k=req.rerank_top_k, session_id=req.session_id, history=history,
        )
        sources = [Source(**s) for s in final.get("sources", [])]
        return AskResponse(answer=final.get("answer", ""), sources=sources)

    # Gate 1: pure greeting / chit-chat — reply without retrieving or generating.
    canned = smalltalk_reply(req.query)
    if canned is not None:
        return AskResponse(answer=canned, sources=[])

    # Resolve follow-ups: condense history + query into a standalone retrieval query.
    history, search_query = _history_and_search_query(req, _state["cfg"])
    results = retriever.retrieve(
        search_query, top_k=req.top_k, rerank_top_k=req.rerank_top_k, session_id=req.session_id
    )

    # Gate 2: real question, but no relevant source in the book — skip generation.
    floor = _state["cfg"].get("retrieval", {}).get("relevance_floor", 0.0)
    if is_off_topic(results, floor):
        return AskResponse(answer=OUT_OF_SCOPE_REPLY, sources=[])

    answer = generate_answer(
        req.query, results, _state["cfg"], session_id=req.session_id, history=history
    )
    return AskResponse(answer=answer, sources=_build_sources(results))


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream the answer as NDJSON events: one 'sources' event, then 'token' events, then 'done'.

    Retrieval runs first (so sources are sent immediately), then generation tokens
    stream as they arrive — letting the client measure time-to-first-token.
    """
    retriever: HybridRetriever | None = _state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready")

    def canned_stream(message: str) -> Iterator[str]:
        """Emit a gated reply (smalltalk / out-of-scope) in the normal event shape.

        Gated replies make no LLM call, so cost is zero.
        """
        yield json.dumps({"type": "sources", "sources": []}) + "\n"
        yield json.dumps({"type": "token", "text": message}) + "\n"
        yield json.dumps(
            {"type": "done", "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
        ) + "\n"

    # History for the answer LLM; set below for both graph and direct paths.
    history: list[dict[str, Any]] | None

    # Agentic mode: run the smalltalk → domain gate → agent⇄tools loop to gather
    # textbook + web sources, then stream the final synthesis (its own generator +
    # combined source shape). Kept separate from the book-only graph/direct flow below.
    if USE_AGENTIC:
        from upsc_rag.agent import prepare_stream as agentic_prepare_stream

        history = [t.model_dump() for t in req.history] if req.history else None
        prep = agentic_prepare_stream(
            retriever, _state["cfg"], req.query, top_k=req.top_k,
            rerank_top_k=req.rerank_top_k, session_id=req.session_id, history=history,
        )
        if prep.get("route") != "answer":
            return StreamingResponse(
                canned_stream(prep["answer"]), media_type="application/x-ndjson"
            )

        contexts = prep.get("contexts") or []
        web = prep.get("web") or []
        sources = [Source(**s) for s in build_agentic_sources(contexts, web)]

        def agentic_event_stream() -> Iterator[str]:
            yield json.dumps(
                {"type": "sources", "sources": [s.model_dump() for s in sources]}
            ) + "\n"
            usage_sink: dict[str, Any] = {}
            for delta in generate_agentic_answer_stream(
                req.query, contexts, web, _state["cfg"],
                session_id=req.session_id, usage_sink=usage_sink, history=history,
            ):
                yield json.dumps({"type": "token", "text": delta}) + "\n"
            yield json.dumps({
                "type": "done",
                "cost_usd": usage_sink.get("cost_usd", 0.0),
                "input_tokens": usage_sink.get("input_tokens", 0),
                "output_tokens": usage_sink.get("output_tokens", 0),
            }) + "\n"

        return StreamingResponse(agentic_event_stream(), media_type="application/x-ndjson")

    # Graph mode: run the smalltalk → retrieve → gate prefix through the graph nodes,
    # then stream tokens below exactly as the direct path does (same NDJSON contract).
    if USE_GRAPH:
        from upsc_rag.graph import prepare_stream

        history = [t.model_dump() for t in req.history] if req.history else None
        prep = prepare_stream(
            retriever, _state["cfg"], req.query, top_k=req.top_k,
            rerank_top_k=req.rerank_top_k, session_id=req.session_id, history=history,
        )
        if prep["route"] != "answer":
            return StreamingResponse(
                canned_stream(prep["answer"]), media_type="application/x-ndjson"
            )
        results = prep["results"]
    else:
        # Gate 1: pure greeting / chit-chat — reply without retrieving or generating.
        canned = smalltalk_reply(req.query)
        if canned is not None:
            return StreamingResponse(canned_stream(canned), media_type="application/x-ndjson")

        # Resolve follow-ups: condense history + query into a standalone retrieval query.
        history, search_query = _history_and_search_query(req, _state["cfg"])
        results = retriever.retrieve(
            search_query, top_k=req.top_k, rerank_top_k=req.rerank_top_k, session_id=req.session_id
        )

        # Gate 2: real question, but no relevant source in the book — skip generation.
        floor = _state["cfg"].get("retrieval", {}).get("relevance_floor", 0.0)
        if is_off_topic(results, floor):
            return StreamingResponse(
                canned_stream(OUT_OF_SCOPE_REPLY), media_type="application/x-ndjson"
            )

    sources = _build_sources(results)

    def event_stream() -> Iterator[str]:
        yield json.dumps({"type": "sources", "sources": [s.model_dump() for s in sources]}) + "\n"
        # generate_answer_stream traces its own LLM generation (see generation/answer.py).
        # usage_sink is filled with token counts + estimated cost once the stream ends.
        usage_sink: dict[str, Any] = {}
        for delta in generate_answer_stream(
            req.query, results, _state["cfg"], session_id=req.session_id,
            usage_sink=usage_sink, history=history,
        ):
            yield json.dumps({"type": "token", "text": delta}) + "\n"
        yield json.dumps({
            "type": "done",
            "cost_usd": usage_sink.get("cost_usd", 0.0),
            "input_tokens": usage_sink.get("input_tokens", 0),
            "output_tokens": usage_sink.get("output_tokens", 0),
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
