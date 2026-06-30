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
from typing import Any, Iterator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from upsc_rag.config import get_settings, load_runtime_config
from upsc_rag.generation.answer import generate_answer, generate_answer_stream
from upsc_rag.generation.router import OUT_OF_SCOPE_REPLY, is_off_topic, smalltalk_reply
from upsc_rag.generation.sources import build_source_dicts
from upsc_rag.retrieval.hybrid import HybridRetriever

# Default book served by the API; override with UPSC_RAG_BOOK env var.
BOOK_ID = os.environ.get("UPSC_RAG_BOOK", "laxmikanth_6")

# Orchestration backend: "graph" routes /ask + /ask/stream through the LangGraph
# pipeline (built once in lifespan); anything else uses the direct path (default).
USE_GRAPH = os.environ.get("UPSC_RAG_PIPELINE", "").lower() == "graph"

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
    # Build the LangGraph pipeline once if graph mode is selected (reused per request).
    if USE_GRAPH:
        from upsc_rag.graph import build_ask_graph

        _state["graph"] = build_ask_graph(retriever, cfg)
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


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int | None = Field(default=None, description="Dense+BM25 candidate pool size")
    rerank_top_k: int | None = Field(default=None, description="Sources passed to the LLM")
    session_id: str | None = Field(
        default=None,
        description="Conversation id; groups this question's traces in Langfuse Sessions",
    )


class Source(BaseModel):
    n: int
    section_path: list[str]
    chapter_title: str
    page_start: int | None
    page_end: int | None


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

    # Graph mode: the LangGraph pipeline runs the same gates + retrieve + generate.
    if USE_GRAPH:
        from upsc_rag.graph import run_ask

        final = run_ask(
            _state["graph"], req.query, top_k=req.top_k,
            rerank_top_k=req.rerank_top_k, session_id=req.session_id,
        )
        sources = [Source(**s) for s in final.get("sources", [])]
        return AskResponse(answer=final.get("answer", ""), sources=sources)

    # Gate 1: pure greeting / chit-chat — reply without retrieving or generating.
    canned = smalltalk_reply(req.query)
    if canned is not None:
        return AskResponse(answer=canned, sources=[])

    results = retriever.retrieve(
        req.query, top_k=req.top_k, rerank_top_k=req.rerank_top_k, session_id=req.session_id
    )

    # Gate 2: real question, but no relevant source in the book — skip generation.
    floor = _state["cfg"].get("retrieval", {}).get("relevance_floor", 0.0)
    if is_off_topic(results, floor):
        return AskResponse(answer=OUT_OF_SCOPE_REPLY, sources=[])

    answer = generate_answer(req.query, results, _state["cfg"], session_id=req.session_id)
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

    # Graph mode: run the smalltalk → retrieve → gate prefix through the graph nodes,
    # then stream tokens below exactly as the direct path does (same NDJSON contract).
    if USE_GRAPH:
        from upsc_rag.graph import prepare_stream

        prep = prepare_stream(
            retriever, _state["cfg"], req.query, top_k=req.top_k,
            rerank_top_k=req.rerank_top_k, session_id=req.session_id,
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

        results = retriever.retrieve(
            req.query, top_k=req.top_k, rerank_top_k=req.rerank_top_k, session_id=req.session_id
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
            req.query, results, _state["cfg"], session_id=req.session_id, usage_sink=usage_sink
        ):
            yield json.dumps({"type": "token", "text": delta}) + "\n"
        yield json.dumps({
            "type": "done",
            "cost_usd": usage_sink.get("cost_usd", 0.0),
            "input_tokens": usage_sink.get("input_tokens", 0),
            "output_tokens": usage_sink.get("output_tokens", 0),
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
