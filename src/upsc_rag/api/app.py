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
from upsc_rag.retrieval.hybrid import HybridRetriever

# Default book served by the API; override with UPSC_RAG_BOOK env var.
BOOK_ID = os.environ.get("UPSC_RAG_BOOK", "laxmikanth_6")

# Populated at startup, reused across requests.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the retriever + config once when the server boots."""
    settings = get_settings()
    cfg = load_runtime_config(BOOK_ID)
    chunks_path = settings.resolve(settings.processed_dir) / BOOK_ID / "chunks.jsonl"
    _state["cfg"] = cfg
    _state["retriever"] = HybridRetriever(cfg, chunks_path)
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
    seen: set[str] = set()
    sources: list[Source] = []
    for r in results:
        path = r.get("section_path") or []
        key = f"{' > '.join(path)}|{r.get('page_start')}-{r.get('page_end')}"
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                n=len(sources) + 1,
                section_path=path,
                chapter_title=r.get("chapter_title", ""),
                page_start=r.get("page_start"),
                page_end=r.get("page_end"),
            )
        )
    return sources


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

    results = retriever.retrieve(
        req.query, top_k=req.top_k, rerank_top_k=req.rerank_top_k, session_id=req.session_id
    )
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

    results = retriever.retrieve(
        req.query, top_k=req.top_k, rerank_top_k=req.rerank_top_k, session_id=req.session_id
    )
    sources = _build_sources(results)

    def event_stream() -> Iterator[str]:
        yield json.dumps({"type": "sources", "sources": [s.model_dump() for s in sources]}) + "\n"
        # generate_answer_stream traces its own LLM generation (see generation/answer.py).
        for delta in generate_answer_stream(req.query, results, _state["cfg"], session_id=req.session_id):
            yield json.dumps({"type": "token", "text": delta}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
