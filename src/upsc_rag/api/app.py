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
    generate_agentic_answer,
    generate_agentic_answer_stream,
    generate_answer,
    generate_answer_stream,
)
from upsc_rag.generation.condense import condense_query
from upsc_rag.generation.live_scores import record_answer_scores
from upsc_rag.generation.router import OUT_OF_SCOPE_REPLY, is_off_topic, smalltalk_reply
from upsc_rag.generation.sources import build_agentic_sources, build_source_dicts
from upsc_rag.llm.clients import get_openai_client
from upsc_rag.observability import NOOP_CONTEXT, trace_manager
from upsc_rag.retrieval.hybrid import HybridRetriever
from upsc_rag.retrieval.rewrite import rewrite_query
from upsc_rag.retrieval.sufficiency import sources_answer_question
from upsc_rag.retrieval.web import web_search_multi

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


def _history_and_search_query(
    req: "AskRequest", cfg: dict[str, Any], parent: Any = None
) -> tuple[list[dict[str, Any]] | None, str]:
    """Resolve conversation history + the (possibly condensed) query used for retrieval.

    Returns ``(history_dicts, search_query)``. When conversation is enabled and history
    is present, the follow-up is condensed into a standalone search query; otherwise the
    raw query is used and history is None. The condensed question feeds BOTH retrieval and
    generation — it has references and abbreviations resolved, which the answer LLM needs
    too (see the /ask handler). ``parent`` nests the condense trace under the request root.
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
        parent=parent,
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


def _run_web_search(
    query: str, cfg: dict[str, Any], session_id: str | None, root: Any
) -> list[dict[str, Any]]:
    """Search the web for ``query``; ``[]`` on failure (caller then answers book-only).

    Runs only when ``sources_answer_question`` says the retrieved sections don't answer the
    question, and only after the relevance floor — so off-topic questions never reach the
    web. Kept separate from the sufficiency check so /ask/stream can emit a "searching the
    web" status between the two.

    The user's wording is REWRITTEN before it is sent to the engine. A search engine takes
    the string literally, so "who is the current CJI and UPSC chairmain" returns SEO filler
    (opaque abbreviation, typo, two questions in one) — while "Chief Justice of India
    current appointment" returns the answer outright. rewrite_query also splits a compound
    question into one query per part, and web_search_multi round-robins the results so both
    parts survive truncation.
    """
    ws_cfg = cfg.get("web_search", {}) or {}
    rw_cfg = (cfg.get("retrieval", {}) or {}).get("rewrite", {}) or {}

    queries = [query]
    try:
        variants = rewrite_query(
            query,
            get_openai_client(),
            model=rw_cfg.get("model", "gpt-4.1-nano"),
            num_variants=ws_cfg.get("num_queries", 3),
            obs=root or NOOP_CONTEXT,
        )
        # rewrite_query returns [original, *rewrites]; the rewrites are the keyword-y ones
        # the engine can actually use. Keep the original only if it produced nothing.
        queries = [v for v in variants if v != query] or [query]
    except Exception:
        pass  # rewriting is an optimization — fall back to the raw query

    return web_search_multi(
        queries,
        max_results=ws_cfg.get("max_results", 5),
        region=ws_cfg.get("region", "in-en"),
        session_id=session_id,
        parent=root,
    )


def _status(stage: str, label: str) -> str:
    """One NDJSON status event — narrates the pipeline stage the client is waiting on."""
    return json.dumps({"type": "status", "stage": stage, "label": label}) + "\n"


# Pipeline stages surfaced to the UI while it waits (see _status).
_S_RETRIEVING = ("retrieving", "Searching the textbook…")
_S_CHECKING = ("checking", "Checking whether the textbook covers this…")
_S_WEB = ("web", "The textbook doesn't cover this — searching the web…")
_S_GENERATING = ("generating", "Writing the answer…")


def _score_sources(
    results: list[dict[str, Any]], web: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Sources to score the answer against — score_answer only reads each dict's "text".

    When the answer was written over web results too, they must be included or
    groundedness is measured against book sections the answer never used.
    """
    return results + [{"text": w.get("snippet", "")} for w in web]


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

    # One root trace per question: condense + retrieve + generate nest under it, so
    # Langfuse rolls up total cost/latency with a per-step breakdown in a single view.
    with trace_manager.trace(
        "ask", input={"query": req.query}, session_id=req.session_id
    ) as root:
        # Resolve follow-ups: condense history + query into a standalone retrieval query.
        history, search_query = _history_and_search_query(req, _state["cfg"], parent=root)
        results = retriever.retrieve(
            search_query, top_k=req.top_k, rerank_top_k=req.rerank_top_k,
            session_id=req.session_id, parent=root,
        )

        # Gate 2: real question, but no relevant source in the book — skip generation.
        floor = _state["cfg"].get("retrieval", {}).get("relevance_floor", 0.0)
        if is_off_topic(results, floor):
            root.end(output={"route": "out_of_scope"})
            return AskResponse(answer=OUT_OF_SCOPE_REPLY, sources=[])

        # Gate 3: on-topic, but do the retrieved sections actually ANSWER it? If not (the
        # framers' intent, post-2011 events, current office-holders), search the web and
        # answer over textbook + web instead of replying "the sources do not specify".
        web: list[dict[str, Any]] = []
        if not sources_answer_question(
            search_query, results, _state["cfg"], session_id=req.session_id, parent=root
        ):
            web = _run_web_search(search_query, _state["cfg"], req.session_id, root)

        # Generate from the CONDENSED question, not the raw one: it has the follow-up's
        # references and abbreviations resolved ("from where was the FD taken?" ->
        # "...Fundamental Duties..."). Given the raw form, the answer LLM decodes the
        # abbreviation from the conversation's topic and answers the wrong question even
        # when retrieval fetched the right sources. On a first turn (no history) condense
        # is a no-op, so this IS req.query.
        if web:
            answer = generate_agentic_answer(
                search_query, results, web, _state["cfg"], session_id=req.session_id,
                history=history, parent=root,
            )
            sources = [Source(**s) for s in build_agentic_sources(results, web)]
        else:
            answer = generate_answer(
                search_query, results, _state["cfg"], session_id=req.session_id,
                history=history, parent=root,
            )
            sources = _build_sources(results)

        record_answer_scores(
            root, search_query, answer, _score_sources(results, web), _state["cfg"]
        )
        # Filterable in Langfuse: how often does the book fail to answer on its own?
        root.score("used_web", 1.0 if web else 0.0)
        root.end(output={"answer_chars": len(answer), "used_web": bool(web), "num_web": len(web)})
    return AskResponse(answer=answer, sources=sources)


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream the answer as NDJSON events: 'status' × N, then 'sources', 'token' × N, 'done'.

    'status' events narrate the stage the client is waiting on (searching the textbook,
    checking whether it covers the question, falling back to the web, writing the answer),
    so the direct path runs its pipeline inside the generator and flushes each stage as it
    reaches it. Then generation tokens stream as they arrive — letting the client measure
    time-to-first-token. Clients that ignore 'status' see the original event contract.
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

    # Graph mode precomputes retrieval out here (it keeps its own tracing and emits no
    # status events). The DIRECT path instead runs its whole pipeline *inside* the
    # generator below: the client must receive "Searching the textbook…" while that search
    # is happening, and nothing can be sent from a response that hasn't started streaming.
    graph_results: list[dict[str, Any]] | None = None
    graph_history: list[dict[str, Any]] | None = None

    if USE_GRAPH:
        from upsc_rag.graph import prepare_stream

        graph_history = [t.model_dump() for t in req.history] if req.history else None
        prep = prepare_stream(
            retriever, _state["cfg"], req.query, top_k=req.top_k,
            rerank_top_k=req.rerank_top_k, session_id=req.session_id, history=graph_history,
        )
        if prep["route"] != "answer":
            return StreamingResponse(
                canned_stream(prep["answer"]), media_type="application/x-ndjson"
            )
        graph_results = prep["results"]
    else:
        # Gate 1: pure greeting / chit-chat — reply without retrieving or generating.
        # Stays outside the generator: it does no work, so there is no wait to narrate.
        canned = smalltalk_reply(req.query)
        if canned is not None:
            return StreamingResponse(canned_stream(canned), media_type="application/x-ndjson")

    def event_stream() -> Iterator[str]:
        root: Any = None                        # request root trace (direct path only)
        history: list[dict[str, Any]] | None = graph_history
        results: list[dict[str, Any]] = graph_results or []
        web: list[dict[str, Any]] = []          # non-empty once the web fallback fires
        gen_query: str = req.query
        usage_sink: dict[str, Any] = {}
        parts: list[str] = []
        trace_out: dict[str, Any] | None = None  # set on an early gated exit
        try:
            if graph_results is None:
                # Direct path: run the pipeline here, narrating each stage as we go.
                root = trace_manager.open_root(
                    "ask", input={"query": req.query}, session_id=req.session_id
                )
                yield _status(*_S_RETRIEVING)
                # Resolve follow-ups into a standalone query, then search the textbook.
                history, gen_query = _history_and_search_query(req, _state["cfg"], parent=root)
                results = retriever.retrieve(
                    gen_query, top_k=req.top_k, rerank_top_k=req.rerank_top_k,
                    session_id=req.session_id, parent=root,
                )

                # Gate 2: real question, but nothing relevant in the book — no generation.
                floor = _state["cfg"].get("retrieval", {}).get("relevance_floor", 0.0)
                if is_off_topic(results, floor):
                    trace_out = {"route": "out_of_scope"}
                    yield json.dumps({"type": "sources", "sources": []}) + "\n"
                    yield json.dumps({"type": "token", "text": OUT_OF_SCOPE_REPLY}) + "\n"
                    yield json.dumps(
                        {"type": "done", "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
                    ) + "\n"
                    return

                # Gate 3: on-topic, but do the sections actually ANSWER it? If not, say so
                # in the UI and fall back to the web rather than "the sources do not specify".
                yield _status(*_S_CHECKING)
                if not sources_answer_question(
                    gen_query, results, _state["cfg"],
                    session_id=req.session_id, parent=root,
                ):
                    yield _status(*_S_WEB)
                    web = _run_web_search(gen_query, _state["cfg"], req.session_id, root)

            # Web sources are numbered after the book ones, matching build_agentic_prompt.
            sources = (
                [Source(**s) for s in build_agentic_sources(results, web)] if web
                else _build_sources(results)
            )
            yield json.dumps(
                {"type": "sources", "sources": [s.model_dump() for s in sources]}
            ) + "\n"
            yield _status(*_S_GENERATING)

            # Web fallback fired -> synthesize over textbook + web (same citation numbering
            # as the `sources` event); otherwise stream from the textbook exactly as before.
            # In graph mode root is None, so the generator emits its own trace as before.
            stream = (
                generate_agentic_answer_stream(
                    gen_query, results, web, _state["cfg"], session_id=req.session_id,
                    usage_sink=usage_sink, history=history, parent=root,
                )
                if web
                else generate_answer_stream(
                    gen_query, results, _state["cfg"], session_id=req.session_id,
                    usage_sink=usage_sink, history=history, parent=root,
                )
            )
            for delta in stream:
                parts.append(delta)
                yield json.dumps({"type": "token", "text": delta}) + "\n"
            yield json.dumps({
                "type": "done",
                "cost_usd": usage_sink.get("cost_usd", 0.0),
                "input_tokens": usage_sink.get("input_tokens", 0),
                "output_tokens": usage_sink.get("output_tokens", 0),
            }) + "\n"
            # Score the full answer once streaming completes (tokens already sent).
            record_answer_scores(
                root, gen_query, "".join(parts), _score_sources(results, web), _state["cfg"]
            )
            # Filterable in Langfuse: how often does the book fail to answer on its own?
            if root is not None:
                root.score("used_web", 1.0 if web else 0.0)
        finally:
            # Close + flush the request root once the stream ends (or the client
            # disconnects). No-op in graph mode where root is None.
            if root is not None:
                root.end(output=trace_out or {
                    "output_tokens": usage_sink.get("output_tokens", 0),
                    "cost_usd": usage_sink.get("cost_usd", 0.0),
                    "used_web": bool(web),
                    "num_web": len(web),
                })
                trace_manager.flush()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
