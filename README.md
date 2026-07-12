# Structured-RAG

Structured **retrieval-augmented generation (RAG)** for massive UPSC polity textbooks—starting with *Indian Polity* (6th ed.) by M. Laxmikanth (~1,500 pages).

Unlike naive RAG (fixed-size text splits), this project preserves the book’s hierarchy—**PART → Chapter → Section**—so retrieval returns contextually correct snippets (e.g. Fundamental Rights vs Centre–State relations) with page citations. A FastAPI backend serves the pipeline over HTTP, and a Next.js chat UI streams grounded answers token-by-token.

---

## How it works (pipeline)

```mermaid
flowchart LR
  PDF[Source PDF] --> Parse[parsing/]
  Parse --> TOC[TOC + sections]
  TOC --> Chunk[chunking/]
  Chunk --> Enrich[enrichment/]
  Enrich --> Index[indexing/ + Qdrant]
  Index --> Retrieve[retrieval/ hybrid]
  Retrieve --> Gen[generation/ LLM]
  Gen --> API[FastAPI /ask]
  API --> UI[Next.js chat UI]
```

| Stage | Package | Purpose | Status |
|-------|---------|---------|--------|
| 1. Parse | `parsing/` | Extract text from PDF pages (PyMuPDF) | ✅ Done |
| 2. Structure | `parsing/toc.py`, `align.py` | Build PART / chapter / section tree from Contents | ✅ Done |
| 3. Chunk | `chunking/` | Split text inside section boundaries with overlap | ✅ Done |
| 4. Enrich | `enrichment/` | Add syllabus tags, entities, content types | ✅ Done |
| 5. Index | `indexing/` | Save `chunks.jsonl`; embed + upsert to Qdrant | ✅ Done |
| 6. Retrieve | `retrieval/` | Dense (Qdrant) + BM25 hybrid search, RRF fusion, cross-encoder rerank | ✅ Done |
| 7. Answer | `generation/` | LLM prompt with cited sources (OpenAI) | ✅ Done |
| 8. Serve | `api/` | FastAPI `/ask` + streaming `/ask/stream` | ✅ Done |
| 9. UI | `web/` | Next.js streaming chat with citations | ✅ Done |
| 10. Evaluate | `eval/` | Retrieval, cheap generation-quality, and LLM-judge harnesses on a labeled gold set | 🟡 In progress |

**Current status:** Full pipeline works end-to-end — ingest → embed → hybrid retrieve → LLM answer, exposed via FastAPI and a Next.js chat UI with token streaming (restyled as a retro **government-dossier theme**: typewriter fonts, aged-paper palette, rubber-stamp provenance badges). The direct path now carries a **sufficiency-gated web fallback** (post-2011 topics the 2011 book can't cover get DuckDuckGo results synthesized alongside the textbook), follow-up questions are **condensed with conversation history**, every chat call can route through a **LiteLLM AI gateway**, and each request emits a unified **Langfuse** trace with live quality scores. Three evaluation harnesses are in place (see [Evaluation](#evaluation)): retrieval quality (**70-question gold set** — hit@k 98.6 %, MRR 0.802), a cheap generation-quality harness (groundedness + citation), and a nuanced **LLM-judge rubric** (`gpt-5-mini`) that cross-checks the cheap signals. The judge drove a tuning loop (prompt-faithfulness pass + `gpt-4.1` generator) that lifted overall answer quality from **3.43 → 4.01 / 5**; the open levers are citation precision and the production-generator cost decision. See `progress.json` for details.

---

## Architecture

> 📐 **Full diagrams:** see [`docs/architecture.md`](docs/architecture.md) for editable Mermaid
> diagrams — a system overview (indexing · serving · evaluation) and the detailed multi-query
> retrieval flow (gated rewrite + RRF fusion). The whole-system diagram is also kept as a
> standalone Mermaid source, [`docs/architecture.mmd`](docs/architecture.mmd), rendered to
> [`docs/architecture-mermaid.svg`](docs/architecture-mermaid.svg) (regenerate with
> `npx @mermaid-js/mermaid-cli -i docs/architecture.mmd -o docs/architecture-mermaid.svg`).

```
Browser (Next.js chat)            FastAPI (Python)               Services
┌────────────────────┐  POST     ┌──────────────────────┐
│ web/app/page.tsx    │ /api/ask  │ /ask        (JSON)   │
│  - streams tokens   │ ────────► │ /ask/stream (NDJSON) │ ──► Qdrant  :6333
│  - markdown answers │ (route    │   HybridRetriever    │ ──► OpenAI  (embed + chat)
│  - source citations │  handler  │   generate_answer()  │
└────────────────────┘  pipes    └──────────────────────┘
```

The Next.js route handler (`web/app/api/ask/route.ts`) is a same-origin proxy that forwards to FastAPI and pipes the NDJSON stream back — so the browser never deals with CORS and the backend URL stays server-side.

### Orchestration backends

The request flow (smalltalk gate → retrieve → relevance gate → generate) ships in **three**
interchangeable forms behind the `UPSC_RAG_PIPELINE` env flag — the default keeps the original
direct code path, `graph` runs the same steps as a **LangGraph** state machine, and `agentic`
is a tool-calling **ReAct** agent. All three reuse the same `HybridRetriever` + `generate_answer`,
so retrieval/generation behavior (and the eval numbers) are unchanged; only orchestration differs.

| `UPSC_RAG_PIPELINE` | What it does |
|---------------------|--------------|
| *(unset)* / `direct` | FastAPI calls the functions directly (default) |
| `graph` | A compiled **LangGraph** state machine wraps the same functions as thin nodes |
| `agentic` | A **ReAct** loop where the LLM chooses per polity question between the textbook (`HybridRetriever`) and **DuckDuckGo web search** (`retrieval/web.py`) — gated so only polity queries reach the tools; web may answer standalone for post-2011 topics the 2011 book can't cover |

The direct vs graph paths run identical steps:

```
default                                   UPSC_RAG_PIPELINE=graph
FastAPI calls the functions directly      FastAPI invokes a compiled LangGraph:
  smalltalk_reply()                         START → smalltalk ─(smalltalk)→ END
  → HybridRetriever.retrieve()                       └(answer)→ retrieve → gate ─(off_topic)→ END
  → is_off_topic()                                                          └(answer)→ generate → END
  → generate_answer[_stream]()
```

The graph **wraps the existing functions as thin nodes** — it does *not* reimplement the
retriever, RRF, FlashRank rerank, or catalog logic — so retrieval/generation behavior (and the
eval numbers) are unchanged; only orchestration differs. Sources are deduped identically by a
shared `generation/sources.py` helper, and `/ask/stream` keeps the same NDJSON contract (the
graph drives routing + retrieval, then tokens stream through `generate_answer_stream`). An
optional `UPSC_RAG_LLM_BACKEND=langchain` flag routes generation through `ChatOpenAI`
(`llm/clients.py`) as a provider-portability seam (off by default). See the `graph/` package and
the `orchestration` stage in `progress.json`.

```powershell
# run the backend on the LangGraph path (or "agentic" for the ReAct + web-search agent)
$env:UPSC_RAG_PIPELINE = "graph"
python -m uvicorn upsc_rag.api.app:app --reload --port 8000
```

### Web fallback on the direct path (sufficiency-gated)

The off-topic relevance floor only measures whether retrieval found *related* sections — not
whether they actually answer the question. So after the floor, a cheap **sufficiency check**
(`retrieval/sufficiency.py`) asks whether the retrieved sections contain what was asked; on
"no", the API runs `retrieval/web.py::web_search` (DuckDuckGo) and answers over **textbook +
web** via the same agentic synthesis prompt. This is what stops *"the sources do not specify"*
on questions the 2011 book can't cover — framers' intent, post-2011 amendments, current
office-holders. It runs *after* the off-topic gate, so junk never reaches the web. Toggle via
`retrieval.web_fallback.enabled` in `config/default.yaml` (off = textbook-only, no classifier
call). The UI stamps each answer with its provenance: **from the textbook / from the web /
textbook + web**.

### Conversation history (follow-up questions)

The frontend sends the last few completed exchanges with each request (`history` +
`session_id`); the backend **condenses** a follow-up ("what about *his* removal?") into a
standalone question using that history, and the condensed query feeds both retrieval and
generation. One browser session = one Langfuse session for trace grouping.

### Observability (Langfuse)

Every `/ask` request on the direct path emits **one unified `ask` root trace** with per-step
spans — rewrite, retrieve, sufficiency, web_search, generate — plus **live scores** computed
right after the answer finishes (grounded_fraction, citation validity, `used_web`, retrieval
top-score). Scoring runs *after* the `done` NDJSON event is sent, so it never delays the
answer: the UI reveals the finished answer at `done` while the backend spends a few more
seconds scoring and flushing the trace. Langfuse ships in `docker-compose.yml`; keys live in
`.env` (note: they reset if the container volume is wiped — a silent 401 in the logs is the
symptom).

### AI gateway (LiteLLM)

Every **chat** call the app makes (answer generation, query rewrite, history condense, the
agentic domain gate + tool router, and the LLM judge) can be routed through a self-hosted
**[LiteLLM](https://github.com/BerriAI/litellm) proxy** — an OpenAI-compatible gateway for
central cost tracking, virtual keys, provider fallbacks, and one-place model swaps. It's
**off by default** (direct OpenAI, so eval numbers are unchanged); set `UPSC_RAG_LLM_GATEWAY=litellm`
to turn it on. **Embeddings never go through it** — they stay on the direct OpenAI endpoint
(dimension-locked to the Qdrant collection).

A single factory, `get_openai_client()` in `llm/clients.py`, is the seam: it returns a client
pointed at the proxy when the gateway is enabled, or a direct OpenAI client when it isn't, so no
call site changes. Model aliases in `litellm/config.yaml` mirror the model names in
`config/default.yaml`, so the app passes the same names either way.

```powershell
# start the proxy (+ Qdrant) via docker-compose; config in litellm/config.yaml
docker-compose up -d qdrant-db litellm      # gateway + admin UI at http://localhost:4000

# enable it for the app (.env): UPSC_RAG_LLM_GATEWAY=litellm
#   LITELLM_BASE_URL=http://localhost:4000
#   LITELLM_API_KEY=<same value as the proxy's LITELLM_MASTER_KEY>
```

> The proxy's **admin UI / virtual-key / spend storage** needs a Postgres (the `litellm-db`
> service in `docker-compose.yml`); routing chat calls does not. Log in at `/ui` with
> `admin` + the master key.

---

## Quick start

Prereqs: Python 3.13 venv, Node 18+, Docker (for Qdrant), and an `OPENAI_API_KEY` in `.env`.

```powershell
# 1. Start Qdrant (vector DB)
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant
#    or: docker-compose up -d qdrant-db   (compose also offers Langfuse + the LiteLLM gateway)

# 2. Build the index (once): parse + chunk, then embed + upsert
python scripts/ingest.py --book laxmikanth_6
python scripts/embed.py  --book laxmikanth_6

# 3. Start the API backend
python -m uvicorn upsc_rag.api.app:app --reload --port 8000

# 4. Start the frontend (in another terminal)
cd web
npm install      # first time only
npm run dev      # opens http://localhost:3000
```

Ask a question in the browser, or hit the API directly at **http://localhost:8000/docs**.

---

## Project layout

```
Structured-RAG/
├── .env                      # OPENAI_API_KEY (gitignored)
├── pyproject.toml            # Package metadata, dependencies
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev deps + editable install
├── progress.json             # Stage-by-stage status (source of truth)
├── CLAUDE.md                 # Quick reference for the codebase
├── README.md                 # This file
├── docker-compose.yml        # Qdrant + Langfuse + LiteLLM proxy (+ its Postgres)
├── litellm/
│   └── config.yaml           # LiteLLM proxy model routes (AI gateway; off by default)
│
├── docs/
│   ├── architecture.md       # Mermaid diagrams: system overview + multi-query retrieval
│   ├── architecture.mmd      # Whole-system Mermaid source (edit this)
│   ├── architecture-mermaid.svg  # Rendered from architecture.mmd
│   └── architecture.svg      # Older hand-drawn static render
│
├── config/
│   ├── default.yaml          # Global defaults (chunking, indexing, retrieval, generation)
│   └── books/
│       └── laxmikanth_6.yaml # Book PDF path, page ranges, structure regex
│
├── data/processed/laxmikanth_6/
│   ├── manifest.json         # Book metadata + page count
│   ├── toc.json              # Parsed TOC tree (cached)
│   └── chunks.jsonl          # Parent + child chunks (index input)
│
├── scripts/                  # Thin CLI wrappers
│   ├── calibrate_pages.py    # Verify PDF page ranges before ingest
│   ├── ingest.py             # Parse + chunk → chunks.jsonl
│   ├── embed.py              # Embed children + upsert to Qdrant
│   ├── retrieve.py           # Run a hybrid query, print ranked results
│   ├── ask.py                # Retrieve → generate → print cited answer
│   └── evaluate.py           # Score retrieval over the gold set
│
├── src/upsc_rag/             # Main Python package
│   ├── config.py             # Load YAML + .env; resolve paths
│   ├── parsing/              # PDF + TOC + alignment
│   ├── chunking/             # Hierarchy-aware splitting
│   ├── enrichment/           # syllabus_tags, entities
│   ├── indexing/             # JSONL store, OpenAI embedder, Qdrant store
│   ├── retrieval/            # HybridRetriever (dense + BM25 + RRF), rewrite, cross-encoder rerank, web.py (DuckDuckGo)
│   ├── generation/           # build_answer_prompt, generate_answer[_stream], condense, agentic synthesis, sources (dedupe)
│   ├── graph/                # LangGraph orchestration (UPSC_RAG_PIPELINE=graph): state, nodes, build, runner
│   ├── agent/                # Agentic ReAct pipeline (UPSC_RAG_PIPELINE=agentic): tools, nodes, build, runner
│   ├── llm/                  # clients.py — get_openai_client() gateway factory + ChatOpenAI/OpenAIEmbeddings seam
│   ├── api/                  # FastAPI app (/ask, /ask/stream, /health)
│   ├── eval/                 # retrieval-quality harness (hit@k, MRR, article_recall)
│   └── pipeline/             # run_ingest, run_embed orchestration
│
├── web/                      # Next.js 16 frontend (React 19, TS, Tailwind v4)
│   ├── app/page.tsx          # Streaming chat UI (retro government-dossier theme)
│   ├── app/api/ask/route.ts  # BFF proxy → FastAPI stream
│   ├── app/types.ts          # Shared AskResponse/Source types
│   └── .env.local            # BACKEND_URL (default http://localhost:8000)
│
└── tests/                    # Pytest suite
```

---

## `config/` — YAML settings

Configuration is split so one codebase can ingest many books. `default.yaml` holds shared defaults; `books/<id>.yaml` holds per-book overrides (PDF path, page ranges, structure regex). They are deep-merged at load time.

### Key `default.yaml` blocks

| Key | Meaning |
|-----|---------|
| `chunking.child_chunk_tokens` / `child_chunk_overlap` | Child chunk size (~600 tokens) and overlap |
| `indexing.collection_name` | Qdrant collection (`upsc_polity`) |
| `indexing.embedding_model` / `embedding_dim` | `text-embedding-3-large`, 3072-dim, cosine |
| `indexing.qdrant_url` | `http://localhost:6333` |
| `retrieval.top_k` / `rerank_top_k` | Candidate pool size and final results |
| `generation.model` / `temperature` / `max_tokens` | LLM settings (`gpt-4o-mini`) |

Add new books by creating `config/books/<book_id>.yaml` (see "Adding a new book" in `CLAUDE.md`).

---

## Chunk schema (`chunks.jsonl`)

One JSON object per line:

```json
{
  "id": "sec_0002_001_6c6329fcfb10",
  "text": "...",
  "book_id": "laxmikanth_6",
  "part": "PART-I",
  "chapter_num": 1,
  "chapter_title": "Historical Background",
  "section_path": ["PART-I", "1 Historical Background", "The Company Rule (1773–1858)"],
  "page_start": 49,
  "page_end": 53,
  "content_type": "child",
  "parent_id": "sec_0002",
  "entities": ["Article 14"]
}
```

**Parent chunks** = full section text (stored as `parent_text` in the Qdrant payload, not embedded).
**Child chunks** = overlapping ~600-token splits, embedded into Qdrant for precise retrieval. On a hit, the parent text is returned for full context.

---

## API

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/health` | GET | Liveness + which book is loaded |
| `/ask` | POST | `{ answer, sources[] }` (complete JSON) |
| `/ask/stream` | POST | NDJSON event stream: `status` × N (pipeline stage labels), one `sources` event, `token` × N, then `done` (cost + token counts) |

Request body: `{ "query": "...", "history"?: [{role, content}], "session_id"?: str, "top_k"?: int, "rerank_top_k"?: int }`.
The retriever is built once at startup (FastAPI lifespan) and reused per request. Sources are deduplicated by section + page range and renumbered. The `done` event is sent **before** the post-answer Langfuse scoring/flush, so clients should treat `done` — not stream close — as end-of-answer (the bundled UI does).

Interactive docs: **http://localhost:8000/docs**.

---

## Commands

```powershell
# Verify PDF page ranges before ingest
python scripts/calibrate_pages.py --book laxmikanth_6

# Ingest (parse + chunk) → chunks.jsonl
python scripts/ingest.py --book laxmikanth_6

# Embed children + upsert to Qdrant
python scripts/embed.py --book laxmikanth_6

# Hybrid retrieval (print ranked chunks)
python scripts/retrieve.py "How is the Constitution amended?"

# Full answer in the terminal (retrieve → generate)
python scripts/ask.py "How is the Constitution amended?" --rerank 5

# Run tests
pytest
```

> **Venv note:** this `.venv` was cloned from another project, so always install with
> `python -m pip install ...` (not bare `pip`). See `CLAUDE.md` for details.

---

## Evaluation

Retrieval quality is measured against a **labeled gold set** (`data/eval/laxmikanth_6.jsonl`)
rather than by eyeballing. Each question is labeled with the section it *should* retrieve
(`section_contains` — substrings that must all appear in the matched `section_path`) and,
where relevant, the governing Constitutional **Article(s)**. The harness runs the real
`HybridRetriever` over every question and scores deterministic metrics — **no LLM judge**,
so it's cheap, repeatable, and exactly the signal needed to tune retrieval.

```powershell
python scripts/evaluate.py --rerank 8            # run the eval
python scripts/evaluate.py --rerank 8 --no-rerank    # A/B a single layer
#   flags: --no-rewrite | --no-graph | --no-catalog | --no-rerank
```

### Metrics and what they mean

| Metric | Meaning | Why it matters |
|--------|---------|----------------|
| **hit@k** | Fraction of questions where a relevant section appears anywhere in the top-*k* results | Is the right section retrieved *at all*? |
| **MRR** | Mean Reciprocal Rank — average of `1/rank` of the first relevant section (rank 1 → 1.0, rank 2 → 0.5) | Is the right section ranked *high*, not just present? |
| **article_recall** | Of questions labeled with a governing Article, the fraction whose Article appears in some result's `entities` | Is the *citable* Article surfaced for generation? |
| **avg_articles_on_hit** | Average number of Articles attached to the gold section (lower is better) | Over-attach / citation-precision proxy — are we flooding answers with irrelevant Articles? |

### The gold set

**70 questions** (expanded from 30 on 2026-07-10), deliberately split to stress different
failure modes:

- **10 clean** — textbook-phrased (e.g. *"How is a judge of the Supreme Court appointed?"*).
- **20 messy** — colloquial and abbreviation-heavy, like real aspirants ask
  (e.g. *"ok so who actually picks SC judges, the collegium or the govt?"*, *"if an MLA jumps
  ship to another party can he lose his seat?"*). These exercise the query-rewrite layer and
  expose vocabulary gaps the clean set hides.
- **40 breadth** — added across previously under-represented chapters (Vice-President,
  President's veto/ordinance/pardon, Parliament procedure, Finance Commission, Preamble,
  individual Fundamental Rights, citizenship, local government, tribunals, Lokpal, NHRC, …),
  keeping the same clean/messy mix. Every question is grounded to a real parent section, and
  Article labels are attached only where the chapter's Articles-at-a-Glance catalog backs them.

### Current results

70 questions, `rerank_top_k=8`, all retrieval layers on (incl. cross-encoder rerank):

| Metric | Result |
|--------|--------|
| hit@k | **98.6 %** (69/70) |
| MRR | **0.802** |
| article_recall | **97.4 %** (38/39) |
| avg_articles_on_hit | **3.6** |

Every metric *improved* over the 30-question set despite 40 harder/broader questions. The one
remaining miss is a known catalog data gap (Article 361 immunity). This harness is
**retrieval-only**; a complementary **generation-quality** harness scores the answers
themselves — see [Generation-quality eval](#generation-quality-eval-3-cheap-signals--no-llm-judge) below.

### Cross-encoder reranking (FlashRank)

RRF fuses two *bi-encoders* (dense + BM25), which encode the query and each passage
*separately* and so struggle to separate sibling sections in the same chapter — the gold
section often landed at rank #2–#3. A **cross-encoder**
([`ms-marco-MiniLM-L-12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
via [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) — ONNX/CPU, **no torch**)
re-scores the widened deduped candidate pool by reading `(question, full section text)`
*jointly*, so query and passage tokens cross-attend and lookalike sections separate.

**Blend, don't replace.** The cross-encoder score is min-max normalized over the pool and
combined with the (also normalized) RRF score — it refines the fusion order instead of
overriding it, which keeps a safety net for sections both retrievers already agreed on:

```
blend_score = weight · norm(cross_encoder) + (1 − weight) · norm(RRF)
```

Ablation on the 30-question gold set (`weight` swept; `weight=0.5` shipped):

| `weight` | hit@k | MRR | note |
|---------:|:-----:|:---:|------|
| 0.0 | 93.3 % | 0.693 | pure RRF (rerank off) — baseline |
| 0.3 | 93.3 % | 0.744 | |
| **0.5** | **96.7 %** | **0.745** | **shipped** — best |
| 0.7 | 96.7 % | 0.721 | |
| 1.0 | 93.3 % | 0.719 | pure cross-encoder — *regresses*: demotes correct #1s, drops one section out of top-8 |

Net: **hit@k 93.3 % → 96.7 %**, **MRR 0.693 → 0.744**, article_recall unchanged. Note the
pure reorder (`weight=1.0`) is *worse* than the blend — the lesson is to fuse, not replace.

**Usage.** It's on by default; tune or A/B it via config and the `--no-rerank` flag:

```yaml
# config/default.yaml → retrieval.rerank
rerank:
  enabled: true
  model: ms-marco-MiniLM-L-12-v2   # FlashRank model; ~22 MB, downloaded + cached on first use
  candidate_pool: 25               # deduped sections fed to the cross-encoder (then cut to rerank_top_k)
  max_chars: 2000                  # section text truncated before scoring (~512-token model limit)
  weight: 0.5                      # blend weight: 1.0 = pure rerank, 0.0 = pure RRF
```

```powershell
python scripts/evaluate.py --rerank 8               # rerank ON (default)
python scripts/evaluate.py --rerank 8 --no-rerank   # A/B: rerank OFF
```

> The eval is also a regression gate: it surfaced — and quantified the fix for — a section
> **alignment bug** where some chunks held the wrong section's text. Correcting it moved hit@k
> from 76.7 % → 93.3 % and article_recall from 77.3 % → 95.5 %.

### Generation-quality eval (3 cheap signals — no LLM judge)

Retrieval being correct doesn't mean the *answer* is. On top of the retrieval harness, a
second harness scores the **generated answer** with three signals that are cheap enough to run
on every answer — a deterministic regex check plus two small-embedding cosines (no GPT judge,
which is reserved for a later nuanced rubric):

| Signal | How it's measured | Catches |
|--------|-------------------|---------|
| **article_recall** | deterministic — does the answer name the gold Constitutional Article? | Missing the citable Article |
| **uncited_answer_rate** | deterministic — fraction of answers with **no** `[n]` source marker at all | Answers that cite nothing |
| **grounded_fraction** | embedding cosine — share of answer sentences whose best match to a source *sentence-window* clears a threshold (0.65) | Claims not supported by the sources (hallucination / corpus gap) |
| **answer_relevance** | embedding cosine — answer centroid vs the question | Off-topic drift |

The three signals trade off cost against what they can see. Two are **deterministic** (a regex
and a string check — free, exact, reproducible); two are **embedding cosines** (one cheap
`text-embedding-3-small` call per answer — far cheaper than asking a second LLM to judge). None
of them call a judge model, which is the whole point: they're cheap enough to run on *every*
answer, so an LLM judge can be reserved for the nuanced rubric (completeness, exam-appropriateness)
later.

**1. Citation correctness (deterministic).** Two checks read straight off the answer text.
*`article_recall`* reuses the same `\bArticle\s+\d+[A-Z]?\b` regex as the rest of the pipeline to
ask: did the answer actually print the governing Article the gold set expects (e.g. *Article 124*)?
*`uncited_answer_rate`* counts answers that carry **no** `[n]` source marker at all — the real
"did it cite anything?" signal. (A companion `cited_fraction` exists but is intentionally *not* a
target: it divides by all the sources supplied, and a focused answer legitimately uses only a
subset, so < 100 % is correct.) Because these are exact string operations, the same answer always
yields the same score.

**2. Groundedness (embedding cosine).** The hallucination proxy. The answer is split into
sentences and each source section into *sentence-windows*; every answer sentence is embedded and
matched to its best-cosine source window. `grounded_fraction` is the share of sentences whose best
match clears **0.65** (`mean_support` reports the average best-cosine). The intuition: a faithful
sentence has a near-paraphrase somewhere in the sources, so it scores high; a sentence the model
invented (or pulled from its training memory) has no close match and falls below the threshold.
Matching against *windows* rather than whole sections matters — embedding a 2 000-char section as
one vector washes out short, reworded claims, which is what made procedural answers look
ungrounded before the fix.

**3. Answer relevance (embedding cosine).** A lightweight "did it stay on topic?" check: cosine
between the centroid of the answer's sentence embeddings and the question embedding. It won't
catch a *wrong* answer (groundedness does that), but it flags an answer that drifts off the
question entirely.

```powershell
python scripts/evaluate_generation.py --rerank 8            # score generation over the gold set
python scripts/evaluate_generation.py --rerank 8 --limit 5  # sample a few (cost control)
```

**Results** (30 questions, `rerank_top_k=8`). Two fixes were driven directly by these signals:

| Signal | Before | After | Fix |
|--------|:------:|:-----:|-----|
| article_recall (in answer) | 100 % | 95.5 %\* | — |
| **uncited_answer_rate** | **17 %** (5/30) | **0 %** | Strengthened the prompt to require a `[n]` per claim |
| grounded_fraction | 68 %† | **81 %** | Match answer sentences to source *sentence-windows*, not whole sections |
| mean_support (cosine) | 0.555 | 0.758 | — |
| answer_relevance | 0.669 | 0.682 | — |

<sub>\* one run-to-run flip at `temperature 0.2`, not a regression. † the 68 % was a *measurement
artifact*: embedding a whole ~2 000-char section as one vector diluted support for short
paraphrased steps. Sentence-window matching removed it (procedural answers like impeachment went
8 %→100 %); the threshold was then raised 0.5→0.65 to keep the metric discriminative.</sub>

**Why groundedness earns its keep.** Spot-checking a flagged answer (*"is privacy a fundamental
right?"*, 25 % grounded) exposed a real failure the other signals miss: the book is the **6th ed.
(2011), which predates the Puttaswamy privacy judgment (2017)**, so the corpus has no
privacy-as-a-fundamental-right content. Retrieval behaved correctly, but the LLM answered
confidently **from its own training knowledge** — citing valid `[2][3]` markers that don't
actually support the claim. `uncited_answer_rate` and citation-validity both pass; only
**groundedness catches off-source embellishment**, doubling as a corpus-gap / hallucination
detector. Tightening generation faithfulness (abstain when sources are thin) was the next lever —
pulled in the [LLM-as-judge](#llm-as-judge-nuanced-rubric--cross-checks-the-cheap-signals) section
below, which quantifies it per-criterion and drove the 3.43 → 4.01 climb.

### LLM-as-judge (nuanced rubric — cross-checks the cheap signals)

The cheap signals are deliberately shallow — a regex and two cosines. They can't score whether an
answer is **complete** or **exam-appropriate**, and their faithfulness signal is only a cosine
proxy. A rubric **LLM judge** grades each answer 1–5 on four criteria the cheap signals can't see —
*faithfulness, completeness, exam_appropriateness, citation_quality* — and, run on the *same*
answers, tells us whether the cheap gates are trustworthy stand-ins for it.

The judge is **`gpt-5-mini` on OpenAI** — deliberately a *different, stronger* model than the
`gpt-4o-mini` generator, so it doesn't rubber-stamp the generator's own style (same-provider
self-preference is reduced, not eliminated). Because gpt-5 is a *reasoning* model that rejects
`temperature != 1`, the judge call omits `temperature` and passes `reasoning_effort` instead.

```powershell
python scripts/evaluate_judge.py --rerank 8                     # judge + cheap signals on the same answers
python scripts/evaluate_judge.py --rerank 8 --gen-model gpt-4.1 # override the generator for this run only
python scripts/evaluate_judge.py --rerank 8 --limit 5           # sample a few (cost control)
```

**Generator sweep** (30 questions, `rerank_top_k=8`, judge `gpt-5-mini`). The judge turned "the
answers seem good" into an actual tuning loop — a prompt-faithfulness pass, then a generator
upgrade, then two refinements — that moved **overall from 3.43 → 4.01 (crossing 4.0)**:

| Generator · prompt | faithfulness | completeness | exam_approp. | citation | overall |
|--------------------|:---:|:---:|:---:|:---:|:---:|
| gpt-4o-mini · baseline | 3.37 | 3.67 | 3.70 | 3.00 | 3.43 (69 %) |
| gpt-4o-mini · tightened prompt | 3.73 | 3.27 | 3.60 | 3.17 | 3.44 |
| gpt-4.1 · tightened prompt | 3.57 | 4.23 | 4.40 | 3.20 | 3.85 |
| **gpt-4.1 · + refinements** | **4.07** | **4.13** | **4.30** | **3.53** | **4.01 (80 %)** |

> The eval overrides the generator with `--gen-model gpt-4.1`; production `generation.model` stays
> `gpt-4o-mini` until adopted. The prompt refinements are live in `generation/answer.py`.

**What moved and why.** A stronger generator (`gpt-4.1`) lifted **completeness** (3.27 → 4.13) and
**exam_appropriateness** (3.60 → 4.30) — the two criteria a small model was weakest on — but at
first *dented* faithfulness because it confidently added true-but-unsourced specifics. Two
refinements fixed that: a prompt rule against sharpening a source's general statement into a
specific one, and raising the judge's per-source window (1.5k → 3k chars) so it stopped docking
claims it simply couldn't see. Net: **faithfulness 3.57 → 4.07**, citation 3.20 → 3.53. **Three of
four criteria now clear 4**; citation_quality (3.53) is the laggard, held down partly by corpus-gap
questions that *should* score low (e.g. privacy — the 2011 book predates Puttaswamy).

**The cheap signals track the judge better as quality rises.** At the winning config the
per-question correlations are the strongest measured: faithfulness ↔ grounded_fraction `r ≈ +0.47`,
citation_quality ↔ cited_fraction `r ≈ +0.33` (both were near zero at the 3.43 baseline). Still,
the cheap signals remain a **gate, not a substitute** — the judge is what catches:

- **Misattributed citation** — the *HC-judge-removal* answer scored 100 % grounded but the judge
  gave faithfulness **3/5**: it cites "Article 217" that isn't in the supplied sources. Citation
  *validity* passes (a real marker); only the judge sees the source doesn't support the claim.
- **Cheap false-negative** — the *DPSP-binding* answer scored ~40 % grounded (heavy paraphrase) but
  the judge confirmed the claims **are** supported (5/5) — the cosine was overly pessimistic.
- **Corpus-gap floor** — *privacy* scores low on both, correctly: the source lacks the content, so
  the honest answer is to abstain; no generator fixes this.

**Takeaway:** the judge is what let a vibe become a measured 3.43 → 4.01 climb. The remaining lever
is **citation precision** (cite only the source that supports each claim), plus the product
decision of whether the completeness/exam gains justify running `gpt-4.1` in production (~13× the
per-answer cost of `gpt-4o-mini`, or `gpt-4.1-mini` at ~2×).

---

## Design principles

1. **Structure first** — Chunk inside TOC sections, not across chapter boundaries.
2. **Rich metadata** — Every chunk carries `part`, `chapter`, `section_path`, pages, and entities for filtered retrieval.
3. **Parent-child retrieval** — Embed children for precision, return the parent for full context.
4. **Config-driven books** — New textbooks = new YAML under `config/books/`, same code path.
5. **Reproducible artifacts** — `data/processed/` can be deleted and rebuilt from the PDF + config.
6. **Layered** — CLI, API, and UI all call the same core (`HybridRetriever` + `generate_answer`); no duplicated logic.

---

## Roadmap

- [x] TOC page-range detection and section body extraction
- [x] Populate `chunks.jsonl` from Laxmikanth hierarchy
- [x] Vector store (Qdrant) + BM25 hybrid retrieval with RRF
- [x] `generation/` LLM integration (OpenAI, grounded + cited)
- [x] FastAPI backend with token streaming
- [x] Next.js streaming chat UI with markdown + citations
- [x] Retrieval-quality eval harness + labeled gold set (hit@k, MRR, article_recall)
- [x] Reranker (cross-encoder, FlashRank) blended with RRF candidates
- [x] Generation-quality eval — 3 cheap signals (article recall, citation, groundedness) — no LLM judge
- [x] LangGraph orchestration backend (parallel path behind `UPSC_RAG_PIPELINE=graph`) + LangChain LLM portability seam
- [x] LLM-judge rubric (faithfulness / completeness / exam-appropriateness / citation) cross-checked against the cheap signals (`gpt-5-mini`)
- [x] Judge-driven tuning loop — prompt-faithfulness refinements + `gpt-4.1` generator lifted judge overall 3.43 → 4.01 (faithfulness 3.37 → 4.07)
- [x] Conversation history — follow-ups condensed into standalone questions (`history` + `session_id` in the API)
- [x] Agentic ReAct path with polity-gated DuckDuckGo web search (`UPSC_RAG_PIPELINE=agentic`)
- [x] Gold set expanded 30 → 70 questions — hit@k 98.6 %, MRR 0.802, article_recall 97.4 %
- [x] LiteLLM AI gateway for all chat calls (`UPSC_RAG_LLM_GATEWAY=litellm`, off by default; embeddings stay direct)
- [x] Sufficiency-gated web fallback on the direct path (textbook + web synthesis for post-2011 topics) + provenance stamps in the UI
- [x] Langfuse observability — unified `ask` root trace with per-step spans + live scores (post-`done`, non-blocking)
- [x] Retro government-dossier UI theme (typewriter type, aged-paper palette, rubber-stamp provenance badges)
- [ ] Citation precision — cite only the source that supports each claim (judge citation_quality 3.53, the remaining laggard)
- [ ] Adopt `gpt-4.1` (or `gpt-4.1-mini`) as the production generator if the completeness/exam gains justify the cost
- [ ] Multi-book support in the UI (book selector)
```
