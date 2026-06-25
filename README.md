# UPSC-RAG

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
| 10. Evaluate | `eval/` | Retrieval-quality harness on a labeled gold set | 🟡 In progress |

**Current status:** Full pipeline works end-to-end — ingest → embed → hybrid retrieve → LLM answer, exposed via FastAPI and a Next.js chat UI with token streaming. A retrieval-quality evaluation harness is in place (see [Evaluation](#evaluation)); a generation-quality (groundedness/citation) judge is the remaining gap. See `progress.json` for details.

---

## Architecture

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

---

## Quick start

Prereqs: Python 3.13 venv, Node 18+, Docker (for Qdrant), and an `OPENAI_API_KEY` in `.env`.

```powershell
# 1. Start Qdrant (vector DB)
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant

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
│   ├── retrieval/            # HybridRetriever (dense + BM25 + RRF), rewrite, cross-encoder rerank
│   ├── generation/           # build_answer_prompt, generate_answer[_stream]
│   ├── api/                  # FastAPI app (/ask, /ask/stream, /health)
│   ├── eval/                 # retrieval-quality harness (hit@k, MRR, article_recall)
│   └── pipeline/             # run_ingest, run_embed orchestration
│
├── web/                      # Next.js 16 frontend (React 19, TS, Tailwind v4)
│   ├── app/page.tsx          # Streaming chat UI
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
| `/ask/stream` | POST | NDJSON event stream: one `sources` event, then `token` events, then `done` |

Request body: `{ "query": "...", "top_k"?: int, "rerank_top_k"?: int }`.
The retriever is built once at startup (FastAPI lifespan) and reused per request. Sources are deduplicated by section + page range and renumbered.

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

30 questions, deliberately split to stress different failure modes:

- **10 clean** — textbook-phrased (e.g. *"How is a judge of the Supreme Court appointed?"*).
- **20 messy** — colloquial and abbreviation-heavy, like real aspirants ask
  (e.g. *"ok so who actually picks SC judges, the collegium or the govt?"*, *"if an MLA jumps
  ship to another party can he lose his seat?"*). These exercise the query-rewrite layer and
  expose vocabulary gaps the clean set hides.

### Current results

30 questions, `rerank_top_k=8`, all retrieval layers on (incl. cross-encoder rerank):

| Metric | Result |
|--------|--------|
| hit@k | **96.7 %** (29/30) |
| MRR | **0.744** |
| article_recall | **95.5 %** (21/22) |
| avg_articles_on_hit | **4.2** |

The one remaining miss is a known catalog data gap (Article 361 immunity). The harness is
**retrieval-only**; an end-to-end generation-quality eval (groundedness + citation
correctness) is the next addition.

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
- [ ] Generation-quality eval (groundedness + citation correctness, LLM judge)
- [ ] Multi-book support in the UI (book selector)
```
