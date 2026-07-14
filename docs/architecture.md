# Architecture

Editable (Mermaid) architecture diagrams for **Structured-RAG**. These render on GitHub and
in most Markdown viewers. `docs/architecture.svg` is a static end-to-end render of the same
system; this file is the source-of-truth, diff-able version and adds the detailed retrieval
view.

- [1. System overview](#1-system-overview) — indexing, serving, and evaluation
- [2. Multi-query retrieval](#2-multi-query-retrieval-gated-rewrite--rrf-fusion) — the gated rewrite + RRF fusion

---

## 1. System overview

Two-phase system: build the index **offline** (once per book), serve queries **online** (per
request). A separate set of **offline eval harnesses** scores retrieval and generation on a
labeled gold set.

```mermaid
flowchart TB
  subgraph INDEX["Phase 1 — Indexing pipeline (offline, once per book)"]
    direction TB
    PDF["Laxmikanth PDF (~1500 pp)"]
    PARSE["parsing/<br/>PyMuPDF · TOC tree · align TOC to body"]
    CHUNK["chunking/<br/>split inside sections<br/>parent (full) + child (~600 tok, overlap)"]
    ENRICH["enrichment/<br/>Article entities · syllabus tags<br/>chapter Article catalog"]
    EMBED["indexing/<br/>embed CHILDREN (text-embedding-3-large, 3072-d)<br/>parent text stored in payload, not embedded"]
    PDF --> PARSE --> CHUNK --> ENRICH --> EMBED
    EMBED --> JSONL[("chunks.jsonl")]
    EMBED --> QDRANT[("Qdrant<br/>collection upsc_polity")]
  end

  subgraph SERVE["Phase 2 — Query serving (online, per request)"]
    direction TB
    UI["Browser · Next.js chat (web/)"]
    BFF["BFF proxy · app/api/ask/route.ts<br/>same-origin, pipes NDJSON"]
    API["FastAPI · api/app.py<br/>/ask · /ask/stream · /health"]
    ORCH["Orchestration<br/>direct (default) | LangGraph (UPSC_RAG_PIPELINE=graph)"]
    SMALL{"smalltalk<br/>gate"}
    CACHE{"answer cache<br/>exact-match sqlite (direct only)"}
    QAC[("qa_cache.sqlite")]
    RET["HybridRetriever<br/>dense + BM25 → RRF → rerank → catalog"]
    OFF{"off-topic<br/>gate"}
    GEN["generate_answer<br/>OpenAI chat · grounded + cited [n]"]
    CANNED["canned reply"]
    OOS["out-of-scope reply"]
    UI --> BFF --> API --> ORCH --> SMALL
    SMALL -->|chit-chat| CANNED
    SMALL -->|answer| CACHE
    CACHE -->|"hit · replay, $0"| UI
    CACHE -->|miss| RET --> OFF
    OFF -->|below floor| OOS
    OFF -->|in-scope| GEN --> UI
    GEN -.->|"store (textbook-only)"| QAC
    QAC -.-> CACHE
  end

  subgraph EVAL["Evaluation (offline harnesses, data/eval gold set)"]
    direction TB
    GOLD[("gold set (labeled Qs)")]
    RQ["retrieval quality<br/>hit@k · MRR · article_recall"]
    CQ["cheap generation signals<br/>citation · groundedness · relevance"]
    JQ["LLM-as-judge · gpt-5-mini<br/>faithfulness · completeness · exam · citation"]
    GOLD --> RQ
    GOLD --> CQ
    GOLD --> JQ
  end

  QDRANT -.->|"dense vector search"| RET
  JSONL -.->|"BM25 corpus + parent text"| RET
```

**Load-bearing ideas**

- **Structure preserved end-to-end** — every chunk carries `PART ▸ Chapter ▸ Section ▸ pages ▸ Articles`, so retrieval is filterable and answers are citable.
- **Parent–child retrieval** — embed small children for *precision*, return the parent section for *context*.
- **Hybrid + rerank** — dense and BM25 fused by RRF, then a cross-encoder blended in to separate lookalike sibling sections.
- **One core, three surfaces** — CLI, API, and UI all call the same `HybridRetriever` + `generate_answer`.
- **Config-driven** — a new book = new YAML under `config/books/`, same code path.
- **Two orchestration backends** — direct calls or a LangGraph state machine, same steps, identical eval numbers.

---

## 2. Multi-query retrieval (gated rewrite + RRF fusion)

The retriever can expand a question into **up to 3 rewrite variants** (synonyms/abbreviations)
and fuse retrieval over all of them — but only when the original query retrieves *weakly*.
Well-phrased queries skip the rewrite entirely, so the cost is paid only where it helps.

```mermaid
flowchart TB
  Q["user query<br/>e.g. 'who picks SC judges?'"]
  Q --> FP["FIRST PASS · original query only (1 embed)<br/>dense (Qdrant) + BM25 → ranks + top_score (cosine)"]
  FP --> GATE{"rewrite gate<br/>top_score in [0.30, 0.55) ?"}

  GATE -->|"no · strong (0.55+) or off-topic (below 0.30)"| FUSE
  GATE -->|"yes · weak / vocabulary gap"| RW["rewrite_query() → gpt-4.1-nano<br/>temp 0 · JSON mode · process-cached<br/>expand SC→Supreme Court, picks→appointed<br/>→ up to 3 variants (+ original)"]

  RW --> VS["variant search · ThreadPoolExecutor (1 worker/variant)<br/>v1 dense+BM25 · v2 dense+BM25 · v3 dense+BM25"]
  VS --> FUSE

  FUSE["RRF fusion · _rrf_fuse, k=60<br/>score = Σ 1/(60 + rank + 1) over all rank lists<br/>strong path: 2 lists · weak path: up to 8"]
  FUSE --> POST["dedup to distinct PARENT sections (drop 'Notes and References')<br/>→ cross-encoder rerank · FlashRank, blend 0.5<br/>→ catalog Article attribution + parent-text expansion"]
  POST --> OUT["rerank_top_k results → generation"]
```

**Why it's built this way**

- **"3 variants" = `retrieval.rewrite.num_variants`.** The original query is always kept, so
  fusion runs over **up to 4 queries → up to 8 rank lists** (each query yields a dense list and
  a BM25 list).
- **The gate is the point** (`retrieval/hybrid.py`). Rewriting costs an LLM call + 3× embeds +
  3× searches, so it fires only when the first pass is weak *but still on-topic*:
  `relevance_floor (0.30) ≤ top_score < score_threshold (0.55)`.
  - **Strong** (≥ 0.55) → single-query path, ~0.2–1 s.
  - **Off-topic** (< 0.30) → skip (the relevance gate will reject it downstream anyway).
  - **Only weak/colloquial** queries pay the full multi-query cost (~2 s).
- **Fan-out is concurrent** — each variant needs its own embedding round-trip, so `_search_one`
  runs across a `ThreadPoolExecutor` (measured ~1.24 s → ~0.33 s for the embed/search portion).
- **Rewrite model** is `gpt-4.1-nano` (config `retrieval.rewrite.model`), temperature 0,
  JSON-mode, **process-cached** per `(query, model, num_variants)`.
- **RRF generalizes to N lists** — one formula handles the 2-list strong path and the 8-list
  weak path identically; no special-casing.

> **Note:** on the current 30-question gold set almost every question clears the 0.55 gate, so
> the rewrite layer rarely fires in eval — its lift shows up on genuinely messy/abbreviation-heavy
> queries. The plumbing is correct and gated; demonstrating its value needs a more colloquial
> gold set (the gold-set-expansion item).

---

## Config knobs (see `config/default.yaml`)

| Block | Key knobs |
|-------|-----------|
| `retrieval` | `top_k`, `rerank_top_k`, `relevance_floor` |
| `retrieval.rewrite` | `enabled`, `num_variants`, `model`, `score_threshold` |
| `retrieval.rerank` | `enabled`, `model`, `candidate_pool`, `weight` (cross-encoder blend) |
| `retrieval.catalog` | `enabled`, `match` (embedding\|chapter), `score_threshold`, `max_articles` |
| `retrieval.graph` | `enabled` (shelved — sparse entities on this corpus) |
