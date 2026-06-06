# CLAUDE.md — Project Reference

Quick-start for Claude Code. Read this instead of scanning all source files.
See `progress.json` for what's done and what's next.

---

## Project

**UPSC-RAG** — Structured retrieval-augmented generation over M. Laxmikanth's
*Indian Polity* (6th ed., 1531 pages). Preserves PART → Chapter → Section
hierarchy so retrieval returns contextually correct, citable snippets.

---

## Environment

| Item | Value |
|------|-------|
| Python venv | `P:\ML-AI\git repo\Structured-RAG\.venv\Scripts\python.exe` (Python 3.13) |
| Qdrant (Docker) | `http://localhost:6333` |
| API keys | `.env` at project root — needs `OPENAI_API_KEY` |
| Config dir | `config/default.yaml` + `config/books/laxmikanth_6.yaml` |
| Processed data | `data/processed/laxmikanth_6/` |

Always invoke scripts with the full path to this interpreter, e.g.:

```powershell
"P:/ML-AI/git repo/Structured-RAG/.venv/Scripts/python.exe" scripts/embed.py --book laxmikanth_6
```

**Venv caveat:** this `.venv` was cloned from `P:\ML-AI\projects\UPSC-RAG\.venv`,
so its `pyvenv.cfg` and bare `pip.exe` still reference that other path — a bare
`pip install` writes to the WRONG site-packages and the local python won't see it.
Always install with `python.exe -m pip install ...` (which targets the interpreter
that actually imports). To fix permanently, recreate: delete `.venv`, then
`python -m venv .venv` and reinstall `requirements.txt`.

---

## Pipeline stages

```
PDF → parsing/ → chunking/ → enrichment/ → indexing/ → retrieval/ → generation/
```

| Stage | Script / command | Status |
|-------|-----------------|--------|
| 1. Ingest (parse + chunk) | `python scripts/ingest.py --book laxmikanth_6` | Done |
| 2. Embed + upsert Qdrant | `python scripts/embed.py --book laxmikanth_6` | Blocked on `.env` |
| 3. Hybrid retrieval | `retrieval/hybrid.py` — skeleton only | Planned |
| 4. Answer generation | `generation/answer.py` — prompt builder only | Planned |

---

## Key files

```
config/
  default.yaml              global defaults (chunk sizes, Qdrant URL, embedding model)
  books/laxmikanth_6.yaml   PDF path, TOC/content page ranges, regex patterns

data/processed/laxmikanth_6/
  manifest.json             book metadata + page count
  toc.json                  parsed TOC tree (cached — ingest reuses if present)
  chunks.jsonl              1835 chunks (366 parents + 1469 children)

src/upsc_rag/
  config.py                 AppSettings, BookConfig, load_runtime_config()
  parsing/pdf.py            PyMuPDF wrapper — extract_page_text(), iter_pages()
  parsing/toc.py            TocNode tree from Contents pages
  parsing/align.py          align TOC entries to body pages, extract_sections()
  chunking/structured.py    ChunkRecord, chunk_section_text() — paragraph-aware splits
  enrichment/metadata.py    enrich_chunk() — adds syllabus_tags
  indexing/store.py         save_chunks_jsonl()
  indexing/embedder.py      embed_texts() — batched OpenAI API calls with retry
  indexing/qdrant_store.py  ensure_collection(), upsert_points()
  retrieval/hybrid.py       load_chunks_jsonl() — BM25 + dense search (planned)
  generation/answer.py      build_answer_prompt() — formats sources for LLM
  pipeline/ingest.py        run_ingest() orchestration
  pipeline/embed.py         run_embed() orchestration

scripts/
  calibrate_pages.py        verify PDF page ranges in config before ingest
  ingest.py                 thin CLI wrapper → pipeline/ingest.py
  embed.py                  thin CLI wrapper → pipeline/embed.py
```

---

## Chunk schema (`chunks.jsonl`)

Each line is a JSON object:

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
  "content_type": "child",   // "parent" | "child"
  "parent_id": "sec_0002",   // null for parent chunks
  "entities": ["Article 14"]
}
```

**Parent chunks** = full section text (not embedded).
**Child chunks** = overlapping ~600-token splits (embedded into Qdrant).
Parent text is stored as `parent_text` in the Qdrant payload for context expansion.

---

## Qdrant collection

- **Name:** `upsc_polity`
- **Model:** `text-embedding-3-large` (OpenAI), 3072 dimensions, cosine similarity
- **Points:** child chunks only (~1469 expected)
- **Payload fields:** `chunk_id`, `parent_id`, `parent_text`, `text`, `book_id`,
  `part`, `chapter_num`, `chapter_title`, `section_path`, `page_start`,
  `page_end`, `entities`

---

## Config keys (default.yaml)

```yaml
indexing:
  collection_name: upsc_polity
  qdrant_url: http://localhost:6333
  embedding_model: text-embedding-3-large
  embedding_dim: 3072
  embed_batch_size: 100

chunking:
  child_chunk_tokens: 600
  child_chunk_overlap: 80
  min_chunk_chars: 200

retrieval:
  top_k: 30
  rerank_top_k: 8
```

---

## Adding a new book

1. Create `config/books/<book_id>.yaml` with `book:`, `parsing:` page ranges, `structure:` regex.
2. Place the PDF at the path specified in that YAML.
3. Run `python scripts/calibrate_pages.py --book <book_id>` to verify pages.
4. Run `python scripts/ingest.py --book <book_id>` → produces `chunks.jsonl`.
5. Run `python scripts/embed.py --book <book_id>` → upserts into Qdrant.

---

## Design principles

1. Chunk inside TOC sections — never across chapter boundaries.
2. Every chunk carries full hierarchy metadata for filtered retrieval.
3. Parent-child structure: embed children for precision, retrieve parent for full context.
4. Config-driven — new book = new YAML, same code.
5. Incremental — each stage outputs an artifact the next stage reads.
