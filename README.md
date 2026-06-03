# UPSC-RAG

Structured **retrieval-augmented generation (RAG)** for massive UPSC polity textbooks—starting with *Indian Polity* (6th ed.) by M. Laxmikanth (~1,500 pages).

Unlike naive RAG (fixed-size text splits), this project preserves the book’s hierarchy—**PART → Chapter → Section**—so retrieval returns contextually correct snippets (e.g. Fundamental Rights vs Centre–State relations) with citations.

---

## How it works (pipeline)

```mermaid
flowchart LR
  PDF[Source PDF] --> Parse[parsing/]
  Parse --> TOC[TOC + sections]
  TOC --> Chunk[chunking/]
  Chunk --> Enrich[enrichment/]
  Enrich --> Index[indexing/]
  Index --> Retrieve[retrieval/]
  Retrieve --> Gen[generation/]
  Gen --> Answer[Grounded answer]
```

| Stage | Package | Purpose |
|-------|---------|---------|
| 1. Parse | `parsing/` | Extract text from PDF pages (PyMuPDF) |
| 2. Structure | `parsing/toc.py` | Build PART / chapter / section tree from Contents |
| 3. Chunk | `chunking/` | Split text inside section boundaries with overlap |
| 4. Enrich | `enrichment/` | Add syllabus tags, entities, content types |
| 5. Index | `indexing/` | Save `chunks.jsonl`; later vector + BM25 stores |
| 6. Retrieve | `retrieval/` | Hybrid search + rerank (planned) |
| 7. Answer | `generation/` | LLM prompt with cited sources (planned) |

**Current status:** Config, PDF parsing, chunking utilities, and ingest skeleton are in place. Full TOC → section → chunk export and vector retrieval are next.

---

## Project layout

```
UPSC-RAG/
├── .env.example              # Environment variable template (copy to .env)
├── .gitignore                # Ignores venv, caches, processed data, vector DB dirs
├── pyproject.toml            # Package metadata, dependencies, CLI entry point
├── requirements.txt          # Runtime dependencies (mirrors pyproject.toml)
├── requirements-dev.txt      # Dev deps + editable install (`pip install -e .`)
├── README.md                 # This file
│
├── config/                   # YAML configuration (no secrets)
│   ├── default.yaml          # Global defaults for all books
│   └── books/
│       └── laxmikanth_6.yaml # Book-specific PDF path, structure, exclusions
│
├── data/                     # Data on disk (not the Python package)
│   ├── raw/                  # Optional folder for future source PDFs
│   ├── *.pdf                 # Your textbook PDF(s) (e.g. Laxmikanth)
│   └── processed/            # Generated artifacts per book (gitignored)
│       └── laxmikanth_6/
│           ├── manifest.json # Book metadata + page count from ingest
│           └── chunks.jsonl  # One JSON object per chunk (main index input)
│
├── scripts/
│   └── ingest.py             # Thin CLI: runs the ingest pipeline
│
├── src/upsc_rag/             # Main Python package (installed editable)
│   ├── __init__.py           # Package version
│   ├── config.py             # Load YAML + .env; resolve paths
│   ├── parsing/              # PDF and table-of-contents logic
│   ├── chunking/             # Hierarchy-aware text splitting
│   ├── enrichment/           # Extra metadata before indexing
│   ├── indexing/             # Persist chunks (JSONL today; vectors later)
│   ├── retrieval/            # Load/search chunks (hybrid search planned)
│   ├── generation/           # Build grounded LLM prompts
│   └── pipeline/             # Orchestrates end-to-end ingest
│
├── tests/                    # Pytest suite
│   ├── conftest.py           # Adds `src/` to Python path
│   ├── test_config.py        # Config loading smoke tests
│   └── test_chunking.py      # Chunking + entity extraction tests
│
└── .venv/                    # Local virtual environment (gitignored)
```

---

## Root files

| File | Description |
|------|-------------|
| **`.env.example`** | Template for environment variables. Copy to `.env` to override paths (`UPSC_RAG_DATA_DIR`, `UPSC_RAG_PROCESSED_DIR`) and later API keys for embeddings/LLMs. |
| **`.gitignore`** | Keeps `.venv`, `__pycache__`, `.env`, test caches, vector DB folders (`chroma/`, `qdrant_storage/`), and `data/processed/` out of version control. |
| **`pyproject.toml`** | Defines the `upsc-rag` package (Python ≥3.11), dependencies, pytest/ruff settings, and the `upsc-rag-ingest` console script. |
| **`requirements.txt`** | Runtime pins: `pydantic`, `pydantic-settings`, `python-dotenv`, `PyYAML`, `pymupdf`. |
| **`requirements-dev.txt`** | Installs runtime deps plus `pytest`, `ruff`, and the project in editable mode. |

---

## `config/` — YAML settings

Configuration is split so one codebase can ingest many books.

### `config/default.yaml`

Shared defaults merged into every book config:

| Key | Meaning |
|-----|---------|
| `project.processed_dir` | Where generated files are written (`data/processed`) |
| `parsing.content_start_page` | First page of main body (overridden per book) |
| `chunking.child_chunk_tokens` | Target size for child chunks (~600 tokens) |
| `chunking.child_chunk_overlap` | Overlap between consecutive chunks in the same section |
| `chunking.min_chunk_chars` | Minimum chunk length to avoid tiny fragments |
| `indexing.collection_name` | Name for the future vector collection |
| `retrieval.top_k` / `rerank_top_k` | How many chunks to retrieve and rerank |

### `config/books/laxmikanth_6.yaml`

Book-specific overrides:

| Key | Meaning |
|-----|---------|
| `book.id` | Identifier used in CLI (`--book laxmikanth_6`) and output paths |
| `book.pdf_path` | Path to the PDF relative to project root |
| `parsing.content_start_page` | First page of chapter body (after front matter and Contents) |
| `parsing.content_end_page` | Last page of chapter body (before appendices / index) |
| `structure.part_pattern` | Regex to detect `PART-I`, `PART-II`, etc. |
| `structure.chapter_pattern` | Regex to detect numbered chapters (`7 Fundamental Rights`) |

Add new books by creating `config/books/<book_id>.yaml`.

---

## `data/` — inputs and outputs

| Path | Description |
|------|-------------|
| **`data/raw/`** | Optional staging area for PDFs you add later. The Laxmikanth file currently lives at `data/M laxmikanth 6th edition.pdf` as configured in YAML. |
| **`data/processed/<book_id>/`** | All generated artifacts for one book. Regenerated by ingest; safe to delete and rebuild. |
| **`manifest.json`** | Written by ingest: `book_id`, title, resolved PDF path, `page_count`. |
| **`chunks.jsonl`** | Newline-delimited JSON—one record per chunk. Primary input for indexing and retrieval. |

Example chunk record (target schema):

```json
{
  "id": "ch07_right_to_equality_003_a1b2c3",
  "text": "...",
  "book_id": "laxmikanth_6",
  "part": "PART-I",
  "chapter_num": 7,
  "chapter_title": "Fundamental Rights",
  "section_path": ["Fundamental Rights", "Right to Equality"],
  "page_start": 112,
  "page_end": 118,
  "content_type": "body",
  "entities": ["Article 14", "Article 15"]
}
```

---

## `src/upsc_rag/` — Python package

### `config.py`

Central configuration loader:

- **`AppSettings`** — Reads `.env` with prefix `UPSC_RAG_` (e.g. `UPSC_RAG_PROCESSED_DIR`).
- **`BookConfig`** — Validated book metadata (id, title, author, edition, `pdf_path`).
- **`load_runtime_config(book_id)`** — Deep-merges `default.yaml` + `books/<book_id>.yaml`.
- **`PROJECT_ROOT`** — Absolute path to the repo root for resolving relative paths.

### `parsing/` — PDF extraction and structure

| Module | Description |
|--------|-------------|
| **`pdf.py`** | Opens PDFs with PyMuPDF (`fitz`). `extract_page_text()` returns plain text for a 1-based page. `iter_pages()` streams page number + text for a range. |
| **`toc.py`** | Parses the book **Contents** pages into a tree (`TocNode`: PART → chapter → subsection). Uses regex for `PART-I` and numbered chapters. Page boundaries are assigned in a later ingest step. |

### `chunking/` — Structured splits

| Module | Description |
|--------|-------------|
| **`structured.py`** | **`ChunkRecord`** dataclass holds chunk text plus hierarchy metadata. **`chunk_section_text()`** splits within a section using paragraph boundaries and token estimates—not arbitrary character cuts. **`extract_entities()`** pulls references like `Article 14` for filtering and citations. |

### `enrichment/` — Metadata tags

| Module | Description |
|--------|-------------|
| **`metadata.py`** | **`enrich_chunk()`** adds derived fields before indexing (e.g. `syllabus_tags: ["GS2_Polity"]`). Will grow to support content types (`body`, `table`, `mcq`) and exam focus. |

### `indexing/` — Persistence

| Module | Description |
|--------|-------------|
| **`store.py`** | **`save_chunks_jsonl()`** writes chunk dicts to JSONL. Future: Chroma/Qdrant/pgvector with the same metadata payload. |

### `retrieval/` — Search (planned)

| Module | Description |
|--------|-------------|
| **`hybrid.py`** | **`load_chunks_jsonl()`** reads chunks back from disk. Will add BM25 + dense vectors + reranking per `config/default.yaml`. |

### `generation/` — Answers (planned)

| Module | Description |
|--------|-------------|
| **`answer.py`** | **`build_answer_prompt()`** formats retrieved chunks with source numbers and chapter/page citations for a grounded LLM response. |

### `pipeline/` — Orchestration

| Module | Description |
|--------|-------------|
| **`ingest.py`** | **`run_ingest(book_id)`** runs the full ingest: load config → open PDF → write `manifest.json` → (future) TOC + chunk → `chunks.jsonl`. **`main()`** exposes CLI flags `--book` and `--output`. |

Registered as console script: **`upsc-rag-ingest`**.

---

## `scripts/` — Command-line entry

| File | Description |
|------|-------------|
| **`ingest.py`** | Wrapper so you can run `python scripts/ingest.py` without remembering module paths. Delegates to `upsc_rag.pipeline.ingest:main`. |

---

## `tests/` — Quality checks

| File | Description |
|------|-------------|
| **`conftest.py`** | Ensures `src/` is on `sys.path` for pytest. |
| **`test_config.py`** | Verifies project root exists and Laxmikanth YAML loads correctly. |
| **`test_chunking.py`** | Tests entity regex and that `chunk_section_text()` yields valid `ChunkRecord`s. |

---

## Setup

```powershell
cd p:\ML-AI\projects\UPSC-RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Optional: copy environment template and adjust paths.

```powershell
copy .env.example .env
```

---

## Commands

**Run ingest** (validates PDF, writes manifest; chunk export still stub):

```powershell
python scripts/ingest.py --book laxmikanth_6
# equivalent:
upsc-rag-ingest --book laxmikanth_6
```

**Run tests:**

```powershell
pytest
```

**Lint (optional):**

```powershell
ruff check src tests scripts
```

---

## Design principles

1. **Structure first** — Chunk inside TOC sections, not across chapter boundaries.
2. **Rich metadata** — Every chunk carries `part`, `chapter`, `section_path`, pages, and entities for filtered retrieval.
3. **Config-driven books** — New textbooks = new YAML under `config/books/`, same code path.
4. **Reproducible artifacts** — `data/processed/` can be deleted and rebuilt from the PDF + config.
5. **Incremental build** — JSONL indexing works before vector DBs; embeddings and hybrid search plug in later.

---

## Roadmap

- [ ] Wire TOC page-range detection and section body extraction
- [ ] Populate `chunks.jsonl` from Laxmikanth hierarchy
- [ ] Vector store + BM25 hybrid retrieval
- [ ] Reranker + `generation/` LLM integration
- [ ] Evaluation set from chapter MCQs / past UPSC questions
