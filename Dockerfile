# FastAPI backend image. Pure-Python deps ship as wheels, so a single slim stage
# is enough — no compiler toolchain needed.
#
# Build (from repo root; the processed data under data/processed/ must exist locally —
# run scripts/ingest.py first, it is gitignored but IS copied into the image):
#   docker build -t upsc-rag-api .
#
# Run (needs a reachable Qdrant + an OpenAI key):
#   docker run -p 8000:8000 \
#     -e OPENAI_API_KEY=sk-... \
#     -e UPSC_RAG_QDRANT_URL=http://host.docker.internal:6333 \
#     upsc-rag-api
FROM python:3.13-slim

WORKDIR /app

# Install deps first so the layer caches when only source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code + config + the prebuilt index data (chunks.jsonl / section_articles.json
# are read at startup to build the BM25 index; Qdrant holds the vectors separately).
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/ ./config/
COPY data/processed/ ./data/processed/
RUN pip install --no-cache-dir --no-deps -e .

# Run as non-root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# FastAPI has no /health that pings Qdrant yet, so probe the OpenAPI schema route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/openapi.json')" || exit 1

CMD ["uvicorn", "upsc_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
