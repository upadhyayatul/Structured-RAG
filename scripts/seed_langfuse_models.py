"""CLI: seed Langfuse model-pricing definitions for the app's models.

Langfuse computes each generation's cost by matching its ``model`` string against a
model-pricing table and multiplying token usage by the per-token prices. Langfuse ships
built-ins for common OpenAI models (e.g. gpt-4o-mini, text-embedding-3-large), but the
custom names this project uses (gpt-4.1-nano, gpt-5-mini) are NOT priced out of the box,
so those steps log zero cost and the unified `ask` roll-up undercounts. This script upserts
project-scoped definitions so every model the app calls is priced.

Idempotent: existing project definitions (matched by modelName) are left alone.

Run:
    "P:/ML-AI/git repo/Structured-RAG/.venv/Scripts/python.exe" scripts/seed_langfuse_models.py

Prices are USD PER TOKEN (OpenAI list prices ÷ 1e6). Edit MODELS below to adjust — in
particular gpt-5-mini is a PLACEHOLDER; set it to the real price for your deployment.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()

# modelName, matchPattern (case-insensitive, anchored), input $/token, output $/token.
# Embeddings bill on input tokens only -> output price 0.
MODELS = [
    # name,                      match regex,                          input,   output
    ("gpt-4o-mini",             r"(?i)^gpt-4o-mini$",                 1.5e-7,  6.0e-7),
    ("gpt-4.1-nano",            r"(?i)^gpt-4\.1-nano$",               1.0e-7,  4.0e-7),
    ("gpt-5-mini",              r"(?i)^gpt-5-mini$",                  2.5e-7,  2.0e-6),  # PLACEHOLDER — set real price
    ("text-embedding-3-large",  r"(?i)^text-embedding-3-large$",      1.3e-7,  0.0),
    ("text-embedding-3-small",  r"(?i)^text-embedding-3-small$",      2.0e-8,  0.0),
]


def _auth() -> tuple[str, str]:
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3001").rstrip("/")
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not (pub and sec):
        sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing from environment (.env).")
    basic = base64.b64encode(f"{pub}:{sec}".encode()).decode()
    return host, f"Basic {basic}"


def _request(url: str, auth: str, body: dict | None = None) -> dict:
    """GET (body=None) or POST JSON with basic auth; returns the parsed JSON response."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": auth, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read() or b"{}")


def _existing_names(host: str, auth: str) -> set[str]:
    """Return modelNames already defined for the project (all pages)."""
    names: set[str] = set()
    page = 1
    while True:
        params = urllib.parse.urlencode({"page": page, "limit": 100})
        data = _request(f"{host}/api/public/models?{params}", auth).get("data", [])
        if not data:
            break
        names.update(m.get("modelName", "") for m in data)
        if len(data) < 100:
            break
        page += 1
    return names


def main() -> None:
    host, auth = _auth()
    try:
        existing = _existing_names(host, auth)
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"Could not reach Langfuse at {host} (is the container up?): {exc}")

    created, skipped = [], []
    for name, pattern, in_price, out_price in MODELS:
        if name in existing:
            skipped.append(name)
            continue
        body = {
            "modelName": name,
            "matchPattern": pattern,
            "unit": "TOKENS",
            "inputPrice": in_price,
            "outputPrice": out_price,
        }
        try:
            _request(f"{host}/api/public/models", auth, body)
        except urllib.error.HTTPError as exc:
            print(f"  ! {name}: HTTP {exc.code} {exc.read()[:200]!r}")
            continue
        created.append(name)

    print(f"Langfuse @ {host}")
    print(f"  created: {created or '—'}")
    print(f"  skipped (already defined): {skipped or '—'}")


if __name__ == "__main__":
    main()
