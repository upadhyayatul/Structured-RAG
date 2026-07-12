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

import os
import sys

import requests
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


def _auth() -> tuple[str, tuple[str, str]]:
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3001").rstrip("/")
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not (pub and sec):
        sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing from environment (.env).")
    return host, (pub, sec)


def _existing_names(host: str, auth: tuple[str, str]) -> set[str]:
    """Return modelNames already defined for the project (all pages)."""
    names: set[str] = set()
    page = 1
    while True:
        r = requests.get(
            f"{host}/api/public/models",
            params={"page": page, "limit": 100},
            auth=auth,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
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
    except requests.RequestException as exc:
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
        r = requests.post(f"{host}/api/public/models", json=body, auth=auth, timeout=15)
        if r.status_code >= 300:
            print(f"  ! {name}: HTTP {r.status_code} {r.text[:200]}")
            continue
        created.append(name)

    print(f"Langfuse @ {host}")
    print(f"  created: {created or '—'}")
    print(f"  skipped (already defined): {skipped or '—'}")


if __name__ == "__main__":
    main()
