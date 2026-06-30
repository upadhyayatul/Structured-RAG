"""Optional LangChain LLM/embedding adapters — a thin provider-portability seam.

The default pipeline talks to the OpenAI SDK directly (see ``generation/answer.py``,
``indexing/embedder.py``). When ``UPSC_RAG_LLM_BACKEND=langchain`` the graph's generate
node routes through ``ChatOpenAI`` instead, so the model provider can later be swapped
(Anthropic, Azure, a local server) by changing one factory rather than every call site.
Off by default — set the env var to opt in.

These adapters are intentionally minimal: they do NOT replace the custom retrieval,
batched corpus embedding, or Qdrant payload handling, which stay on the existing code.
"""
from __future__ import annotations

import os
from typing import Any


def langchain_backend_enabled() -> bool:
    """True when the LangChain LLM backend is selected via env (default: OpenAI SDK)."""
    return os.environ.get("UPSC_RAG_LLM_BACKEND", "openai").lower() == "langchain"


def get_chat_model(cfg: dict[str, Any]) -> Any:
    """Build a ``ChatOpenAI`` from the ``generation`` config block.

    Reads model / temperature / max_tokens from ``cfg["generation"]`` so it matches the
    OpenAI-SDK path's parameters exactly.
    """
    from langchain_openai import ChatOpenAI

    gen_cfg = cfg.get("generation", {})
    return ChatOpenAI(
        model=gen_cfg.get("model", "gpt-4o-mini"),
        temperature=gen_cfg.get("temperature", 0.2),
        max_tokens=gen_cfg.get("max_tokens", 1024),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def get_embeddings(cfg: dict[str, Any]) -> Any:
    """Build an ``OpenAIEmbeddings`` matching the indexing embedding model.

    Provided for symmetry / future use; the current retrieval path keeps its own
    query-embedding call (it also needs token usage for cost tracing).
    """
    from langchain_openai import OpenAIEmbeddings

    model = cfg.get("indexing", {}).get("embedding_model", "text-embedding-3-large")
    return OpenAIEmbeddings(model=model, api_key=os.environ.get("OPENAI_API_KEY"))
