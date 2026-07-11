"""LLM client factory + optional LangChain adapters — the provider-portability seam.

The default pipeline talks to the OpenAI SDK directly (see ``generation/answer.py``,
``indexing/embedder.py``). Two opt-in seams live here:

* ``UPSC_RAG_LLM_GATEWAY=litellm`` routes every CHAT/completion call through a
  LiteLLM **proxy** (an OpenAI-compatible gateway) for central cost tracking, virtual
  keys, fallbacks, and provider swaps. ``get_openai_client()`` is the single factory the
  chat call sites use; when the gateway is off it returns a plain direct-OpenAI client,
  so the default path is byte-for-byte unchanged. Embeddings deliberately do NOT go
  through the gateway — they stay on the direct client (dimension-locked to Qdrant).
* ``UPSC_RAG_LLM_BACKEND=langchain`` routes the graph's generate node through
  ``ChatOpenAI`` instead of the OpenAI SDK, so the model provider can later be swapped
  by changing one factory rather than every call site.

Both are off by default — set the env var to opt in.
"""
from __future__ import annotations

import os
from typing import Any


def gateway_enabled() -> bool:
    """True when the LiteLLM proxy gateway is selected via env (default: direct OpenAI)."""
    return os.environ.get("UPSC_RAG_LLM_GATEWAY", "").lower() == "litellm"


def _gateway_base_url() -> str:
    """Base URL of the LiteLLM proxy (OpenAI-compatible ``/v1`` endpoint)."""
    return os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")


def _gateway_api_key() -> str:
    """Key presented to the proxy — its virtual/master key, or the OpenAI key as fallback."""
    return os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def get_openai_client() -> Any:
    """Build an OpenAI SDK client for CHAT/completion calls.

    Routes through the LiteLLM proxy when ``UPSC_RAG_LLM_GATEWAY=litellm``, else talks to
    OpenAI directly. The returned object is a drop-in ``openai.OpenAI`` instance, so call
    sites use ``.chat.completions.create(...)`` exactly as before. Embeddings must NOT use
    this — they keep their own direct client (the proxy fronts chat only).
    """
    from openai import OpenAI

    if gateway_enabled():
        return OpenAI(base_url=_gateway_base_url(), api_key=_gateway_api_key())
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def langchain_backend_enabled() -> bool:
    """True when the LangChain LLM backend is selected via env (default: OpenAI SDK)."""
    return os.environ.get("UPSC_RAG_LLM_BACKEND", "openai").lower() == "langchain"


def get_chat_model(cfg: dict[str, Any]) -> Any:
    """Build a ``ChatOpenAI`` from the ``generation`` config block.

    Reads model / temperature / max_tokens from ``cfg["generation"]`` so it matches the
    OpenAI-SDK path's parameters exactly. When the LiteLLM gateway is enabled it points at
    the proxy (same as ``get_openai_client``) so the langchain path is gated identically.
    """
    from langchain_openai import ChatOpenAI

    gen_cfg = cfg.get("generation", {})
    kwargs: dict[str, Any] = {
        "model": gen_cfg.get("model", "gpt-4o-mini"),
        "temperature": gen_cfg.get("temperature", 0.2),
        "max_tokens": gen_cfg.get("max_tokens", 1024),
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
    if gateway_enabled():
        kwargs["base_url"] = _gateway_base_url()
        kwargs["api_key"] = _gateway_api_key()
    return ChatOpenAI(**kwargs)


def get_embeddings(cfg: dict[str, Any]) -> Any:
    """Build an ``OpenAIEmbeddings`` matching the indexing embedding model.

    Provided for symmetry / future use; the current retrieval path keeps its own
    query-embedding call (it also needs token usage for cost tracing). Embeddings stay on
    the direct OpenAI endpoint — intentionally NOT routed through the LiteLLM gateway.
    """
    from langchain_openai import OpenAIEmbeddings

    model = cfg.get("indexing", {}).get("embedding_model", "text-embedding-3-large")
    return OpenAIEmbeddings(model=model, api_key=os.environ.get("OPENAI_API_KEY"))
