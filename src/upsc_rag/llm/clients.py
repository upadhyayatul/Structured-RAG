"""LLM client factory — the single seam every chat call site builds its client through.

The default pipeline talks to the OpenAI SDK directly (see ``generation/answer.py``,
``indexing/embedder.py``). One opt-in seam lives here: ``UPSC_RAG_LLM_GATEWAY=litellm``
routes every CHAT/completion call through a LiteLLM **proxy** (an OpenAI-compatible
gateway) for central cost tracking, virtual keys, fallbacks, and provider swaps.
``get_openai_client()`` is the single factory the chat call sites use; when the gateway
is off it returns a plain direct-OpenAI client, so the default path is byte-for-byte
unchanged. Embeddings deliberately do NOT go through the gateway — they stay on the
direct client (dimension-locked to Qdrant).
"""
from __future__ import annotations

import os
from typing import Any

# Bound how long a single chat call may hang before it fails, so a wedged upstream
# (OpenAI or the proxy black-holing) can't pin a worker forever. The SDK already retries
# transient errors with backoff; this only caps the per-attempt wait. Tunable via env.
_TIMEOUT_S = float(os.environ.get("UPSC_RAG_LLM_TIMEOUT", "30"))


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
        return OpenAI(base_url=_gateway_base_url(), api_key=_gateway_api_key(), timeout=_TIMEOUT_S)
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=_TIMEOUT_S)
