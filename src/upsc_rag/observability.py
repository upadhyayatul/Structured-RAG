"""Langfuse observability helpers — thin wrappers so the rest of the code stays clean.

If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set, all calls become no-ops
so development without Langfuse works without any changes.

Usage
-----
    from upsc_rag.observability import trace_manager

    with trace_manager.trace("retrieve", input={"query": q}) as span:
        with span.span("embed") as s:
            result = embed(q)
            s.end(output={"tokens": len(result)})
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Generator


class _NoopSpan:
    """Drop-in when Langfuse is not configured."""
    def end(self, **kwargs: Any) -> None: ...
    def span(self, name: str, **kwargs: Any) -> "_NoopContext": ...  # type: ignore[empty-body]
    def generation(self, name: str, **kwargs: Any) -> "_NoopContext": ...  # type: ignore[empty-body]


class _NoopContext:
    def __enter__(self) -> _NoopSpan:
        return _NoopSpan()
    def __exit__(self, *_: Any) -> None: ...
    def end(self, **kwargs: Any) -> None: ...
    def span(self, name: str, **kwargs: Any) -> "_NoopContext":
        return _NoopContext()
    def generation(self, name: str, **kwargs: Any) -> "_NoopContext":
        return _NoopContext()


class _LangfuseSpanContext:
    """Wraps a Langfuse span/generation as a context manager."""

    def __init__(self, span: Any) -> None:
        self._span = span
        self._t0 = time.perf_counter()

    def __enter__(self) -> "_LangfuseSpanContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._span.update(level="ERROR", status_message=str(exc_val))
        self._finish()

    def end(self, **kwargs: Any) -> None:
        if kwargs:
            self._span.update(**kwargs)
        self._finish()

    def _finish(self) -> None:
        # Spans/generations have .end(); a top-level trace does not (it is closed
        # on flush). Call .end() only when the wrapped object actually exposes it.
        end = getattr(self._span, "end", None)
        if callable(end):
            end()

    def span(self, name: str, **kwargs: Any) -> "_LangfuseSpanContext":
        return _LangfuseSpanContext(self._span.span(name=name, **kwargs))

    def generation(self, name: str, **kwargs: Any) -> "_LangfuseSpanContext":
        return _LangfuseSpanContext(self._span.generation(name=name, **kwargs))

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000


class TraceManager:
    """Creates top-level Langfuse traces, or returns no-ops when not configured."""

    def __init__(self) -> None:
        self._client: Any = None
        self._enabled = False
        self._try_init()

    def _try_init(self) -> None:
        pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3001")
        if not (pub and sec):
            return
        try:
            from langfuse import Langfuse
            self._client = Langfuse(public_key=pub, secret_key=sec, host=host)
            self._enabled = True
        except Exception:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def trace(
        self, name: str, *, input: dict[str, Any] | None = None, **kwargs: Any
    ) -> Generator[_LangfuseSpanContext | _NoopContext, None, None]:
        if not self._enabled:
            yield _NoopContext()
            return
        t = self._client.trace(name=name, input=input or {}, **kwargs)
        ctx = _LangfuseSpanContext(t)
        try:
            yield ctx
        except Exception as exc:
            t.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            if self._client:
                self._client.flush()

    def flush(self) -> None:
        if self._enabled and self._client:
            self._client.flush()


# Module-level singleton — import this everywhere.
trace_manager = TraceManager()

# Shared no-op, handed to helpers that take an optional parent observation so they
# can unconditionally call .span()/.generation() without a None check.
NOOP_CONTEXT = _NoopContext()
