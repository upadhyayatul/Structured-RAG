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
    def score(self, name: str, value: float, comment: str | None = None) -> None: ...


class _NoopContext:
    def __enter__(self) -> _NoopSpan:
        return _NoopSpan()
    def __exit__(self, *_: Any) -> None: ...
    def end(self, **kwargs: Any) -> None: ...
    def span(self, name: str, **kwargs: Any) -> "_NoopContext":
        return _NoopContext()
    def generation(self, name: str, **kwargs: Any) -> "_NoopContext":
        return _NoopContext()
    def score(self, name: str, value: float, comment: str | None = None) -> None: ...


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

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        """Attach a numeric score. On a root trace this is a trace-level score;
        on a span/generation it scores that observation. Silently skips if the
        wrapped object has no ``.score`` (older SDK / unexpected object)."""
        fn = getattr(self._span, "score", None)
        if callable(fn):
            fn(name=name, value=value, comment=comment)

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

    @contextmanager
    def start(
        self,
        name: str,
        *,
        parent: "_LangfuseSpanContext | _NoopContext | None" = None,
        input: dict[str, Any] | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Generator[_LangfuseSpanContext | _NoopContext, None, None]:
        """Open ``name`` as a child span of ``parent``, or as a root trace if none.

        Lets a stage nest under a request-level root trace (one question = one trace
        with rolled-up cost/latency) while still working standalone (own root trace)
        when called without a parent — e.g. from scripts or eval. A child span does
        not flush; the root trace flushes once when it closes.
        """
        if parent is not None:
            with parent.span(name, input=input or {}, **kwargs) as span:
                yield span
            return
        with self.trace(name, input=input, session_id=session_id, **kwargs) as t:
            yield t

    def open_root(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> _LangfuseSpanContext | _NoopContext:
        """Return a root trace context WITHOUT the auto-flush context manager.

        For callers whose root must outlive a ``with`` block — e.g. the streaming
        handler, where retrieval runs before the response generator and generation
        runs inside it. The caller owns the lifetime: call ``root.end(...)`` and
        ``trace_manager.flush()`` in a ``finally``.
        """
        if not self._enabled:
            return _NoopContext()
        return _LangfuseSpanContext(self._client.trace(name=name, input=input or {}, session_id=session_id, **kwargs))

    def flush(self) -> None:
        if self._enabled and self._client:
            self._client.flush()


# Module-level singleton — import this everywhere.
trace_manager = TraceManager()

# Shared no-op, handed to helpers that take an optional parent observation so they
# can unconditionally call .span()/.generation() without a None check.
NOOP_CONTEXT = _NoopContext()
