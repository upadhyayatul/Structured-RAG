"""Graceful-degradation checks for /ask: upstream failures become clean responses,
not 500s. Exercises the handler directly with a fake retriever + monkeypatched
generation, so no Qdrant/OpenAI is needed.
"""
import httpx
import pytest
from fastapi import HTTPException

import upsc_rag.api.app as app_mod
from upsc_rag.api.app import AskRequest, GENERATION_UNAVAILABLE_MSG, ask


class _FakeRetriever:
    def __init__(self, results=None, boom=None):
        self._results = results or []
        self._boom = boom

    def retrieve(self, *a, **k):
        if self._boom:
            raise self._boom
        return self._results


@pytest.fixture(autouse=True)
def _wire_state(monkeypatch):
    # Minimal cfg: no cache, conversation off, gates neutral.
    monkeypatch.setitem(app_mod._state, "cfg", {"retrieval": {"relevance_floor": 0.0}})
    monkeypatch.setitem(app_mod._state, "answer_cache", None)
    # Not smalltalk, on-topic, sources sufficient (skip web) — isolate the failure paths.
    monkeypatch.setattr(app_mod, "smalltalk_reply", lambda q: None)
    monkeypatch.setattr(app_mod, "is_off_topic", lambda results, floor: False)
    monkeypatch.setattr(app_mod, "sources_answer_question", lambda *a, **k: True)
    monkeypatch.setattr(app_mod, "record_answer_scores", lambda *a, **k: None)
    monkeypatch.setattr(app_mod, "_build_sources", lambda results: [])


def test_retrieval_down_returns_503(monkeypatch):
    monkeypatch.setitem(
        app_mod._state, "retriever", _FakeRetriever(boom=httpx.ConnectError("qdrant down"))
    )
    with pytest.raises(HTTPException) as exc:
        ask(AskRequest(query="who appoints the CJI"))
    assert exc.value.status_code == 503


def test_generation_down_returns_sources(monkeypatch):
    monkeypatch.setitem(app_mod._state, "retriever", _FakeRetriever(results=[{"x": 1}]))

    def _boom(*a, **k):
        raise httpx.ReadTimeout("openai timeout")

    monkeypatch.setattr(app_mod, "generate_answer", _boom)
    resp = ask(AskRequest(query="who appoints the CJI"))
    # Graceful: fallback message, NOT a 500 — retrieval succeeded so we still respond.
    assert resp.answer == GENERATION_UNAVAILABLE_MSG


def test_happy_path_still_works(monkeypatch):
    monkeypatch.setitem(app_mod._state, "retriever", _FakeRetriever(results=[{"x": 1}]))
    monkeypatch.setattr(app_mod, "generate_answer", lambda *a, **k: "Article 124 governs it.")
    resp = ask(AskRequest(query="who appoints the CJI"))
    assert resp.answer == "Article 124 governs it."
