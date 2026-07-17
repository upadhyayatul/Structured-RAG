"""Source dedup + renumbering — parent-text expansion maps several child chunks onto
one section, and these helpers must collapse them and number from 1. Regressed before.
"""
from upsc_rag.generation.sources import (
    build_source_dicts,
    build_web_source_dicts,
    dedupe_results,
)


def _book(path, ps, pe):
    return {"section_path": path, "chapter_title": path[-1], "page_start": ps, "page_end": pe}


def test_dedupe_collapses_same_section_and_pages():
    dupes = [_book(["P1", "Fundamental Rights"], 40, 45)] * 3
    assert len(dedupe_results(dupes)) == 1


def test_dedupe_keeps_distinct_sections_in_order():
    results = [_book(["P1", "A"], 1, 2), _book(["P1", "B"], 3, 4)]
    out = dedupe_results(results)
    assert [r["chapter_title"] for r in out] == ["A", "B"]


def test_build_source_dicts_renumbers_from_one():
    results = [_book(["P1", "A"], 1, 2), _book(["P1", "A"], 1, 2), _book(["P1", "B"], 3, 4)]
    sources = build_source_dicts(results)
    assert [s["n"] for s in sources] == [1, 2]          # deduped then renumbered
    assert all(s["type"] == "book" for s in sources)


def test_web_sources_dedupe_by_url_and_offset_numbering():
    web = [
        {"title": "a", "url": "http://x", "snippet": "s"},
        {"title": "a2", "url": "http://x", "snippet": "s"},   # dup URL
        {"title": "b", "url": "http://y", "snippet": "s"},
    ]
    sources = build_web_source_dicts(web, start_n=2)          # book had 2 sources before
    assert [s["n"] for s in sources] == [3, 4]                # numbered after the book ones
    assert all(s["type"] == "web" for s in sources)
