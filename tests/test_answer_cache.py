"""Roundtrip + normalization checks for the sqlite answer cache."""
from upsc_rag.api.cache import AnswerCache


def test_roundtrip_and_normalization(tmp_path):
    cache = AnswerCache(tmp_path / "qa_cache.sqlite")

    sources = [{"n": 1, "type": "book", "chapter_title": "Amendment of the Constitution"}]
    cache.put("what is article 368", "Article 368 deals with amendments.", sources)

    # Caps / extra spaces / trailing punctuation all hit the same row.
    for variant in [
        "what is article 368",
        "What   is Article 368?",
        "WHAT IS ARTICLE 368!",
    ]:
        hit = cache.get(variant)
        assert hit is not None, variant
        answer, got_sources = hit
        assert answer == "Article 368 deals with amendments."
        assert got_sources == sources

    # Different wording misses.
    assert cache.get("explain article 368") is None

    # put overwrites.
    cache.put("What is Article 368?", "Updated answer.", [])
    assert cache.get("what is article 368") == ("Updated answer.", [])


def test_persists_across_instances(tmp_path):
    db = tmp_path / "qa_cache.sqlite"
    AnswerCache(db).put("q", "a", [])
    assert AnswerCache(db).get("q") == ("a", [])
