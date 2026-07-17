"""Reciprocal Rank Fusion — the core of hybrid retrieval. Generalized to N rank lists
(dense, BM25, per-variant); a regression here silently degrades every query.
"""
from upsc_rag.retrieval.hybrid import HybridRetriever, _rrf_score

fuse = HybridRetriever._rrf_fuse  # staticmethod, no instance / Qdrant needed


def test_higher_rank_scores_more():
    fused = fuse([{"a": 0, "b": 5}])  # a ranked above b in the one list
    assert fused["a"] > fused["b"]


def test_appearing_in_more_lists_wins():
    # "a" is rank-1 in both lists; "b" is rank-0 in only one.
    fused = fuse([{"a": 1, "b": 0}, {"a": 1}])
    assert fused["a"] > fused["b"]


def test_score_matches_rrf_formula():
    fused = fuse([{"a": 0}, {"a": 2}])
    assert fused["a"] == _rrf_score(0) + _rrf_score(2)


def test_empty_input_is_empty():
    assert fuse([]) == {}
