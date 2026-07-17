"""Eval-harness metric math (hit@k, MRR, article_recall) with a fake retriever — the
quality numbers are only trustworthy if the scorer itself is correct. No API/Qdrant.
"""
from upsc_rag.eval.harness import GoldQuestion, evaluate


class _FakeRetriever:
    """Returns a fixed result list regardless of the query."""
    def __init__(self, results):
        self._results = results

    def retrieve(self, question, rerank_top_k=None):
        return self._results


def _res(section, entities=None):
    return {"section_path": ["PART", section], "entities": entities or []}


def test_hit_and_mrr_when_gold_section_ranks_first():
    retr = _FakeRetriever([_res("Fundamental Rights", ["Article 14"]), _res("Other")])
    gold = [GoldQuestion("what are FRs", section_contains=["fundamental rights"],
                         articles=["Article 14"])]
    rep = evaluate(retr, gold)
    assert rep["hit_at_k"] == 1.0
    assert rep["mrr"] == 1.0
    assert rep["article_recall"] == 1.0


def test_mrr_reflects_second_position():
    retr = _FakeRetriever([_res("Other"), _res("Fundamental Rights")])
    gold = [GoldQuestion("q", section_contains=["fundamental rights"])]
    rep = evaluate(retr, gold)
    assert rep["hit_at_k"] == 1.0
    assert rep["mrr"] == 0.5          # first relevant at rank 2 -> 1/2


def test_miss_scores_zero():
    retr = _FakeRetriever([_res("Other")])
    gold = [GoldQuestion("q", section_contains=["nonexistent section"])]
    rep = evaluate(retr, gold)
    assert rep["hit_at_k"] == 0.0
    assert rep["mrr"] == 0.0


def test_article_recall_none_when_no_labels():
    retr = _FakeRetriever([_res("Fundamental Rights")])
    gold = [GoldQuestion("q", section_contains=["fundamental rights"])]  # no articles labeled
    assert evaluate(retr, gold)["article_recall"] is None
