"""Pre-retrieval gates: smalltalk short-circuit and the off-topic relevance floor.
Pure logic (regex + threshold), gates every request, so worth pinning.
"""
from upsc_rag.generation.router import is_off_topic, smalltalk_reply


def test_greeting_is_smalltalk():
    for q in ["hi", "Hello there!", "how are you doing today?", "thanks!", "bye"]:
        assert smalltalk_reply(q) is not None, q


def test_real_question_is_not_smalltalk():
    # Contains "hi"/"is"/"the" but is a genuine question — must fall through to the RAG path.
    for q in ["What is Article 14?", "Hello, what is Article 14?", "How is the President elected?"]:
        assert smalltalk_reply(q) is None, q


def test_off_topic_floor():
    strong = [{"dense_top_score": 0.7}]
    weak = [{"dense_top_score": 0.4}]
    assert is_off_topic(weak, floor=0.5) is True
    assert is_off_topic(strong, floor=0.5) is False
    assert is_off_topic([], floor=0.5) is True       # nothing retrieved
    assert is_off_topic(weak, floor=0.0) is False     # floor 0 disables the gate
