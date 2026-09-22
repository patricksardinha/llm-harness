import pytest

from retrieval import CORPUS, QUESTIONS, get_index, recall_at, search_reranked

@pytest.mark.slow
def test_recall_at_1_no_regression():
    score, _ = recall_at(QUESTIONS, CORPUS, get_index(), 1, search_reranked)
    assert score >= 0.75


@pytest.mark.slow
def test_recall_at_3_no_regression():
    score, _ = recall_at(QUESTIONS, CORPUS, get_index(), 3, search_reranked)
    assert score == 1.0