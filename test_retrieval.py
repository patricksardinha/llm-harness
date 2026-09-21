from retrieval import CORPUS, QUESTIONS, mat, recall_at, search_reranked

def test_recall_at_1_no_regression():
    score, _ = recall_at(QUESTIONS, CORPUS, mat, 1, search_reranked)
    assert score >= 0.75


def test_recall_at_3_no_regression():
    score, _ = recall_at(QUESTIONS, CORPUS, mat, 3, search_reranked)
    assert score == 1.0