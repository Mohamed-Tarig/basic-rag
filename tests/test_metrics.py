import pytest

from rag_app.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    retrieved = ["2", "4", "3"]
    relevant = ["3", "5"]

    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_does_not_double_count_pages():
    retrieved = ["3", "3", "4"]
    relevant = ["3", "5"]

    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_with_no_relevant_pages():
    assert recall_at_k(["1", "2"], [], k=2) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(["1"], ["1"], k=k)


def test_reciprocal_rank():
    assert reciprocal_rank(["2", "4", "3"], ["3"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["1", "2"], ["3"]) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([1.0, 0.5, 0.25]) == pytest.approx(
        0.5833333333
    )


def test_mrr_empty_dataset():
    assert mean_reciprocal_rank([]) == 0.0