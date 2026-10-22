import json
from unittest.mock import Mock, patch

import pytest

from rag_app.evaluation.runner import evaluate_retrieval


@patch("rag_app.evaluation.runner.create_reranker")
@patch("rag_app.evaluation.runner.create_retriever")
def test_evaluation_runner(mock_retriever, mock_reranker, tmp_path):
    dataset = [
        {
            "query": "What is DSPy?",
            "relevant_pages": ["3"],
        }
    ]

    path = tmp_path / "eval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    def make_result(page):
        node = Mock()
        node.metadata = {"page_label": page}
        return Mock(node=node)

    candidates = [
        make_result("2"),
        make_result("4"),
        make_result("3"),
    ]

    reranked = [
        candidates[2],
        candidates[0],
        candidates[1],
    ]

    mock_retriever.return_value.retrieve.return_value = candidates
    mock_reranker.return_value.postprocess_nodes.return_value = reranked

    result = evaluate_retrieval(path)

    assert result["queries"] == 1
    assert result["retrieval_recall_at_3"] == 1.0
    assert result["reranked_recall_at_3"] == 1.0

    assert result["retrieval_mrr_at_3"] == pytest.approx(1 / 3)
    assert result["reranked_mrr_at_3"] == 1.0

    mock_retriever.assert_called_once_with(top_k=10)
    mock_reranker.assert_called_once_with(top_n=3)