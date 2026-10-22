from unittest.mock import patch

import pytest

from rag_app.retrieval.reranker import create_reranker


@pytest.mark.parametrize("top_n", [0, -1])
def test_invalid_top_n(top_n):
    with pytest.raises(ValueError, match="top_n must be positive"):
        create_reranker(top_n=top_n)


@patch("rag_app.retrieval.reranker.SentenceTransformerRerank")
def test_create_reranker(mock_reranker):
    result = create_reranker(
        model_name="test-model",
        top_n=3,
        device="cpu",
    )

    mock_reranker.assert_called_once_with(
        model="test-model",
        top_n=3,
        device="cpu",
    )

    assert result == mock_reranker.return_value