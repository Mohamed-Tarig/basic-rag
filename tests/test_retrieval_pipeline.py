from unittest.mock import Mock, patch

import pytest

from rag_app.retrieval.pipeline import retrieve_and_rerank


def test_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_and_rerank("   ")


def test_rejects_invalid_top_n():
    with pytest.raises(ValueError, match="cannot exceed"):
        retrieve_and_rerank(
            "What is DSPy?",
            retrieval_top_k=3,
            rerank_top_n=5,
        )


@patch("rag_app.retrieval.pipeline.create_reranker")
@patch("rag_app.retrieval.pipeline.create_retriever")
def test_empty_retrieval_skips_reranking(mock_retriever, mock_reranker):
    mock_retriever.return_value.retrieve.return_value = []

    results = retrieve_and_rerank("What is DSPy?")

    assert results == []
    mock_reranker.assert_not_called()


@patch("rag_app.retrieval.pipeline.create_reranker")
@patch("rag_app.retrieval.pipeline.create_retriever")
def test_successful_retrieval_and_reranking(mock_retriever, mock_reranker):
    candidates = [Mock(), Mock(), Mock()]
    reranked = [candidates[1], candidates[0]]

    mock_retriever.return_value.retrieve.return_value = candidates
    mock_reranker.return_value.postprocess_nodes.return_value = reranked

    result = retrieve_and_rerank(
        "What is DSPy?",
        retrieval_top_k=10,
        rerank_top_n=2,
    )

    mock_retriever.assert_called_once_with(top_k=10)
    mock_reranker.assert_called_once_with(top_n=2)

    assert result == reranked

    call_args = mock_reranker.return_value.postprocess_nodes.call_args

    assert call_args.args[0] == candidates
    assert call_args.kwargs["query_bundle"].query_str == "What is DSPy?"