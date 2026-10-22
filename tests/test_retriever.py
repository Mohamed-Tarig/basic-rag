from unittest.mock import Mock, patch

import pytest

from rag_app.retrieval.retriever import create_retriever


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k must be positive"):
        create_retriever(top_k=top_k)


@patch("rag_app.retrieval.retriever.create_embedder")
@patch("rag_app.retrieval.retriever.VectorStoreIndex")
@patch("rag_app.retrieval.retriever.QdrantVectorStore")
@patch("rag_app.retrieval.retriever.create_qdrant_client")
def test_create_retriever(
    mock_client_factory,
    mock_vector_store,
    mock_index_class,
    mock_embedder,
):
    client = Mock()
    mock_client_factory.return_value = client

    retriever = create_retriever(
        collection_name="basic_rag_active",
        top_k=5,
    )

    mock_vector_store.assert_called_once_with(
        client=client,
        collection_name="basic_rag_active",
    )

    mock_index_class.from_vector_store.assert_called_once_with(
        vector_store=mock_vector_store.return_value,
        embed_model=mock_embedder.return_value,
    )

    index = mock_index_class.from_vector_store.return_value
    index.as_retriever.assert_called_once_with(similarity_top_k=5)

    assert retriever == index.as_retriever.return_value