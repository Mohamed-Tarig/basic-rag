from unittest.mock import Mock, patch

from rag_app.ingestion.indexer import create_index


@patch("rag_app.ingestion.indexer.uuid4")
@patch("rag_app.ingestion.indexer.VectorStoreIndex")
@patch("rag_app.ingestion.indexer.StorageContext")
@patch("rag_app.ingestion.indexer.QdrantVectorStore")
@patch("rag_app.ingestion.indexer.create_qdrant_client")
def test_create_versioned_index(
    mock_create_client,
    mock_vector_store,
    mock_storage_context,
    mock_index,
    mock_uuid,
):
    # Make the collection version deterministic
    mock_uuid.return_value.hex = "abc123def4567890"

    client = Mock()
    mock_create_client.return_value = client

    index, collection_name = create_index(
        nodes=[],
        embed_model=Mock(),
        collection_prefix="basic_rag",
    )

    assert collection_name == "basic_rag_abc123def456"

    mock_vector_store.assert_called_once_with(
        client=client,
        collection_name=collection_name,
    )

    mock_index.assert_called_once()
    assert index == mock_index.return_value

    # Existing collections must remain untouched
    client.delete_collection.assert_not_called()