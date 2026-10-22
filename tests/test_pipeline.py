from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from rag_app.ingestion.pipeline import run_ingestion


@patch("rag_app.ingestion.pipeline.activate_collection")
@patch("rag_app.ingestion.pipeline.create_qdrant_client")
@patch("rag_app.ingestion.pipeline.create_index")
@patch("rag_app.ingestion.pipeline.create_embedder")
@patch("rag_app.ingestion.pipeline.assign_stable_ids")
@patch("rag_app.ingestion.pipeline.chunk_documents")
@patch("rag_app.ingestion.pipeline.load_documents")
def test_validation_failure_does_not_activate_index(
    mock_load,
    mock_chunk,
    mock_assign_ids,
    mock_embedder,
    mock_create_index,
    mock_client_factory,
    mock_activate,
):
    mock_load.return_value = [Mock()]
    mock_chunk.return_value = [Mock(), Mock()]
    mock_assign_ids.return_value = [Mock(), Mock()]

    mock_create_index.return_value = (
        Mock(),
        "basic_rag_new",
    )

    client = Mock()
    client.count.return_value.count = 1
    mock_client_factory.return_value = client

    with pytest.raises(RuntimeError, match="Index validation failed"):
        run_ingestion(Path("sample.pdf"))

    mock_activate.assert_not_called()

@patch("rag_app.ingestion.pipeline.activate_collection")
@patch("rag_app.ingestion.pipeline.create_qdrant_client")
@patch("rag_app.ingestion.pipeline.create_index")
@patch("rag_app.ingestion.pipeline.create_embedder")
@patch("rag_app.ingestion.pipeline.assign_stable_ids")
@patch("rag_app.ingestion.pipeline.chunk_documents")
@patch("rag_app.ingestion.pipeline.load_documents")
def test_successful_ingestion_activates_new_index(
    mock_load,
    mock_chunk,
    mock_assign_ids,
    mock_embedder,
    mock_create_index,
    mock_client_factory,
    mock_activate,
):
    mock_load.return_value = [Mock(), Mock()]
    mock_chunk.return_value = [Mock(), Mock(), Mock()]
    mock_assign_ids.return_value = [Mock(), Mock(), Mock()]

    mock_create_index.return_value = (
        Mock(),
        "basic_rag_new",
    )

    client = Mock()
    client.count.return_value.count = 3
    mock_client_factory.return_value = client

    result = run_ingestion(
        Path("sample.pdf"),
        collection_prefix="basic_rag",
    )

    mock_activate.assert_called_once_with(
        client=client,
        collection_name="basic_rag_new",
        alias_name="basic_rag_active",
    )

    assert result["documents"] == 2
    assert result["chunks"] == 3
    assert result["stored_points"] == 3
    assert result["collection"] == "basic_rag_new"
    assert result["active_alias"] == "basic_rag_active"