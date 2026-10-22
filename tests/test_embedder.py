from unittest.mock import patch

from rag_app.ingestion.embedder import create_embedder


@patch("rag_app.ingestion.embedder.HuggingFaceEmbedding")
@patch("rag_app.ingestion.embedder.torch.cuda.is_available")
def test_uses_gpu_when_available(mock_cuda, mock_embedding):
    mock_cuda.return_value = True

    create_embedder()

    assert mock_embedding.call_args.kwargs["device"] == "cuda"
    assert mock_embedding.call_args.kwargs["embed_batch_size"] == 16


@patch("rag_app.ingestion.embedder.HuggingFaceEmbedding")
@patch("rag_app.ingestion.embedder.torch.cuda.is_available")
def test_falls_back_to_cpu(mock_cuda, mock_embedding):
    mock_cuda.return_value = False

    create_embedder()

    assert mock_embedding.call_args.kwargs["device"] == "cpu"


@patch("rag_app.ingestion.embedder.HuggingFaceEmbedding")
def test_explicit_device_override(mock_embedding):
    create_embedder(device="cpu")

    assert mock_embedding.call_args.kwargs["device"] == "cpu"