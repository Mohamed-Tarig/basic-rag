import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def create_embedder(
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str | None = None,
) -> HuggingFaceEmbedding:
    """Create a HuggingFace embedding model."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        embed_batch_size=16,
    )