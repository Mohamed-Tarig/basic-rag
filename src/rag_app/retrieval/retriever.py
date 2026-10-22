from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag_app.ingestion.embedder import create_embedder
from rag_app.storage.vector_store import create_qdrant_client


def create_retriever(
    collection_name: str = "basic_rag_active",
    top_k: int = 10,
):
    """Connect to an existing Qdrant index for semantic retrieval."""

    if top_k < 1:
        raise ValueError("top_k must be positive")

    client = create_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=create_embedder(),
    )

    return index.as_retriever(similarity_top_k=top_k)