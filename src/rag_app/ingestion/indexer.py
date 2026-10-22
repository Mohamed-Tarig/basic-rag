from uuid import uuid4

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag_app.storage.vector_store import create_qdrant_client


def create_index(
    nodes,
    embed_model,
    collection_prefix: str = "basic_rag",
) -> tuple[VectorStoreIndex, str]:
    """Build a new versioned index without touching the active collection."""

    client = create_qdrant_client()

    collection_name = f"{collection_prefix}_{uuid4().hex[:12]}"

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    return index, collection_name