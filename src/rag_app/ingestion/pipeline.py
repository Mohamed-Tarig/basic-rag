from pathlib import Path

from rag_app.ingestion.chunker import assign_stable_ids, chunk_documents
from rag_app.ingestion.embedder import create_embedder
from rag_app.ingestion.indexer import create_index
from rag_app.ingestion.loader import load_documents
from rag_app.storage.aliases import activate_collection
from rag_app.storage.vector_store import create_qdrant_client


def run_ingestion(
    pdf_path: Path,
    collection_prefix: str = "basic_rag",
) -> dict:
    """Build, validate, and activate a versioned RAG index."""

    # 1. Load
    documents = load_documents(pdf_path)

    # 2. Chunk
    nodes = assign_stable_ids(chunk_documents(documents))

    if not nodes:
        raise ValueError("No chunks generated from document")

    # 3. Embed and persist to a new collection
    embed_model = create_embedder()
    _, new_collection = create_index(
        nodes=nodes,
        embed_model=embed_model,
        collection_prefix=collection_prefix,
    )

    # 4. Validate the new collection
    client = create_qdrant_client()

    stored_count = client.count(
        collection_name=new_collection,
        exact=True,
    ).count

    if stored_count != len(nodes):
        raise RuntimeError(
            f"Index validation failed: "
            f"expected {len(nodes)}, got {stored_count}"
        )

    # 5. Activate only after successful validation
    alias_name = f"{collection_prefix}_active"

    activate_collection(
        client=client,
        collection_name=new_collection,
        alias_name=alias_name,
    )

    return {
        "documents": len(documents),
        "chunks": len(nodes),
        "stored_points": stored_count,
        "collection": new_collection,
        "active_alias": alias_name,
    }