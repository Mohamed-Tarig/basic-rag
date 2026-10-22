from qdrant_client import QdrantClient


def create_qdrant_client(
    url: str = "http://localhost:6333",
) -> QdrantClient:
    """Create a client for the Qdrant server."""
    return QdrantClient(url=url, timeout=10)