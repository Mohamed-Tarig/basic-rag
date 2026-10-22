import uuid

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[BaseNode]:
    """Split documents into chunks while preserving metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.get_nodes_from_documents(documents)

def assign_stable_ids(nodes: list[BaseNode]) -> list[BaseNode]:
    """Assign deterministic IDs based on document identity and chunk content."""

    for node in nodes:
        source = node.metadata.get("file_name", "")
        page = node.metadata.get("page_label", "")
        content = node.get_content(metadata_mode="none")

        fingerprint = f"{source}:{page}:{content}"
        node.id_ = str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint))

    return nodes