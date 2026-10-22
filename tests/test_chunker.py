import pytest
from llama_index.core import Document

from rag_app.ingestion.chunker import assign_stable_ids, chunk_documents


def test_chunking_preserves_content_and_metadata():
    document = Document(
        text=("Retrieval augmented generation improves responses. " * 100),
        metadata={"file_name": "sample.pdf", "page_label": "1"},
    )

    nodes = chunk_documents(
        [document],
        chunk_size=128,
        chunk_overlap=20,
    )

    assert len(nodes) > 1
    assert all(node.get_content().strip() for node in nodes)
    assert all(node.metadata["file_name"] == "sample.pdf" for node in nodes)
    assert all(node.metadata["page_label"] == "1" for node in nodes)


def test_empty_documents():
    nodes = chunk_documents([])

    assert nodes == []


def test_invalid_chunk_settings():
    document = Document(text="Sample content")

    with pytest.raises(ValueError):
        chunk_documents(
            [document],
            chunk_size=50,
            chunk_overlap=50,
        )


def test_stable_ids_are_deterministic():
    doc = Document(
        text="This is a sample document.",
        metadata={
            "file_name": "sample.pdf",
            "page_label": "1",
        },
    )

    first = assign_stable_ids(chunk_documents([doc]))
    second = assign_stable_ids(chunk_documents([doc]))

    assert [node.node_id for node in first] == [
        node.node_id for node in second
    ]


def test_different_content_produces_different_ids():
    doc1 = Document(
        text="First document content.",
        metadata={"file_name": "sample.pdf", "page_label": "1"},
    )

    doc2 = Document(
        text="Second document content.",
        metadata={"file_name": "sample.pdf", "page_label": "1"},
    )

    nodes1 = assign_stable_ids(chunk_documents([doc1]))
    nodes2 = assign_stable_ids(chunk_documents([doc2]))

    assert nodes1[0].node_id != nodes2[0].node_id