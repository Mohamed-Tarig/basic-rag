from unittest.mock import Mock

from rag_app.generation.context import build_context


def make_node(content: str, file_name: str, page: str):
    node = Mock()
    node.metadata = {
        "file_name": file_name,
        "page_label": page,
    }
    node.get_content.return_value = content

    return Mock(node=node, score=0.9)


def test_context_includes_content_and_sources():
    nodes = [
        make_node("DSPy is a programming model.", "dspy.pdf", "3"),
        make_node("DSPy uses signatures.", "dspy.pdf", "4"),
    ]

    context = build_context(nodes)

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Page: 3" in context
    assert "Page: 4" in context
    assert "DSPy uses signatures." in context


def test_empty_nodes_returns_empty_context():
    assert build_context([]) == ""


def test_empty_content_is_skipped():
    nodes = [
        make_node("   ", "empty.pdf", "1"),
        make_node("Useful information.", "dspy.pdf", "3"),
    ]

    context = build_context(nodes)

    assert "Useful information." in context
    assert "empty.pdf" not in context