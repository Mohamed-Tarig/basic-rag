from unittest.mock import Mock, patch

import pytest

from rag_app.generation.pipeline import generate_answer


def test_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        generate_answer("   ")


@patch("rag_app.generation.pipeline.create_llm")
@patch("rag_app.generation.pipeline.retrieve_and_rerank")
def test_no_retrieval_results(mock_retrieve, mock_llm):
    mock_retrieve.return_value = []

    result = generate_answer("What is DSPy?")

    assert result["sources"] == []
    assert "don't have enough information" in result["answer"]
    mock_llm.assert_not_called()


@patch("rag_app.generation.pipeline.create_llm")
@patch("rag_app.generation.pipeline.retrieve_and_rerank")
def test_successful_generation(mock_retrieve, mock_llm):
    node = Mock()
    node.metadata = {
        "file_name": "dspy.pdf",
        "page_label": "3",
    }
    node.get_content.return_value = (
        "DSPy is a framework for programming language models."
    )

    mock_retrieve.return_value = [Mock(node=node, score=3.85)]
    mock_llm.return_value.complete.return_value.text = (
        "DSPy is a programming framework. [Source 1]"
    )

    result = generate_answer("What is DSPy?")

    assert result["answer"] == (
        "DSPy is a programming framework. [Source 1]"
    )
    assert result["sources"] == [
        {"id": 1, "file": "dspy.pdf", "page": "3"}
    ]

    mock_llm.return_value.complete.assert_called_once()
    prompt = mock_llm.return_value.complete.call_args.args[0]

    assert "What is DSPy?" in prompt
    assert "DSPy is a framework" in prompt
    assert "[Source 1]" in prompt