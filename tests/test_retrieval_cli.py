import json
from unittest.mock import Mock, patch

from rag_app.retrieval.cli import main


@patch("rag_app.retrieval.cli.retrieve_and_rerank")
def test_cli_success(mock_pipeline, monkeypatch, capsys):
    node = Mock()
    node.metadata = {"file_name": "dspy.pdf", "page_label": "3"}
    node.get_content.return_value = "DSPy is a programming framework."

    mock_pipeline.return_value = [
        Mock(node=node, score=3.857)
    ]

    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--query", "What is DSPy?", "--top-k", "5", "--top-n", "1"],
    )

    exit_code = main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(output) == 1
    assert output[0]["rank"] == 1
    assert output[0]["source"] == "dspy.pdf"
    assert output[0]["page"] == "3"
    assert output[0]["score"] == 3.857

    mock_pipeline.assert_called_once_with(
        query="What is DSPy?",
        retrieval_top_k=5,
        rerank_top_n=1,
    )


@patch("rag_app.retrieval.cli.retrieve_and_rerank")
def test_cli_failure(mock_pipeline, monkeypatch, capsys):
    mock_pipeline.side_effect = RuntimeError("Qdrant unavailable")

    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--query", "What is DSPy?"],
    )

    exit_code = main()

    assert exit_code == 1
    assert "Retrieval failed" in capsys.readouterr().err