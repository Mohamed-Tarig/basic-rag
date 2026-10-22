import json
from unittest.mock import patch

from rag_app.generation.cli import main


@patch("rag_app.generation.cli.generate_answer")
def test_cli_success(mock_generate, monkeypatch, capsys):
    mock_generate.return_value = {
        "answer": "DSPy is a programming model. [Source 1]",
        "sources": [
            {"id": 1, "file": "dspy.pdf", "page": "3"},
        ],
    }

    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--query", "What is DSPy?"],
    )

    exit_code = main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["answer"] == (
        "DSPy is a programming model. [Source 1]"
    )
    assert output["sources"][0]["page"] == "3"

    mock_generate.assert_called_once_with(
        query="What is DSPy?"
    )


@patch("rag_app.generation.cli.generate_answer")
def test_cli_failure(mock_generate, monkeypatch, capsys):
    mock_generate.side_effect = RuntimeError("Ollama unavailable")

    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--query", "What is DSPy?"],
    )

    exit_code = main()

    assert exit_code == 1
    assert "RAG generation failed" in capsys.readouterr().err