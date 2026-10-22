from pathlib import Path
from unittest.mock import patch

from rag_app.ingestion.cli import main


@patch("rag_app.ingestion.cli.run_ingestion")
def test_cli_success(mock_ingestion, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--file", "sample.pdf"],
    )

    mock_ingestion.return_value = {
        "documents": 1,
        "chunks": 3,
        "stored_points": 3,
        "collection": "basic_rag_v1",
        "active_alias": "basic_rag_active",
    }

    exit_code = main()

    assert exit_code == 0
    mock_ingestion.assert_called_once_with(
        pdf_path=Path("sample.pdf"),
        collection_prefix="basic_rag",
    )

    assert '"stored_points": 3' in capsys.readouterr().out


@patch("rag_app.ingestion.cli.run_ingestion")
def test_cli_failure(mock_ingestion, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--file", "missing.pdf"],
    )

    mock_ingestion.side_effect = FileNotFoundError("Document not found")

    exit_code = main()

    assert exit_code == 1
    assert "Ingestion failed" in capsys.readouterr().err