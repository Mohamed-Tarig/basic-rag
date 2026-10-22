from pathlib import Path
from unittest.mock import patch

import pytest
from llama_index.core import Document

from rag_app.ingestion.loader import load_documents, validate_document_path


def test_missing_document():
    with pytest.raises(FileNotFoundError):
        validate_document_path(Path("missing.pdf"))


def test_invalid_file_type(tmp_path):
    file = tmp_path / "document.txt"
    file.write_text("Hello RAG")

    with pytest.raises(ValueError):
        validate_document_path(file)


def test_valid_pdf(tmp_path):
    file = tmp_path / "document.pdf"
    file.write_bytes(b"%PDF-1.4\n")

    result = validate_document_path(file)

    assert result == file

def test_load_documents(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    expected = [Document(text="Sample document content")]

    with patch(
        "rag_app.ingestion.loader.SimpleDirectoryReader"
    ) as mock_reader:
        mock_reader.return_value.load_data.return_value = expected

        result = load_documents(pdf)

    assert result == expected