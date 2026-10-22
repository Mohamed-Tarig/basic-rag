from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_app.api.main import app

client = TestClient(app)


def test_upload_success():
    expected = {
        "documents": 1,
        "chunks": 10,
        "stored_points": 10,
        "collection": "basic_rag_test123",
        "active_alias": "basic_rag_active",
    }

    with patch(
        "rag_app.api.upload.run_ingestion",
        return_value=expected,
    ) as mock_ingestion:
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "test.pdf",
                    BytesIO(b"%PDF-1.4\nsample"),
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200
    assert response.json() == expected
    mock_ingestion.assert_called_once()


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"


def test_upload_rejects_empty_file():
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Empty PDF file"


def test_upload_rejects_oversized_file():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "large.pdf",
                b"x" * (20 * 1024 * 1024 + 1),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413


def test_upload_ingestion_failure():
    with patch(
        "rag_app.api.upload.run_ingestion",
        side_effect=RuntimeError("Indexing failed"),
    ):
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", b"%PDF-1.4\nsample", "application/pdf")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Document ingestion failed"
    assert "Indexing failed" not in response.text


def test_upload_rejects_fake_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "fake.pdf",
                b"This is not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF file"