from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_app.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_success():
    expected_result = {
        "answer": "DSPy is a programming framework. [Source 1]",
        "sources": [
            {"id": 1, "file": "dspy.pdf", "page": "3"}
        ],
    }

    with patch(
        "rag_app.api.main.generate_answer",
        return_value=expected_result,
    ) as mock_generate:
        response = client.post(
            "/query",
            json={"query": "What is DSPy?"},
        )

    assert response.status_code == 200
    assert response.json() == expected_result
    mock_generate.assert_called_once_with(
        query="What is DSPy?"
    )


def test_query_empty_string():
    response = client.post(
        "/query",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_query_whitespace_only():
    response = client.post(
        "/query",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_query_missing_field():
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422


def test_query_service_failure():
    with patch(
        "rag_app.api.main.generate_answer",
        side_effect=RuntimeError("Ollama unavailable"),
    ):
        response = client.post(
            "/query",
            json={"query": "What is DSPy?"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "RAG service temporarily unavailable"
    )
    assert "Ollama unavailable" not in response.text