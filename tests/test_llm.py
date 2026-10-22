from unittest.mock import patch

import pytest

from rag_app.generation.llm import create_llm


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        create_llm(temperature=temperature)


@pytest.mark.parametrize("context_window", [0, -1])
def test_invalid_context_window(context_window):
    with pytest.raises(ValueError, match="context_window"):
        create_llm(context_window=context_window)


@patch("rag_app.generation.llm.Ollama")
def test_create_llm(mock_ollama):
    result = create_llm(
        model_name="test-model",
        temperature=0.2,
        context_window=8192,
    )

    mock_ollama.assert_called_once_with(
        model="test-model",
        temperature=0.2,
        context_window=8192,
        request_timeout=120.0,
    )

    assert result == mock_ollama.return_value