from llama_index.llms.ollama import Ollama


def create_llm(
    model_name: str = "llama3.2:1b",
    temperature: float = 0.1,
    context_window: int = 4096,
) -> Ollama:
    """Create a local LLM client using Ollama."""

    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")

    if context_window < 1:
        raise ValueError("context_window must be positive")

    return Ollama(
        model=model_name,
        temperature=temperature,
        context_window=context_window,
        request_timeout=120.0,
    )