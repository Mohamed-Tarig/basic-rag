from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank


def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 3,
    device: str = "cuda",
) -> SentenceTransformerRerank:
    """Create a Cross-Encoder reranker."""

    if top_n < 1:
        raise ValueError("top_n must be positive")

    return SentenceTransformerRerank(
        model=model_name,
        top_n=top_n,
        device=device,
    )