from llama_index.core.schema import NodeWithScore, QueryBundle

from rag_app.retrieval.reranker import create_reranker
from rag_app.retrieval.retriever import create_retriever


def retrieve_and_rerank(
    query: str,
    retrieval_top_k: int = 10,
    rerank_top_n: int = 3,
) -> list[NodeWithScore]:
    """Retrieve relevant chunks and rerank them."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if rerank_top_n > retrieval_top_k:
        raise ValueError(
            "rerank_top_n cannot exceed retrieval_top_k"
        )

    # 1. Retrieve candidates
    retriever = create_retriever(top_k=retrieval_top_k)
    candidates = retriever.retrieve(query)

    if not candidates:
        return []

    # 2. Rerank candidates
    reranker = create_reranker(top_n=rerank_top_n)

    return reranker.postprocess_nodes(
        candidates,
        query_bundle=QueryBundle(query_str=query),
    )