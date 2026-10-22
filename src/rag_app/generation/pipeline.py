from rag_app.generation.context import build_context
from rag_app.generation.llm import create_llm
from rag_app.generation.prompts import RAG_PROMPT
from rag_app.retrieval.pipeline import retrieve_and_rerank


def generate_answer(query: str) -> dict:
    """Generate an answer grounded in retrieved documents."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    # 1. Retrieve and rerank
    nodes = retrieve_and_rerank(
        query=query,
        retrieval_top_k=10,
        rerank_top_n=3,
    )

    if not nodes:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": [],
        }

    # 2. Build context
    context = build_context(nodes)

    if not context:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": [],
        }

    # 3. Build prompt
    prompt = RAG_PROMPT.format(
        context=context,
        query=query,
    )

    # 4. Generate answer
    llm = create_llm()
    response = llm.complete(prompt)

    # 5. Return answer and source metadata
    sources = [
        {
            "id": index,
            "file": item.node.metadata.get("file_name"),
            "page": item.node.metadata.get("page_label"),
        }
        for index, item in enumerate(nodes, start=1)
        if item.node.get_content(metadata_mode="none").strip()
    ]

    return {
        "answer": response.text,
        "sources": sources,
    }