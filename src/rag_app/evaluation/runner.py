import json
from pathlib import Path

from llama_index.core.schema import QueryBundle

from rag_app.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from rag_app.retrieval.reranker import create_reranker
from rag_app.retrieval.retriever import create_retriever


def evaluate_retrieval(dataset_path: Path) -> dict:
    """Compare retrieval and reranking using page-level relevance."""

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    if not dataset:
        raise ValueError("Evaluation dataset cannot be empty")

    retriever = create_retriever(top_k=10)
    reranker = create_reranker(top_n=3)

    before_recalls = []
    after_recalls = []
    before_rrs = []
    after_rrs = []

    for item in dataset:
        query = item["query"]
        relevant = item["relevant_pages"]

        candidates = retriever.retrieve(query)

        reranked = reranker.postprocess_nodes(
            candidates,
            query_bundle=QueryBundle(query_str=query),
        )

        # Compare the same output budget: Top 3 vs Top 3
        before_pages = [
            node.node.metadata.get("page_label")
            for node in candidates[:3]
        ]

        after_pages = [
            node.node.metadata.get("page_label")
            for node in reranked[:3]
        ]

        before_recalls.append(recall_at_k(before_pages, relevant, 3))
        after_recalls.append(recall_at_k(after_pages, relevant, 3))

        before_rrs.append(reciprocal_rank(before_pages, relevant))
        after_rrs.append(reciprocal_rank(after_pages, relevant))

    n = len(dataset)

    return {
        "queries": n,
        "retrieval_recall_at_3": sum(before_recalls) / n,
        "reranked_recall_at_3": sum(after_recalls) / n,
        "retrieval_mrr_at_3": mean_reciprocal_rank(before_rrs),
        "reranked_mrr_at_3": mean_reciprocal_rank(after_rrs),
    }