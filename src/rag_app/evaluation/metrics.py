def recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Calculate recall over relevant document/page IDs."""

    if k < 1:
        raise ValueError("k must be positive")

    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    retrieved_set = set(retrieved[:k])

    return len(retrieved_set & relevant_set) / len(relevant_set)


def reciprocal_rank(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    """Return reciprocal rank of the first relevant result."""

    relevant_set = set(relevant)

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(scores: list[float]) -> float:
    """Calculate mean reciprocal rank across queries."""

    if not scores:
        return 0.0

    return sum(scores) / len(scores)