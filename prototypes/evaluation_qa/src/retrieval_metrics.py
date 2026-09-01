"""Retrieval metrics for the Evaluation / QA prototype."""


def hit_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str],
    k: int,
) -> float:
    """
    Return 1.0 if at least one gold evidence chunk
    appears in the top-k retrieved results.
    """

    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]

    return float(
        bool(set(top_k) & gold_ids)
    )


def precision_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str],
    k: int,
) -> float:
    """
    Calculate the proportion of top-k retrieved chunks
    that are relevant gold evidence.
    """

    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant = len(
        set(top_k) & gold_ids
    )

    return relevant / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    gold_ids: set[str],
    k: int,
) -> float:
    """
    Calculate the proportion of all gold evidence chunks
    retrieved within the top-k results.
    """

    if not gold_ids:
        return 0.0

    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant = len(
        set(top_k) & gold_ids
    )

    return relevant / len(gold_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    gold_ids: set[str],
) -> float:
    """
    Return the reciprocal rank of the first relevant
    retrieved chunk.
    """

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in gold_ids:
            return 1.0 / rank

    return 0.0
