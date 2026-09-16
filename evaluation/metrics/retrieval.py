"""Project retrieval measures over a frozen, compatible judged candidate pool."""

from __future__ import annotations

import math


def retrieval_metrics(returned: list[str], qrels: dict[str, int | None] | None, *, k: int) -> dict:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(set(returned)) != len(returned):
        raise ValueError("Duplicate retrieved chunk IDs are not valid ranks")
    selected = returned[:k]
    names = ("hit_at_k", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")
    if qrels is None or not any(value is not None for value in qrels.values()):
        return dict.fromkeys(names) | {
            "actual_returned_count": len(selected),
            "status": "unavailable_qrels",
        }
    if any(
        value is not None
        and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3)
        for value in qrels.values()
    ):
        raise ValueError("Relevance grades must be integers 0–3 or null")
    judged = {key: value for key, value in qrels.items() if value is not None}
    relevant = {key for key, value in judged.items() if value > 0}
    hits = [index for index, key in enumerate(selected, 1) if key in relevant]
    dcg = sum(
        (2 ** judged.get(key, 0) - 1) / math.log2(index + 1)
        for index, key in enumerate(selected, 1)
    )
    ideal = sorted(judged.values(), reverse=True)[:k]
    idcg = sum((2**value - 1) / math.log2(index + 1) for index, value in enumerate(ideal, 1))
    return {
        "hit_at_k": float(bool(hits)),
        "precision_at_k": len(hits) / len(selected) if selected else 0.0,
        "recall_at_k": len(hits) / len(relevant) if relevant else None,
        "mrr": 1 / hits[0] if hits else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else None,
        "actual_returned_count": len(selected),
        "judged_positive_count": len(relevant),
        "unjudged_returned_count": sum(key not in judged for key in selected),
        "status": "partial_judgments" if any(key not in judged for key in selected) else "judged",
    }


def citation_validity(response_type: str, citations: list[str], evidence: list[dict]) -> dict:
    if response_type in {"social", "clarification", "refusal"}:
        return {"applicable": False, "identifier_validity": None, "actual_context_identity": None}
    import hashlib

    mapping = {row["evidence_id"]: row for row in evidence}
    if not citations:
        return {
            "applicable": True,
            "identifier_validity": None,
            "actual_context_identity": None,
            "missing_citations": True,
        }
    ids_valid = len(set(citations)) == len(citations) and all(key in mapping for key in citations)
    identity = ids_valid and all(
        hashlib.sha256(mapping[key]["text"].encode()).hexdigest() == mapping[key]["text_hash"]
        for key in citations
    )
    return {
        "applicable": True,
        "identifier_validity": float(ids_valid),
        "actual_context_identity": float(identity),
        "factual_support": None,
    }
