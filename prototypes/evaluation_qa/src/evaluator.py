"""Evaluation engine for the independent Evaluation / QA prototype."""

from typing import Optional

from .metrics import normalize_answer
from .retrieval_metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .schema import EvaluationExample, EvaluationResult


def evaluate_response(
    example: EvaluationExample,
    configuration: str,
    predicted_answer: Optional[str],
    retrieved_chunk_ids: Optional[list[str]] = None,
    k: int = 5,
    latency_ms: Optional[float] = None,
) -> EvaluationResult:
    """
    Evaluate one system response against the fixed
    evaluation example.
    """

    prediction = normalize_answer(
        predicted_answer
    )

    answer_correct = (
        prediction == example.gold_answer
    )

    retrieved = (
        retrieved_chunk_ids
        if retrieved_chunk_ids is not None
        else []
    )

    gold_ids = {
        evidence.chunk_id
        for evidence in example.gold_evidence
    }

    retrieval_available = bool(
        gold_ids
    )

    if retrieval_available:
        hit_score = hit_at_k(
            retrieved,
            gold_ids,
            k,
        )

        precision_score = precision_at_k(
            retrieved,
            gold_ids,
            k,
        )

        recall_score = recall_at_k(
            retrieved,
            gold_ids,
            k,
        )

        rr_score = reciprocal_rank(
            retrieved,
            gold_ids,
        )

    else:
        hit_score = None
        precision_score = None
        recall_score = None
        rr_score = None

    return EvaluationResult(
        question_id=example.question_id,
        configuration=configuration,
        gold_answer=example.gold_answer,
        predicted_answer=prediction,
        answer_correct=answer_correct,
        retrieved_chunk_ids=retrieved,
        hit_at_k=hit_score,
        precision_at_k=precision_score,
        recall_at_k=recall_score,
        reciprocal_rank=rr_score,
        latency_ms=latency_ms,
    )
