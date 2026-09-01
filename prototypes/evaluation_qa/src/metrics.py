"""Answer-level metrics for the Evaluation / QA prototype."""

import re
from typing import Optional


VALID_LABELS = {"A", "B", "C", "D"}


def normalize_answer(
    answer: Optional[str],
) -> Optional[str]:
    """
    Convert a model answer into a standard A-D label.

    Returns None when no valid answer label can be identified.
    """

    if answer is None:
        return None

    answer = answer.strip().upper()

    if answer in VALID_LABELS:
        return answer

    match = re.search(
        r"\b(?:ANSWER\s*:\s*)?([A-D])\b",
        answer,
    )

    if match:
        return match.group(1)

    return None


def exact_match_accuracy(
    predictions: list[Optional[str]],
    references: list[str],
) -> float:
    """
    Calculate MCQ exact-match accuracy.
    """

    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have the same length"
        )

    correct = sum(
        normalize_answer(prediction)
        == normalize_answer(reference)
        for prediction, reference
        in zip(predictions, references)
    )

    return correct / len(references)


def invalid_answer_rate(
    predictions: list[Optional[str]],
) -> float:
    """
    Calculate the proportion of predictions that cannot be
    mapped to A, B, C or D.
    """

    if not predictions:
        return 0.0

    invalid = sum(
        normalize_answer(prediction) is None
        for prediction in predictions
    )

    return invalid / len(predictions)
