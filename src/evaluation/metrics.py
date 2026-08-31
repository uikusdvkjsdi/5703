"""Evaluation metrics."""

from typing import Optional


def exact_match_accuracy(predictions: list[Optional[str]], references: list[str]) -> float:
    """Compute exact-match accuracy."""
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if p is not None and p.upper() == r.upper())
    return correct / len(predictions)


def extract_letter(text: str) -> Optional[str]:
    """Best-effort extraction of an A-D answer from raw text."""
    import re

    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None
