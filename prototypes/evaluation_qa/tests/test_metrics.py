"""Tests for answer-level evaluation metrics."""

from src.metrics import (
    exact_match_accuracy,
    invalid_answer_rate,
    normalize_answer,
)


def test_normalize_answer_accepts_single_label():
    assert normalize_answer("A") == "A"
    assert normalize_answer("b") == "B"


def test_normalize_answer_extracts_answer_format():
    assert normalize_answer("Answer: C") == "C"
    assert normalize_answer("answer: d") == "D"


def test_normalize_answer_rejects_invalid_response():
    assert normalize_answer("I do not know") is None
    assert normalize_answer(None) is None


def test_exact_match_accuracy():
    predictions = [
        "A",
        "B",
        "C",
        "D",
    ]

    references = [
        "A",
        "B",
        "A",
        "D",
    ]

    score = exact_match_accuracy(
        predictions,
        references,
    )

    assert score == 0.75


def test_invalid_answer_rate():
    predictions = [
        "A",
        "Answer: B",
        "I do not know",
        None,
    ]

    score = invalid_answer_rate(
        predictions
    )

    assert score == 0.5
