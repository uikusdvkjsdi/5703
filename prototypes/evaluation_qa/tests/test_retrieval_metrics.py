"""Tests for retrieval evaluation metrics."""

from src.retrieval_metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k():
    retrieved = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]

    gold = {
        "chunk_2",
    }

    assert hit_at_k(
        retrieved,
        gold,
        1,
    ) == 0.0

    assert hit_at_k(
        retrieved,
        gold,
        3,
    ) == 1.0


def test_precision_at_k():
    retrieved = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]

    gold = {
        "chunk_2",
    }

    score = precision_at_k(
        retrieved,
        gold,
        3,
    )

    assert score == 1 / 3


def test_recall_at_k():
    retrieved = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]

    gold = {
        "chunk_2",
        "chunk_4",
    }

    score = recall_at_k(
        retrieved,
        gold,
        3,
    )

    assert score == 0.5


def test_reciprocal_rank():
    retrieved = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]

    gold = {
        "chunk_2",
    }

    score = reciprocal_rank(
        retrieved,
        gold,
    )

    assert score == 0.5


def test_no_relevant_retrieval():
    retrieved = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]

    gold = {
        "chunk_9",
    }

    assert hit_at_k(
        retrieved,
        gold,
        3,
    ) == 0.0

    assert recall_at_k(
        retrieved,
        gold,
        3,
    ) == 0.0

    assert reciprocal_rank(
        retrieved,
        gold,
    ) == 0.0
