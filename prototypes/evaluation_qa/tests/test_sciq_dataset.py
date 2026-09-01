"""Tests for deterministic SciQ evaluation-set preparation."""

from src.sciq_dataset import build_sciq_example


SAMPLE_EXAMPLE = {
    "question": "Which organelle produces most cellular ATP?",
    "correct_answer": "Mitochondrion",
    "distractor1": "Nucleus",
    "distractor2": "Ribosome",
    "distractor3": "Golgi apparatus",
}


def test_same_question_produces_same_choice_order():
    """
    The same question ID and seed must always produce
    the same A-D answer order.
    """

    first = build_sciq_example(
        SAMPLE_EXAMPLE,
        question_id="sciq_00001",
    )

    second = build_sciq_example(
        SAMPLE_EXAMPLE,
        question_id="sciq_00001",
    )

    assert first.choices == second.choices
    assert first.gold_answer == second.gold_answer


def test_gold_answer_matches_gold_answer_text():
    """
    The labelled gold answer must always point to
    the correct SciQ answer text.
    """

    item = build_sciq_example(
        SAMPLE_EXAMPLE,
        question_id="sciq_00001",
    )

    assert (
        item.choices[item.gold_answer]
        == item.gold_answer_text
    )


def test_correct_answer_is_not_always_option_a():
    """
    Different question IDs should produce different
    deterministic answer positions rather than fixing
    the correct answer to A.
    """

    labels = []

    for index in range(1, 6):
        item = build_sciq_example(
            SAMPLE_EXAMPLE,
            question_id=f"sciq_{index:05d}",
        )

        labels.append(item.gold_answer)

    assert len(set(labels)) > 1
