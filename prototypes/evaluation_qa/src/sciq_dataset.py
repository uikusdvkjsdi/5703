"""SciQ test-set preparation for the Evaluation / QA prototype."""

import hashlib
import random

from schema import EvaluationExample


LABELS = ["A", "B", "C", "D"]


def _question_seed(
    question_id: str,
    base_seed: int = 5703,
) -> int:
    """
    Create a stable integer seed for one evaluation question.

    The same question_id and base_seed always produce
    the same answer-choice order.
    """

    raw_value = f"{base_seed}:{question_id}"

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def build_sciq_example(
    example: dict,
    question_id: str,
    base_seed: int = 5703,
) -> EvaluationExample:
    """
    Convert one raw SciQ record into the common
    EvaluationExample schema.

    The answer choices are shuffled deterministically
    so the correct answer is not always option A.
    """

    correct_answer = example["correct_answer"]

    answers = [
        example["correct_answer"],
        example["distractor1"],
        example["distractor2"],
        example["distractor3"],
    ]

    rng = random.Random(
        _question_seed(
            question_id=question_id,
            base_seed=base_seed,
        )
    )

    rng.shuffle(answers)

    choices = {
        label: answer
        for label, answer
        in zip(LABELS, answers)
    }

    correct_index = answers.index(
        correct_answer
    )

    gold_answer = LABELS[
        correct_index
    ]

    return EvaluationExample(
        question_id=question_id,
        question=example["question"],
        choices=choices,
        gold_answer=gold_answer,
        gold_answer_text=correct_answer,
        gold_evidence=[],
        annotation_status="pending",
    )
