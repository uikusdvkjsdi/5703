"""SciQ-derived stem-only OpenQA adapter; references never enter the command."""

from __future__ import annotations

from evaluation.datasets.sciq import PrivateSciQRecord


def openqa_projection(record: PrivateSciQRecord, *, run_id: str, item_id: str) -> tuple[dict, dict]:
    command = {
        "mode": "benchmark_openqa",
        "run_id": run_id,
        "item_id": item_id,
        "question_id": record.question_id,
        "question_text": record.question,
    }
    private = {"reference": record.correct_answer, "support": record.support}
    return command, private
