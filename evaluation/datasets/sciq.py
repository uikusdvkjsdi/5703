"""Read private SciQ-shaped records and freeze actual split checksums."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.common import fingerprint, read_json


def normalize_choice(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class PrivateSciQRecord:
    question_id: str
    question: str
    correct_answer: str
    distractors: tuple[str, str, str]
    support: str

    @classmethod
    def from_mapping(
        cls, value: dict[str, Any], *, fallback_id: str, require_distinct_choices: bool = True
    ) -> "PrivateSciQRecord":
        required = ("question", "correct_answer", "distractor1", "distractor2", "distractor3")
        if any(not isinstance(value.get(key), str) or not value[key].strip() for key in required):
            raise ValueError(
                "SciQ question, reference and three distractors must be nonempty strings"
            )
        options = [value[key] for key in required[1:]]
        if require_distinct_choices and len({normalize_choice(option) for option in options}) != 4:
            raise ValueError("SciQ candidates must be four normalized-distinct texts")
        question_id = value.get("question_id", value.get("id", fallback_id))
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Question identity must be a nonempty string")
        support = value.get("support", "")
        if not isinstance(support, str):
            raise ValueError("SciQ support must be text when present")
        return cls(
            question_id,
            value["question"],
            value["correct_answer"],
            tuple(value[f"distractor{n}"] for n in range(1, 4)),
            support,
        )


def load_split(
    path: str | Path, *, split: str, revision: str, require_distinct_choices: bool = True
) -> tuple[list[PrivateSciQRecord], dict]:
    if split not in {"train", "validation", "test", "authored"} or not revision.strip():
        raise ValueError("A named split and nonempty dataset revision are required")
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        values = read_json(path)
        if isinstance(values, dict):
            values = values.get(split, values.get("records"))
    if not isinstance(values, list) or not values:
        raise ValueError("Dataset split must contain at least one record")
    records = [
        PrivateSciQRecord.from_mapping(
            value,
            fallback_id=f"sciq:{split}:{index:06}",
            require_distinct_choices=require_distinct_choices,
        )
        for index, value in enumerate(values)
    ]
    if len({row.question_id for row in records}) != len(records):
        raise ValueError("Dataset question IDs must be unique")
    manifest = {
        "name": "authored_sciq_fixture" if split == "authored" else "SciQ",
        "revision": revision,
        "split": split,
        "count": len(records),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_kind": "authored_fixture" if split == "authored" else "provided_dataset_file",
        "candidate_policy": "require_four_distinct"
        if require_distinct_choices
        else "unused_by_openqa",
    }
    return records, manifest


def mcq_projection(
    record: PrivateSciQRecord, *, run_id: str, item_id: str, seed: int
) -> tuple[dict, dict]:
    # Sort before shuffling so changing which existing candidate is gold does
    # not change its presentation. Correct labels are computed afterward.
    candidates = sorted(
        (record.correct_answer, *record.distractors),
        key=lambda text: (normalize_choice(text), text),
    )
    seeded = int(fingerprint({"seed": seed, "question_id": record.question_id}), 16)
    random.Random(seeded).shuffle(candidates)
    options = dict(zip("ABCD", candidates, strict=True))
    command = {
        "mode": "benchmark_mcq",
        "run_id": run_id,
        "item_id": item_id,
        "question_id": record.question_id,
        "question_text": record.question,
        "options": options,
    }
    private = {
        "reference": record.correct_answer,
        "correct_label": next(
            label for label, value in options.items() if value == record.correct_answer
        ),
        "support": record.support,
    }
    return command, private
