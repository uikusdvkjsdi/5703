"""Independent blinded profile ratings with explicit missingness and paired units."""

from __future__ import annotations

import csv
import io
import random
from collections import defaultdict
from statistics import mean

from evaluation.common import fingerprint
from evaluation.metrics.paired import paired_analysis
from personalisation.rubric import ANCHORS, DIMENSIONS, GATES

RUBRIC_VERSION = "profile_rating_v1"
RUBRIC = {
    "version": RUBRIC_VERSION,
    "anchors": {str(key): value for key, value in ANCHORS.items()},
    "dimensions": {
        "level_fit": "Terminology, depth and assumptions suit the stated level.",
        "clarity": "The response is readable and coherent.",
        "prerequisite_support": "Necessary terms and prerequisites are handled appropriately.",
        "usefulness": "The explanation addresses the learner's question with a useful application where appropriate.",
        "guidance": "Any next step or learning check is relevant and does not distract.",
        "consistency": "Meaning, formulas, negation and facts remain consistent with the frozen base and sources.",
    },
    "gates": {
        "correctness": "Factual correctness rated 0–3; gate passes at 2 or 3, null is unavailable.",
        "groundedness": "Support by frozen evidence rated 0–3; gate passes at 2 or 3, null is unavailable.",
    },
    "interpretation": "Independent output review, not a measure of student learning gain. Missing ratings stay null.",
}


def create_blind_package(
    outputs: list[dict], *, seed: int = 0, require_complete_design: bool = True
) -> dict:
    grouped = defaultdict(list)
    seen = set()
    for row in outputs:
        identity = (row["question_id"], row["target_level"], row["condition"])
        if (
            identity in seen
            or row["target_level"] not in {"beginner", "intermediate", "advanced"}
            or row["condition"] not in {"C0", "C1", "C2"}
        ):
            raise ValueError("Profile study needs unique question × level × condition items")
        seen.add(identity)
        grouped[row["question_id"]].append(row)
    for rows in grouped.values():
        required = {
            (level, condition)
            for level in ("beginner", "intermediate", "advanced")
            for condition in ("C0", "C1", "C2")
        }
        if (
            require_complete_design
            and {(row["target_level"], row["condition"]) for row in rows} != required
        ):
            raise ValueError(
                "The full matched profile design requires all nine conditions per question, including failed outputs"
            )
        frozen = {
            (row["base_answer_hash"], row["evidence_hash"], row["model_configuration_hash"])
            for row in rows
        }
        if len(frozen) != 1:
            raise ValueError(
                "Matched explanations must share the exact base answer, evidence and model"
            )
    rng = random.Random(seed)
    shuffled = list(outputs)
    rng.shuffle(shuffled)
    key, review_items = [], []
    for index, row in enumerate(shuffled):
        blind_id = fingerprint({"seed": seed, "order": index, "item_id": row["item_id"]})[:20]
        key.append(
            {
                "blind_id": blind_id,
                "item_id": row["item_id"],
                "question_id": row["question_id"],
                "target_level": row["target_level"],
                "condition": row["condition"],
                "status": row["status"],
            }
        )
        review_items.append(
            {
                "blind_id": blind_id,
                "target_level": row["target_level"],
                "question": row["question"],
                "explanation": row.get("explanation") if row["status"] == "completed" else None,
                "evidence": row.get("evidence", []),
                "status": row["status"],
            }
        )
    return {
        "rubric": RUBRIC,
        "seed": seed,
        "private_key": key,
        "review_items": review_items,
        "scheduled_count": len(outputs),
        "output_hash": fingerprint(outputs),
    }


def blank_ratings_csv(package: dict) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=["reviewer_id", "blind_id", *GATES, *DIMENSIONS, "notes"]
    )
    writer.writeheader()
    for item in package["review_items"]:
        writer.writerow({"blind_id": item["blind_id"]})
    return stream.getvalue()


def parse_ratings_csv(text: str, package: dict) -> list[dict]:
    known = {row["blind_id"]: row for row in package["private_key"]}
    rows, seen = [], set()
    reader = csv.DictReader(io.StringIO(text))
    required = {"reviewer_id", "blind_id", *GATES, *DIMENSIONS}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise ValueError("Rating sheet lacks required columns")
    for values in reader:
        blind_id, reviewer = values["blind_id"].strip(), values["reviewer_id"].strip()
        if blind_id not in known:
            raise ValueError("Rating sheet contains an unknown blinded item")
        if not reviewer:
            if any(values[name].strip() for name in (*GATES, *DIMENSIONS)):
                raise ValueError("Actual ratings need an independent reviewer identity")
            continue
        identity = (reviewer, blind_id)
        if identity in seen:
            raise ValueError(
                "Duplicate reviewer/item rating; resolve explicitly instead of overwriting"
            )
        seen.add(identity)
        row = {
            "reviewer_id": reviewer,
            "blind_id": blind_id,
            "notes": values.get("notes", ""),
            "rubric_version": package["rubric"]["version"],
        }
        for name in (*GATES, *DIMENSIONS):
            value = values[name].strip()
            if value not in {"", "0", "1", "2", "3"}:
                raise ValueError("Anchored dimension scores must be 0–3 or empty")
            row[name] = int(value) if value else None
        if known[blind_id]["status"] != "completed" and any(
            row[name] is not None for name in (*GATES, *DIMENSIONS)
        ):
            raise ValueError("A failed/missing explanation cannot receive an invented rating")
        rows.append(row)
    return rows


def analyze_ratings(
    package: dict, ratings: list[dict], *, expected_reviewers: list[str], seed: int = 0
) -> dict:
    if len(set(expected_reviewers)) != len(expected_reviewers):
        raise ValueError("Reviewer identities must be distinct")
    key = {row["blind_id"]: row for row in package["private_key"]}
    if any(
        row["reviewer_id"] not in expected_reviewers or row["blind_id"] not in key
        for row in ratings
    ):
        raise ValueError("Rating is outside the preregistered reviewer/item population")
    seen = {(row["reviewer_id"], row["blind_id"]) for row in ratings}
    if len(seen) != len(ratings):
        raise ValueError("Duplicate independent ratings")
    grouped = defaultdict(list)
    for row in ratings:
        item = key[row["blind_id"]]
        grouped[(item["question_id"], item["target_level"], item["condition"])].append(row)
    summaries = []
    for identity, rows in sorted(grouped.items()):
        metrics = {
            name: mean([row[name] for row in rows if row[name] is not None])
            if any(row[name] is not None for row in rows)
            else None
            for name in DIMENSIONS
        }
        gate_rows = [
            row for row in rows if all(row[gate] is not None and row[gate] >= 2 for gate in GATES)
        ]
        gated = {
            name: mean([row[name] for row in gate_rows if row[name] is not None])
            if any(row[name] is not None for row in gate_rows)
            else None
            for name in DIMENSIONS
        }
        disagreement = {
            name: max(values) - min(values) if len(values) >= 2 else None
            for name in DIMENSIONS
            if (values := [row[name] for row in rows if row[name] is not None]) is not None
        }
        summaries.append(
            {
                "question_id": identity[0],
                "target_level": identity[1],
                "condition": identity[2],
                "scores": metrics,
                "correct_and_grounded_scores": gated,
                "rating_count": len(rows),
                "gate_pass_count": len(gate_rows),
                "disagreement_range": disagreement,
            }
        )
    lookup = {(row["question_id"], row["target_level"], row["condition"]): row for row in summaries}
    comparisons = []
    for level in ("beginner", "intermediate", "advanced"):
        for before_condition, after_condition in (("C0", "C1"), ("C0", "C2"), ("C1", "C2")):
            for dimension in DIMENSIONS:
                pairs = []
                for question in sorted({row["question_id"] for row in package["private_key"]}):
                    first = (
                        lookup.get((question, level, before_condition), {})
                        .get("scores", {})
                        .get(dimension)
                    )
                    second = (
                        lookup.get((question, level, after_condition), {})
                        .get("scores", {})
                        .get(dimension)
                    )
                    if first is not None and second is not None:
                        pairs.append((first, second))
                comparisons.append(
                    {
                        "target_level": level,
                        "before": before_condition,
                        "after": after_condition,
                        "dimension": dimension,
                        "analysis": paired_analysis(
                            [pair[0] for pair in pairs], [pair[1] for pair in pairs], seed=seed
                        )
                        if pairs
                        else None,
                        "paired_question_count": len(pairs),
                    }
                )
    expected = len(key) * len(expected_reviewers)
    complete = sum(all(row[name] is not None for name in (*GATES, *DIMENSIONS)) for row in ratings)
    return {
        "rubric_version": package["rubric"]["version"],
        "scheduled_outputs": len(key),
        "completed_outputs": sum(row["status"] == "completed" for row in key.values()),
        "expected_independent_ratings": expected,
        "received_rating_rows": len(ratings),
        "complete_rating_rows": complete,
        "missing_or_incomplete_rating_rows": expected - complete,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "unit": "paired_question_after_independent_reviewer_aggregation",
        "claim_limit": "Output-quality ratings do not measure student learning gains.",
    }
