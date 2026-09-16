"""Private reviews, compatibility boundaries and actual conversation workload."""

import copy
import csv
import hashlib
import io
from pathlib import Path

import pytest

from evaluation.analysis.ablations import validate_comparison
from evaluation.annotations.blind import (
    DIMENSIONS,
    GATES,
    analyze_ratings,
    blank_ratings_csv,
    create_blind_package,
    parse_ratings_csv,
)
from evaluation.annotations.qrels import create_template, validate_judgements
from evaluation.common import read_json
from evaluation.conversations.runner import aggregate_scenarios, rendered_turn, run_scenario

ROOT = Path(__file__).resolve().parents[2]


def test_qrels_require_actual_review_identity_and_frozen_sources():
    text = "Plants use light energy."
    candidate = {
        "chunk_id": "c1",
        "processing_id": "p1",
        "text_hash": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "source_spans": [{"unit_id": "u1", "start": 0, "end": len(text)}],
    }
    template = create_template(
        corpus_release_id="r1", processing_ids=["p1"], question_candidates={"q1": [candidate]}
    )
    result = validate_judgements(template, corpus_release_id="r1", processing_ids=["p1"])
    assert result["judged_count"] == 0 and result["qrels"]["q1"]["c1"] is None
    template["rows"][0]["grade"] = 3
    with pytest.raises(ValueError, match="reviewer"):
        validate_judgements(template, corpus_release_id="r1", processing_ids=["p1"])
    template["rows"][0].update(reviewer_id="reviewer-a", reviewed_at="2026-09-08T00:00:00Z")
    assert (
        validate_judgements(template, corpus_release_id="r1", processing_ids=["p1"])["judged_count"]
        == 1
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_judgements(template, corpus_release_id="r2", processing_ids=["p1"])
    template["rows"][0]["text"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        validate_judgements(template, corpus_release_id="r1", processing_ids=["p1"])


def outputs():
    return [
        {
            "item_id": f"q1-{level}-{condition}",
            "question_id": "q1",
            "question": "What is photosynthesis?",
            "target_level": level,
            "condition": condition,
            "base_answer_hash": "base1",
            "evidence_hash": "evidence1",
            "model_configuration_hash": "model1",
            "status": "completed",
            "explanation": "Plants store light-derived energy in sugars.",
            "evidence": [{"text": "Plants store light energy in sugar."}],
        }
        for level in ("beginner", "intermediate", "advanced")
        for condition in ("C0", "C1", "C2")
    ]


def filled_csv(package, *, reviewer="r1", empty=False):
    reader = csv.DictReader(io.StringIO(blank_ratings_csv(package)))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        row["reviewer_id"] = reviewer
        for name in GATES:
            row[name] = "" if empty else "2"
        for name in DIMENSIONS:
            row[name] = "" if empty else "2"
        writer.writerow(row)
    return stream.getvalue()


def test_blinding_hides_conditions_and_retains_all_nine_items():
    package = create_blind_package(outputs(), seed=7)
    assert package["scheduled_count"] == 9
    assert all(
        "condition" not in row and "item_id" not in row and "question_id" not in row
        for row in package["review_items"]
    )
    assert package == create_blind_package(outputs(), seed=7)
    assert len({row["blind_id"] for row in package["private_key"]}) == 9
    with pytest.raises(ValueError, match="nine"):
        create_blind_package(outputs()[:-1])
    wrong = outputs()
    wrong[0]["evidence_hash"] = "different"
    with pytest.raises(ValueError, match="exact base"):
        create_blind_package(wrong)


def test_missing_ratings_remain_null_and_pairs_use_questions():
    package = create_blind_package(outputs())
    assert parse_ratings_csv(blank_ratings_csv(package), package) == []
    missing = analyze_ratings(package, [], expected_reviewers=["r1", "r2"])
    assert missing["expected_independent_ratings"] == 18
    assert missing["missing_or_incomplete_rating_rows"] == 18
    assert all(row["analysis"] is None for row in missing["paired_comparisons"])
    ratings = parse_ratings_csv(filled_csv(package), package)
    result = analyze_ratings(package, ratings, expected_reviewers=["r1", "r2"])
    assert result["complete_rating_rows"] == 9
    assert result["paired_comparisons"][0]["paired_question_count"] == 1
    assert result["paired_comparisons"][0]["analysis"]["n_pairs"] == 1
    assert result["missing_or_incomplete_rating_rows"] == 9


def test_duplicate_and_failed_output_ratings_are_rejected():
    package = create_blind_package(outputs())
    text = filled_csv(package)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_ratings_csv(text + text.splitlines()[1] + "\n", package)
    failed = outputs()
    failed[0]["status"] = "error"
    package = create_blind_package(failed)
    with pytest.raises(ValueError, match="failed/missing"):
        parse_ratings_csv(filled_csv(package), package)


def base_comparison():
    return {
        "protocol_id": "sciq_openqa",
        "mode": "benchmark_openqa",
        "question_ids": ["q1"],
        "dataset_revision": "d1",
        "dataset_split": "test",
        "model_configuration_hash": "m1",
        "prompt_configuration_hash": "prompt1",
        "scorer_version": "v1",
        "seed": 0,
        "source_asset_hashes": ["source1"],
        "condition": "E1",
        "retrieval_variant": "R0",
        "top_k": 5,
    }


def test_controlled_comparisons_reject_test_tuning_and_multiple_changes():
    base = base_comparison()
    variant = base | {"condition": "R1", "retrieval_variant": "R1"}
    result = validate_comparison(base, variant, factor="retrieval", selected_on_split="validation")
    assert result["changed_fields"] == ["retrieval_variant"]
    with pytest.raises(ValueError, match="validation"):
        validate_comparison(base, variant, factor="retrieval", selected_on_split="test")
    with pytest.raises(ValueError, match="declared factor"):
        validate_comparison(
            base, variant | {"top_k": 10}, factor="retrieval", selected_on_split="validation"
        )
    with pytest.raises(ValueError, match="E1"):
        validate_comparison(
            base,
            base | {"retrieval_variant": "R2"},
            factor="retrieval",
            selected_on_split="validation",
        )


def test_twelve_authored_families_fit_composer_without_private_claims():
    fixture = read_json(ROOT / "evaluation/conversations/scenarios.json")
    assert len(fixture["scenarios"]) == fixture["family_count"] == 12
    assert len({row["scenario_id"] for row in fixture["scenarios"]}) == 12
    for scenario in fixture["scenarios"]:
        assert 3 <= len(scenario["turns"]) <= 5
        for turn in scenario["turns"]:
            assert 0 < len(rendered_turn(scenario, turn)) <= 4000
            assert "reference_claims" not in turn and "expected_answer" not in turn


class ChatSpy:
    def __init__(self):
        self.session_count = 0
        self.calls = []
        self.profiles = []
        self.relogins = 0

    def capabilities(self):
        return {"model_mode": "mock"}

    def new_session(self):
        self.session_count += 1
        return f"session-{self.session_count}"

    def set_profile(self, value):
        self.profiles.append(value)

    def relogin(self):
        self.relogins += 1

    def send_and_wait(self, session_id, content, *, use_profile, idempotency_key):
        self.calls.append(
            {
                "session": session_id,
                "content": content,
                "use_profile": use_profile,
                "idempotency_key": idempotency_key,
            }
        )
        return {"status": "completed"}


def test_conversation_profile_off_keeps_session_and_new_chat_is_distinct():
    scenario = read_json(ROOT / "evaluation/conversations/scenarios.json")["scenarios"][-1]
    spy = ChatSpy()
    result = run_scenario(scenario, spy, run_id="test-run")
    assert spy.calls[1]["session"] == spy.calls[2]["session"]
    assert spy.calls[2]["use_profile"] is False
    assert spy.calls[3]["session"] != spy.calls[2]["session"]
    assert spy.relogins == 1 and len(spy.profiles) == 1
    assert len({row["idempotency_key"] for row in spy.calls}) == 4
    summary = aggregate_scenarios([result])
    assert summary["scheduled_turn_count"] == 4
    assert summary["semantic_correctness"] is None
