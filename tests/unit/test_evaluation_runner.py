"""Durable evaluator state and gold-free shared-service boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evaluation.common import read_json
from evaluation.datasets.sciq import PrivateSciQRecord, load_split, mcq_projection
from evaluation.datasets.sciq_openqa import openqa_projection
from evaluation.runner import EvaluationRun, validate_public

ROOT = Path(__file__).resolve().parents[2]


def record(**changes):
    fields = {
        "question_id": "q",
        "question": " Where is oxygen produced? ",
        "correct_answer": "chloroplast",
        "distractors": ("mitochondrion", "ribosome", "nucleus"),
        "support": "EVALUATOR_PRIVATE_SUPPORT",
    }
    return PrivateSciQRecord(**(fields | changes))


def test_stem_projection_unchanged_when_private_labels_change():
    first, _ = openqa_projection(record(), run_id="r", item_id="i")
    second, _ = openqa_projection(
        record(correct_answer="changed", support="different", distractors=("a", "b", "c")),
        run_id="r",
        item_id="i",
    )
    assert first == second
    assert first["question_text"] == " Where is oxygen produced? "
    assert set(first) == {"mode", "run_id", "item_id", "question_id", "question_text"}


def test_mcq_shuffle_is_stable_symmetric_and_gold_stays_private():
    first, gold = mcq_projection(record(), run_id="r", item_id="i", seed=17)
    swapped = record(
        correct_answer="nucleus", distractors=("mitochondrion", "ribosome", "chloroplast")
    )
    second, second_gold = mcq_projection(swapped, run_id="r", item_id="i", seed=17)
    assert first == second
    assert gold["correct_label"] != second_gold["correct_label"]
    assert first["options"][gold["correct_label"]] == "chloroplast"
    assert "support" not in first and "correct_label" not in first


def test_duplicate_candidates_and_split_counts(tmp_path):
    payload = {
        "question": "q",
        "correct_answer": "A",
        "distractor1": " a  ",
        "distractor2": "b",
        "distractor3": "c",
    }
    with pytest.raises(ValueError):
        PrivateSciQRecord.from_mapping(payload, fallback_id="q")
    path = ROOT / "evaluation/private_fixture/sciq_authored.json"
    records, manifest = load_split(path, split="authored", revision="test-v1")
    assert len(records) == manifest["count"] == 4
    assert manifest["source_kind"] == "authored_fixture"


def test_duplicate_source_choices_do_not_block_stem_only_openqa_freeze(tmp_path):
    source = {
        "question": "Which organelle produces oxygen?",
        "correct_answer": "chloroplast",
        "distractor1": "nucleus",
        "distractor2": " nucleus ",
        "distractor3": "ribosome",
        "support": "PRIVATE_SUPPORT_SENTINEL",
    }
    path = tmp_path / "native-duplicate.json"
    path.write_text(json.dumps([source]), encoding="utf-8")
    original = path.read_bytes()
    backend = DurableBackend()
    run = EvaluationRun.freeze(config(tmp_path / "openqa", dataset_path=str(path)), backend)
    assert run.manifest["scheduled_count"] == 1
    assert run.manifest["dataset"]["candidate_policy"] == "unused_by_openqa"
    assert run.commands[0]["question_text"] == source["question"]
    assert set(run.commands[0]) == {"mode", "run_id", "item_id", "question_id", "question_text"}
    assert "PRIVATE_SUPPORT_SENTINEL" not in json.dumps(run.commands)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="four normalized-distinct"):
        EvaluationRun.freeze(
            config(tmp_path / "mcq", dataset_path=str(path), protocol_id="sciq_mcq"), backend
        )
    assert not (tmp_path / "mcq").exists()
    # Standalone loading keeps the strict MCQ default; no global weakening.
    with pytest.raises(ValueError, match="four normalized-distinct"):
        load_split(path, split="validation", revision="source-test")


def answer():
    return {
        "schema_version": "chat_response_v1",
        "response_type": "answer",
        "answer_text": "Plants store chemical energy in sugars.",
        "short_answer": "chemical energy",
        "citations": [],
        "refusal_reason": None,
        "follow_up_questions": [],
        "confidence": None,
    }


class DurableBackend:
    def __init__(self):
        self.env = {
            "model_mode": "mock",
            "corpus_release_id": "release-a",
            "source_visibility_hash": "visible",
        }
        self.receipts = {}
        self.commands = []
        self.pending = False
        self.fail_after_receipt = False
        self.fail_without_receipt = False
        self.run_directory = None

    def environment(self):
        return dict(self.env)

    def register_run(self, manifest):
        validate_public(manifest)
        return {"run_id": manifest["run_id"]}

    def submit(self, command, *, run_context, idempotency_key):
        if self.run_directory:
            state = read_json(self.run_directory / "state.json")
            assert len(state["items"]) == run_context["scheduled_count"]
            assert state["items"][0]["status"] == "submitting"
        if self.fail_without_receipt:
            raise ConnectionError("Uncertain submission")
        if idempotency_key not in self.receipts:
            self.commands.append(copy.deepcopy(command))
            self.receipts[idempotency_key] = {
                "request_id": "req-" + command["item_id"],
                "job_id": "job-" + command["item_id"],
            }
        if self.fail_after_receipt:
            raise ConnectionError("Receipt response lost")
        return self.receipts[idempotency_key]

    def lookup(self, idempotency_key):
        return self.receipts.get(idempotency_key)

    def poll(self, receipt):
        return (
            {"status": "pending"}
            if self.pending
            else {
                "status": "completed",
                "response": answer(),
                "model_mode": "mock",
                "usage": {"cost": None},
            }
        )


def config(tmp_path, **changes):
    return {
        "protocol_id": "sciq_openqa",
        "condition": "E1",
        "dataset_path": str(ROOT / "evaluation/private_fixture/sciq_authored.json"),
        "dataset_revision": "test-v1",
        "split": "authored",
        "code_revision": "test",
        "backend_factory": "unused:factory",
        "private_root": str(tmp_path),
        "limit": 1,
        **changes,
    }


def test_freeze_preregisters_and_resume_does_not_repeat_calls(tmp_path):
    backend = DurableBackend()
    backend.pending = True
    run = EvaluationRun.freeze(config(tmp_path), backend)
    backend.run_directory = run.directory
    first = run.advance(backend)
    assert first["items"][0]["status"] == "submitted"
    assert len(backend.commands) == 1
    backend.run_directory = None
    backend.pending = False
    resumed = EvaluationRun(run.directory)
    assert resumed.advance(backend)["status"] == "completed"
    resumed.advance(backend)
    assert len(backend.commands) == 1
    assert not any(
        key in backend.commands[0]
        for key in ("reference", "support", "options", "history", "profile")
    )


def test_interrupted_submission_reconciles_receipt_without_new_call(tmp_path):
    backend = DurableBackend()
    backend.fail_after_receipt = True
    run = EvaluationRun.freeze(config(tmp_path), backend)
    assert run.advance(backend)["status"] == "needs_reconciliation"
    backend.fail_after_receipt = False
    assert EvaluationRun(run.directory).advance(backend)["status"] == "completed"
    assert len(backend.commands) == 1


def test_ambiguous_unrecorded_submission_never_automatically_repeats(tmp_path):
    backend = DurableBackend()
    backend.fail_without_receipt = True
    run = EvaluationRun.freeze(config(tmp_path), backend)
    assert run.advance(backend)["items"][0]["status"] == "uncertain"
    backend.fail_without_receipt = False
    assert EvaluationRun(run.directory).advance(backend)["items"][0]["status"] == "uncertain"
    assert len(backend.commands) == 0


def test_environment_change_stops_new_calls_preserves_scheduled_set(tmp_path):
    backend = DurableBackend()
    run = EvaluationRun.freeze(config(tmp_path, limit=2), backend)
    run.advance(backend, max_new_submissions=1)
    backend.env["source_visibility_hash"] = "revoked"
    state = run.advance(backend)
    assert state["status"] == "environment_changed"
    assert state["items"][0]["status"] == "completed"
    assert state["items"][1]["status"] == "pending"
    assert len(backend.commands) == 1


def test_tampered_manifest_or_private_reference_rejected(tmp_path):
    run = EvaluationRun.freeze(config(tmp_path), DurableBackend())
    references = read_json(run.directory / "references.json")
    references[0]["reference"] = "post-hoc changed"
    (run.directory / "references.json").write_text(json.dumps(references), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen evaluator"):
        EvaluationRun(run.directory)


def test_cancelled_export_retains_denominator_and_has_no_private_reference(tmp_path):
    run = EvaluationRun.freeze(config(tmp_path / "private", limit=2), DurableBackend())
    run.cancel()
    destination = run.export(tmp_path / "exports")
    summary = read_json(destination / "aggregate.json")
    assert summary["scheduled_count"] == 2
    assert summary["status_counts"] == {"cancelled": 2}
    assert summary["metrics"]["em"] == 0
    rows = (destination / "items.jsonl").read_text()
    assert '"reference":' not in rows and '"support":' not in rows
    assert len((destination / "items.csv").read_text().splitlines()) == 3


def test_live_defaults_and_e1_redefinition_are_rejected(tmp_path):
    backend = DurableBackend()
    backend.env["model_mode"] = "live"
    with pytest.raises(ValueError, match="mock-only"):
        EvaluationRun.freeze(config(tmp_path), backend)
    backend.env["model_mode"] = "mock"
    with pytest.raises(ValueError, match="R0"):
        EvaluationRun.freeze(config(tmp_path, configuration={"retrieval_variant": "R2"}), backend)
    with pytest.raises(ValueError, match="Private field"):
        EvaluationRun.freeze(config(tmp_path, configuration={"support": "oracle"}), backend)


def test_repeat_export_preserves_existing_independent_review(tmp_path):
    from evaluation.common import atomic_json

    backend = DurableBackend()
    run = EvaluationRun.freeze(config(tmp_path / "private"), backend)
    run.advance(backend)
    destination = run.export(tmp_path / "exports")
    review = read_json(destination / "full_response_review.json")
    review["reviews"][0].update(
        reviewer_id="actual-reviewer-test-fixture",
        correctness=2,
        groundedness=2,
        notes="Preserve independently entered review",
    )
    atomic_json(destination / "full_response_review.json", review)
    run.export(tmp_path / "exports")
    assert read_json(destination / "full_response_review.json") == review


def test_malformed_shared_response_is_retained_as_invalid(tmp_path):
    backend = DurableBackend()
    backend.poll = lambda receipt: {
        "status": "completed",
        "response": {"answer_text": "not a valid contract"},
    }
    run = EvaluationRun.freeze(config(tmp_path), backend)
    state = run.advance(backend)
    assert state["items"][0]["status"] == "invalid"
    assert state["items"][0]["scores"]["em"] == 0
