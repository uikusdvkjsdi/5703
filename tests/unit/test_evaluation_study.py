"""Matched profile orchestration calls the shared teaching implementation."""

import copy
import hashlib

import pytest

from evaluation.annotations.blind import analyze_ratings
from evaluation.common import fingerprint, read_json
from evaluation.study import ProfileStudy
from personalisation.study import TeachingStudyService


def question():
    text = "Photosynthesis captures light energy and stores it as chemical energy in sugars."
    evidence = {
        "evidence_id": "ev_001",
        "chunk_id": "chunk-1",
        "asset_id": "asset-1",
        "processing_id": "processing-1",
        "source_title": "Authored biology notes",
        "section": "Photosynthesis",
        "pages": [1],
        "locator": "page 1",
        "text": text,
        "text_hash": hashlib.sha256(text.encode()).hexdigest(),
        "context_order": 0,
        "inherited_from": None,
    }
    base = {
        "schema_version": "chat_response_v1",
        "response_type": "answer",
        "answer_text": text + " [ev_001]",
        "short_answer": "chemical energy",
        "citations": ["ev_001"],
        "refusal_reason": None,
        "follow_up_questions": [],
        "confidence": None,
    }
    return {
        "question_id": "photosynthesis",
        "question": "What is photosynthesis?",
        "base_answer": base,
        "evidence": [evidence],
    }


class CountingTeachingService(TeachingStudyService):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs["item_id"])
        return super().generate(**kwargs)


def test_shared_mock_study_freezes_nine_conditions_and_resumes_without_repeating(tmp_path):
    original = question()
    original_hash = fingerprint(original)
    service = CountingTeachingService()
    study = ProfileStudy.freeze(root=tmp_path / "runs", questions=[original], seed=11)
    assert study.manifest["scheduled_count"] == 9
    assert study.advance(service, max_items=4)["status"] == "running"
    resumed = ProfileStudy(study.directory)
    assert resumed.advance(service)["status"] == "completed"
    resumed.advance(service)
    assert len(service.calls) == len(set(service.calls)) == 9
    assert all(item["status"] == "completed" for item in resumed.state["items"])
    assert all(item["budget"]["consumed_calls"] == 1 for item in resumed.state["items"])
    assert fingerprint(original) == original_hash
    package = resumed.export_blind(tmp_path / "blind")
    assert len(package["review_items"]) == 9
    assert all("condition" not in row for row in package["review_items"])
    analysis = analyze_ratings(package, [], expected_reviewers=["independent-reviewer"])
    assert analysis["missing_or_incomplete_rating_rows"] == 9
    assert analysis["complete_rating_rows"] == 0


def test_interrupted_study_call_is_not_silently_repeated(tmp_path):
    study = ProfileStudy.freeze(root=tmp_path, questions=[question()])
    state = read_json(study.directory / "state.json")
    state["items"][0]["status"] = "running"
    from evaluation.common import atomic_json

    atomic_json(study.directory / "state.json", state)
    service = CountingTeachingService()
    result = study.advance(service)
    assert result["items"][0]["status"] == "uncertain"
    assert result["status"] == "needs_reconciliation"
    assert len(service.calls) == 8


def test_study_rejects_live_mode_and_mutated_frozen_inputs(tmp_path):
    with pytest.raises(ValueError, match="mock-only"):
        ProfileStudy.freeze(
            root=tmp_path, questions=[question()], model_config={"provider": "openai"}
        )
    study = ProfileStudy.freeze(root=tmp_path, questions=[question()])
    inputs = read_json(study.directory / "inputs.json")
    inputs["questions"][0]["base_answer"]["answer_text"] = "changed"
    from evaluation.common import atomic_json

    atomic_json(study.directory / "inputs.json", inputs)
    with pytest.raises(ValueError, match="base/evidence"):
        ProfileStudy(study.directory)


def test_study_prompt_or_rubric_environment_change_stops_remaining_calls(tmp_path, monkeypatch):
    import evaluation.study as module

    study = ProfileStudy.freeze(root=tmp_path, questions=[question()])
    service = CountingTeachingService()
    study.advance(service, max_items=1)
    monkeypatch.setattr(
        module, "study_environment", lambda: {"prompt_configuration_hash": "changed"}
    )
    assert study.advance(service)["status"] == "environment_changed"
    assert len(service.calls) == 1
    assert study.state["items"][0]["status"] == "completed"
