"""Frozen evaluator and independent teaching jobs on migrated PostgreSQL."""

from pathlib import Path
from uuid import uuid4
from datetime import timedelta

import pytest
from sqlalchemy import func, select

from app.core.exceptions import AppError
from app.db.base import new_uuid, utcnow
from app.modules.answering.models import Answer, AnswerRequest, Attempt, Job, Message, Snapshot
from app.modules.experiment.bridge import DatabaseAnswerBackend, execute_teaching, run_results
from app.modules.experiment.models import ExperimentRun, TeachingStudyRecord
from app.modules.identity.models import User
from app.modules.knowledge import service as knowledge
from app.modules.knowledge.models import Document, ProcessingRun
from evaluation.annotations.blind import create_blind_package
from evaluation.common import read_json
from evaluation.runner import EvaluationRun
from personalisation.study import TeachingStudyService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evaluation_backend(runtime):
    with runtime.db() as db:
        actor = db.scalar(select(User).where(User.email == "admin@example.com"))
        actor_id = actor.id
        passages = read_json(ROOT / "evaluation/conversations/source_requirements.json")["passages"]
        raw = (
            "\n\n".join("# " + passage["section"] + "\n" + passage["text"] for passage in passages)
            + "\n\n# Source identity\n"
            + uuid4().hex
        ).encode()
        document, version, _ = knowledge.ingest(
            db,
            runtime.settings,
            actor_id,
            "evaluation-source.txt",
            raw,
            "Authored PostgreSQL evaluation fixture",
        )
        processing, job = knowledge.queue_process(db, document.id, actor_id)
        processing_id, document_id = processing.id, document.id
        db.commit()
    assert runtime.work()
    with runtime.db() as db:
        assert db.get(ProcessingRun, processing_id).state == "ready"
        release, job = knowledge.queue_release(db, actor_id, [processing_id])
        release_id = release.id
        db.commit()
    assert runtime.work()
    with runtime.db() as db:
        knowledge.activate(db, release_id)
        config = knowledge.config_create(
            db,
            "evaluation",
            "Actual evaluation parameters",
            {"top_k": 1, "model": {"seed": 5703, "temperature": 0.0, "max_tokens": 512}},
        )
        config_id = config.id
        db.commit()
    backend = DatabaseAnswerBackend(
        runtime.settings,
        runtime.engine,
        actor_id=actor_id,
        configuration_id=config_id,
        inline_worker=True,
    )
    return backend, document_id


def config(tmp_path, protocol="sciq_openqa", limit=2):
    return {
        "protocol_id": protocol,
        "condition": "E1",
        "dataset_path": str(ROOT / "evaluation/private_fixture/sciq_authored.json"),
        "dataset_revision": "authored-v1",
        "split": "authored",
        "code_revision": "software-test",
        "backend_factory": "app.modules.experiment.bridge:create_backend",
        "private_root": str(tmp_path / "private-runs"),
        "limit": limit,
        "configuration": {"top_k": 1},
    }


@pytest.mark.parametrize("protocol", ["sciq_openqa", "sciq_mcq"])
def test_postgres_frozen_runner_resume_preserves_receipts_and_actual_parameters(
    runtime, evaluation_backend, tmp_path, protocol
):
    backend, _ = evaluation_backend
    run = EvaluationRun.freeze(config(tmp_path, protocol), backend)
    with runtime.db() as db:
        before = {
            table.__tablename__: db.scalar(select(func.count()).select_from(table))
            for table in (Message, Snapshot)
        }
    assert run.advance(backend, max_new_submissions=1)["status"] == "running"
    first_receipt = run.state["items"][0]["receipt"]
    resumed = EvaluationRun(run.directory)
    assert resumed.advance(backend)["status"] == "completed"
    assert resumed.state["items"][0]["receipt"] == first_receipt
    resumed.advance(backend)
    with runtime.db() as db:
        requests = list(
            db.scalars(select(AnswerRequest).where(AnswerRequest.run_id == run.manifest["run_id"]))
        )
        assert len(requests) == 2
        for req in requests:
            assert req.profile_snapshot_id is req.context_snapshot_id is req.session_id is None
            assert req.command["model_config"]["max_tokens"] == 512
            assert req.command["model_config"]["seed"] == 5703
            assert req.command["top_k"] == 1 and req.command["retriever"] == "R0"
            assert (
                set(req.command)
                & {"gold", "support", "reference", "correct_answer", "correct_label"}
                == set()
            )
            assert len(req.trace["retrieval_candidates"]) <= 1
            assert req.budget["consumed_calls"] == 1
        for table in (Message, Snapshot):
            assert db.scalar(select(func.count()).select_from(table)) == before[table.__tablename__]
        assert (
            db.scalar(
                select(func.count())
                .select_from(Answer)
                .where(Answer.request_id.in_([req.id for req in requests]))
            )
            == 2
        )
        summary = run_results(db, db.get(ExperimentRun, run.manifest["run_id"]))
        assert summary["scheduled_count"] == summary["scored_count"] == 2
        assert summary["human_review"] is None
    exported = resumed.export(tmp_path / "exports")
    assert exported.parent.name == protocol
    assert read_json(exported / "aggregate.json")["evidence_class"] == "mock_software_rehearsal"


def test_postgres_public_freeze_rejects_hash_and_stops_revoked_environment(
    runtime, evaluation_backend, tmp_path
):
    backend, document_id = evaluation_backend
    run = EvaluationRun.freeze(config(tmp_path, limit=1), backend)
    backend.register_run(run.manifest)
    changed = {**run.commands[0], "question_text": "Altered after freezing"}
    key = f"evaluation:{run.manifest['run_id']}:{changed['item_id']}"
    with pytest.raises(AppError, match="preregistered"):
        backend.submit(changed, run_context=run.manifest, idempotency_key=key)
    receipt = backend.submit(run.commands[0], run_context=run.manifest, idempotency_key=key)
    with runtime.db() as db:
        document = db.get(Document, document_id)
        document.revoked = True
        db.commit()
    result = backend.poll(receipt)
    assert result["status"] == "error"
    with runtime.db() as db:
        assert db.get(ExperimentRun, run.manifest["run_id"]).state == "environment_changed"
        request = db.get(AnswerRequest, receipt["request_id"])
        assert request.budget["consumed_calls"] == 0
        assert (
            db.scalar(
                select(func.count()).select_from(Answer).where(Answer.request_id == request.id)
            )
            == 0
        )


@pytest.mark.parametrize("protocol", ["sciq_openqa", "sciq_mcq"])
def test_postgres_e0_both_schemas_never_call_retriever(
    runtime, evaluation_backend, tmp_path, monkeypatch, protocol
):
    from app.modules.answering import service as answering

    backend, _ = evaluation_backend

    def forbidden_retrieval(*args, **kwargs):
        raise AssertionError("E0 must not call any retriever")

    monkeypatch.setattr(answering, "retrieve", forbidden_retrieval)
    run = EvaluationRun.freeze({**config(tmp_path, protocol, limit=1), "condition": "E0"}, backend)
    assert run.advance(backend)["status"] == "completed"
    assert run.state["items"][0]["status"] == "completed"
    with runtime.db() as db:
        request = db.scalar(
            select(AnswerRequest).where(AnswerRequest.run_id == run.manifest["run_id"])
        )
        assert request.release_id is None
        assert request.trace["retrieval_candidates"] == []
        assert request.budget["consumed_calls"] == 1
        assert (
            request.profile_snapshot_id is request.context_snapshot_id is request.session_id is None
        )
        assert run.state["items"][0]["outcome"]["evidence"] == []


def base_answer(runtime, backend, tmp_path):
    run = EvaluationRun.freeze(config(tmp_path, limit=1), backend)
    assert run.advance(backend)["status"] == "completed"
    return run.state["items"][0]["outcome"]["answer_id"]


def test_postgres_teaching_api_nine_matched_jobs_and_blind_export_without_chat_mutation(
    runtime, evaluation_backend, tmp_path
):
    backend, _ = evaluation_backend
    base_id = base_answer(runtime, backend, tmp_path)
    headers = runtime.headers("admin@example.com")
    with runtime.db() as db:
        original = db.get(Answer, base_id).response
        before = {
            table.__tablename__: db.scalar(select(func.count()).select_from(table))
            for table in (Message, Snapshot, Answer, AnswerRequest)
        }
    response = runtime.client.post(
        "/api/v1/teaching-studies",
        headers=headers,
        json={
            "base_answer_ids": [base_id],
            "seed": 42,
            "configuration_id": backend.configuration_id,
        },
    )
    assert response.status_code == 201, response.text
    run_id = response.json()["data"]["id"]
    for _ in range(2):
        response = runtime.client.post(f"/api/v1/experiments/{run_id}/start", headers=headers)
        assert response.status_code == 202, response.text
    for _ in range(9):
        assert runtime.work()
    result = runtime.client.get(f"/api/v1/experiments/{run_id}/results", headers=headers)
    assert result.status_code == 200, result.text
    result = result.json()["data"]
    assert result["scheduled_count"] == 9
    assert result["outcome_counts"] == {"completed": 9}, result
    assert result["human_review"] is None and result["metrics"] is None
    assert len(create_blind_package(result["items"], seed=42)["review_items"]) == 9
    csv_export = runtime.client.get(
        f"/api/v1/experiments/{run_id}/export?format=csv", headers=headers
    )
    assert csv_export.status_code == 200 and csv_export.headers["content-type"].startswith(
        "text/csv"
    )
    assert len(csv_export.text.splitlines()) == 10
    jsonl_export = runtime.client.get(
        f"/api/v1/experiments/{run_id}/export?format=jsonl", headers=headers
    )
    assert jsonl_export.status_code == 200 and len(jsonl_export.text.splitlines()) == 9
    assert '"reference":' not in jsonl_export.text
    with runtime.db() as db:
        assert db.get(Answer, base_id).response == original
        for table in (Message, Snapshot, Answer, AnswerRequest):
            assert db.scalar(select(func.count()).select_from(table)) == before[table.__tablename__]
        rows = list(
            db.scalars(select(TeachingStudyRecord).where(TeachingStudyRecord.run_id == run_id))
        )
        assert len(rows) == 9 and len({row.input_hash for row in rows}) == 1
        assert all(row.budget["consumed_calls"] == 1 for row in rows)
        jobs = list(
            db.scalars(
                select(Job).where(
                    Job.kind == "teaching", Job.payload["run_id"].as_string() == run_id
                )
            )
        )
        assert len(jobs) == 9 and all(job.request_id is None for job in jobs)
        assert (
            db.scalar(
                select(func.count())
                .select_from(Attempt)
                .where(Attempt.job_id.in_([job.id for job in jobs]))
            )
            == 18
        )
    denied = runtime.client.post(
        "/api/v1/teaching-studies", headers=runtime.headers(), json={"base_answer_ids": [base_id]}
    )
    assert denied.status_code == 403


def test_postgres_teaching_cancel_keeps_success_and_all_scheduled_outcomes(
    runtime, evaluation_backend, tmp_path
):
    backend, _ = evaluation_backend
    base_id = base_answer(runtime, backend, tmp_path)
    headers = runtime.headers("admin@example.com")
    created = runtime.client.post(
        "/api/v1/teaching-studies", headers=headers, json={"base_answer_ids": [base_id]}
    ).json()["data"]
    run_id = created["id"]
    assert (
        runtime.client.post(f"/api/v1/experiments/{run_id}/start", headers=headers).status_code
        == 202
    )
    assert runtime.work()
    assert (
        runtime.client.post(f"/api/v1/experiments/{run_id}/cancel", headers=headers).status_code
        == 200
    )
    result = runtime.client.get(f"/api/v1/experiments/{run_id}/results", headers=headers).json()[
        "data"
    ]
    assert result["outcome_counts"] == {"completed": 1, "cancelled": 8}
    assert result["scheduled_count"] == 9
    assert (
        runtime.client.post(f"/api/v1/experiments/{run_id}/start", headers=headers).status_code
        == 409
    )
    with runtime.db() as db:
        rows = list(
            db.scalars(select(TeachingStudyRecord).where(TeachingStudyRecord.run_id == run_id))
        )
        assert sum(row.budget["consumed_calls"] for row in rows) == 1


def test_postgres_teaching_late_source_revocation_discards_publication(
    runtime, evaluation_backend, tmp_path, monkeypatch
):
    backend, document_id = evaluation_backend
    base_id = base_answer(runtime, backend, tmp_path)
    headers = runtime.headers("admin@example.com")
    original_generate = TeachingStudyService.generate

    def revoke_after_generation(self, **kwargs):
        output = original_generate(self, **kwargs)
        with runtime.db() as db:
            db.get(Document, document_id).revoked = True
            db.commit()
        return output

    monkeypatch.setattr(TeachingStudyService, "generate", revoke_after_generation)
    run_id = runtime.client.post(
        "/api/v1/teaching-studies", headers=headers, json={"base_answer_ids": [base_id]}
    ).json()["data"]["id"]
    assert (
        runtime.client.post(f"/api/v1/experiments/{run_id}/start", headers=headers).status_code
        == 202
    )
    with runtime.db() as db:
        base_before = db.get(Answer, base_id).response
    assert runtime.work()
    with runtime.db() as db:
        run = db.get(ExperimentRun, run_id)
        assert run.state == "environment_changed"
        failed = db.scalar(
            select(TeachingStudyRecord).where(
                TeachingStudyRecord.run_id == run_id, TeachingStudyRecord.state == "error"
            )
        )
        assert failed and failed.response is None and failed.budget["consumed_calls"] == 1
        assert failed.error["code"] == "SOURCE_UNAVAILABLE"
        assert db.get(Answer, base_id).response == base_before
    assert (
        runtime.client.post(f"/api/v1/experiments/{run_id}/cancel", headers=headers).status_code
        == 200
    )


def test_postgres_teaching_stale_claim_retains_charged_budget_and_fences_late_result(
    runtime, evaluation_backend, tmp_path
):
    from app.worker import recover_stale

    backend, _ = evaluation_backend
    base_id = base_answer(runtime, backend, tmp_path)
    headers = runtime.headers("admin@example.com")
    run_id = runtime.client.post(
        "/api/v1/teaching-studies", headers=headers, json={"base_answer_ids": [base_id]}
    ).json()["data"]["id"]
    assert (
        runtime.client.post(f"/api/v1/experiments/{run_id}/start", headers=headers).status_code
        == 202
    )
    with runtime.db() as db:
        job = db.scalar(
            select(Job)
            .where(Job.kind == "teaching", Job.payload["run_id"].as_string() == run_id)
            .order_by(Job.created_at)
        )
        job.state, job.execution_token = "running", new_uuid()
        job.updated_at = utcnow() - timedelta(seconds=1000)
        item = db.get(TeachingStudyRecord, job.payload["teaching_id"])
        item.state = "running"
        item.budget = {**item.budget, "consumed_calls": 1, "active_seconds": 1.0}
        job_id, token, item_id = job.id, job.execution_token, item.id
        db.commit()
    assert recover_stale(runtime.engine, 181) == 1
    execute_teaching(runtime.engine, runtime.settings, job_id, token)
    with runtime.db() as db:
        item = db.get(TeachingStudyRecord, item_id)
        assert item.state == "error" and item.response is None
        assert item.budget["consumed_calls"] == 1
        assert item.error["details"]["uncertain_external_execution"] is True
        assert db.get(Job, job_id).state == "failed"
    assert (
        runtime.client.post(f"/api/v1/experiments/{run_id}/cancel", headers=headers).status_code
        == 200
    )


def test_postgres_all_twelve_authored_scenario_families_use_real_chat_api(
    runtime, evaluation_backend
):
    from evaluation.conversations.http_backend import HttpChatBackend
    from evaluation.conversations.runner import aggregate_scenarios, run_scenario

    scenarios = read_json(ROOT / "evaluation/conversations/scenarios.json")["scenarios"]
    backend = HttpChatBackend(
        runtime.client,
        email="student@example.com",
        password="Passw0rd!",
        poll_interval=0.001,
        on_poll=runtime.work,
    )
    run_id = uuid4().hex
    results = [run_scenario(scenario, backend, run_id=run_id) for scenario in scenarios]
    aggregate = aggregate_scenarios(results)
    assert aggregate["scenario_count"] == 12
    assert aggregate["scheduled_turn_count"] == sum(
        len(scenario["turns"]) for scenario in scenarios
    )
    assert aggregate["completed_turn_count"] == aggregate["scheduled_turn_count"], [
        (result["scenario_id"], result["turns"])
        for result in results
        if not result["complete_scenario"]
    ]
    assert aggregate["semantic_correctness"] is None
    for result in results:
        assert result["human_semantic_review"] is None
        for turn in result["turns"]:
            assert set(turn["command"]) == {"session_id", "content", "use_profile"}
            assert turn["observation"]["answer"]["model_mode"] == "mock"
    profile_case = next(
        result for result in results if result["scenario_id"] == "profile_continuity"
    )
    continued = profile_case["turns"][2]["observation"]["answer"]
    assert continued["profile_snapshot"]["use_profile"] is False
    assert continued["conversation_snapshot"]["messages"]
    fresh = profile_case["turns"][3]["observation"]["answer"]
    assert fresh["conversation_snapshot"]["messages"] == []
