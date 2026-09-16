"""Shared real application/worker evaluation rehearsal on an isolated SQLite DB.

PostgreSQL migration/transaction acceptance remains a separate integration gate.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.exceptions import AppError
from app.db.base import Base
from app.db import models as all_models
from app.modules.answering.models import Answer, AnswerRequest, Job, Message, Snapshot
from app.modules.experiment.models import ExperimentItem, ExperimentRun, TeachingStudyRecord
from app.modules.experiment.bridge import (
    DatabaseAnswerBackend,
    checked_manifest,
    run_results,
    submit_item,
)
from app.modules.identity.models import Role, User, Workspace
from app.modules.knowledge import service as knowledge
from app.modules.knowledge.models import ProcessingRun
from app.worker import run_once
from evaluation.common import fingerprint
from evaluation.runner import EvaluationRun

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def shared_backend(tmp_path):
    settings = Settings(
        env="test",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_root=str(tmp_path / "storage"),
        mock_delay_seconds=0,
    )
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        workspace = Workspace(name="Test", slug="test")
        role = Role(name="admin")
        db.add_all([workspace, role])
        db.flush()
        actor = User(
            email="operator@example.test",
            full_name="Test operator",
            hashed_password="unused-test-password-hash",
            workspace_id=workspace.id,
            role_id=role.id,
        )
        db.add(actor)
        db.commit()
        actor_id = actor.id
        text = b"# Photosynthesis\nPhotosynthesis captures light energy and stores it as chemical energy in sugars. In plants it occurs in chloroplasts. Plants use carbon dioxide and water and release oxygen."
        document, version, _ = knowledge.ingest(
            db, settings, actor_id, "authored.txt", text, "Authored evaluation source"
        )
        processing, job = knowledge.queue_process(db, document.id, actor_id)
        processing_id = processing.id
        db.commit()
    assert run_once(engine, settings)
    with factory() as db:
        assert db.get(ProcessingRun, processing_id).state == "ready"
        release, job = knowledge.queue_release(db, actor_id, [processing_id])
        release_id = release.id
        db.commit()
    assert run_once(engine, settings)
    with factory() as db:
        knowledge.activate(db, release_id)
        db.commit()
    yield DatabaseAnswerBackend(settings, engine, actor_id=actor_id, inline_worker=True), factory
    engine.dispose()


def config(tmp_path, mode="sciq_openqa", limit=1):
    return {
        "protocol_id": mode,
        "condition": "E1",
        "dataset_path": str(ROOT / "evaluation/private_fixture/sciq_authored.json"),
        "dataset_revision": "authored-v1",
        "split": "authored",
        "code_revision": "test",
        "backend_factory": "app.modules.experiment.bridge:create_backend",
        "private_root": str(tmp_path / "private-runs"),
        "limit": limit,
    }


@pytest.mark.parametrize("protocol", ["sciq_openqa", "sciq_mcq"])
def test_actual_shared_worker_persists_typed_benchmark_without_dialogue_or_gold(
    tmp_path, shared_backend, protocol
):
    backend, factory = shared_backend
    run = EvaluationRun.freeze(config(tmp_path, protocol), backend)
    assert run.advance(backend)["status"] == "completed"
    assert run.state["items"][0]["status"] == "completed"
    assert run.state["items"][0]["scores_published"] is True
    with factory() as db:
        request = db.scalar(
            select(AnswerRequest).where(AnswerRequest.run_id == run.manifest["run_id"])
        )
        assert (
            request.context_snapshot_id is None
            and request.profile_snapshot_id is None
            and request.session_id is None
        )
        assert request.command["question"] == run.commands[0]["question_text"]
        assert (
            "support" not in request.command
            and "reference" not in request.command
            and "correct_label" not in request.command
        )
        assert ("options" in request.command) == (protocol == "sciq_mcq")
        assert db.scalar(select(func.count()).select_from(Message)) == 0
        assert db.scalar(select(func.count()).select_from(Snapshot)) == 0
        assert db.scalar(select(func.count()).select_from(Answer)) == 1
        result = run_results(db, db.get(ExperimentRun, run.manifest["run_id"]))
        assert result["scored_count"] == 1 and result["scheduled_count"] == 1
        assert result["human_review"] is None
    run.advance(backend)
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Answer)) == 1


def test_registered_hash_rejects_post_freeze_question_or_gold_injection(tmp_path, shared_backend):
    backend, factory = shared_backend
    run = EvaluationRun.freeze(config(tmp_path), backend)
    backend.register_run(run.manifest)
    command = dict(run.commands[0])
    command["question_text"] = "A different question"
    with pytest.raises(AppError, match="preregistered"):
        backend.submit(
            command,
            run_context=run.manifest,
            idempotency_key=f"evaluation:{run.manifest['run_id']}:{command['item_id']}",
        )
    command = dict(run.commands[0], support="leaked private source")
    with pytest.raises(AppError, match="references"):
        backend.submit(command, run_context=run.manifest, idempotency_key="key")
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(AnswerRequest)) == 0


def test_server_cancellation_retains_all_scheduled_items(tmp_path, shared_backend):
    backend, factory = shared_backend
    run = EvaluationRun.freeze(config(tmp_path, limit=2), backend)
    backend.register_run(run.manifest)
    run.state["registered"] = True
    run._save()
    run.cancel(backend)
    with factory() as db:
        result = run_results(db, db.get(ExperimentRun, run.manifest["run_id"]))
        assert result["scheduled_count"] == 2
        assert result["outcome_counts"] == {"cancelled": 2}
        assert result["metrics"]["em"] is None


def test_admin_api_create_freeze_start_results_and_role_denial(shared_backend):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.v1.deps import get_current_user
    from app.core.config import get_settings
    from app.core.exceptions import register_exception_handlers
    from app.db.session import get_db
    from app.modules.experiment.router import router

    backend, factory = shared_backend
    with factory() as db:
        configuration = knowledge.config_create(
            db,
            "evaluation",
            "Authored OpenQA",
            {
                "evaluation_items": [
                    {"question_id": "authored-question", "question_text": "What is photosynthesis?"}
                ]
            },
        )
        configuration_id = configuration.id
        db.commit()
        actor = db.get(User, backend.actor_id)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    register_exception_handlers(app)

    def get_test_db():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = get_test_db
    app.dependency_overrides[get_settings] = lambda: backend.settings
    app.dependency_overrides[get_current_user] = lambda: actor
    with TestClient(app) as client:
        result = client.post(
            "/api/v1/experiments",
            json={
                "protocol_id": "sciq_openqa",
                "mode": "benchmark_openqa",
                "condition": "E1",
                "dataset": "authored-api-fixture",
                "dataset_revision": "v1",
                "split": "authored",
                "seed": 17,
                "configuration_id": configuration_id,
                "scheduled_count": 1,
            },
        )
        assert result.status_code == 201, result.text
        run_id = result.json()["data"]["id"]
        assert result.json()["data"]["can_freeze"] is True
        result = client.post(f"/api/v1/experiments/{run_id}/freeze")
        assert result.status_code == 200, result.text
        assert result.json()["data"]["can_start"] is True
        result = client.post(f"/api/v1/experiments/{run_id}/start")
        assert result.status_code == 202, result.text
        assert run_once(backend.engine, backend.settings)
        result = client.get(f"/api/v1/experiments/{run_id}/results")
        assert result.status_code == 200
        assert result.json()["data"]["scheduled_count"] == 1
        assert result.json()["data"]["outcome_counts"] == {"completed": 1}
        assert result.json()["data"]["metrics"]["em"] is None
        assert (
            client.get(f"/api/v1/experiments/{run_id}/export").json()["data"]["manifest"][
                "protocol_id"
            ]
            == "sciq_openqa"
        )
        actor.role = Role(name="student")
        assert client.get("/api/v1/experiments").status_code == 403
