"""Export timing from stored answers without mixing worker and legacy stages."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db import models as all_models
from app.modules.answering.models import Answer, AnswerRequest, Job
from app.modules.experiment.bridge import item_outcome
from app.modules.identity.models import Role, User, Workspace


WORKER_SCOPE = "worker_execution_before_answer_insert_v1"


@pytest.mark.parametrize(
    "answer_timing, trace, expected",
    [
        pytest.param(
            {"preparation_ms": 46.328, "retrieval_ms": 4624.836, "timing_scope": WORKER_SCOPE},
            {"preparation_ms": 4671.163},
            {"preparation_ms": 46.328, "retrieval_ms": 4624.836, "timing_scope": WORKER_SCOPE},
            id="measured-preparation-excludes-retrieval",
        ),
        pytest.param(
            {"preparation_ms": 0, "retrieval_ms": 12, "timing_scope": WORKER_SCOPE},
            {"preparation_ms": 12},
            {"preparation_ms": 0, "retrieval_ms": 12, "timing_scope": WORKER_SCOPE},
            id="measured-zero-is-not-missing",
        ),
        pytest.param(
            {"preparation_ms": 3.25, "generation_ms": 10},
            {"preparation_ms": 900},
            {"preparation_ms": 3.25, "generation_ms": 10},
            id="existing-unscoped-answer-value-is-preserved",
        ),
        pytest.param(
            {"generation_ms": 10},
            {"preparation_ms": 900},
            {
                "generation_ms": 10,
                "preparation_ms": 900,
                "preparation_timing_scope": "legacy_preparation_including_retrieval",
                "preparation_timing_source": "request_trace.preparation_ms",
            },
            id="legacy-missing-answer-value-has-labelled-fallback",
        ),
        pytest.param(
            {"generation_ms": 10},
            {},
            {"generation_ms": 10, "preparation_ms": None},
            id="unmeasured-legacy-value-stays-null",
        ),
        pytest.param(
            {"preparation_ms": None, "retrieval_ms": 12, "timing_scope": WORKER_SCOPE},
            {"preparation_ms": 900},
            {"preparation_ms": None, "retrieval_ms": 12, "timing_scope": WORKER_SCOPE},
            id="scoped-missing-value-is-not-an-aggregate",
        ),
        pytest.param(
            {"timing_scope": "future_measured_scope"},
            {"preparation_ms": 900},
            {"timing_scope": "future_measured_scope"},
            id="unknown-scoped-measurement-is-not-legacy",
        ),
    ],
)
def test_persisted_item_outcome_preserves_authoritative_timing(answer_timing, trace, expected):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            workspace = Workspace(name="Timing fixture", slug="timing")
            role = Role(name="admin")
            db.add_all([workspace, role])
            db.flush()
            actor = User(
                email="timing@example.test",
                full_name="Timing fixture",
                hashed_password="unused-test-hash",
                workspace_id=workspace.id,
                role_id=role.id,
            )
            db.add(actor)
            db.flush()
            request = AnswerRequest(
                owner_id=actor.id,
                route="/experiments/timing/items/item-0",
                idempotency_key="timing-fixture",
                body_hash="0" * 64,
                mode="benchmark_openqa",
                response_schema="chat_response_v1",
                state="refused",
                trace=deepcopy(trace),
            )
            db.add(request)
            db.flush()
            job = Job(owner_id=actor.id, request_id=request.id, state="succeeded")
            db.add(job)
            db.flush()
            answer = Answer(
                request_id=request.id,
                job_id=job.id,
                response_schema="chat_response_v1",
                response={"response_type": "refusal", "answer_text": "Timing fixture."},
                model_mode="mock",
                timing=deepcopy(answer_timing),
            )
            db.add(answer)
            db.flush()
            job.answer_id = answer.id
            db.commit()

            outcome = item_outcome(db, SimpleNamespace(request_id=request.id))

            assert outcome["status"] == "refused"
            assert outcome["answer_id"] == answer.id
            assert outcome["timings"] == expected
            assert answer.timing == answer_timing
            assert request.trace == trace
            assert not db.dirty
            db.expire_all()
            assert db.get(Answer, answer.id).timing == answer_timing
            assert db.get(AnswerRequest, request.id).trace == trace
    finally:
        engine.dispose()
