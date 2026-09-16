"""Administrator experiment lifecycle and all-outcome results, without gold data."""

from __future__ import annotations

import csv
import io
import json
from typing import Literal

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import Response
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.v1.deps import require_roles
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.responses import ok
from app.db.base import utcnow
from app.db.session import get_db
from app.modules.answering import service as answering
from app.modules.answering.models import Job
from app.modules.experiment.models import ExperimentItem, ExperimentRun
from app.modules.experiment import bridge
from app.modules.identity.models import User
from app.modules.knowledge.models import Configuration
from app.modules.knowledge.service import digest
from contracts.models import Contract, ExperimentSpec, MCQCommand, OpenQACommand
from contracts.http import Envelope

router = APIRouter(tags=["experiments"])


class ManifestInput(Contract):
    manifest: dict


class ItemsInput(Contract):
    commands: list[dict] = Field(min_length=1, max_length=10000)


class CommandInput(Contract):
    command: dict


class ScoresInput(Contract):
    scores: dict


class TeachingInput(Contract):
    base_answer_ids: list[str] = Field(min_length=1, max_length=1000)
    seed: int = 0
    configuration_id: str | None = None


def get_run(db, run_id, *, lock=False):
    query = select(ExperimentRun).where(ExperimentRun.id == run_id)
    if lock:
        query = query.with_for_update()
    run = db.scalar(query)
    if not run:
        raise AppError("NOT_FOUND")
    return run


def serialize_run(db, run, *, detail=False):
    items = bridge.refresh_run(db, run)
    if run.mode == "profile_study":
        result = {
            "id": run.id,
            "protocol_id": run.protocol_id,
            "mode": run.mode,
            "condition": run.condition,
            "status": run.state,
            "state": run.state,
            "spec": run.spec,
            "scheduled_count": run.scheduled_count,
            "completed_count": sum(item.state == "completed" for item in items),
            "failed_count": sum(item.state == "error" for item in items),
            "cancelled_count": sum(item.state == "cancelled" for item in items),
            "missing_input_count": 0,
            "model_mode": run.model_mode,
            "configuration_id": run.configuration_id,
            "manifest_hash": run.manifest_hash,
            "environment_change": run.environment_change,
            "can_start": run.state in {"frozen", "running"}
            and any(item.state == "pending" for item in items),
            "can_freeze": False,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
        }
        if detail:
            result["manifest"] = run.manifest
            result["items"] = [
                {
                    "item_id": item.item_id,
                    "base_answer_id": item.base_answer_id,
                    "state": item.state,
                    "condition": item.condition,
                    "target_level": item.target_level,
                }
                for item in items
            ]
        return result
    result = {
        "id": run.id,
        "protocol_id": run.protocol_id,
        "mode": run.mode,
        "condition": run.condition,
        "status": run.state,
        "state": run.state,
        "spec": run.spec,
        "scheduled_count": run.scheduled_count,
        "completed_count": sum(item.state in {"completed", "refused"} for item in items),
        "failed_count": sum(item.state in {"error", "invalid", "incomplete"} for item in items),
        "cancelled_count": sum(item.state == "cancelled" for item in items),
        "missing_input_count": sum(item.command is None for item in items),
        "model_mode": run.model_mode,
        "configuration_id": run.configuration_id,
        "manifest_hash": run.manifest_hash,
        "environment_change": run.environment_change,
        "can_start": run.state in {"frozen", "running"}
        and any(item.state == "pending" and item.command for item in items),
        "can_freeze": run.state == "draft" and all(item.command for item in items),
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }
    if detail:
        result["manifest"] = run.manifest
        result["items"] = [
            {
                "item_id": item.item_id,
                "question_id": item.question_id,
                "command_hash": item.command_hash,
                "state": item.state,
                "request_id": item.request_id,
            }
            for item in items
        ]
        result["input_guidance"] = (
            "Attach the frozen gold-free question list through the evaluator or POST /experiments/{id}/items before freezing. Private labels/support never enter this API."
        )
    return result


@router.get("/experiments", response_model=Envelope[list[dict]])
def list_runs(db: Session = Depends(get_db), actor: User = Depends(require_roles("admin"))):
    rows = [
        serialize_run(db, run)
        for run in db.scalars(select(ExperimentRun).order_by(ExperimentRun.created_at.desc()))
    ]
    db.commit()
    return ok(rows)


@router.post("/experiments", status_code=201, response_model=Envelope[dict])
def create_run(
    body: ExperimentSpec,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
    settings: Settings = Depends(get_settings),
):
    spec = body.model_dump()
    expected_mode = {"sciq_openqa": "benchmark_openqa", "sciq_mcq": "benchmark_mcq"}.get(
        body.protocol_id
    )
    if expected_mode != body.mode:
        raise AppError(
            "VALIDATION_FAILED",
            detail="Use the explicit SciQ-derived OpenQA or MCQ protocol. Conversation and profile studies use their separate tools.",
        )
    config = db.get(Configuration, body.configuration_id)
    if not config:
        raise AppError("NOT_FOUND", detail="Select an existing immutable configuration.")
    bridge.public_only(config.values)
    run = ExperimentRun(
        owner_id=actor.id,
        protocol_id=body.protocol_id,
        mode=body.mode,
        condition=body.condition,
        scheduled_count=body.scheduled_count,
        spec=spec,
        configuration_id=config.id,
        model_mode=settings.model_mode,
    )
    db.add(run)
    db.flush()
    for index in range(body.scheduled_count):
        db.add(ExperimentItem(run_id=run.id, item_id=f"item_{index:06}", ordinal=index))
    db.flush()
    configured_items = config.values.get("evaluation_items")
    if configured_items:
        attach_items(db, run, configured_items)
    db.commit()
    return ok(serialize_run(db, run, detail=True))


@router.post("/teaching-studies", status_code=201, response_model=Envelope[dict])
def create_teaching(
    body: TeachingInput,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
    settings: Settings = Depends(get_settings),
):
    run = bridge.create_teaching_run(
        db,
        settings,
        actor,
        body.base_answer_ids,
        seed=body.seed,
        configuration_id=body.configuration_id,
    )
    db.commit()
    return ok(serialize_run(db, run, detail=True))


@router.get("/evaluation/environment", response_model=Envelope[dict])
def evaluation_environment(
    configuration_id: str | None = None,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
    settings: Settings = Depends(get_settings),
):
    return ok(bridge.environment(db, settings, configuration_id))


@router.post("/experiments/register", status_code=201, response_model=Envelope[dict])
def register_run(
    body: ManifestInput,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
    settings: Settings = Depends(get_settings),
):
    run = bridge.register_manifest(db, settings, actor, body.manifest)
    db.commit()
    return ok(serialize_run(db, run, detail=True))


@router.get("/experiments/{run_id}", response_model=Envelope[dict])
def detail_run(
    run_id: str, db: Session = Depends(get_db), actor: User = Depends(require_roles("admin"))
):
    result = serialize_run(db, get_run(db, run_id), detail=True)
    db.commit()
    return ok(result)


def attach_items(db, run, commands):
    if run.state != "draft" or len(commands) != run.scheduled_count:
        raise AppError(
            "CONFLICT", detail="Supply the complete scheduled question list to a draft run."
        )
    rows = list(
        db.scalars(
            select(ExperimentItem)
            .where(ExperimentItem.run_id == run.id)
            .order_by(ExperimentItem.ordinal)
        )
    )
    seen = set()
    for item, value in zip(rows, commands, strict=True):
        bridge.public_only(value)
        # A configured dataset may provide only public question fields; runtime
        # identities are assigned by this already-owned experiment.
        command = {**value, "mode": run.mode, "run_id": run.id, "item_id": item.item_id}
        try:
            (OpenQACommand if run.mode == "benchmark_openqa" else MCQCommand).model_validate(
                command
            )
        except ValueError as exc:
            raise AppError(
                "VALIDATION_FAILED",
                detail="Question list violates the selected gold-free command contract.",
            ) from exc
        if command["question_id"] in seen:
            raise AppError(
                "VALIDATION_FAILED", detail="Question IDs must be unique in the frozen workload."
            )
        seen.add(command["question_id"])
        item.question_id = command["question_id"]
        item.command = command
        item.command_hash = digest(command)
    db.flush()


@router.post("/experiments/{run_id}/items", response_model=Envelope[dict])
def attach(
    run_id: str,
    body: ItemsInput,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
):
    run = get_run(db, run_id, lock=True)
    attach_items(db, run, body.commands)
    db.commit()
    return ok(serialize_run(db, run, detail=True))


@router.post("/experiments/{run_id}/freeze", response_model=Envelope[dict])
def freeze_run(
    run_id: str,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
    settings: Settings = Depends(get_settings),
):
    run = get_run(db, run_id, lock=True)
    if run.manifest_hash:
        return ok(serialize_run(db, run, detail=True))
    items = list(
        db.scalars(
            select(ExperimentItem)
            .where(ExperimentItem.run_id == run.id)
            .order_by(ExperimentItem.ordinal)
        )
    )
    if run.state != "draft" or not items or any(item.command is None for item in items):
        raise AppError(
            "CONFLICT",
            detail="Freeze requires the actual gold-free question list; attach it with the evaluator or item-import API.",
        )
    environment = bridge.environment(db, settings, run.configuration_id)
    configuration = db.get(Configuration, run.configuration_id)
    variant = "none" if run.condition == "E0" else "R0" if run.condition == "E1" else run.condition
    runtime = {
        "model_mode": settings.model_mode,
        "retrieval_variant": variant,
        "history_policy": "empty",
        "summary_policy": "empty",
        "profile_policy": "off",
    }
    if configuration.values.get("top_k") is not None:
        runtime["top_k"] = configuration.values["top_k"]
    spec = run.spec
    manifest = {
        "version": "evaluation-run-v1",
        "run_id": run.id,
        "created_at": utcnow().isoformat(),
        "protocol_id": run.protocol_id,
        "mode": run.mode,
        "condition": run.condition,
        "dataset": {
            "name": spec["dataset"],
            "revision": spec["dataset_revision"],
            "split": spec["split"],
            "count": run.scheduled_count,
            "sha256": digest([item.command for item in items]),
            "source_kind": "provided_gold_free_commands",
        },
        "seed": spec["seed"],
        "scheduled_count": run.scheduled_count,
        "items": [
            {
                "item_id": item.item_id,
                "question_id": item.question_id,
                "command_hash": item.command_hash,
            }
            for item in items
        ],
        "environment": environment,
        "configuration": runtime,
        "scorer_version": spec["scorer_version"],
        "rubric_version": spec["rubric_version"],
        "code_revision": "workspace-development-uncommitted",
        "reference_policy": "evaluator_only_unavailable_to_application",
        "full_response_review_item_ids": [item.item_id for item in items[:20]],
        "evidence_class": "mock_software_rehearsal"
        if settings.model_mode == "mock"
        else "live_model_run",
    }
    manifest["manifest_hash"] = digest(manifest)
    bridge.register_manifest(db, settings, actor, manifest)
    db.commit()
    return ok(serialize_run(db, run, detail=True))


@router.post("/experiments/{run_id}/start", status_code=202, response_model=Envelope[dict])
def start_run(
    run_id: str,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
    settings: Settings = Depends(get_settings),
):
    run = get_run(db, run_id, lock=True)
    if run.state not in {"frozen", "running"} or not run.manifest:
        raise AppError(
            "CONFLICT",
            detail="Start a frozen run; cancelled outcomes remain terminal and require a new run.",
        )
    if run.mode == "profile_study":
        bridge.start_teaching(db, settings, actor, run)
        db.commit()
        return ok(serialize_run(db, run, detail=True))
    items = list(
        db.scalars(
            select(ExperimentItem)
            .where(ExperimentItem.run_id == run.id)
            .order_by(ExperimentItem.ordinal)
        )
    )
    if any(item.state == "pending" and item.command is None for item in items):
        raise AppError(
            "CONFLICT",
            detail="The frozen run awaits evaluator-supplied commands; no fabricated questions are scheduled.",
        )
    for item in items:
        if item.state == "pending":
            bridge.submit_item(
                db,
                settings,
                actor,
                item.command,
                run.manifest,
                f"evaluation:{run.id}:{item.item_id}",
                commit=False,
            )
    db.commit()
    return ok(serialize_run(db, run, detail=True))


@router.post("/experiments/{run_id}/cancel", response_model=Envelope[dict])
def cancel_run(
    run_id: str, db: Session = Depends(get_db), actor: User = Depends(require_roles("admin"))
):
    run = get_run(db, run_id, lock=True)
    if run.mode == "profile_study":
        bridge.cancel_teaching(db, run)
        db.commit()
        return ok(serialize_run(db, run, detail=True))
    items = bridge.refresh_run(db, run)
    for item in items:
        if item.state in bridge.TERMINAL:
            continue
        if item.request_id:
            for job in db.scalars(
                select(Job).where(
                    Job.request_id == item.request_id, Job.state.in_(answering.ACTIVE)
                )
            ):
                answering.cancel_job(db, job)
        item.state = "cancelled"
    run.state = "cancelled"
    db.commit()
    return ok(serialize_run(db, run, detail=True))


@router.post(
    "/experiments/{run_id}/items/{item_id}/submit", status_code=202, response_model=Envelope[dict]
)
def submit(
    run_id: str,
    item_id: str,
    body: CommandInput,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
    settings: Settings = Depends(get_settings),
):
    run = get_run(db, run_id)
    if body.command.get("run_id") != run_id or body.command.get("item_id") != item_id:
        raise AppError("VALIDATION_FAILED", detail="Path and command identities disagree.")
    return ok(
        bridge.submit_item(db, settings, actor, body.command, run.manifest or {}, idempotency_key)
    )


@router.post("/experiments/{run_id}/items/{item_id}/scores", response_model=Envelope[dict])
def record_scores(
    run_id: str,
    item_id: str,
    body: ScoresInput,
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
):
    bridge.scores_for_item(db, actor, run_id, item_id, body.scores)
    db.commit()
    return ok({"run_id": run_id, "item_id": item_id, "status": "scored"})


@router.get("/experiments/{run_id}/results", response_model=Envelope[dict])
def results(
    run_id: str, db: Session = Depends(get_db), actor: User = Depends(require_roles("admin"))
):
    run = get_run(db, run_id)
    result = bridge.run_results(db, run)
    result["manifest"] = run.manifest
    db.commit()
    return ok(result)


@router.get(
    "/experiments/{run_id}/export",
    response_model=Envelope[dict],
    responses={200: {"content": {"text/csv": {}, "application/x-ndjson": {}}}},
)
def export_results(
    run_id: str,
    format: Literal["json", "jsonl", "csv"] = Query(default="json"),
    db: Session = Depends(get_db),
    actor: User = Depends(require_roles("admin")),
):
    run = get_run(db, run_id)
    result = bridge.run_results(db, run)
    result["manifest"] = run.manifest
    db.commit()
    if format == "json":
        return ok(result)
    headers = {"Content-Disposition": f'attachment; filename="{run.protocol_id}-{run.id}.{format}"'}
    if format == "jsonl":
        content = "".join(
            json.dumps(
                {
                    "run_id": run.id,
                    "protocol_id": run.protocol_id,
                    "model_mode": run.model_mode,
                    **item,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for item in result["items"]
        )
        return Response(content=content, media_type="application/x-ndjson", headers=headers)
    stream = io.StringIO(newline="")
    columns = [
        "run_id",
        "protocol_id",
        "model_mode",
        "item_id",
        "question_id",
        "status",
        "request_id",
        "job_id",
        "em",
        "token_f1",
        "accuracy",
        "condition",
        "target_level",
        "error_code",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for item in result["items"]:
        outcome = item.get("outcome", item)
        writer.writerow(
            {
                "run_id": run.id,
                "protocol_id": run.protocol_id,
                "model_mode": run.model_mode,
                "item_id": item["item_id"],
                "question_id": item.get("question_id"),
                "status": item.get("state", item.get("status")),
                "request_id": item.get("request_id"),
                "job_id": outcome.get("job_id"),
                "condition": item.get("condition", run.condition),
                "target_level": item.get("target_level"),
                "error_code": (outcome.get("error") or {}).get("code"),
                **{
                    key: (item.get("scores") or {}).get(key)
                    for key in ("em", "token_f1", "accuracy")
                },
            }
        )
    return Response(content=stream.getvalue(), media_type="text/csv", headers=headers)
