"""Gold-free experiments use the same durable answer requests and worker.

This module deliberately imports no evaluator, dataset or scoring implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import random
from collections import Counter
from pathlib import Path

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.exceptions import AppError
from app.db.base import new_uuid, utcnow
from app.modules.answering.models import Answer, AnswerRequest, Attempt, Job, Snapshot
from app.modules.answering import service as answering
from app.modules.identity.models import Role, User
from app.modules.knowledge.models import (
    ActiveCorpus,
    Chunk,
    Configuration,
    CorpusRelease,
    Document,
    ReleaseChunk,
)
from app.modules.knowledge.service import digest
from app.modules.experiment.models import ExperimentItem, ExperimentRun, TeachingStudyRecord
from contracts.models import (
    ChatResponseV1,
    MCQResponseV1,
    TeachingStudyResponseV1,
    MCQCommand,
    OpenQACommand,
)
from generation.prompt_builder import PROMPTS
from generation.types import ModelConfig, RequestBudget
from personalisation.compiler import COMPILER_VERSION
from personalisation.study import TeachingStudyService

FORBIDDEN_FIELDS = {
    "correct_answer",
    "correct_label",
    "reference",
    "references",
    "support",
    "gold",
    "distractors",
    "distractor1",
    "distractor2",
    "distractor3",
    "evidence_status",
    "api_key",
    "secret_key",
    "password",
    "access_token",
}
TERMINAL = {"completed", "refused", "error", "cancelled", "invalid", "incomplete"}
MANIFEST_FIELDS = {
    "version",
    "run_id",
    "created_at",
    "protocol_id",
    "mode",
    "condition",
    "dataset",
    "seed",
    "scheduled_count",
    "items",
    "environment",
    "configuration",
    "scorer_version",
    "rubric_version",
    "code_revision",
    "reference_policy",
    "full_response_review_item_ids",
    "evidence_class",
    "manifest_hash",
}


def public_only(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in FORBIDDEN_FIELDS:
                raise AppError(
                    "VALIDATION_FAILED",
                    detail="Evaluator references and credentials are not accepted in application payloads.",
                )
            public_only(child)
    elif isinstance(value, list):
        for child in value:
            public_only(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise AppError("VALIDATION_FAILED", detail="Nonfinite metadata is invalid.")


def require_admin(actor):
    if actor is None or actor.status != "active" or actor.role.name != "admin":
        raise AppError("FORBIDDEN")


def experiment_model_config(settings, configuration=None):
    values = configuration.values if configuration else {}
    if any(
        key in values
        for key in ("model_config", "prompt", "prompt_template", "prompt_configuration_id")
    ):
        raise AppError(
            "VALIDATION_FAILED",
            detail="Use the documented model object; arbitrary prompt overrides are unsupported by the frozen protocol.",
        )
    overrides = values.get("model", {})
    allowed = {
        "model",
        "window_tokens",
        "max_tokens",
        "timeout_seconds",
        "temperature",
        "seed",
        "token_limit_parameter",
        "structured_output_mode",
        "reasoning_effort",
    }
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise AppError(
            "VALIDATION_FAILED",
            detail="Model overrides must use supported noncredential configuration fields.",
        )
    effective = {**answering.model_config(settings), **overrides}
    if type(effective["max_tokens"]) is not int or effective["max_tokens"] > 1024:
        raise AppError(
            "VALIDATION_FAILED", detail="Frozen output budgets cannot exceed 1024 tokens."
        )
    try:
        ModelConfig.from_dict(effective).validate()
    except (ValueError, TypeError) as exc:
        raise AppError(
            "VALIDATION_FAILED", detail="Invalid effective model configuration."
        ) from exc
    return effective


def environment(db, settings, configuration_id=None):
    config = db.get(Configuration, configuration_id) if configuration_id else None
    if configuration_id and not config:
        raise AppError(
            "NOT_FOUND", detail="The configured experiment configuration does not exist."
        )
    pointer = db.get(ActiveCorpus, 1)
    release = db.get(CorpusRelease, pointer.release_id) if pointer and pointer.release_id else None
    documents = []
    if release:
        documents = list(
            db.execute(
                select(Document.id, Document.active, Document.revoked, Document.version)
                .join(Chunk, Chunk.document_id == Document.id)
                .join(ReleaseChunk, ReleaseChunk.chunk_id == Chunk.id)
                .where(ReleaseChunk.release_id == release.id)
                .distinct()
                .order_by(Document.id)
            )
        )
    prompts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(PROMPTS.glob("*.txt"))
    }
    schemas = {
        model.__name__: model.model_json_schema()
        for model in (ChatResponseV1, MCQResponseV1, TeachingStudyResponseV1)
    }
    return {
        "model_mode": settings.model_mode,
        "configuration_id": config.id if config else None,
        "configuration_hash": config.content_hash if config else None,
        "corpus_release_id": release.id if release else None,
        "corpus_manifest_hash": digest(release.manifest) if release else None,
        "model_configuration_hash": digest(experiment_model_config(settings, config)),
        "prompt_configuration_hash": digest(prompts),
        "response_schema_hash": digest(schemas),
        "embedding_configuration_hash": digest(release.configuration) if release else None,
        "source_visibility_hash": digest([list(row) for row in documents]),
    }


def validate_request_environment(db, settings, req):
    if req.mode == "interactive_chat" or not req.run_id:
        return True
    run = db.get(ExperimentRun, req.run_id)
    if not run or not run.manifest:
        raise AppError("CONFLICT", detail="A registered frozen experiment is required.")
    if run.state == "cancelled":
        raise AppError("CONFLICT", detail="The frozen experiment has been cancelled.")
    current = environment(db, settings, run.configuration_id)
    differences = {
        key: {"frozen": value, "current": current.get(key)}
        for key, value in run.manifest["environment"].items()
        if current.get(key) != value
    }
    if differences or run.state == "environment_changed":
        run.state = "environment_changed"
        run.environment_change = {
            "changes": differences,
            "detected_at": utcnow().isoformat(),
            "policy": "Stop affected new calls and preserve previous outcomes",
        }
        db.flush()
        raise AppError(
            "SOURCE_UNAVAILABLE",
            detail="The frozen experiment environment changed; prior results are preserved.",
        )
    return True


def checked_manifest(manifest):
    public_only(manifest)
    if set(manifest) != MANIFEST_FIELDS:
        raise AppError(
            "VALIDATION_FAILED", detail="A complete versioned public run manifest is required."
        )
    if set(manifest["environment"]) != {
        "model_mode",
        "configuration_id",
        "configuration_hash",
        "corpus_release_id",
        "corpus_manifest_hash",
        "model_configuration_hash",
        "prompt_configuration_hash",
        "response_schema_hash",
        "embedding_configuration_hash",
        "source_visibility_hash",
    }:
        raise AppError(
            "VALIDATION_FAILED",
            detail="Freeze the complete actual environment, including source visibility and model/prompt identities.",
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,36}", manifest.get("run_id", "")):
        raise AppError(
            "VALIDATION_FAILED",
            detail="Run IDs must fit the shared 36-character identity contract.",
        )
    if manifest["mode"] not in {"benchmark_openqa", "benchmark_mcq"} or manifest["protocol_id"] != (
        "sciq_openqa" if manifest["mode"] == "benchmark_openqa" else "sciq_mcq"
    ):
        raise AppError("VALIDATION_FAILED", detail="Protocol and mode disagree.")
    if manifest["condition"] not in {"E0", "E1", "R1", "R2", "R3"}:
        raise AppError("VALIDATION_FAILED", detail="Unknown controlled condition.")
    if (
        not isinstance(manifest["scheduled_count"], int)
        or isinstance(manifest["scheduled_count"], bool)
        or not 1 <= manifest["scheduled_count"] <= 10000
        or len(manifest["items"]) != manifest["scheduled_count"]
    ):
        raise AppError("VALIDATION_FAILED", detail="Every scheduled item must be preregistered.")
    seen = set()
    questions = set()
    for item in manifest["items"]:
        if (
            set(item) != {"item_id", "question_id", "command_hash"}
            or not isinstance(item["item_id"], str)
            or len(item["item_id"]) > 160
            or not item["item_id"]
            or item["item_id"] in seen
            or not isinstance(item["question_id"], str)
            or not item["question_id"]
            or len(item["question_id"]) > 250
            or item["question_id"] in questions
            or not re.fullmatch(r"[a-f0-9]{64}", item["command_hash"])
        ):
            raise AppError(
                "VALIDATION_FAILED",
                detail="Frozen item identities/hashes must be valid and unique.",
            )
        seen.add(item["item_id"])
        questions.add(item["question_id"])
    if manifest["manifest_hash"] != digest(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    ):
        raise AppError("VALIDATION_FAILED", detail="Frozen manifest hash differs from its content.")
    configuration = manifest["configuration"]
    if set(configuration) - {
        "model_mode",
        "retrieval_variant",
        "top_k",
        "history_policy",
        "summary_policy",
        "profile_policy",
    }:
        raise AppError(
            "VALIDATION_FAILED",
            detail="Unsupported runtime parameter; model settings belong to the frozen application configuration.",
        )
    if (
        configuration.get("model_mode", manifest["environment"]["model_mode"])
        != manifest["environment"]["model_mode"]
    ):
        raise AppError(
            "VALIDATION_FAILED", detail="Runtime model mode must equal the actual environment."
        )
    if any(
        configuration.get(key) != value
        for key, value in {
            "history_policy": "empty",
            "summary_policy": "empty",
            "profile_policy": "off",
        }.items()
    ):
        raise AppError(
            "VALIDATION_FAILED",
            detail="Isolated benchmark history, summary and profile policy must be disabled.",
        )
    if manifest["condition"] == "E1" and configuration.get("retrieval_variant", "R0") != "R0":
        raise AppError("VALIDATION_FAILED", detail="Frozen E1 is basic dense R0.")
    if manifest["condition"] == "E0" and configuration.get("retrieval_variant", "none") != "none":
        raise AppError("VALIDATION_FAILED", detail="E0 cannot retrieve evidence.")
    if (
        manifest["condition"] in {"R1", "R2", "R3"}
        and configuration.get("retrieval_variant", manifest["condition"]) != manifest["condition"]
    ):
        raise AppError(
            "VALIDATION_FAILED",
            detail="The selected controlled retrieval variant differs from its condition.",
        )
    if configuration.get("top_k") is not None and (
        type(configuration["top_k"]) is not int or not 1 <= configuration["top_k"] <= 100
    ):
        raise AppError("VALIDATION_FAILED", detail="top_k must be a bounded positive integer.")
    return manifest


def register_manifest(db, settings, actor, manifest):
    require_admin(actor)
    checked_manifest(manifest)
    current = environment(db, settings, manifest["environment"].get("configuration_id"))
    if any(current.get(key) != value for key, value in manifest["environment"].items()):
        raise AppError(
            "CONFLICT",
            detail="The requested frozen environment differs from the actual application.",
        )
    if manifest["condition"] != "E0" and not current["corpus_release_id"]:
        raise AppError(
            "SOURCE_UNAVAILABLE", detail="Grounded evaluation requires a validated active corpus."
        )
    run = db.scalar(
        select(ExperimentRun).where(ExperimentRun.id == manifest["run_id"]).with_for_update()
    )
    if run and run.manifest_hash:
        if run.manifest_hash != manifest["manifest_hash"]:
            raise AppError(
                "IDEMPOTENCY_CONFLICT", detail="A frozen manifest cannot be edited in place."
            )
        return run
    dataset = manifest["dataset"]
    spec = {
        "protocol_id": manifest["protocol_id"],
        "mode": manifest["mode"],
        "condition": manifest["condition"],
        "dataset": dataset["name"],
        "dataset_revision": dataset["revision"],
        "split": dataset["split"],
        "seed": manifest["seed"],
        "configuration_id": current["configuration_id"],
        "scheduled_count": manifest["scheduled_count"],
        "scorer_version": manifest["scorer_version"],
        "rubric_version": manifest["rubric_version"],
    }
    if run:
        if run.state != "draft" or any(run.spec.get(key) != value for key, value in spec.items()):
            raise AppError(
                "CONFLICT",
                detail="The draft specification differs from the supplied frozen manifest.",
            )
    else:
        run = ExperimentRun(
            id=manifest["run_id"],
            owner_id=actor.id,
            protocol_id=manifest["protocol_id"],
            mode=manifest["mode"],
            condition=manifest["condition"],
            scheduled_count=manifest["scheduled_count"],
            spec=spec,
            model_mode=settings.model_mode,
            configuration_id=current["configuration_id"],
        )
        db.add(run)
        db.flush()
    run.manifest = manifest
    run.manifest_hash = manifest["manifest_hash"]
    run.state = "frozen"
    existing = {
        item.item_id: item
        for item in db.scalars(select(ExperimentItem).where(ExperimentItem.run_id == run.id))
    }
    if existing and set(existing) != {row["item_id"] for row in manifest["items"]}:
        raise AppError("CONFLICT", detail="Frozen item IDs differ from the preregistered draft.")
    for ordinal, value in enumerate(manifest["items"]):
        item = existing.get(value["item_id"])
        if item is None:
            item = ExperimentItem(run_id=run.id, item_id=value["item_id"], ordinal=ordinal)
            db.add(item)
        if item.command and digest(item.command) != value["command_hash"]:
            raise AppError("CONFLICT", detail="Draft command differs from frozen command hash.")
        item.question_id = value["question_id"]
        item.command_hash = value["command_hash"]
    db.flush()
    return run


def submit_item(db, settings, actor, command, run_context, idempotency_key, *, commit=True):
    require_admin(actor)
    public_only(command)
    try:
        (OpenQACommand if command.get("mode") == "benchmark_openqa" else MCQCommand).model_validate(
            command
        )
    except ValueError as exc:
        raise AppError(
            "VALIDATION_FAILED", detail="The evaluator command failed its gold-free mode contract."
        ) from exc
    run = db.scalar(
        select(ExperimentRun).where(ExperimentRun.id == command["run_id"]).with_for_update()
    )
    if (
        not run
        or not run.manifest
        or run.manifest_hash != run_context.get("manifest_hash")
        or digest({key: value for key, value in run_context.items() if key != "manifest_hash"})
        != run.manifest_hash
    ):
        raise AppError("CONFLICT", detail="Submit only against the registered frozen run manifest.")
    if run.state not in {"frozen", "running", "completed"} or command["mode"] != run.mode:
        raise AppError("CONFLICT", detail="Run state or mode does not permit submission.")
    item = db.scalar(
        select(ExperimentItem)
        .where(ExperimentItem.run_id == run.id, ExperimentItem.item_id == command["item_id"])
        .with_for_update()
    )
    if (
        not item
        or item.question_id != command["question_id"]
        or item.command_hash != digest(command)
    ):
        raise AppError(
            "CONFLICT", detail="Question differs from its preregistered item identity or hash."
        )
    if item.request_id:
        return answering.receipt(db, db.get(AnswerRequest, item.request_id))
    current = environment(db, settings, run.configuration_id)
    if any(current.get(key) != value for key, value in run.manifest["environment"].items()):
        run.state = "environment_changed"
        run.environment_change = {
            "detected_at": utcnow().isoformat(),
            "policy": "Stop affected new calls",
        }
        if commit:
            db.commit()
        raise AppError("SOURCE_UNAVAILABLE", detail="The frozen run environment changed.")
    expected_key = f"evaluation:{run.id}:{item.item_id}"
    if idempotency_key != expected_key:
        raise AppError("VALIDATION_FAILED", detail="Use the frozen run/item idempotency key.")
    route = f"/experiments/{run.id}/items/{item.item_id}"
    prior = answering.idempotent(db, actor.id, route, idempotency_key, command)
    if prior:
        item.request_id = prior.id
        return answering.receipt(db, prior)
    config = run.manifest["configuration"]
    retriever = (
        "R0"
        if run.condition == "E1"
        else run.condition
        if run.condition in {"R1", "R2", "R3"}
        else None
    )
    selected_configuration = (
        db.get(Configuration, run.configuration_id) if run.configuration_id else None
    )
    top_k = config.get(
        "top_k", selected_configuration.values.get("top_k") if selected_configuration else None
    )
    request_command = {
        "question": command["question_text"],
        "question_id": command["question_id"],
        "model_config": experiment_model_config(settings, selected_configuration),
        "condition": run.condition,
        "retriever": retriever,
        "top_k": top_k,
    }
    if run.mode == "benchmark_mcq":
        request_command["options"] = command["options"]
    req = AnswerRequest(
        owner_id=actor.id,
        route=route,
        idempotency_key=idempotency_key,
        body_hash=digest(command),
        mode=run.mode,
        response_schema="chat_response_v1" if run.mode == "benchmark_openqa" else "mcq_response_v1",
        session_id=None,
        user_message_id=None,
        context_snapshot_id=None,
        profile_snapshot_id=None,
        release_id=current["corpus_release_id"] if run.condition != "E0" else None,
        config_id=run.configuration_id,
        run_id=run.id,
        item_id=item.item_id,
        command=request_command,
        budget=RequestBudget(
            max_calls=settings.max_provider_calls,
            max_active_seconds=settings.request_timeout_seconds,
        ).to_dict(),
    )
    db.add(req)
    db.flush()
    job = Job(request_id=req.id, owner_id=actor.id, kind="answer")
    db.add(job)
    db.flush()
    item.request_id = req.id
    item.command = command
    item.state = "submitted"
    run.state = "running"
    if commit:
        db.commit()
    return answering.receipt(db, req, job)


def item_outcome(db, item):
    if not item.request_id:
        return {"status": item.state if item.state in TERMINAL else "pending"}
    job = db.scalar(
        select(Job).where(Job.request_id == item.request_id).order_by(Job.created_at.desc())
    )
    if not job or job.state in answering.ACTIVE:
        return {
            "status": "pending",
            "request_id": item.request_id,
            "job_id": job.id if job else None,
        }
    if job.state == "succeeded" and job.answer_id:
        answer = db.get(Answer, job.answer_id)
        result = answering.answer_out(db, answer)
        req = db.get(AnswerRequest, item.request_id)
        timings = dict(result["timing"] or {})
        # Scoped answer timings are authoritative. The historical request trace
        # aggregates preparation and retrieval, so it is not a stage replacement.
        if timings.get("preparation_ms") is None and not timings.get("timing_scope"):
            timings["preparation_ms"] = req.trace.get("preparation_ms")
            if timings["preparation_ms"] is not None:
                timings["preparation_timing_scope"] = "legacy_preparation_including_retrieval"
                timings["preparation_timing_source"] = "request_trace.preparation_ms"
        return {
            "status": "refused" if result["status"] == "refused" else "completed",
            "response": result["response"],
            "model_mode": result["model_mode"],
            "request_id": item.request_id,
            "job_id": job.id,
            "answer_id": answer.id,
            "evidence": result["evidence"],
            "timings": timings,
            "usage": req.trace.get("usage", {}),
            "context_snapshot_id": req.context_snapshot_id,
            "profile_snapshot_id": req.profile_snapshot_id,
        }
    return {
        "status": "cancelled" if job.state == "cancelled" else "error",
        "error": job.error,
        "request_id": item.request_id,
        "job_id": job.id,
    }


def refresh_run(db, run):
    if run.mode == "profile_study":
        return refresh_teaching(db, run)
    items = list(
        db.scalars(
            select(ExperimentItem)
            .where(ExperimentItem.run_id == run.id)
            .order_by(ExperimentItem.ordinal)
        )
    )
    for item in items:
        outcome = item_outcome(db, item)
        if outcome["status"] in TERMINAL:
            item.state = outcome["status"]
    if (
        run.state not in {"draft", "cancelled", "environment_changed"}
        and items
        and all(item.state in TERMINAL for item in items)
    ):
        run.state = "completed"
    db.flush()
    return items


def scores_for_item(db, actor, run_id, item_id, scores):
    require_admin(actor)
    item = db.scalar(
        select(ExperimentItem).where(
            ExperimentItem.run_id == run_id, ExperimentItem.item_id == item_id
        )
    )
    run = db.get(ExperimentRun, run_id)
    if not item or not run:
        raise AppError("NOT_FOUND")
    allowed = (
        {
            "em",
            "token_f1",
            "compact_answer_present",
            "valid_response",
            "lexically_scorable",
            "scorer_version",
        }
        if run.mode == "benchmark_openqa"
        else {"accuracy", "valid_response", "scorer_version"}
    )
    if set(scores) != allowed or scores.get("scorer_version") != run.spec["scorer_version"]:
        raise AppError(
            "VALIDATION_FAILED",
            detail="Only frozen scorer result fields are accepted; no evaluator references.",
        )
    for key, value in scores.items():
        if key == "scorer_version":
            continue
        if key in {"em", "token_f1", "accuracy"}:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise AppError("VALIDATION_FAILED")
        elif type(value) is not bool:
            raise AppError("VALIDATION_FAILED")
    outcome = item_outcome(db, item)
    if outcome["status"] not in TERMINAL:
        raise AppError("CONFLICT", detail="Only terminal scheduled items can receive scores.")
    if item.scores is not None and item.scores != scores:
        raise AppError(
            "CONFLICT", detail="Frozen item scores are immutable; use an explicit new scorer run."
        )
    item.scores = scores
    item.scoring_evidence = {
        "source": "separate_evaluator",
        "received_at": utcnow().isoformat(),
        "actor_id": actor.id,
    }
    db.flush()


def run_results(db, run):
    if run.mode == "profile_study":
        return teaching_results(db, run)
    items = refresh_run(db, run)
    metric_names = ("em", "token_f1") if run.mode == "benchmark_openqa" else ("accuracy",)
    scored = sum(item.scores is not None for item in items)
    return {
        "run_id": run.id,
        "protocol_id": run.protocol_id,
        "mode": run.mode,
        "condition": run.condition,
        "model_mode": run.model_mode,
        "status": run.state,
        "scheduled_count": run.scheduled_count,
        "outcome_counts": dict(Counter(item.state for item in items)),
        "scored_count": scored,
        "missing_scoring_count": run.scheduled_count - scored,
        "metrics": {
            name: sum((item.scores or {}).get(name, 0) for item in items) / run.scheduled_count
            if scored
            else None
            for name in metric_names
        },
        "provisional": scored != run.scheduled_count
        or any(item.state not in TERMINAL for item in items),
        "human_review": None,
        "items": [
            {
                "item_id": item.item_id,
                "question_id": item.question_id,
                "state": item.state,
                "request_id": item.request_id,
                "scores": item.scores,
                "outcome": item_outcome(db, item),
            }
            for item in items
        ],
    }


def create_teaching_run(db, settings, actor, base_answer_ids, *, seed=0, configuration_id=None):
    """Freeze all nine matched variants of each existing neutral grounded answer."""
    require_admin(actor)
    if (
        not base_answer_ids
        or len(base_answer_ids) > 1000
        or len(set(base_answer_ids)) != len(base_answer_ids)
    ):
        raise AppError("VALIDATION_FAILED", detail="Provide distinct saved base answer identities.")
    frozen_environment = environment(db, settings, configuration_id)
    run = ExperimentRun(
        owner_id=actor.id,
        protocol_id="profile_study",
        mode="profile_study",
        condition="C0_C1_C2",
        state="frozen",
        scheduled_count=len(base_answer_ids) * 9,
        configuration_id=configuration_id,
        model_mode=settings.model_mode,
        spec={
            "seed": seed,
            "base_answer_ids": base_answer_ids,
            "rubric_version": "profile_rating_v1",
            "profile_rule_version": COMPILER_VERSION,
            "scorer_version": "independent-human-review",
        },
    )
    db.add(run)
    db.flush()
    rows = []
    for base_id in base_answer_ids:
        answer = db.get(Answer, base_id)
        if (
            not answer
            or answer.response_schema != "chat_response_v1"
            or answer.response.get("response_type") != "answer"
        ):
            raise AppError(
                "VALIDATION_FAILED",
                detail="Teaching studies require a saved completed grounded base answer.",
            )
        request = answering.request_owned(db, answer.request_id, actor)
        profile = (
            db.get(Snapshot, request.profile_snapshot_id) if request.profile_snapshot_id else None
        )
        if profile and profile.payload.get("use_profile"):
            raise AppError(
                "VALIDATION_FAILED",
                detail="Select a neutral base generated with learner profile disabled.",
            )
        output = answering.answer_out(db, answer)
        evidence = output["evidence"]
        if (
            not evidence
            or not answer.response.get("citations")
            or not set(answer.response["citations"]) <= {e["evidence_id"] for e in evidence}
        ):
            raise AppError(
                "SOURCE_UNAVAILABLE",
                detail="All original base citations require available frozen source evidence.",
            )
        for passage in evidence:
            document = db.get(Document, passage["asset_id"])
            if not document or not document.active or document.revoked:
                raise AppError("SOURCE_UNAVAILABLE")
        if answer.model_mode != settings.model_mode:
            raise AppError(
                "CONFLICT",
                detail="The base and teaching workload must have the same mock/live evidence class.",
            )
        selected_configuration = (
            db.get(Configuration, configuration_id) if configuration_id else None
        )
        inputs = {
            "question_id": base_id,
            "question": request.command["question"],
            "base_answer": answer.response,
            "evidence": evidence,
            "model_config": experiment_model_config(settings, selected_configuration),
            "base_answer_hash": digest(answer.response),
            "evidence_hash": digest(evidence),
            "model_configuration_hash": frozen_environment["model_configuration_hash"],
        }
        for level in ("beginner", "intermediate", "advanced"):
            for condition in ("C0", "C1", "C2"):
                item_id = f"{base_id}:{level}:{condition}"
                item = TeachingStudyRecord(
                    run_id=run.id,
                    item_id=item_id,
                    owner_id=actor.id,
                    base_answer_id=base_id,
                    condition=condition,
                    target_level=level,
                    frozen_inputs=inputs,
                    input_hash=digest(inputs),
                    budget=RequestBudget(
                        max_calls=settings.max_provider_calls,
                        max_active_seconds=settings.request_timeout_seconds,
                    ).to_dict(),
                )
                db.add(item)
                rows.append(item)
    random.Random(seed).shuffle(rows)
    manifest = {
        "version": "teaching-study-run-v1",
        "run_id": run.id,
        "created_at": utcnow().isoformat(),
        "protocol_id": "profile_study",
        "mode": "profile_study",
        "scheduled_count": run.scheduled_count,
        "seed": seed,
        "environment": frozen_environment,
        "profile_rule_version": COMPILER_VERSION,
        "rubric_version": "profile_rating_v1",
        "response_schema": "teaching_study_response_v1",
        "base_answer_ids": base_answer_ids,
        "items": [
            {
                "item_id": item.item_id,
                "input_hash": item.input_hash,
                "condition": item.condition,
                "target_level": item.target_level,
            }
            for item in rows
        ],
        "evidence_class": "mock_software_rehearsal"
        if settings.model_mode == "mock"
        else "live_model_run",
    }
    manifest["manifest_hash"] = digest(manifest)
    run.manifest, run.manifest_hash = manifest, manifest["manifest_hash"]
    db.flush()
    return run


def teaching_job(db, item):
    return db.scalar(
        select(Job)
        .where(Job.kind == "teaching", Job.payload["teaching_id"].as_string() == item.id)
        .order_by(Job.created_at.desc())
    )


def validate_teaching_environment(db, settings, run, item):
    current = environment(db, settings, run.configuration_id)
    differences = {
        key: {"frozen": value, "current": current.get(key)}
        for key, value in run.manifest["environment"].items()
        if current.get(key) != value
    }
    if COMPILER_VERSION != run.manifest["profile_rule_version"]:
        differences["profile_rule_version"] = {
            "frozen": run.manifest["profile_rule_version"],
            "current": COMPILER_VERSION,
        }
    if differences or run.state == "environment_changed":
        run.state = "environment_changed"
        run.environment_change = {"changes": differences, "detected_at": utcnow().isoformat()}
        raise AppError(
            "SOURCE_UNAVAILABLE", detail="Frozen teaching configuration or sources changed."
        )
    if run.state == "cancelled" or item.state == "cancelled":
        raise answering.ExecutionCancelled()
    if digest(item.frozen_inputs) != item.input_hash:
        raise AppError("CONFLICT", detail="Frozen teaching input hash no longer matches.")
    for passage in item.frozen_inputs["evidence"]:
        document = db.get(Document, passage["asset_id"])
        if not document or not document.active or document.revoked:
            raise AppError(
                "SOURCE_UNAVAILABLE", detail="A frozen teaching source became unavailable."
            )
    return True


def start_teaching(db, settings, actor, run):
    require_admin(actor)
    if run.mode != "profile_study" or run.state not in {"frozen", "running"}:
        raise AppError("CONFLICT", detail="Start only a frozen or running teaching study.")
    rows = {item.item_id: item for item in refresh_teaching(db, run)}
    for identity in run.manifest["items"]:
        item = rows[identity["item_id"]]
        if item.state != "pending":
            continue
        validate_teaching_environment(db, settings, run, item)
        if teaching_job(db, item) is None:
            db.add(
                Job(
                    owner_id=actor.id,
                    kind="teaching",
                    payload={"teaching_id": item.id, "run_id": run.id},
                )
            )
        item.state = "submitted"
    run.state = "running"
    db.flush()


def refresh_teaching(db, run):
    items = list(
        db.scalars(
            select(TeachingStudyRecord)
            .where(TeachingStudyRecord.run_id == run.id)
            .order_by(TeachingStudyRecord.item_id)
        )
    )
    for item in items:
        job = teaching_job(db, item)
        if job and job.state in {"cancelled", "failed"} and item.state not in TERMINAL:
            item.state = "cancelled" if job.state == "cancelled" else "error"
            item.error = job.error
    if (
        run.state not in {"cancelled", "environment_changed"}
        and items
        and all(item.state in TERMINAL for item in items)
    ):
        run.state = "completed"
    db.flush()
    return items


def cancel_teaching(db, run):
    for item in refresh_teaching(db, run):
        if item.state not in TERMINAL:
            job = teaching_job(db, item)
            if job:
                answering.cancel_job(db, job)
            item.state = "cancelled"
    run.state = "cancelled"


def interrupt_teaching(db, job, error):
    """Worker recovery/cancellation hook; preserves every recorded call charge."""
    item = db.get(TeachingStudyRecord, job.payload.get("teaching_id"))
    if item and item.state not in TERMINAL:
        item.state = "cancelled" if job.state == "cancelled" else "error"
        item.error = error
        run = db.get(ExperimentRun, item.run_id)
        if run:
            refresh_teaching(db, run)
    db.flush()


def teaching_results(db, run):
    items = refresh_teaching(db, run)
    outputs = []
    for item in items:
        frozen = item.frozen_inputs
        available = all(
            (doc := db.get(Document, evidence["asset_id"])) and not doc.revoked
            for evidence in frozen["evidence"]
        )
        job = teaching_job(db, item)
        outputs.append(
            {
                "item_id": item.item_id,
                "question_id": frozen["question_id"],
                "question": frozen["question"],
                "target_level": item.target_level,
                "condition": item.condition,
                "base_answer_hash": frozen["base_answer_hash"],
                "evidence_hash": frozen["evidence_hash"],
                "model_configuration_hash": frozen["model_configuration_hash"],
                "status": item.state,
                "response": item.response if available else None,
                "explanation": item.response.get("explanation")
                if item.response and available
                else None,
                "evidence": frozen["evidence"] if available else [],
                "source_available": available,
                "error": item.error,
                "job_id": job.id if job else None,
                "budget": item.budget,
                "execution": job.payload.get("outcome") if job else None,
            }
        )
    return {
        "run_id": run.id,
        "protocol_id": run.protocol_id,
        "mode": run.mode,
        "condition": run.condition,
        "status": run.state,
        "model_mode": run.model_mode,
        "scheduled_count": run.scheduled_count,
        "outcome_counts": dict(Counter(item.state for item in items)),
        "provisional": any(item.state not in TERMINAL for item in items),
        "human_review": None,
        "metrics": None,
        "items": outputs,
    }


def execute_teaching(engine, settings, job_id, token):
    """Execute one independent item with the shared teaching service and fence."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def finish_failure(db, job, item, error):
        answering.fail_job(db, job, error)
        item.state, item.error = "error", error
        db.commit()

    with factory() as db:
        job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if not job or job.execution_token != token or job.state != "running":
            return
        item = db.get(TeachingStudyRecord, job.payload["teaching_id"])
        run = db.get(ExperimentRun, item.run_id)
        try:
            validate_teaching_environment(db, settings, run, item)
            # A claimed call without a durable result is ambiguous, never a free retry.
            if item.budget.get("consumed_calls", 0):
                raise AppError(
                    "CONFLICT",
                    detail="A prior teaching attempt is uncertain; retain it and create a new explicit run.",
                )
        except AppError as exc:
            finish_failure(db, job, item, {"code": exc.code, "message": exc.detail, "details": {}})
            return
        item.state, job.stage = "running", "teaching"
        frozen = dict(item.frozen_inputs)
        budget = RequestBudget.from_dict(item.budget)
        values = {
            "run_id": run.id,
            "item_id": item.item_id,
            "condition": item.condition,
            "target_level": item.target_level,
        }
        db.commit()

    def on_attempt(event):
        with factory() as db:
            job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if job.execution_token != token or job.state != "running":
                raise answering.ExecutionCancelled()
            item = db.get(TeachingStudyRecord, job.payload["teaching_id"])
            item.budget = event.get("budget", item.budget)
            job.updated_at = utcnow()
            count = db.scalar(
                select(func.count()).select_from(Attempt).where(Attempt.job_id == job_id)
            )
            db.add(Attempt(job_id=job_id, sequence=count + 1, payload=event))
            db.commit()

    result = TeachingStudyService().generate(
        **values,
        base_answer=frozen["base_answer"],
        evidence=frozen["evidence"],
        config=ModelConfig.from_dict(frozen["model_config"]),
        budget=budget,
        on_attempt=on_attempt,
    )
    with factory() as db:
        job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if job.execution_token != token or job.state != "running":
            return
        item = db.get(TeachingStudyRecord, job.payload["teaching_id"])
        run = db.get(ExperimentRun, item.run_id)
        item.budget = result["budget"]
        job.payload = {
            **job.payload,
            "outcome": {
                key: result.get(key)
                for key in (
                    "usage",
                    "latency_ms",
                    "token_budget",
                    "model_mode",
                    "provider",
                    "model",
                )
            },
        }
        try:
            validate_teaching_environment(db, settings, run, item)
        except AppError as exc:
            finish_failure(db, job, item, {"code": exc.code, "message": exc.detail, "details": {}})
            return
        if result["error"]:
            finish_failure(db, job, item, result["error"])
            return
        item.state, item.response = "completed", result["response"]
        job.state, job.stage, job.execution_token = "succeeded", "complete", None
        refresh_teaching(db, run)
        db.commit()


class DatabaseAnswerBackend:
    def __init__(
        self,
        settings,
        engine,
        *,
        actor_id=None,
        actor_email=None,
        configuration_id=None,
        inline_worker=False,
    ):
        self.settings, self.engine = settings, engine
        self.factory = sessionmaker(bind=engine, expire_on_commit=False)
        self.configuration_id = configuration_id
        self.inline_worker = inline_worker
        with self.factory() as db:
            actor = (
                db.get(User, actor_id)
                if actor_id
                else db.scalar(select(User).where(User.email == actor_email))
                if actor_email
                else None
            )
            if actor is None and settings.env in {"dev", "test", "demo"}:
                actor = db.scalar(select(User).where(User.email == "admin@example.com"))
            require_admin(actor)
            self.actor_id = actor.id

    def _actor(self, db):
        actor = db.get(User, self.actor_id)
        require_admin(actor)
        return actor

    def environment(self):
        with self.factory() as db:
            self._actor(db)
            return environment(db, self.settings, self.configuration_id)

    def register_run(self, manifest):
        with self.factory() as db:
            run = register_manifest(db, self.settings, self._actor(db), manifest)
            db.commit()
            return {"run_id": run.id}

    def submit(self, command, *, run_context, idempotency_key):
        with self.factory() as db:
            return submit_item(
                db, self.settings, self._actor(db), command, run_context, idempotency_key
            )

    def poll(self, receipt):
        if self.inline_worker:
            # Explicit local rehearsal only; production uses the single worker process.
            from app.worker import run_once

            run_once(self.engine, self.settings)
        with self.factory() as db:
            self._actor(db)
            job = db.get(Job, receipt["job_id"])
            if not job or job.request_id != receipt["request_id"]:
                raise AppError("NOT_FOUND")
            item = db.scalar(
                select(ExperimentItem).where(ExperimentItem.request_id == job.request_id)
            )
            if not item:
                raise AppError("NOT_FOUND")
            result = item_outcome(db, item)
            refresh_run(db, db.get(ExperimentRun, item.run_id))
            db.commit()
            return result

    def lookup(self, idempotency_key):
        with self.factory() as db:
            actor = self._actor(db)
            req = db.scalar(
                select(AnswerRequest).where(
                    AnswerRequest.owner_id == actor.id,
                    AnswerRequest.idempotency_key == idempotency_key,
                    AnswerRequest.run_id.is_not(None),
                )
            )
            return answering.receipt(db, req) if req else None

    def record_scores(self, run_id, item_id, scores):
        with self.factory() as db:
            scores_for_item(db, self._actor(db), run_id, item_id, scores)
            db.commit()

    def cancel(self, receipt):
        with self.factory() as db:
            self._actor(db)
            job = db.get(Job, receipt["job_id"])
            if not job or job.request_id != receipt["request_id"]:
                raise AppError("NOT_FOUND")
            answering.cancel_job(db, job)
            db.commit()

    def cancel_run(self, run_id):
        with self.factory() as db:
            self._actor(db)
            run = db.scalar(
                select(ExperimentRun).where(ExperimentRun.id == run_id).with_for_update()
            )
            if not run:
                raise AppError("NOT_FOUND")
            if run.mode == "profile_study":
                cancel_teaching(db, run)
                db.commit()
                return
            for item in refresh_run(db, run):
                if item.state in TERMINAL:
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


def create_backend(options):
    allowed = {"database_url", "actor_id", "actor_email", "configuration_id", "inline_worker"}
    if set(options) - allowed:
        raise ValueError(
            "Unknown backend options; provider settings come from the application environment"
        )
    settings = Settings()
    if options.get("database_url"):
        settings.database_url = options["database_url"]
    inline = options.get("inline_worker", False)
    if inline and (settings.env not in {"dev", "test", "demo"} or settings.model_mode != "mock"):
        raise ValueError("Inline worker is only permitted for explicit local mock rehearsal")
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False}
        if settings.database_url.startswith("sqlite")
        else {},
        pool_pre_ping=True,
    )
    return DatabaseAnswerBackend(
        settings,
        engine,
        actor_id=options.get("actor_id"),
        actor_email=options.get("actor_email"),
        configuration_id=options.get("configuration_id"),
        inline_worker=inline,
    )
