"""Four preregistered real-SciQ/E5 conditions with a mock answer provider."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from sqlalchemy import select
from app.modules.answering.models import Answer, AnswerRequest, Job
from app.modules.experiment.bridge import create_backend
from evaluation.datasets.sciq import load_split
from evaluation.runner import EvaluationRun, validate_public


REVISION = "2c94ad3e1aafab77146f384e23536f97a4849815"
FULL_VALIDATION_HASH = "cf71bab38a36bdc7b6a81b6ebd20c0ab19d385b4abe1b833bd9b723ce90c1147"
PRIVATE = Path("artifacts/evaluator-private/sciq") / REVISION / "technical_rehearsal"
PLAN = Path("evidence/sciq/technical-rehearsal-plan.json")
REPORT = Path("evidence/sciq/technical-rehearsal.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute"))
    args = parser.parse_args()
    if args.action == "plan":
        if PLAN.exists():
            raise RuntimeError("The preregistered plan already exists; do not resample")
        source = PRIVATE.parent / "derived/json_ascii_v1/validation.jsonl"
        if sha(source) != FULL_VALIDATION_HASH:
            raise RuntimeError("The acquired full validation source differs")
        lines = source.read_bytes().splitlines(keepends=True)
        if len(lines) != 1000:
            raise RuntimeError("The acquired full validation split must retain all1000 rows")
        PRIVATE.mkdir(parents=True, exist_ok=True)
        prefix = PRIVATE / "validation-first2-exact.jsonl"
        if prefix.exists():
            raise RuntimeError("The exact source prefix already exists")
        prefix.write_bytes(b"".join(lines[:2]))
        try:
            records, _ = load_split(prefix, split="validation", revision=REVISION)
            preflight = {"status": "ready", "count": len(records)}
        except ValueError as exc:
            preflight = {"status": "blocked_native_input", "error": str(exc)}
        value = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": "actual-sciq-prefix-technical-rehearsal-v1",
            "dataset_revision": REVISION,
            "source_split": "validation",
            "full_source_sha256": sha(source),
            "full_source_count": 1000,
            "selection": "Exactly source row indexes0 and1 before any output inspection; no filtering or replacement.",
            "prefix_path": prefix.as_posix(),
            "prefix_sha256": sha(prefix),
            "prefix_count": 2,
            "preflight": preflight,
            "full_split_mcq_native_rejections": 3,
            "full_dataset_mcq_native_rejections": 48,
            "full_split_policy": "The unmodified full MCQ split still fails strict whole-split preflight. This explicitly labeled2-row source prefix does not change that policy or act as a default formal dataset.",
            "runs": [
                {"protocol_id": mode, "run_id": uuid4().hex} for mode in ("sciq_openqa", "sciq_mcq")
            ],
            "scheduled_count": 4,
            "condition": "E1",
            "retrieval_variant": "R0",
            "top_k": 5,
            "seed": 5703,
            "answer_model_mode": "mock",
            "formal_quality_metrics": None,
            "human_scores": None,
        }
        validate_public(value)
        write(PLAN, value)
        print(
            json.dumps(
                {"planned": 4, "preflight": preflight, "prefix_sha256": value["prefix_sha256"]}
            )
        )
        return
    if REPORT.exists():
        raise RuntimeError("Preserve the prior rehearsal; reconcile existing run IDs explicitly")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (
        plan["preflight"]["status"] != "ready"
        or sha(Path(plan["prefix_path"])) != plan["prefix_sha256"]
    ):
        raise RuntimeError("The fixed native prefix is not valid and unchanged; no substitution")
    backend = create_backend({"actor_email": "admin@example.com", "inline_worker": False})
    environment = backend.environment()
    if environment["model_mode"] != "mock" or not environment["corpus_release_id"]:
        raise RuntimeError("A real active corpus and explicit mock answer provider are required")
    frozen = []
    for item in plan["runs"]:
        configuration = {
            "protocol_id": item["protocol_id"],
            "run_id": item["run_id"],
            "condition": "E1",
            "dataset_path": plan["prefix_path"],
            "dataset_revision": REVISION,
            "split": "validation",
            "expected_count": 2,
            "expected_dataset_sha256": plan["prefix_sha256"],
            "private_root": str((PRIVATE / "runs").resolve()),
            "backend_factory": "app.modules.experiment.bridge:create_backend",
            "backend_options": {"actor_email": "admin@example.com", "inline_worker": False},
            "configuration": {"top_k": 5},
            "seed": 5703,
            "full_response_review_count": 0,
            "code_revision": "answer-service-sha256:"
            + sha(Path("backend/app/modules/answering/service.py")),
        }
        frozen.append(EvaluationRun.freeze(configuration, backend))
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "plan_sha256": sha(PLAN),
        "environment": environment,
        "scheduled_count": 4,
        "results": [],
        "model_mode": "mock",
        "provider_reported_usage": None,
        "formal_quality_metrics": None,
        "human_scores": None,
        "scope": "Actual official source prefix and real E5/PostgreSQL; deterministic mock answers. Existing evaluator lexical scores remain classified as mock software diagnostics and are not promoted to benchmark accuracy.",
    }
    for run in frozen:
        report["results"].extend(
            {
                "run_id": run.manifest["run_id"],
                "protocol_id": run.manifest["protocol_id"],
                "item_id": i["item_id"],
                "question_id": i["question_id"],
                "status": "pending",
            }
            for i in run.state["items"]
        )
    write(REPORT, report)
    for run in frozen:
        started = time.perf_counter()
        while time.perf_counter() - started < 240:
            state = run.advance(backend, max_new_submissions=1)
            if state["status"] != "running":
                break
            time.sleep(0.25)
        elapsed = time.perf_counter() - started
        for item in run.state["items"]:
            row = next(
                r
                for r in report["results"]
                if r["run_id"] == run.manifest["run_id"] and r["item_id"] == item["item_id"]
            )
            row.update(
                status=item["status"],
                run_wall_seconds=round(elapsed, 6),
                manifest_hash=run.manifest["manifest_hash"],
            )
            starts = [e["at"] for e in item["events"] if e["event"] == "submit_started"]
            terminals = [e["at"] for e in item["events"] if e["event"] == "terminal"]
            row["observer_wall_seconds"] = (
                (
                    datetime.fromisoformat(terminals[-1]) - datetime.fromisoformat(starts[0])
                ).total_seconds()
                if starts and terminals
                else None
            )
            row["observer_wall_scope"] = (
                "Evaluator submit-start to observed terminal; includes queue and polling delay, distinct from worker stage timing."
            )
            if item.get("receipt"):
                with backend.factory() as db:
                    req = db.get(AnswerRequest, item["receipt"]["request_id"])
                    job = db.get(Job, item["receipt"]["job_id"])
                    answer = db.scalar(select(Answer).where(Answer.request_id == req.id))
                    validate_public(req.command)
                    row.update(
                        request_id=req.id,
                        job_id=job.id,
                        answer_id=answer.id if answer else None,
                        release_id=req.release_id,
                        job_state=job.state,
                        error=job.error,
                        answer_timing=answer.timing if answer else None,
                        answer_model_mode=answer.model_mode if answer else None,
                        provider_reported_usage=None,
                        request_budget=req.budget,
                        isolated_from_chat=all(
                            getattr(req, k) is None
                            for k in (
                                "session_id",
                                "user_message_id",
                                "assistant_message_id",
                                "context_snapshot_id",
                                "profile_snapshot_id",
                            )
                        ),
                        private_fields_absent_from_backend_command=True,
                    )
            write(REPORT, report)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = (
        "completed"
        if all(
            r["status"] in ("completed", "refused", "error", "cancelled", "invalid", "incomplete")
            for r in report["results"]
        )
        else "needs_reconciliation"
    )
    validate_public(report)
    write(REPORT, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "scheduled": 4,
                "outcomes": [r["status"] for r in report["results"]],
            }
        )
    )


if __name__ == "__main__":
    main()
