"""Freeze and resume offline evaluations through the application's answer backend.

Examples: python -m evaluation.runner --config configs/evaluation/mock_openqa.json
          python -m evaluation.runner --resume RUN_ID --root evaluation/private_runs
"""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from contracts.models import ChatResponseV1, MCQCommand, MCQResponseV1, OpenQACommand
from evaluation.bridge import AnswerBackend, TERMINAL_STATUSES, check_outcome
from evaluation.common import atomic_json, canonical, fingerprint, read_json, utc_now
from evaluation.datasets.sciq import load_split, mcq_projection
from evaluation.datasets.sciq_openqa import openqa_projection
from evaluation.metrics.scoring import SCORER_VERSION, aggregate, score_outcome

PROTOCOL_MODES = {"sciq_openqa": "benchmark_openqa", "sciq_mcq": "benchmark_mcq"}
PRIVATE_KEYS = {
    "correct_answer",
    "correct_label",
    "reference",
    "references",
    "support",
    "distractors",
    "distractor1",
    "distractor2",
    "distractor3",
    "gold",
    "evidence_status",
    "api_key",
    "secret_key",
    "password",
    "access_token",
}


def validate_public(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in PRIVATE_KEYS:
                raise ValueError(f"Private field cannot enter public run configuration: {key}")
            validate_public(child)
    elif isinstance(value, list):
        for child in value:
            validate_public(child)


def load_backend(factory: str, options: dict) -> AnswerBackend:
    if ":" not in factory:
        raise ValueError("Backend factory must use trusted module:function syntax")
    module_name, callable_name = factory.split(":", 1)
    factory_callable = getattr(importlib.import_module(module_name), callable_name)
    return factory_callable(options)


@contextmanager
def run_lock(directory: Path):
    path = directory / "runner.lock"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "Evaluator run is locked; inspect the recorded process before explicit stale-lock removal"
        ) from exc
    try:
        os.write(handle, canonical({"pid": os.getpid(), "acquired_at": utc_now()}).encode())
        os.close(handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def _safe_run_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,36}", value):
        raise ValueError("Run ID must be a safe single directory component")
    return value


class EvaluationRun:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        recorded = self.manifest["manifest_hash"]
        if (
            fingerprint(
                {key: value for key, value in self.manifest.items() if key != "manifest_hash"}
            )
            != recorded
        ):
            raise ValueError("Frozen manifest was changed")
        private_index = read_json(self.directory / "private_index.json")
        self.commands = read_json(self.directory / "commands.json")
        self.references = read_json(self.directory / "references.json")
        if (
            fingerprint(self.commands) != private_index["commands_hash"]
            or fingerprint(self.references) != private_index["references_hash"]
        ):
            raise ValueError("Frozen evaluator commands/references were changed")
        self._commands = {row["item_id"]: row for row in self.commands}
        self._references = {row["item_id"]: row for row in self.references}
        if len(self._commands) != self.manifest["scheduled_count"] or set(self._commands) != set(
            self._references
        ):
            raise ValueError("Frozen items and private reference identities do not reconcile")
        for item in self.manifest["items"]:
            if fingerprint(self._commands[item["item_id"]]) != item["command_hash"]:
                raise ValueError("Command differs from its publicly frozen identity")
        self.state = read_json(self.directory / "state.json")
        if {row["item_id"] for row in self.state["items"]} != set(self._commands) or len(
            self.state["items"]
        ) != len(self._commands):
            raise ValueError("Mutable result ledger changed the scheduled item set")

    @classmethod
    def freeze(
        cls,
        config: dict,
        backend: AnswerBackend,
        *,
        base_directory: Path = Path("."),
        allow_live: bool = False,
    ) -> "EvaluationRun":
        protocol = config["protocol_id"]
        if protocol not in PROTOCOL_MODES:
            raise ValueError(
                "The lexical runner requires sciq_openqa or sciq_mcq; conversation/profile tools are separate"
            )
        mode = PROTOCOL_MODES[protocol]
        condition = config.get("condition", "E1")
        if condition not in {"E0", "E1", "R1", "R2", "R3"}:
            raise ValueError("Unknown evaluation condition")
        configuration = dict(config.get("configuration", {}))
        if condition == "E1" and configuration.get("retrieval_variant", "R0") != "R0":
            raise ValueError("E1 is frozen basic R0 dense retrieval")
        if condition == "E0" and configuration.get("retrieval_variant", "none") != "none":
            raise ValueError("E0 must disable retrieval")
        configuration.update(history_policy="empty", profile_policy="off", summary_policy="empty")
        validate_public(configuration)
        actual_environment = backend.environment()
        environment = dict(config.get("environment", actual_environment))
        if not environment:
            raise ValueError("A frozen actual application environment is required")
        if actual_environment.get("model_mode") not in {"mock", "live"}:
            raise ValueError("The shared backend must report its actual mock/live mode")
        environment.setdefault("model_mode", actual_environment["model_mode"])
        validate_public(environment)
        if any(actual_environment.get(key) != value for key, value in environment.items()):
            raise ValueError("Configured environment does not match the actual shared backend")
        if actual_environment.get("model_mode") != "mock" and not allow_live:
            raise ValueError(
                "This execution is mock-only; live operation needs an explicit CLI flag and prior authorization"
            )
        records, dataset = load_split(
            base_directory / config["dataset_path"],
            split=config["split"],
            revision=config["dataset_revision"],
            require_distinct_choices=mode == "benchmark_mcq",
        )
        if (
            config.get("expected_dataset_sha256", dataset["sha256"]) != dataset["sha256"]
            or config.get("expected_count", dataset["count"]) != dataset["count"]
        ):
            raise ValueError(
                "Dataset identity/count differs from the configured acquisition manifest"
            )
        limit = config.get("limit")
        if limit is not None:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= len(records)
            ):
                raise ValueError("Limit must select a positive bounded preregistered prefix")
            records = records[:limit]
        run_id = _safe_run_id(config.get("run_id", uuid.uuid4().hex))
        commands, references = [], []
        seed = config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Seed must be an integer")
        for index, record in enumerate(records):
            item_id = f"item_{index:06}"
            if mode == "benchmark_openqa":
                command, reference = openqa_projection(record, run_id=run_id, item_id=item_id)
                OpenQACommand.model_validate(command)
            else:
                command, reference = mcq_projection(
                    record, run_id=run_id, item_id=item_id, seed=seed
                )
                MCQCommand.model_validate(command)
            commands.append(command)
            references.append({"item_id": item_id, **reference})
        review_count = config.get("full_response_review_count", min(len(records), 20))
        if not isinstance(review_count, int) or not 0 <= review_count <= len(records):
            raise ValueError("Full-response review count must be within the scheduled sample")
        import random

        review_ids = sorted(
            random.Random(seed).sample([item["item_id"] for item in commands], review_count)
        )
        manifest = {
            "version": "evaluation-run-v1",
            "run_id": run_id,
            "created_at": utc_now(),
            "protocol_id": protocol,
            "mode": mode,
            "condition": condition,
            "dataset": dataset,
            "seed": seed,
            "scheduled_count": len(commands),
            "items": [
                {
                    "item_id": command["item_id"],
                    "question_id": command["question_id"],
                    "command_hash": fingerprint(command),
                }
                for command in commands
            ],
            "environment": environment,
            "configuration": configuration,
            "scorer_version": SCORER_VERSION,
            "rubric_version": config.get("rubric_version", "review-v1"),
            "code_revision": config["code_revision"],
            "reference_policy": "reject_invalid_before_freeze",
            "full_response_review_item_ids": review_ids,
            "evidence_class": "mock_software_rehearsal"
            if actual_environment.get("model_mode") == "mock"
            else "live_model_run",
        }
        validate_public(manifest)
        manifest["manifest_hash"] = fingerprint(manifest)
        directory = (
            base_directory
            / os.environ.get(
                "EVALUATION_PRIVATE_ROOT", config.get("private_root", "evaluation/private_runs")
            )
            / run_id
        )
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "manifest.json", manifest, immutable=True)
        atomic_json(directory / "commands.json", commands, immutable=True)
        atomic_json(directory / "references.json", references, immutable=True)
        atomic_json(
            directory / "private_index.json",
            {"commands_hash": fingerprint(commands), "references_hash": fingerprint(references)},
            immutable=True,
        )
        atomic_json(
            directory / "backend.json",
            {"factory": config["backend_factory"], "options": config.get("backend_options", {})},
            immutable=True,
        )
        atomic_json(
            directory / "state.json",
            {
                "version": "evaluation-state-v1",
                "run_id": run_id,
                "registered": False,
                "status": "frozen",
                "updated_at": utc_now(),
                "items": [
                    {
                        "item_id": command["item_id"],
                        "question_id": command["question_id"],
                        "status": "pending",
                        "idempotency_key": f"evaluation:{run_id}:{command['item_id']}",
                        "receipt": None,
                        "outcome": None,
                        "scores": None,
                        "events": [],
                    }
                    for command in commands
                ],
            },
        )
        return cls(directory)

    def _save(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_json(self.directory / "state.json", self.state)

    def _environment_matches(self, backend: AnswerBackend) -> bool:
        actual = backend.environment()
        return all(actual.get(key) == value for key, value in self.manifest["environment"].items())

    def _publish_scores(self, backend: AnswerBackend, item: dict) -> None:
        if (
            not item.get("scores")
            or item.get("scores_published")
            or not hasattr(backend, "record_scores")
        ):
            return
        try:
            backend.record_scores(self.manifest["run_id"], item["item_id"], item["scores"])
            item["scores_published"] = True
        except Exception as exc:
            item["events"].append(
                {
                    "at": utc_now(),
                    "event": "score_publication_pending",
                    "error_type": type(exc).__name__,
                }
            )
        self._save()

    def advance(
        self,
        backend: AnswerBackend,
        *,
        max_new_submissions: int | None = None,
        allow_live: bool = False,
    ) -> dict:
        with run_lock(self.directory):
            # Read under the lock so another completed invocation cannot be lost.
            self.state = read_json(self.directory / "state.json")
            if self.manifest["environment"].get("model_mode") != "mock" and not allow_live:
                raise ValueError("Live execution is not authorized by the mock-only runner")
            if self.state["status"] in {"cancelled", "environment_changed"}:
                return self.state
            if not self._environment_matches(backend):
                self.state["status"] = "environment_changed"
                self.state["environment_change"] = {
                    "detected_at": utc_now(),
                    "action": "Stopped affected new calls; existing outcomes preserved",
                }
                self._save()
                return self.state
            if not self.state["registered"]:
                registered = backend.register_run(self.manifest)
                if registered.get("run_id") != self.manifest["run_id"]:
                    raise ValueError("Shared backend changed the frozen run identity")
                self.state["registered"] = True
                self._save()
            submissions = 0
            for item in self.state["items"]:
                if item["status"] in TERMINAL_STATUSES:
                    self._publish_scores(backend, item)
                    continue
                if item["status"] in {"submitting", "uncertain"}:
                    receipt = (
                        backend.lookup(item["idempotency_key"])
                        if hasattr(backend, "lookup")
                        else None
                    )
                    if receipt:
                        item.update(receipt=receipt, status="submitted")
                        self._save()
                    else:
                        item["status"] = "uncertain"
                        self._save()
                        continue
                if item["status"] == "pending":
                    if max_new_submissions is not None and submissions >= max_new_submissions:
                        continue
                    if not self._environment_matches(backend):
                        self.state["status"] = "environment_changed"
                        self._save()
                        return self.state
                    item["status"] = "submitting"
                    item["events"].append({"at": utc_now(), "event": "submit_started"})
                    self._save()
                    try:
                        receipt = backend.submit(
                            self._commands[item["item_id"]],
                            run_context=self.manifest,
                            idempotency_key=item["idempotency_key"],
                        )
                        if not receipt.get("request_id") or not receipt.get("job_id"):
                            raise ValueError(
                                "Durable submission receipt needs request_id and job_id"
                            )
                    except Exception as exc:
                        item["status"] = "uncertain"
                        item["events"].append(
                            {
                                "at": utc_now(),
                                "event": "submission_unconfirmed",
                                "error_type": type(exc).__name__,
                            }
                        )
                        self._save()
                        continue
                    item.update(receipt=receipt, status="submitted")
                    submissions += 1
                    self._save()
                try:
                    outcome = check_outcome(backend.poll(item["receipt"]))
                except Exception as exc:
                    item["events"].append(
                        {
                            "at": utc_now(),
                            "event": "poll_unavailable",
                            "error_type": type(exc).__name__,
                        }
                    )
                    self._save()
                    continue
                if outcome["status"] == "pending":
                    continue
                if outcome["status"] in {"completed", "refused"}:
                    try:
                        response_model = (
                            ChatResponseV1
                            if self.manifest["mode"] == "benchmark_openqa"
                            else MCQResponseV1
                        )
                        response_model.model_validate(outcome.get("response"))
                    except (ValueError, TypeError):
                        outcome = {
                            **outcome,
                            "status": "invalid",
                            "error": {
                                "code": "EVALUATION_RESPONSE_CONTRACT",
                                "message": "Shared backend outcome did not match its frozen response schema",
                            },
                        }
                item["status"] = outcome["status"]
                item["outcome"] = outcome
                item["scores"] = score_outcome(
                    self.manifest["mode"],
                    outcome,
                    self._references[item["item_id"]],
                    self._commands[item["item_id"]],
                )
                item["events"].append(
                    {"at": utc_now(), "event": "terminal", "status": outcome["status"]}
                )
                self._save()
                self._publish_scores(backend, item)
            self.state["status"] = (
                "completed"
                if all(item["status"] in TERMINAL_STATUSES for item in self.state["items"])
                else "needs_reconciliation"
                if any(item["status"] == "uncertain" for item in self.state["items"])
                else "running"
            )
            self._save()
            return self.state

    def cancel(self, backend: AnswerBackend | None = None) -> None:
        with run_lock(self.directory):
            self.state = read_json(self.directory / "state.json")
            if backend is not None and self.state["registered"] and hasattr(backend, "cancel_run"):
                backend.cancel_run(self.manifest["run_id"])
            for item in self.state["items"]:
                if item["status"] in TERMINAL_STATUSES:
                    continue
                if item.get("receipt") and backend is not None and hasattr(backend, "cancel"):
                    backend.cancel(item["receipt"])
                item["status"] = "cancelled"
                item["outcome"] = {
                    "status": "cancelled",
                    "error": {
                        "code": "EVALUATOR_CANCELLED",
                        "message": "Evaluator stopped; a submitted server job may require separate cancellation",
                    },
                }
                item["scores"] = score_outcome(
                    self.manifest["mode"],
                    item["outcome"],
                    self._references[item["item_id"]],
                    self._commands[item["item_id"]],
                )
            self.state["status"] = "cancelled"
            self._save()

    def export(self, root: str | Path) -> Path:
        destination = Path(root) / self.manifest["protocol_id"] / self.manifest["run_id"]
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in self.state["items"]:
            command = self._commands[item["item_id"]]
            outcome = item.get("outcome") or {}
            rows.append(
                {
                    "item_id": item["item_id"],
                    "question_id": item["question_id"],
                    "question_text": command["question_text"],
                    "status": item["status"],
                    "response": outcome.get("response"),
                    "scores": item.get("scores") or {},
                    "outcome": outcome,
                    "request_id": (item.get("receipt") or {}).get("request_id"),
                    "job_id": (item.get("receipt") or {}).get("job_id"),
                }
            )
        summary = aggregate(self.manifest["mode"], rows)
        summary.update(
            run_id=self.manifest["run_id"],
            condition=self.manifest["condition"],
            protocol_id=self.manifest["protocol_id"],
            run_status=self.state["status"],
            evidence_class=self.manifest["evidence_class"],
            provisional=any(row["status"] not in TERMINAL_STATUSES for row in rows),
            human_review_completed=False,
        )
        if (destination / "manifest.json").exists():
            if read_json(destination / "manifest.json") != self.manifest:
                raise ValueError("Export path already belongs to a different frozen manifest")
        else:
            atomic_json(destination / "manifest.json", self.manifest, immutable=True)
        atomic_json(destination / "aggregate.json", summary)
        (destination / "items.jsonl").write_text(
            "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
        )
        columns = [
            "item_id",
            "question_id",
            "status",
            "short_answer",
            "answer",
            "em",
            "token_f1",
            "accuracy",
            "request_id",
            "job_id",
            "error_code",
        ]
        with (destination / "items.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                response = row["response"] or {}
                writer.writerow(
                    {
                        "item_id": row["item_id"],
                        "question_id": row["question_id"],
                        "status": row["status"],
                        "short_answer": response.get("short_answer"),
                        "answer": response.get("answer"),
                        "em": row["scores"].get("em"),
                        "token_f1": row["scores"].get("token_f1"),
                        "accuracy": row["scores"].get("accuracy"),
                        "request_id": row["request_id"],
                        "job_id": row["job_id"],
                        "error_code": (row["outcome"].get("error") or {}).get("code"),
                    }
                )
        review_rows = [
            {
                "item_id": row["item_id"],
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "response": row["response"],
                "status": row["status"],
                "correctness": None,
                "groundedness": None,
                "reviewer_id": None,
            }
            for row in rows
            if row["item_id"] in self.manifest["full_response_review_item_ids"]
        ]
        review_path = destination / "full_response_review.json"
        if review_path.exists():
            prior = read_json(review_path)
            prior_rows = {row["item_id"]: row for row in prior["reviews"]}
            if (
                set(prior_rows) != {row["item_id"] for row in review_rows}
                or prior["rubric_version"] != self.manifest["rubric_version"]
            ):
                raise ValueError("Existing review belongs to a different frozen subset or rubric")
            for row in review_rows:
                previous = prior_rows[row["item_id"]]
                rating_fields = ("correctness", "groundedness", "reviewer_id", "notes")
                if (
                    any(previous.get(name) is not None for name in rating_fields)
                    and previous["response"] != row["response"]
                ):
                    raise ValueError(
                        "A reviewed response changed; preserve it and create a separate explicit review"
                    )
                row.update({name: previous[name] for name in rating_fields if name in previous})
        atomic_json(
            review_path,
            {
                "rubric_version": self.manifest["rubric_version"],
                "selected_before_execution": True,
                "reviews": review_rows,
            },
        )
        return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--config", type=Path)
    actions.add_argument("--resume")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("EVALUATION_PRIVATE_ROOT", "evaluation/private_runs")),
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path(os.environ.get("EVALUATION_EXPORT_ROOT", "evaluation/exports")),
    )
    parser.add_argument("--once", action="store_true", help="Submit/poll one nonblocking pass")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Only use after explicit authorization for the configured live run",
    )
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or not 0.05 <= args.poll_interval <= 60:
        parser.error("Timeout must be positive; poll interval must be 0.05–60 seconds")
    if args.config:
        config = read_json(args.config)
        backend = load_backend(config["backend_factory"], config.get("backend_options", {}))
        run = EvaluationRun.freeze(
            config, backend, base_directory=args.config.resolve().parent, allow_live=args.allow_live
        )
    else:
        run = EvaluationRun(args.root / _safe_run_id(args.resume))
        bridge = read_json(run.directory / "backend.json")
        backend = load_backend(bridge["factory"], bridge["options"])
    if args.cancel:
        run.cancel(backend)
    else:
        deadline = time.monotonic() + args.timeout
        while True:
            state = run.advance(backend, allow_live=args.allow_live)
            if (
                args.once
                or state["status"]
                in {"completed", "cancelled", "environment_changed", "needs_reconciliation"}
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(args.poll_interval)
    export = run.export(args.export_root)
    print(
        canonical(
            {
                "run_id": run.manifest["run_id"],
                "status": run.state["status"],
                "scheduled_count": run.manifest["scheduled_count"],
                "export": str(export),
                "evidence_class": run.manifest["evidence_class"],
            }
        )
    )
    return 0 if run.state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
