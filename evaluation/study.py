"""Persist independent matched-profile studies using the shared teaching service."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from contracts.models import ChatResponseV1, EvidenceSnapshot, TeachingStudyResponseV1
from evaluation.annotations.blind import blank_ratings_csv, create_blind_package
from evaluation.common import atomic_json, fingerprint, read_json, utc_now
from evaluation.runner import _safe_run_id, run_lock
from generation.types import ModelConfig, RequestBudget
from generation.prompt_builder import PROMPTS
from personalisation.compiler import COMPILER_VERSION
from personalisation.study import TeachingStudyService


def study_environment():
    return {
        "prompt_configuration_hash": fingerprint(
            (PROMPTS / "teaching_v1.txt").read_text(encoding="utf-8")
        ),
        "profile_rule_version": COMPILER_VERSION,
        "response_schema_hash": fingerprint(TeachingStudyResponseV1.model_json_schema()),
    }


class ProfileStudy:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        if (
            fingerprint(
                {key: value for key, value in self.manifest.items() if key != "manifest_hash"}
            )
            != self.manifest["manifest_hash"]
        ):
            raise ValueError("Frozen study manifest changed")
        self.inputs = read_json(self.directory / "inputs.json")
        if fingerprint(self.inputs) != self.manifest["inputs_hash"]:
            raise ValueError("Frozen study base/evidence changed")
        self.state = read_json(self.directory / "state.json")

    @classmethod
    def freeze(
        cls,
        *,
        root: Path,
        questions: list[dict],
        model_config: dict | None = None,
        run_id: str | None = None,
        seed: int = 0,
        allow_live: bool = False,
    ):
        config = ModelConfig.from_dict(model_config)
        config.validate()
        if config.provider != "mock" and not allow_live:
            raise ValueError("Profile study defaults to mock-only execution")
        if not questions or len({row["question_id"] for row in questions}) != len(questions):
            raise ValueError("Study needs distinct question identities")
        for row in questions:
            response = ChatResponseV1.model_validate(row["base_answer"])
            if response.response_type != "answer":
                raise ValueError("Matched profile study needs a completed free-form base answer")
            for evidence in row["evidence"]:
                EvidenceSnapshot.model_validate(evidence)
        run_id = _safe_run_id(run_id or uuid.uuid4().hex)
        inputs = {"questions": questions, "model_config": config.to_dict()}
        items = [
            {
                "item_id": f"q{index:05}-{level}-{condition}",
                "question_id": row["question_id"],
                "question_index": index,
                "target_level": level,
                "condition": condition,
            }
            for index, row in enumerate(questions)
            for level in ("beginner", "intermediate", "advanced")
            for condition in ("C0", "C1", "C2")
        ]
        manifest = {
            "version": "matched-profile-study-v1",
            "run_id": run_id,
            "created_at": utc_now(),
            "scheduled_count": len(items),
            "items": items,
            "inputs_hash": fingerprint(inputs),
            "model_configuration_hash": fingerprint(config.to_dict()),
            "environment": study_environment(),
            "seed": seed,
            "model_mode": "mock" if config.provider == "mock" else "live",
            "rubric_version": "profile_rating_v1",
            "human_ratings_completed": False,
        }
        manifest["manifest_hash"] = fingerprint(manifest)
        directory = root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "manifest.json", manifest, immutable=True)
        atomic_json(directory / "inputs.json", inputs, immutable=True)
        atomic_json(
            directory / "state.json",
            {
                "items": [
                    {
                        **item,
                        "status": "pending",
                        "budget": RequestBudget().to_dict(),
                        "outcome": None,
                    }
                    for item in items
                ],
                "status": "frozen",
            },
        )
        return cls(directory)

    def advance(
        self,
        service: TeachingStudyService | None = None,
        *,
        max_items: int | None = None,
        allow_live: bool = False,
    ) -> dict:
        service = service or TeachingStudyService()
        config = ModelConfig.from_dict(self.inputs["model_config"])
        if config.provider != "mock" and not allow_live:
            raise ValueError("Profile study defaults to mock-only execution")
        with run_lock(self.directory):
            self.state = read_json(self.directory / "state.json")
            if self.manifest.get("environment") != study_environment():
                self.state["status"] = "environment_changed"
                atomic_json(self.directory / "state.json", self.state)
                return self.state
            completed_this_pass = 0
            for item in self.state["items"]:
                if item["status"] == "running":
                    item["status"] = "uncertain"
                    atomic_json(self.directory / "state.json", self.state)
                if (
                    item["status"] != "pending"
                    or max_items is not None
                    and completed_this_pass >= max_items
                ):
                    continue
                row = self.inputs["questions"][item["question_index"]]
                item["status"] = "running"
                atomic_json(self.directory / "state.json", self.state)

                def record_attempt(event):
                    item["budget"] = event["budget"]
                    item.setdefault("attempts", []).append(
                        {key: value for key, value in event.items() if key != "raw_text"}
                    )
                    atomic_json(self.directory / "state.json", self.state)

                try:
                    result = service.generate(
                        run_id=self.manifest["run_id"],
                        item_id=item["item_id"],
                        condition=item["condition"],
                        target_level=item["target_level"],
                        base_answer=row["base_answer"],
                        evidence=row["evidence"],
                        config=config,
                        budget=RequestBudget.from_dict(item["budget"]),
                        on_attempt=record_attempt,
                    )
                    item["outcome"] = result
                    item["budget"] = result["budget"]
                    item["status"] = "completed" if result["state"] == "succeeded" else "error"
                except Exception as exc:
                    item["status"] = "uncertain"
                    item["error"] = {
                        "code": "STUDY_EXECUTION_UNCONFIRMED",
                        "error_type": type(exc).__name__,
                    }
                completed_this_pass += 1
                atomic_json(self.directory / "state.json", self.state)
            self.state["status"] = (
                "completed"
                if all(
                    row["status"] in {"completed", "error", "cancelled"}
                    for row in self.state["items"]
                )
                else "needs_reconciliation"
                if any(row["status"] == "uncertain" for row in self.state["items"])
                else "running"
            )
            atomic_json(self.directory / "state.json", self.state)
            return self.state

    def export_blind(self, destination: Path) -> dict:
        outputs = []
        for item in self.state["items"]:
            question = self.inputs["questions"][item["question_index"]]
            response = (item.get("outcome") or {}).get("response") or {}
            outputs.append(
                {
                    "item_id": item["item_id"],
                    "question_id": item["question_id"],
                    "question": question["question"],
                    "condition": item["condition"],
                    "target_level": item["target_level"],
                    "base_answer_hash": fingerprint(question["base_answer"]),
                    "evidence_hash": fingerprint(question["evidence"]),
                    "model_configuration_hash": self.manifest["model_configuration_hash"],
                    "status": item["status"],
                    "explanation": response.get("explanation"),
                    "evidence": question["evidence"],
                }
            )
        package = create_blind_package(outputs, seed=self.manifest["seed"])
        destination.mkdir(parents=True, exist_ok=False)
        atomic_json(destination / "private_package.json", package, immutable=True)
        atomic_json(
            destination / "review_items.json",
            {"rubric": package["rubric"], "review_items": package["review_items"]},
            immutable=True,
        )
        (destination / "ratings.csv").write_text(
            blank_ratings_csv(package), encoding="utf-8", newline=""
        )
        atomic_json(
            destination / "outcomes.json",
            {"manifest": self.manifest, "items": self.state["items"], "actual_human_ratings": 0},
            immutable=True,
        )
        return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path)
    selection.add_argument("--resume", type=Path)
    parser.add_argument("--blind-output", type=Path)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        config = read_json(args.config)
        directory = args.config.resolve().parent
        study = ProfileStudy.freeze(
            root=directory / os.environ.get("EVALUATION_PRIVATE_ROOT", config["private_root"]),
            questions=read_json(directory / config["questions_path"]),
            model_config=config.get("model_config"),
            seed=config.get("seed", 0),
            allow_live=args.allow_live,
        )
    else:
        study = ProfileStudy(args.resume)
    study.advance(allow_live=args.allow_live)
    if args.blind_output:
        study.export_blind(args.blind_output)
    print(
        f"{study.manifest['run_id']}: {study.state['status']}; {study.manifest['scheduled_count']} scheduled outputs; no human ratings claimed"
    )
    return 0 if study.state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
