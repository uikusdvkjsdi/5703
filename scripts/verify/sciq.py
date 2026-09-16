"""Verify actual SciQ provenance and export gold-free, deterministic projections.

No answer generation, scoring, corpus ingestion or database mutation occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from contracts.models import MCQCommand, OpenQACommand
from evaluation.datasets.sciq import PrivateSciQRecord, load_split, mcq_projection
from evaluation.datasets.sciq_openqa import openqa_projection
from evaluation.runner import validate_public
from generation import GenerationRequest
from generation.prompt_builder import build_messages
from scripts.dev.acquire_sciq import immutable, sha256


def jsonl(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()


def assert_prompt_boundary(command: dict) -> None:
    request = GenerationRequest(
        request_id="sciq-source-boundary",
        mode=command["mode"],
        condition="E0",
        question=command["question_text"],
        question_id=command["question_id"],
        options=command.get("options"),
        history=[{"role": "user", "content": "PRIVATE_HISTORY_SENTINEL"}],
        summary="PRIVATE_SUMMARY_SENTINEL",
        profile={"policy": "PRIVATE_PROFILE_SENTINEL"},
    )
    messages, evidence, _, effective = build_messages(request)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert (
        not evidence
        and not effective.history
        and effective.summary is None
        and effective.profile is None
    )
    assert all("PRIVATE_" not in m["content"] for m in messages)
    if request.mode == "benchmark_openqa":
        assert messages[-1]["content"] == command["question_text"]
    else:
        payload = json.loads(messages[-1]["content"])
        assert set(payload) == {"question_id", "question", "options"}
        assert payload["question"] == command["question_text"]
        assert payload["options"] == command["options"]


def verify(acquisition: dict, export_root: Path, evidence_path: Path, seed: int = 5703) -> dict:
    revision = acquisition["revision"]
    private_root = Path(acquisition["private_storage_root"])
    output = export_root.resolve() / revision
    if output.is_relative_to(private_root.resolve()):
        raise ValueError("Public projection and private reference directories must be distinct")
    findings = []
    for split_info in acquisition["splits"]:
        split = split_info["split"]
        raw_path, derived_path = Path(split_info["raw_path"]), Path(split_info["derived_path"])
        if (
            sha256(raw_path.read_bytes()) != split_info["raw_sha256"]
            or sha256(derived_path.read_bytes()) != split_info["derived_sha256"]
        ):
            raise ValueError("Acquired original or derived split no longer matches its manifest")
        # OpenQA must not reject an exact question because unused candidates
        # duplicate each other. The original MCQ anomaly is retained separately.
        records, manifest = load_split(
            derived_path, split=split, revision=revision, require_distinct_choices=False
        )
        assert len(records) == split_info["count"]
        openqa, mcq, ledger, references, private_anomalies = [], [], [], [], []
        original_rows = [
            json.loads(line) for line in derived_path.read_text(encoding="utf-8").splitlines()
        ]
        for index, record in enumerate(records):
            run_id, item_id = f"sciq-projection-{split}-v1", f"item_{index:06}"
            command, private = openqa_projection(record, run_id=run_id, item_id=item_id)
            OpenQACommand.model_validate(command)
            validate_public(command)
            assert set(command) == {"mode", "run_id", "item_id", "question_id", "question_text"}
            assert command["question_text"] == original_rows[index]["question"]
            changed = replace(
                record,
                correct_answer="PRIVATE_CHANGED_ANSWER",
                support="PRIVATE_CHANGED_SUPPORT",
                distractors=("PRIVATE_D1", "PRIVATE_D2", "PRIVATE_D3"),
            )
            assert openqa_projection(changed, run_id=run_id, item_id=item_id)[0] == command
            openqa.append(command)
            reference_row = {
                "item_id": item_id,
                "question_id": record.question_id,
                "openqa": private,
            }
            entry = {
                "item_id": item_id,
                "question_id": record.question_id,
                "openqa_status": "input_ready",
            }
            try:
                PrivateSciQRecord.from_mapping(original_rows[index], fallback_id=record.question_id)
            except ValueError as exc:
                entry.update(mcq_status="input_rejected", error_code="INVALID_DISTINCT_CHOICES")
                private_anomalies.append(
                    {
                        "item_id": item_id,
                        "question_id": record.question_id,
                        "error": str(exc),
                        "original_record": original_rows[index],
                    }
                )
            else:
                mcq_command, mcq_private = mcq_projection(
                    record, run_id=run_id, item_id=item_id, seed=seed
                )
                MCQCommand.model_validate(mcq_command)
                validate_public(mcq_command)
                assert (
                    mcq_projection(record, run_id=run_id, item_id=item_id, seed=seed)[0]
                    == mcq_command
                )
                candidates = [record.correct_answer, *record.distractors]
                for gold_index in range(4):
                    changed_gold = replace(
                        record,
                        correct_answer=candidates[gold_index],
                        distractors=tuple(c for i, c in enumerate(candidates) if i != gold_index),
                        support="PRIVATE_CHANGED_SUPPORT",
                    )
                    other, other_private = mcq_projection(
                        changed_gold, run_id=run_id, item_id=item_id, seed=seed
                    )
                    assert other == mcq_command
                    assert (
                        other["options"][other_private["correct_label"]]
                        == changed_gold.correct_answer
                    )
                assert mcq_command["options"][mcq_private["correct_label"]] == record.correct_answer
                assert set(mcq_command["options"].values()) == set(candidates)
                mcq.append(mcq_command)
                reference_row["mcq"] = mcq_private
                entry["mcq_status"] = "input_ready"
                if index in (0, len(records) // 2, len(records) - 1):
                    assert_prompt_boundary(mcq_command)
            if (
                index in (0, len(records) // 2, len(records) - 1)
                or entry["mcq_status"] == "input_rejected"
            ):
                assert_prompt_boundary(command)
            references.append(reference_row)
            ledger.append(entry)
        assert len(ledger) == len(records) == len(openqa) == len(references)
        assert len(mcq) + len(private_anomalies) == len(records)
        paths = {}
        for name, rows in (("openqa", openqa), ("mcq", mcq), ("preflight", ledger)):
            path = output / f"{split}.{name}.jsonl"
            raw = jsonl(rows)
            immutable(path, raw)
            paths[name] = {"path": str(path), "sha256": sha256(raw), "rows": len(rows)}
        for name, rows in (("references", references), ("mcq-rejections", private_anomalies)):
            immutable(private_root / "projections" / f"{split}.{name}.jsonl", jsonl(rows))
        findings.append(
            {
                "split": split,
                "native_count": len(records),
                "openqa_ready": len(openqa),
                "mcq_ready": len(mcq),
                "mcq_input_rejected": len(private_anomalies),
                "rejected_question_ids": [row["question_id"] for row in private_anomalies],
                "exports": paths,
                "dataset_manifest": manifest,
                "checks": {
                    "exact_stems": len(records),
                    "private_field_mutation_invariant": len(records),
                    "symmetric_four_gold_mcq_checks": len(mcq) * 4,
                    "all_native_records_in_preflight_ledger": True,
                    "no_source_records_changed_or_removed": True,
                },
            }
        )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "passed_with_source_mcq_anomalies",
        "revision": revision,
        "seed": seed,
        "total_native_records": sum(f["native_count"] for f in findings),
        "total_openqa_ready": sum(f["openqa_ready"] for f in findings),
        "total_mcq_ready": sum(f["mcq_ready"] for f in findings),
        "total_mcq_input_rejected": sum(f["mcq_input_rejected"] for f in findings),
        "splits": findings,
        "model_calls": 0,
        "corpus_ingestion": False,
        "scores": None,
        "scope": "Source acquisition, deterministic public projections and actual prompt boundary checks. No model accuracy, full benchmark execution or scientific correctness result.",
        "mcq_policy": "Native duplicate choices are rejected before generation. Every original item remains in the preflight ledger. A future full-split run must retain these input rejections in its scheduled denominator; the smaller ready-command export must not be presented as the complete split.",
    }
    if evidence_path.exists():
        prior = json.loads(evidence_path.read_text())
        if {k: v for k, v in prior.items() if k != "timestamp"} != {
            k: v for k, v in result.items() if k != "timestamp"
        }:
            raise ValueError(
                "Current verification differs from the saved immutable evidence; use a new report path"
            )
        result = prior
    immutable(evidence_path, json.dumps(result, indent=2).encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", default="evidence/sciq/acquisition.json")
    parser.add_argument("--export-root", default="artifacts/runs/sciq_projection")
    parser.add_argument("--evidence", default="evidence/sciq/projection-verification.json")
    args = parser.parse_args()
    result = verify(
        json.loads(Path(args.acquisition).read_text()), Path(args.export_root), Path(args.evidence)
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "revision",
                    "total_native_records",
                    "total_openqa_ready",
                    "total_mcq_ready",
                    "total_mcq_input_rejected",
                    "model_calls",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
