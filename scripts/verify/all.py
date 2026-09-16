"""Run the repeatable mock software gate; no paid provider calls are made."""

import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def source_snapshot():
    """Tie a gate to unchanged executable sources, fixtures and configuration."""
    values = {}
    for directory in (
        "backend",
        "contracts",
        "conversation",
        "evaluation",
        "generation",
        "personalisation",
        "pipelines",
        "retrieval",
        "scripts",
        "tests",
        "configs",
        "frontend/src",
        "frontend/tests",
        "frontend/scripts",
        ".github",
    ):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or any(
                part in {"__pycache__", "private_runs", "exports", "results"} for part in path.parts
            ):
                continue
            if path.suffix not in {
                ".py",
                ".ts",
                ".tsx",
                ".css",
                ".json",
                ".yaml",
                ".yml",
                ".txt",
                ".md",
                ".mjs",
            }:
                continue
            if path.suffix == ".md" and not path.is_relative_to(ROOT / "generation/prompts"):
                continue
            # Schema export is itself a checked stage; record its final content
            # separately rather than interpreting regenerated schema bytes as an edit.
            if path.is_relative_to(ROOT / "contracts/schemas") or path.name in {
                "openapi.json",
                "openapi.yaml",
            }:
                continue
            values[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for path in [
        *ROOT.glob("requirements*.lock"),
        ROOT / "pyproject.toml",
        ROOT / "frontend/package-lock.json",
        ROOT / "frontend/package.json",
        ROOT / "frontend/vite.config.ts",
        ROOT / "frontend/tsconfig.json",
        ROOT / "backend/Dockerfile",
        ROOT / "frontend/Dockerfile",
        ROOT / ".env.example",
        *ROOT.glob("compose*.yaml"),
    ]:
        values[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mock"], required=True)
    args = parser.parse_args()
    output = ROOT / "evidence/final"
    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    before = source_snapshot()
    environment = {
        **os.environ,
        "MODEL_MODE": "mock",
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "backend")]),
    }
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    assert npm, "Node/npm is required for the complete software gate"
    commands = [
        ("python_quality", [sys.executable, "-m", "scripts.verify.python_quality"], ROOT),
        ("foundation", [sys.executable, "-m", "scripts.verify.foundation"], ROOT),
        ("contracts", [sys.executable, "-m", "scripts.verify.contracts"], ROOT),
        ("chat_scope", [sys.executable, "-m", "scripts.verify.chat_scope"], ROOT),
        (
            "python_tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
                "--junitxml=" + str(output / "pytest.xml"),
            ],
            ROOT,
        ),
        ("frontend_types", [npm, "run", "types:check"], ROOT / "frontend"),
        ("frontend_tests", [npm, "test"], ROOT / "frontend"),
        ("frontend_build", [npm, "run", "build"], ROOT / "frontend"),
    ]
    results = []
    for name, command, cwd in commands:
        print("Running " + name, flush=True)
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        results.append(
            {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "log": str((output / (name + ".log")).relative_to(ROOT)),
            }
        )
    after = source_snapshot()
    changed = [
        path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)
    ]
    (output / "source_snapshot.json").write_text(
        json.dumps({"before": before, "after": after, "changed_during_run": changed}, indent=2),
        encoding="utf-8",
    )
    result = {
        "mode": args.mode,
        "started_at": started_at,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed"
        if not changed and all(item["exit_code"] == 0 for item in results)
        else "failed",
        "source_files_unchanged": not changed,
        "changed_during_run": changed,
        "source_snapshot": "evidence/final/source_snapshot.json",
        "checks": results,
        "separate_evidence_required": [
            "real browser responsive and lifecycle journeys",
            "clean Compose install",
            "backup restore",
            "live scientific evaluation",
            "human ratings and physical mobile keyboard",
        ],
    }
    (output / "software_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
