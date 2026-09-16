"""Run non-mutating Python format, correctness lint and contract type gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIRST_PARTY = [
    "backend/app",
    "contracts",
    "conversation",
    "evaluation",
    "generation",
    "personalisation",
    "pipelines",
    "retrieval",
    "scripts",
    "tests",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/devtools")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "backend")]),
        "PYTHONIOENCODING": "utf-8",
        "MODEL_MODE": "mock",
    }
    checks = []

    def run(name: str, command: list[str], *, expected_failure: str | None = None) -> None:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        (output / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        passed = (
            result.returncode == 0
            if expected_failure is None
            else (result.returncode != 0 and expected_failure in result.stdout)
        )
        checks.append(
            {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "expected_failure": expected_failure,
                "passed": passed,
                "log": str(output / f"{name}.log"),
            }
        )
        print(f"{name}: {'passed' if passed else 'FAILED'}", flush=True)

    run(
        "python_lint",
        [sys.executable, "-m", "ruff", "check", *FIRST_PARTY, "--output-format", "concise"],
    )
    run(
        "python_format",
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "--output-format",
            "concise",
            *FIRST_PARTY,
        ],
    )
    run("python_types", [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml"])
    with tempfile.TemporaryDirectory(prefix="cs30-gate-probes-") as temp:
        probe = Path(temp) / "broken.py"
        probe.write_text("value = missing_contract_name\n", encoding="utf-8")
        run(
            "lint_rejects_undefined_name",
            [sys.executable, "-m", "ruff", "check", str(probe), "--select", "F821"],
            expected_failure="F821",
        )
        probe.write_text("def spaced( x ): return x\n", encoding="utf-8")
        run(
            "format_rejects_drift",
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "--output-format",
                "concise",
                str(probe),
            ],
            expected_failure="reformatted",
        )
        probe.write_text(
            "from contracts.models import ChatMessageCreate\n"
            "from generation.types import ModelConfig\n"
            "from evaluation.bridge import AnswerBackend\n"
            "chat = ChatMessageCreate(content=42)\n"
            "config = ModelConfig(max_tokens='1024')\n"
            "def wrong_receipt(backend: AnswerBackend) -> str:\n"
            "    return backend.lookup('request-key')\n",
            encoding="utf-8",
        )
        run(
            "types_reject_bad_contract",
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                str(ROOT / "pyproject.toml"),
                str(probe),
            ],
            expected_failure='Argument "content" to "ChatMessageCreate" has incompatible type',
        )
        # A separate probe proves the protocol return is checked, independently
        # of the model constructor errors above.
        probe.write_text(
            "from evaluation.bridge import AnswerBackend\n"
            "def wrong_receipt(backend: AnswerBackend) -> str:\n"
            "    return backend.lookup('request-key')\n",
            encoding="utf-8",
        )
        run(
            "types_reject_bad_protocol",
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                str(ROOT / "pyproject.toml"),
                str(probe),
            ],
            expected_failure="[return-value]",
        )
        contract_dir = Path(temp) / "contracts"
        contract_dir.mkdir()
        contract = json.loads((ROOT / "contracts/openapi.json").read_text(encoding="utf-8"))
        contract["components"]["schemas"]["ChatMessageCreate"]["properties"]["content"]["type"] = (
            "integer"
        )
        (contract_dir / "openapi.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run(
            "schema_rejects_broken_contract",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from scripts.verify import contracts; import sys; contracts.ROOT=Path(sys.argv.pop()); contracts.main()",
                temp,
            ],
            expected_failure="Runtime OpenAPI drift",
        )
    report = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "versions": {name: version(name) for name in ["ruff", "mypy", "pydantic"]},
        "python": sys.version,
        "first_party_roots": FIRST_PARTY,
        "type_scope": "Canonical Pydantic payloads/HTTP envelopes, generation dataclasses, evaluator protocol and typed consumers. This is not whole-backend strict typing.",
        "checks": checks,
        "live_model_calls": 0,
    }
    (output / "python_quality.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
