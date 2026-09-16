"""Check that normal runtime has no evaluator dependency or private data mount."""

from pathlib import Path
import ast
import json
import yaml
from contracts.models import ChatMessageCreate

ROOT = Path(__file__).resolve().parents[2]


def main():
    inspected = []
    for folder in (
        "backend/app",
        "generation",
        "conversation",
        "personalisation",
        "pipelines",
        "retrieval",
    ):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                assert not any(
                    name == "evaluation" or name.startswith("evaluation.") for name in names
                ), str(path)
            inspected.append(str(path.relative_to(ROOT)))
    assert set(ChatMessageCreate.model_fields) == {"content", "use_profile"}
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    for name in ("api", "worker"):
        assert compose["services"][name]["build"]["target"] == "runtime"
        mounts = compose["services"][name].get("volumes", [])
        assert all(
            not any(word in str(mount).lower() for word in ("evaluator", "sciq", "private", "runs"))
            for mount in mounts
        )
    docker = (ROOT / "backend/Dockerfile").read_text().split("FROM runtime AS evaluator")[0]
    assert "COPY evaluation" not in docker and "COPY . " not in docker
    result = {
        "status": "passed",
        "runtime_python_files": len(inspected),
        "chat_fields": sorted(ChatMessageCreate.model_fields),
        "runtime_evaluator_dependency": False,
        "private_reference_mounts": False,
        "scope": "source and deployment-boundary validation; actual container proof is separate recovery evidence",
    }
    out = ROOT / "evidence/contracts"
    out.mkdir(exist_ok=True, parents=True)
    (out / "chat_scope.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
