"""Prepare an authored mock corpus only inside a dedicated CI database."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.cli import migrate, seed
from app.core.config import Settings
from app.main import create_app
from app.worker import run_once

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    settings = Settings()
    database = make_url(settings.database_url).database or ""
    if (
        not database.startswith("cs30_ci_")
        or settings.env != "test"
        or settings.model_mode != "mock"
        or settings.llm_provider != "mock"
    ):
        raise SystemExit(
            "CI setup requires APP_ENV=test, explicit mock settings and a cs30_ci_* database."
        )
    migrate()
    app = create_app(settings)
    try:
        with sessionmaker(bind=app.state.engine)() as db:
            seed(db)
        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/login", json={"email": "admin@example.com", "password": "Passw0rd!"}
            )
            assert login.status_code == 200, login.status_code
            headers = {"Authorization": "Bearer " + login.json()["data"]["access_token"]}

            def call(method: str, path: str, body: dict | None = None) -> dict:
                result = client.request(method, "/api/v1" + path, json=body, headers=headers)
                assert result.is_success, (path, result.status_code, result.text)
                return result.json()["data"]

            uploaded = client.post(
                "/api/v1/documents",
                headers=headers,
                data={
                    "title": "Authored CI biology fixture",
                    "license": "Original project-authored software fixture",
                },
                files={
                    "file": (
                        "ci-biology.txt",
                        (ROOT / "pipelines/examples/authored_source.txt").read_bytes(),
                        "text/plain",
                    )
                },
            )
            assert uploaded.status_code == 201, uploaded.text
            document_id = uploaded.json()["data"]["document"]["id"]
            process = call("POST", f"/documents/{document_id}/process", {})
            assert run_once(app.state.engine, settings)
            assert call("GET", "/jobs/" + process["job_id"])["state"] == "succeeded"
            build = call(
                "POST", "/corpus/releases", {"processing_run_ids": [process["processing_id"]]}
            )
            assert run_once(app.state.engine, settings)
            assert call("GET", "/jobs/" + build["job_id"])["state"] == "succeeded"
            call("POST", "/corpus/releases/" + build["release_id"] + "/activate", {})
            out = ROOT / "evidence/devtools"
            out.mkdir(parents=True, exist_ok=True)
            (out / "ci_fixture.json").write_text(
                json.dumps(
                    {
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                        "database_name": database,
                        "document_id": document_id,
                        "processing_id": process["processing_id"],
                        "release_id": build["release_id"],
                        "model_mode": "mock",
                        "embedding_provider": "mock",
                        "evidence_class": "authored_software_fixture",
                        "scope": "CI browser prerequisites only; not official OpenStax or live scientific evidence.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print("Authored CI corpus is active.")
    finally:
        app.state.engine.dispose()


if __name__ == "__main__":
    main()
