"""Ordinary authenticated HTTP conversations; references never cross this boundary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from uuid import uuid4

import httpx

from evaluation.common import atomic_json, canonical, fingerprint, read_json, utc_now
from evaluation.conversations.runner import aggregate_scenarios, run_scenario


class HttpChatBackend:
    def __init__(self, client, *, email, password, poll_interval=0.5, timeout=180, on_poll=None):
        self.client, self.email, self.password = client, email, password
        self.poll_interval, self.timeout, self.on_poll = poll_interval, timeout, on_poll
        self.headers = {}
        self.relogin()

    def _request(self, method, path, body=None, headers=None):
        response = self.client.request(
            method, "/api/v1" + path, json=body, headers={**self.headers, **(headers or {})}
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Application request failed: {response.status_code} {method} {path}"
            )
        return response.json()["data"]

    def capabilities(self):
        return self._request("GET", "/capabilities")

    def new_session(self):
        return self._request("POST", "/sessions", {"title": "Authored conversation rehearsal"})[
            "id"
        ]

    def set_profile(self, profile):
        previous = self._request("GET", "/profiles/me")
        self._request("PUT", "/profiles/me", {**previous, **profile})

    def relogin(self):
        login = self._request(
            "POST", "/auth/login", {"email": self.email, "password": self.password}
        )
        self.headers = {"Authorization": "Bearer " + login["access_token"]}

    def send_and_wait(self, session_id, content, *, use_profile, idempotency_key):
        receipt = self._request(
            "POST",
            f"/sessions/{session_id}/messages",
            {"content": content, "use_profile": use_profile},
            {"Idempotency-Key": idempotency_key},
        )
        deadline = time.monotonic() + self.timeout
        while True:
            if self.on_poll:
                self.on_poll()
            job = self._request("GET", "/jobs/" + receipt["job_id"])
            if job["state"] == "succeeded":
                answer = self._request("GET", "/answers/" + job["answer_id"])
                response_type = answer["response"].get("response_type")
                status = {
                    "refusal": "refused",
                    "clarification": "clarification",
                    "social": "social",
                }.get(response_type, "completed")
                return {"status": status, "receipt": receipt, "answer": answer}
            if job["state"] in {"failed", "cancelled"}:
                return {
                    "status": "error" if job["state"] == "failed" else "cancelled",
                    "receipt": receipt,
                    "error": job["error"],
                }
            if time.monotonic() >= deadline:
                return {
                    "status": "incomplete",
                    "receipt": receipt,
                    "error": {
                        "code": "CAPTURE_TIMEOUT",
                        "message": "The server job may still be active; inspect the saved receipt.",
                    },
                }
            time.sleep(self.poll_interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--email",
        required=True,
        help="Use a dedicated rehearsal account; profile-continuity cases update its profile",
    )
    parser.add_argument("--password-env", default="EVALUATION_ACCOUNT_PASSWORD")
    parser.add_argument(
        "--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args(argv)
    password = os.environ.get(args.password_env)
    if not password:
        parser.error(
            "Set the configured account password environment variable; it is never written to exports"
        )
    scenarios = read_json(args.scenarios)
    run_id = uuid4().hex
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        backend = HttpChatBackend(client, email=args.email, password=password)
        capabilities = backend.capabilities()
        if capabilities.get("model_mode") != "mock" and not args.allow_live:
            parser.error(
                "This rehearsal is mock-only unless live execution was explicitly authorized"
            )
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "version": "conversation-capture-v1",
            "protocol_id": "chat_scenarios",
            "run_id": run_id,
            "created_at": utc_now(),
            "scenario_hash": fingerprint(scenarios),
            "scheduled_scenarios": [
                {"scenario_id": scenario["scenario_id"], "scheduled_turns": len(scenario["turns"])}
                for scenario in scenarios["scenarios"]
            ],
            "capabilities": capabilities,
            "model_mode": capabilities["model_mode"],
            "human_semantic_review": None,
        }
        atomic_json(args.output / "manifest.json", manifest, immutable=True)
        atomic_json(args.output / "scenarios.json", scenarios, immutable=True)
        results = []
        for scenario in scenarios["scenarios"]:
            result = run_scenario(scenario, backend, run_id=run_id, allow_live=args.allow_live)
            results.append(result)
            atomic_json(args.output / (scenario["scenario_id"] + ".json"), result, immutable=True)
            atomic_json(
                args.output / "aggregate.json",
                {
                    **aggregate_scenarios(results),
                    "scheduled_scenario_count": len(scenarios["scenarios"]),
                    "provisional": len(results) != len(scenarios["scenarios"]),
                },
            )
        print(
            canonical(
                {"run_id": run_id, "output": str(args.output), **aggregate_scenarios(results)}
            )
        )
    return 0 if all(result["complete_scenario"] for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
