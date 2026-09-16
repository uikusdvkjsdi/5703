"""The evaluator calls a shared application backend; it never generates answers."""

from __future__ import annotations

from typing import Any, Protocol


TERMINAL_STATUSES = frozenset(
    {"completed", "refused", "error", "cancelled", "invalid", "incomplete"}
)


class AnswerBackend(Protocol):
    def environment(self) -> dict[str, Any]:
        """Return current public identities corresponding to manifest environment keys."""

    def register_run(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Idempotently register the public manifest; return {run_id: manifest run_id}."""

    def submit(
        self, command: dict[str, Any], *, run_context: dict[str, Any], idempotency_key: str
    ) -> dict[str, Any]:
        """Return {request_id, job_id, ...}; same key must never duplicate execution."""

    def poll(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Return status pending or terminal, response, model_mode, timings/usage/error."""

    def lookup(self, idempotency_key: str) -> dict[str, Any] | None:
        """Find a durable receipt after interrupted submission; None never authorizes retry."""


def check_outcome(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("status") not in TERMINAL_STATUSES | {"pending"}:
        raise ValueError("Backend returned an unknown outcome status")
    if value["status"] == "completed" and not isinstance(value.get("response"), dict):
        raise ValueError("Completed backend outcomes require a typed response object")
    return value
