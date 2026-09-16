"""Run authored conversations through an injected real chat-session client."""

from __future__ import annotations

from typing import Protocol
import math

from evaluation.common import fingerprint, utc_now
from evaluation.metrics.scoring import quantile


class ChatBackend(Protocol):
    def capabilities(self) -> dict: ...
    def new_session(self) -> str: ...
    def set_profile(self, profile: dict) -> None: ...
    def relogin(self) -> None: ...
    def send_and_wait(
        self, session_id: str, content: str, *, use_profile: bool, idempotency_key: str
    ) -> dict: ...


def rendered_turn(scenario: dict, turn: dict) -> str:
    content = turn["content"]
    if turn.get("use_long_prefix"):
        setup = scenario["setup"]
        content = setup["long_message_prefix"] * setup["long_message_prefix_repetitions"] + content
    if not 0 < len(content) <= 4000:
        raise ValueError("Authored scenario content must satisfy the actual composer contract")
    return content


def run_scenario(
    scenario: dict, backend: ChatBackend, *, run_id: str, allow_live: bool = False
) -> dict:
    if backend.capabilities().get("model_mode") != "mock" and not allow_live:
        raise ValueError("Authored software rehearsal is mock-only by default")
    if not 3 <= len(scenario["turns"]) <= 5:
        raise ValueError("These authored scenario families require three to five scheduled turns")
    if scenario.get("setup", {}).get("profile"):
        backend.set_profile(scenario["setup"]["profile"])
    session_id = backend.new_session()
    result = {
        "run_id": run_id,
        "scenario_id": scenario["scenario_id"],
        "scenario_hash": fingerprint(scenario),
        "started_at": utc_now(),
        "scheduled_turns": len(scenario["turns"]),
        "turns": [],
        "human_semantic_review": None,
    }
    for index, turn in enumerate(scenario["turns"]):
        if turn.get("before") == "relogin":
            backend.relogin()
        elif turn.get("before") == "new_session":
            session_id = backend.new_session()
        command = {
            "session_id": session_id,
            "content": rendered_turn(scenario, turn),
            "use_profile": turn.get("use_profile", True),
        }
        try:
            observation = backend.send_and_wait(
                command["session_id"],
                command["content"],
                use_profile=command["use_profile"],
                idempotency_key=f"conversation:{run_id}:{scenario['scenario_id']}:{index}",
            )
        except Exception as exc:
            observation = {"status": "error", "error_type": type(exc).__name__}
        result["turns"].append(
            {
                "turn_index": index,
                "session_id": session_id,
                "command": command,
                "observation": observation,
            }
        )
    result["completed_turns"] = sum(
        row["observation"].get("status") in {"completed", "refused", "clarification", "social"}
        for row in result["turns"]
    )
    result["complete_scenario"] = result["completed_turns"] == result["scheduled_turns"]
    result["evidence_class"] = (
        "mock_dataflow_rehearsal"
        if backend.capabilities().get("model_mode") == "mock"
        else "live_conversation_capture"
    )
    return result


def aggregate_scenarios(results: list[dict]) -> dict:
    scheduled = sum(row["scheduled_turns"] for row in results)
    completed = sum(row["completed_turns"] for row in results)
    timing = {}
    for phase in ("preparation_ms", "retrieval_ms", "generation_ms", "total_ms"):
        values = [
            turn["observation"].get("answer", {}).get("timing", {}).get(phase)
            for result in results
            for turn in result["turns"]
        ]
        known = [
            value
            for value in values
            if type(value) in (int, float) and math.isfinite(value) and value >= 0
        ]
        timing[phase] = {
            "n": len(known),
            "missing_count": scheduled - len(known),
            "p50": quantile(known, 0.5),
            "p95": quantile(known, 0.95),
        }
    return {
        "scenario_count": len(results),
        "scheduled_turn_count": scheduled,
        "completed_turn_count": completed,
        "turn_completion_rate": completed / scheduled if scheduled else None,
        "complete_scenario_count": sum(row["complete_scenario"] for row in results),
        "scenario_completion_rate": sum(row["complete_scenario"] for row in results) / len(results)
        if results
        else None,
        "timing": timing,
        "cost": None,
        "concurrency": 1,
        "semantic_correctness": None,
        "analysis_unit": "scenario; repeated turns are clustered",
        "interpretation": "Completion records data flow only, not correctness or student learning. Unknown usage/cost is not inferred from latency.",
    }
