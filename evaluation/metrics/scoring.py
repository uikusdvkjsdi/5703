"""Conservative lexical diagnostics; these do not establish semantic correctness."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

SCORER_VERSION = "conservative-v1"
TOKENIZER_VERSION = "words-signed-numbers-symbols-v1"
TOKEN_PATTERN = re.compile(
    r"[+−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+−-]?\d+)?|[^\W\d_]+(?:['’][^\W\d_]+)*|[^\w\s]",
    re.UNICODE,
)


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("Compact answers must be strings or null")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text: str | None) -> list[str]:
    return TOKEN_PATTERN.findall(normalize(text))


def exact_match(prediction: str | None, reference: str) -> float:
    predicted, expected = normalize(prediction), normalize(reference)
    return float(bool(predicted) and bool(expected) and predicted == expected)


def token_f1(prediction: str | None, reference: str) -> float:
    predicted, expected = tokens(prediction), tokens(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    return 2.0 * overlap / (len(predicted) + len(expected))


def score_outcome(mode: str, outcome: dict, reference: dict, command: dict | None = None) -> dict:
    eligible = outcome.get("status") == "completed"
    response = outcome.get("response") or {}
    if mode == "benchmark_openqa":
        eligible = eligible and response.get("response_type") == "answer"
        predicted = response.get("short_answer") if eligible else None
        if predicted is not None and not isinstance(predicted, str):
            predicted = None
        expected = reference["reference"]
        if not normalize(expected):
            raise ValueError("Invalid references must be excluded by a declared pre-freeze policy")
        return {
            "em": exact_match(predicted, expected),
            "token_f1": token_f1(predicted, expected),
            "compact_answer_present": bool(normalize(predicted)),
            "valid_response": eligible,
            "lexically_scorable": bool(normalize(predicted)),
            "scorer_version": SCORER_VERSION,
        }
    if mode == "benchmark_mcq":
        answer = response.get("answer")
        eligible = (
            eligible
            and not response.get("refused", False)
            and isinstance(answer, str)
            and len(answer) == 1
            and answer in "ABCD"
        )
        if command is not None and eligible:
            eligible = response.get("answer_text") == command["options"][answer]
        return {
            "accuracy": float(eligible and answer == reference["correct_label"]),
            "valid_response": bool(eligible),
            "scorer_version": SCORER_VERSION,
        }
    raise ValueError("Lexical scoring supports only explicit OpenQA or MCQ modes")


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0 <= fraction <= 1 or any(not math.isfinite(value) for value in values):
        raise ValueError("Quantiles need finite data and a fraction in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(mode: str, scheduled: list[dict]) -> dict:
    count = len(scheduled)
    names = ("em", "token_f1") if mode == "benchmark_openqa" else ("accuracy",)
    statuses = Counter(item.get("status", "pending") for item in scheduled)
    if mode not in {"benchmark_openqa", "benchmark_mcq"}:
        raise ValueError("Aggregate mode must be an explicit lexical evaluation protocol")
    result = {
        "mode": mode,
        "scheduled_count": count,
        "status_counts": dict(statuses),
        "valid_response_count": sum(
            bool((item.get("scores") or {}).get("valid_response")) for item in scheduled
        ),
        "metrics": {
            name: sum((item.get("scores") or {}).get(name, 0) for item in scheduled) / count
            if count
            else None
            for name in names
        },
        "metric_kind": "lexical_diagnostic_proxies"
        if mode == "benchmark_openqa"
        else "MCQ_selection_accuracy",
        "scorer_version": SCORER_VERSION,
    }
    if mode == "benchmark_openqa":
        result["missing_compact_answer_count"] = count - sum(
            bool((item.get("scores") or {}).get("compact_answer_present")) for item in scheduled
        )
    result["timing"] = {}
    for phase in ("preparation_ms", "retrieval_ms", "generation_ms", "total_ms"):
        values = [(item.get("outcome") or {}).get("timings", {}).get(phase) for item in scheduled]
        values = [
            float(value)
            for value in values
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ]
        result["timing"][phase] = {
            "n": len(values),
            "p50": quantile(values, 0.5),
            "p95": quantile(values, 0.95),
        }
    usage = [(item.get("outcome") or {}).get("usage", {}) for item in scheduled]
    result["usage"] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens", "cost"):
        values = [row.get(name) for row in usage]
        known = [
            value
            for value in values
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ]
        result["usage"][name] = {
            "total": sum(known) if len(known) == count and count else None,
            "known_subtotal": sum(known) if known else None,
            "known_count": len(known),
            "unknown_count": count - len(known),
        }
    result["research_claim"] = "No semantic-correctness or learning-gain claim is implied."
    return result
