"""Hand-computable evaluation examples and critical lexical counterexamples."""

import hashlib

import pytest

from evaluation.metrics.paired import paired_analysis
from evaluation.metrics.retrieval import citation_validity, retrieval_metrics
from evaluation.metrics.scoring import (
    aggregate,
    exact_match,
    normalize,
    score_outcome,
    token_f1,
    tokens,
)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (" Oxygen  ", "oxygen", 1),
        ("ＯＸＹＧＥＮ", "oxygen", 1),
        ("not oxygen", "oxygen", 0),
        ("-2 m", "2 m", 0),
        ("2 mg", "2 g", 0),
        ("", "", 0),
        (None, "oxygen", 0),
    ],
)
def test_conservative_em(left, right, expected):
    assert exact_match(left, right) == expected


def test_multiset_f1_preserves_negation_signs_units_and_repetition():
    assert token_f1("red red blue", "red blue blue") == pytest.approx(2 / 3)
    assert token_f1("not oxygen", "oxygen") == pytest.approx(2 / 3)
    assert token_f1("-2 m", "2 m") == 0.5
    assert token_f1("2 mg", "2 g") == 0.5
    assert tokens("-2.5e-3 m/s + 4") == ["-2.5e-3", "m", "/", "s", "+", "4"]


def test_full_explanation_is_never_used_as_compact_answer():
    result = score_outcome(
        "benchmark_openqa",
        {
            "status": "completed",
            "response": {
                "response_type": "answer",
                "answer_text": "The answer is not oxygen",
                "short_answer": None,
            },
        },
        {"reference": "oxygen"},
    )
    assert result["em"] == result["token_f1"] == 0
    assert not result["compact_answer_present"]


def test_failed_cancelled_and_refused_rows_keep_scheduled_denominator():
    rows = []
    for status in ("completed", "refused", "error", "cancelled"):
        outcome = {
            "status": status,
            "response": {"response_type": "answer", "short_answer": "oxygen"},
            "usage": {"cost": 0 if status == "completed" else None},
        }
        rows.append(
            {
                "status": status,
                "outcome": outcome,
                "scores": score_outcome("benchmark_openqa", outcome, {"reference": "oxygen"}),
            }
        )
    result = aggregate("benchmark_openqa", rows)
    assert result["scheduled_count"] == 4
    assert result["metrics"]["em"] == 0.25
    assert result["missing_compact_answer_count"] == 3
    assert result["usage"]["cost"]["total"] is None
    assert result["usage"]["cost"]["known_subtotal"] == 0


def test_mcq_requires_selected_option_text_not_only_label():
    command = {"options": {"A": "oxygen", "B": "carbon"}}
    response = {
        "status": "completed",
        "response": {"answer": "A", "answer_text": "carbon", "refused": False},
    }
    assert (
        score_outcome("benchmark_mcq", response, {"correct_label": "A"}, command)["accuracy"] == 0
    )


def test_retrieval_uses_actual_returned_denominator_and_graded_ndcg():
    result = retrieval_metrics(["c", "a"], {"a": 3, "b": 1, "c": 0}, k=5)
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.5
    assert 0 < result["ndcg_at_k"] < 1
    assert result["actual_returned_count"] == 2


def test_unavailable_qrels_and_zero_idcg_are_not_fabricated_zeros():
    assert retrieval_metrics([], None, k=3)["hit_at_k"] is None
    assert retrieval_metrics(["a"], {"a": None}, k=3)["ndcg_at_k"] is None
    assert retrieval_metrics(["a"], {"a": 0}, k=3)["ndcg_at_k"] is None
    assert retrieval_metrics([], {"a": 1}, k=3)["precision_at_k"] == 0
    with pytest.raises(ValueError):
        retrieval_metrics(["a", "a"], {"a": 1}, k=3)


def test_citations_check_actual_text_hash_and_have_applicable_denominator():
    evidence = [
        {
            "evidence_id": "ev_001",
            "text": "source",
            "text_hash": hashlib.sha256(b"source").hexdigest(),
        }
    ]
    assert citation_validity("answer", ["ev_001"], evidence)["actual_context_identity"] == 1
    evidence[0]["text"] = "changed"
    assert citation_validity("answer", ["ev_001"], evidence)["actual_context_identity"] == 0
    assert citation_validity("answer", [], evidence)["identifier_validity"] is None
    assert citation_validity("social", [], evidence)["applicable"] is False


def test_exact_binary_discordants_and_continuous_paired_intervals():
    result = paired_analysis([0, 0, 0, 1], [1, 1, 1, 1], binary=True, bootstrap_samples=100)
    assert result["mcnemar"]["improved"] == 3
    assert result["mcnemar"]["p_exact_two_sided"] == 0.25
    assert result["mean_difference"] == 0.75
    continuous = paired_analysis([0.1, 0.3], [0.4, 0.5], bootstrap_samples=100)
    assert "mcnemar" not in continuous
    assert continuous["mean_difference"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        paired_analysis([0.1], [0.2], binary=True)


def test_turns_cluster_by_scenario_without_invalid_per_turn_mcnemar():
    result = paired_analysis(
        [0, 0, 1, 1],
        [1, 1, 1, 1],
        binary=True,
        clusters=["a", "a", "b", "b"],
        bootstrap_samples=100,
    )
    assert result["n_clusters"] == 2
    assert result["mcnemar"]["p_exact_two_sided"] is None
