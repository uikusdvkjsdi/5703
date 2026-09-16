"""Reject uncontrolled variant comparisons and silent E1 redefinitions."""

from __future__ import annotations

from evaluation.common import fingerprint

FACTORS = {
    "retrieval": {"retrieval_variant", "reranker_revision"},
    "chunking": {
        "chunker_version",
        "chunk_configuration_hash",
        "corpus_release_id",
        "processing_ids",
        "qrels_version",
    },
    "embedding": {
        "embedding_model_revision",
        "embedding_dimension",
        "embedding_configuration_hash",
        "corpus_release_id",
    },
    "top_k": {"top_k"},
    "profile_policy": {"profile_condition", "profile_rule_version"},
}


def validate_comparison(
    baseline: dict,
    variant: dict,
    *,
    factor: str,
    selected_on_split: str,
    qrels_compatible: bool = False,
) -> dict:
    if factor not in FACTORS or selected_on_split != "validation":
        raise ValueError("Declare one supported factor and select settings on validation only")
    for key in (
        "protocol_id",
        "mode",
        "question_ids",
        "dataset_revision",
        "dataset_split",
        "model_configuration_hash",
        "prompt_configuration_hash",
        "scorer_version",
        "seed",
        "source_asset_hashes",
    ):
        if key not in baseline or key not in variant or baseline[key] != variant[key]:
            raise ValueError(f"Controlled comparison mismatch: {key}")
    if baseline.get("condition") == "E1" and baseline.get("retrieval_variant") != "R0":
        raise ValueError("Frozen E1 must remain R0 dense retrieval")
    if variant.get("condition") == "E1" and variant.get("retrieval_variant") != "R0":
        raise ValueError("An improved retrieval variant cannot be labelled frozen E1")
    metadata = {"run_id", "condition", "created_at", "manifest_hash", "result_path"}
    changed = {
        key for key in set(baseline) | set(variant) if baseline.get(key) != variant.get(key)
    } - metadata
    if not changed or not changed <= FACTORS[factor]:
        raise ValueError(f"More than the declared factor changed: {sorted(changed)}")
    if factor == "chunking" and not qrels_compatible:
        raise ValueError("Changed chunks need compatible source-span qrels or actual reannotation")
    return {
        "factor": factor,
        "changed_fields": sorted(changed),
        "baseline_hash": fingerprint(baseline),
        "variant_hash": fingerprint(variant),
        "selection_split": selected_on_split,
        "scientific_result_available": False,
        "status": "design_compatible",
    }


def environment_change(frozen: dict, current: dict) -> dict:
    changes = {
        key: {"frozen": value, "current": current.get(key)}
        for key, value in frozen.items()
        if current.get(key) != value
    }
    return {
        "compatible": not changes,
        "changes": changes,
        "action": "continue" if not changes else "stop_new_calls_and_preserve_prior_outcomes",
    }
