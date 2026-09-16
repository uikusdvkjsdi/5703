"""Paired uncertainty and exact binary tests; conversations cluster by scenario."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean

from evaluation.metrics.scoring import quantile


def paired_analysis(
    before: list[float],
    after: list[float],
    *,
    binary: bool = False,
    clusters: list[str] | None = None,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict:
    if len(before) != len(after) or not before:
        raise ValueError("Paired observations require equal nonempty arrays")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) for value in before + after
    ):
        raise ValueError("Paired observations must be finite")
    if binary and any(value not in (0, 1) for value in before + after):
        raise ValueError("McNemar applies only to binary scores")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    if clusters is not None and len(clusters) != len(before):
        raise ValueError("Every observation needs its scenario cluster")
    groups = defaultdict(list)
    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        groups[clusters[index] if clusters is not None else str(index)].append(new - old)
    values = list(groups.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        sampled = [value for group in rng.choices(values, k=len(values)) for value in group]
        draws.append(mean(sampled))
    result = {
        "n_pairs": len(before),
        "n_clusters": len(values),
        "unit": "scenario_cluster" if clusters is not None else "paired_item",
        "mean_difference": mean(new - old for old, new in zip(before, after, strict=True)),
        "bootstrap_95_interval": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    if binary:
        improved = sum(old == 0 and new == 1 for old, new in zip(before, after, strict=True))
        regressed = sum(old == 1 and new == 0 for old, new in zip(before, after, strict=True))
        discordant = improved + regressed
        # Repeated turns are dependent: a per-turn exact test is inapplicable.
        probability = (
            min(
                1.0,
                2
                * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
                / (2**discordant),
            )
            if discordant
            else 1.0
        )
        result["mcnemar"] = {
            "improved": improved,
            "regressed": regressed,
            "discordant": discordant,
            "p_exact_two_sided": None if clusters is not None else probability,
            "status": "not_applicable_to_clustered_turns" if clusters is not None else "exact",
        }
    return result
