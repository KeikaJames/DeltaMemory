"""Statistical summaries for Mneme experiment reports."""

from __future__ import annotations

import random
from statistics import median
from typing import Any

PRIMARY_BASELINES = [
    "no_memory",
    "raw_memory",
    "hidden_retrieval",
    "retrieved_attention",
    "logit_bias",
    "delta_qv_zero",
    "delta_qv_random",
    "delta_qv_shuffled",
    "delta_qv_wrong_layer",
    "delta_qv_wrong_query",
    "delta_qv_identity_gate",
]


def paired_comparison(eval_summary: dict[str, Any], target_mode: str, baseline_mode: str, seed: int = 0) -> dict[str, Any]:
    deltas = []
    for sample in eval_summary.get("samples", []):
        modes = sample.get("modes", {})
        target_nll = _nll(modes, target_mode)
        baseline_nll = _nll(modes, baseline_mode)
        if target_nll is None or baseline_nll is None:
            continue
        deltas.append(float(baseline_nll) - float(target_nll))
    return _paired_delta_stats(deltas, seed=seed) | {
        "target_mode": target_mode,
        "baseline_mode": baseline_mode,
        "num_examples": len(deltas),
    }


def primary_delta_memory_statistics(eval_summary: dict[str, Any], target_mode: str = "delta_qv", seed: int = 0) -> dict[str, Any]:
    comparisons = {
        baseline: paired_comparison(eval_summary, target_mode, baseline, seed=seed)
        for baseline in PRIMARY_BASELINES
        if baseline in eval_summary.get("aggregate", {})
    }
    strongest = _strongest_baseline(eval_summary, list(comparisons))
    return {
        "target_mode": target_mode,
        "strongest_non_prompt_baseline": strongest,
        "comparisons": comparisons,
    }


def _paired_delta_stats(deltas: list[float], seed: int) -> dict[str, Any]:
    if not deltas:
        return {
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "std_delta": 0.0,
            "bootstrap_ci95": [0.0, 0.0],
            "win_rate": 0.0,
            "sign_test_p": 1.0,
            "permutation_p": 1.0,
        }
    mean_delta = sum(deltas) / len(deltas)
    variance = sum((item - mean_delta) ** 2 for item in deltas) / len(deltas)
    return {
        "mean_delta": mean_delta,
        "median_delta": float(median(deltas)),
        "std_delta": variance**0.5,
        "bootstrap_ci95": _bootstrap_ci(deltas, seed=seed),
        "win_rate": sum(1.0 for item in deltas if item > 0.0) / len(deltas),
        "sign_test_p": _sign_test_p(deltas),
        "permutation_p": _sign_flip_permutation_p(deltas, seed=seed),
    }


def _bootstrap_ci(values: list[float], seed: int, rounds: int = 1000) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(0.025 * rounds)], means[int(0.975 * rounds)]]


def _sign_test_p(values: list[float]) -> float:
    wins = sum(1 for item in values if item > 0.0)
    losses = sum(1 for item in values if item < 0.0)
    trials = wins + losses
    if trials == 0:
        return 1.0
    extreme = min(wins, losses)
    prob = sum(_comb(trials, k) for k in range(extreme + 1)) / (2**trials)
    return min(1.0, 2.0 * prob)


def _sign_flip_permutation_p(values: list[float], seed: int, rounds: int = 2000) -> float:
    observed = abs(sum(values) / len(values))
    rng = random.Random(seed + 17)
    at_least = 0
    for _ in range(rounds):
        flipped = [item if rng.random() < 0.5 else -item for item in values]
        if abs(sum(flipped) / len(flipped)) >= observed:
            at_least += 1
    return (at_least + 1) / (rounds + 1)


def _strongest_baseline(eval_summary: dict[str, Any], baselines: list[str]) -> str | None:
    aggregate = eval_summary.get("aggregate", {})
    available = [
        (baseline, aggregate[baseline].get("answer_nll"))
        for baseline in baselines
        if baseline in aggregate and aggregate[baseline].get("answer_nll") is not None
    ]
    if not available:
        return None
    return min(available, key=lambda item: float(item[1]))[0]


def _nll(modes: dict[str, Any], mode: str) -> float | None:
    value = ((modes.get(mode) or {}).get("metrics") or {}).get("answer_nll")
    return None if value is None else float(value)


def bootstrap_value_ci(values: list[float], seed: int = 0, rounds: int = 1000) -> dict[str, float]:
    """Public bootstrap CI + sign test over a list of per-example values.

    Use for binding margin or per-example top1 (0/1) arrays in Stage 6 reports.
    """
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "sign_test_p": 1.0, "n": 0}
    mean_val = sum(values) / len(values)
    ci = _bootstrap_ci(values, seed=seed, rounds=rounds)
    return {
        "mean": float(mean_val),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "sign_test_p": float(_sign_test_p([v - 0.0 for v in values])),
        "n": len(values),
    }


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result
