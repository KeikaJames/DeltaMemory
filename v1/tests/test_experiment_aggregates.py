from __future__ import annotations

import pandas as pd


def test_w1_verdict_passes_small_complete_sweep():
    from experiments.W1_mhc_localize.aggregate import compute_verdict

    verdict = compute_verdict(pd.DataFrame([
        {"model": "m1", "shield": True, "v_scale": True, "alpha": 1.0, "mean_drift": 0.1},
        {"model": "m2", "shield": True, "v_scale": True, "alpha": 1.0, "mean_drift": 0.1},
    ]))
    assert verdict["pass_count"] == 2
    assert verdict["total"] == 2
    assert verdict["verdict"] == "PASS"


def test_w2_qwen_model_labels_are_not_collapsed():
    from experiments.W2_lopi_dissect.aggregate import _get_model_short

    assert _get_model_short("Qwen/Qwen2.5-0.5B") == "Qwen2.5-0.5B"
    assert _get_model_short("Qwen/Qwen2.5-1.5B") == "Qwen2.5-1.5B"
    assert _get_model_short("Qwen/Qwen3-4B-Instruct-2507") == "Qwen3-4B"


def test_w2_verdict_requires_each_model_to_pass():
    from experiments.W2_lopi_dissect.aggregate import compute_verdicts

    rows = []
    for model, a1_drift in [("good", -1.0), ("bad", 10.0)]:
        rows.extend([
            {"model": model, "arm": "A0", "alpha": 2.0, "mean_drift": 0.0},
            {"model": model, "arm": "A1", "alpha": 2.0, "mean_drift": a1_drift},
            {"model": model, "arm": "A2", "alpha": 2.0, "mean_drift": a1_drift - 1.0},
            {"model": model, "arm": "A3", "alpha": 2.0, "mean_drift": a1_drift - 2.0},
        ])
    verdicts = compute_verdicts(pd.DataFrame(rows))
    assert verdicts["Q1"].startswith("FAIL")


def test_w4_stable_seed_ignores_python_hash_salt():
    from experiments.W4_caa_baseline.aggregate import _stable_seed

    assert _stable_seed("Qwen/Qwen2.5-0.5B", 0.25, "caa") == _stable_seed(
        "Qwen/Qwen2.5-0.5B", 0.25, "caa"
    )
    assert _stable_seed("Qwen/Qwen2.5-0.5B", 0.25, "caa") != _stable_seed(
        "Qwen/Qwen2.5-0.5B", 0.25, "none"
    )


def test_w6_significant_harm_is_not_counted_as_success():
    from experiments.W6_counter_prior.aggregate import aggregate

    cells = []
    for i in range(10):
        cells.extend([
            {
                "model": "m",
                "alpha": 1.0,
                "method": "none",
                "seed": 0,
                "prompt_id": f"p{i}",
                "nll_new": 1.0,
                "nll_true": 0.5,
                "kl_unrel": 0.0,
            },
            {
                "model": "m",
                "alpha": 1.0,
                "method": "caa",
                "seed": 0,
                "prompt_id": f"p{i}",
                "nll_new": 2.0,
                "nll_true": 0.5,
                "kl_unrel": 0.1,
            },
        ])

    summary, _pareto = aggregate(cells, "caa")
    assert summary["h6a"]["cells"][0]["holm_reject"] is True
    assert summary["h6a"]["cells"][0]["directional_success"] is False
    assert summary["h6a"]["n_reject"] == 0
