"""Smoke tests for experiments/L_marathon/figures.py.

Runs the analysis pipeline on tiny synthetic data without touching GPU
or loading real model checkpoints.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers


def _write_cells_jsonl(path: pathlib.Path, seed: int, turns: list[int], nll_base: float) -> None:
    """Write a minimal cells.jsonl for one seed."""
    rows = []
    for t in turns:
        rows.append({
            "run_id": "test",
            "model": "fake-model",
            "method": "caa",
            "seed": seed,
            "turn": t,
            "alpha": 1.0,
            "nll_target_new": nll_base + (t - 1) * 0.001,  # tiny drift
            "residual_norm_mu": 100.0 + t * 0.1,
            "mu_layer": 10,
            "mem_rss_mb": 500.0 + t * 0.01,
            "nan_inf_count": 0,
            "kv_cache_size_bytes": 0,
            "abort_reason": None,
        })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# ---------------------------------------------------------------------------
# Unit tests for internal functions


def test_compute_recall_rate_constant():
    from experiments.L_marathon.figures import compute_recall_rate

    nll = [5.0, 5.0, 5.0, 5.0]
    rates = compute_recall_rate(nll)
    assert all(abs(r - 1.0) < 1e-9 for r in rates)


def test_compute_recall_rate_degrading():
    from experiments.L_marathon.figures import compute_recall_rate

    # NLL doubles → recall halves
    nll = [4.0, 8.0]
    rates = compute_recall_rate(nll)
    assert abs(rates[0] - 1.0) < 1e-9
    assert abs(rates[1] - 0.5) < 1e-9


def test_compute_half_life_interpolates():
    from experiments.L_marathon.figures import compute_half_life

    # Drops from 1.0 to 0.0 between turn 100 and 200 → half-life at 150
    turns = [1, 100, 200, 500]
    rates = [1.0, 1.0, 0.0, 0.0]
    hl = compute_half_life(turns, rates)
    assert hl is not None
    assert abs(hl - 150.0) < 1.0


def test_compute_half_life_never_drops():
    from experiments.L_marathon.figures import compute_half_life

    turns = [1, 50, 200, 500]
    rates = [1.0, 1.0, 1.0, 1.0]
    assert compute_half_life(turns, rates) is None


# ---------------------------------------------------------------------------
# Smoke test: full pipeline on synthetic data


def test_run_full_pipeline_synthetic(tmp_path):
    """Smoke: run() on synthetic per-seed data produces all expected outputs."""
    import importlib
    import experiments.L_marathon.figures as fig_mod

    turns = [1, 50, 200, 500]
    out_dir = tmp_path / "L2_out"

    # Build synthetic seed directories
    synth_models = {
        "model_a": {
            "label": "Synthetic Model A",
            "seeds": [
                tmp_path / f"L1_model_a_s{s}_t500" / "cells.jsonl"
                for s in range(2)
            ],
        },
    }

    for s, path in enumerate(synth_models["model_a"]["seeds"]):
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_cells_jsonl(path, seed=s, turns=turns, nll_base=5.0 + s * 0.1)

    # Patch MODELS in the module so the run uses our temp paths
    original_models = fig_mod.MODELS
    fig_mod.MODELS = synth_models
    try:
        result = fig_mod.run(out_dir=out_dir)
    finally:
        fig_mod.MODELS = original_models

    # Outputs exist
    assert (out_dir / "REPORT.md").exists()
    assert (out_dir / "half_life.json").exists()
    assert (out_dir / "recall_vs_turn_model_a.png").exists()
    assert (out_dir / "residual_vs_turn_model_a.png").exists()
    assert (out_dir / "rss_vs_turn_model_a.png").exists()

    # half_life is a dict
    hl = json.loads((out_dir / "half_life.json").read_text())
    assert "model_a" in hl

    # REPORT mentions model label
    report_text = (out_dir / "REPORT.md").read_text()
    assert "Synthetic Model A" in report_text


def test_run_handles_missing_seeds(tmp_path):
    """Smoke: run() gracefully handles missing seed files."""
    import experiments.L_marathon.figures as fig_mod

    out_dir = tmp_path / "L2_out_missing"
    synth_models = {
        "model_missing": {
            "label": "Missing Model",
            "seeds": [tmp_path / "nonexistent" / "cells.jsonl"],
        },
    }
    original_models = fig_mod.MODELS
    fig_mod.MODELS = synth_models
    try:
        result = fig_mod.run(out_dir=out_dir)
    finally:
        fig_mod.MODELS = original_models

    report_text = (out_dir / "REPORT.md").read_text()
    assert "Missing" in report_text or "missing" in report_text
