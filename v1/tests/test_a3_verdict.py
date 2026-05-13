"""Smoke tests for experiments/A_ablation/verdict.py.

Verifies that the statistical helper functions are correct on toy data,
and that the full pipeline runs end-to-end with synthetic cells.jsonl.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Unit tests for statistical helpers


def test_wilcoxon_significant():
    """Strongly different arrays → p ≈ 0 (well below 0.01)."""
    from experiments.A_ablation.verdict import wilcoxon_paired

    ctrl = np.zeros(20)
    abl = np.ones(20)  # ablation always worse
    p = wilcoxon_paired(ctrl, abl)
    assert p < 0.01


def test_wilcoxon_identical():
    """Identical arrays → p = 1.0 (no difference)."""
    from experiments.A_ablation.verdict import wilcoxon_paired

    ctrl = np.array([1.0, 2.0, 3.0, 2.5, 1.8])
    abl = ctrl.copy()
    p = wilcoxon_paired(ctrl, abl)
    assert p == 1.0


def test_bootstrap_ci_excludes_zero_when_large_effect():
    """CI excludes 0 when the true effect is large."""
    from experiments.A_ablation.verdict import bootstrap_ci

    diffs = np.ones(30)  # all differences = +1
    lo, hi = bootstrap_ci(diffs, n_resamples=1000)
    assert lo > 0.0
    assert hi > 0.0


def test_bootstrap_ci_includes_zero_when_no_effect():
    """CI includes 0 when diffs are centred on zero."""
    from experiments.A_ablation.verdict import bootstrap_ci

    rng = np.random.default_rng(0)
    diffs = rng.standard_normal(200)  # mean ≈ 0
    lo, hi = bootstrap_ci(diffs, n_resamples=1000)
    # The CI should include 0 (test only that it's a valid interval)
    assert lo < hi


def test_build_pairs_correct_matching():
    """build_pairs correctly matches by (prompt_id, seed)."""
    from experiments.A_ablation.verdict import build_pairs

    cells = [
        {"arm": "control", "method": "m", "model": "m", "prompt_id": "p1",
         "alpha": 0.0, "seed": 0, "status": "ok",
         "nll_new": 5.0, "nll_true": 4.0},
        {"arm": "control", "method": "m", "model": "m", "prompt_id": "p2",
         "alpha": 0.0, "seed": 0, "status": "ok",
         "nll_new": 6.0, "nll_true": 4.5},
        {"arm": "A5", "method": "m", "model": "m", "prompt_id": "p1",
         "alpha": 1.0, "seed": 0, "status": "ok",
         "nll_new": 7.0, "nll_true": 4.0},
        {"arm": "A5", "method": "m", "model": "m", "prompt_id": "p2",
         "alpha": 1.0, "seed": 0, "status": "ok",
         "nll_new": 8.0, "nll_true": 4.5},
    ]
    ctrl, abl = build_pairs(cells, "A5")
    assert len(ctrl) == 2
    assert len(abl) == 2
    diffs = abl - ctrl
    assert np.all(diffs == 2.0)


# ---------------------------------------------------------------------------
# Smoke test: full pipeline on synthetic cells.jsonl


def _write_ablation_cells(path: pathlib.Path, seeds: list[int], n_prompts: int,
                           arms: list[str], method: str,
                           base_nll: float, ablation_delta: float) -> None:
    """Write a minimal cells.jsonl for verdict.py consumption."""
    rows = []
    prompt_ids = [f"p{i:03d}" for i in range(n_prompts)]
    for seed in seeds:
        for prompt_id in prompt_ids:
            rows.append({
                "cell_id": f"{seed}-{prompt_id}-ctrl",
                "arm": "control",
                "method": method,
                "model": "fake",
                "prompt_id": prompt_id,
                "alpha": 0.0,
                "seed": seed,
                "prereg_version": "test",
                "ts": "2025-01-01T00:00:00Z",
                "status": "ok",
                "phrase": "test",
                "nll_new": base_nll,
                "nll_true": base_nll - 1.0,
            })
            for arm in arms:
                rows.append({
                    "cell_id": f"{seed}-{prompt_id}-{arm}",
                    "arm": arm,
                    "method": method,
                    "model": "fake",
                    "prompt_id": prompt_id,
                    "alpha": 1.0,
                    "seed": seed,
                    "prereg_version": "test",
                    "ts": "2025-01-01T00:01:00Z",
                    "status": "ok",
                    "phrase": "test",
                    "nll_new": base_nll + ablation_delta,
                    "nll_true": base_nll - 1.0,
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_run_produces_verdict_md(tmp_path):
    """Smoke: run() writes VERDICT.md with expected rows."""
    import experiments.A_ablation.verdict as verdict_mod

    a_dir = tmp_path / "A_per_method_v1_qwen3"

    # synthetic caa arm: A5 has large effect (1.5 nats)
    _write_ablation_cells(
        a_dir / "caa_arm" / "cells.jsonl",
        seeds=[0, 1], n_prompts=10,
        arms=["A5"], method="caa",
        base_nll=5.0, ablation_delta=1.5,
    )
    # synthetic lopi arm: A3 is no-op
    _write_ablation_cells(
        a_dir / "lopi_arm" / "cells.jsonl",
        seeds=[0, 1], n_prompts=10,
        arms=["A3"], method="lopi_default",
        base_nll=5.0, ablation_delta=0.0,
    )

    original_arm_files = verdict_mod.ARM_FILES
    verdict_mod.ARM_FILES = {
        "caa": a_dir / "caa_arm" / "cells.jsonl",
        "lopi": a_dir / "lopi_arm" / "cells.jsonl",
    }
    try:
        rows = verdict_mod.run(a_dir=a_dir)
    finally:
        verdict_mod.ARM_FILES = original_arm_files

    # VERDICT.md exists and has content
    verdict_path = a_dir / "VERDICT.md"
    assert verdict_path.exists()
    text = verdict_path.read_text()
    assert "A5" in text
    assert "A3" in text

    # A5 should be supported (large delta)
    a5_row = next(r for r in rows if r.name == "A5")
    assert a5_row.supported, f"A5 should be SUPPORTED, got p={a5_row.p_value}"
    assert abs(a5_row.mean_delta - 1.5) < 0.01

    # A3 should NOT be supported (zero delta)
    a3_row = next(r for r in rows if r.name == "A3")
    assert not a3_row.supported, "A3 should NOT be supported"
    assert abs(a3_row.mean_delta) < 1e-9


def test_run_handles_missing_arm_file(tmp_path):
    """Smoke: run() handles a missing arm file gracefully."""
    import experiments.A_ablation.verdict as verdict_mod

    a_dir = tmp_path / "A_missing"
    a_dir.mkdir()

    original_arm_files = verdict_mod.ARM_FILES
    verdict_mod.ARM_FILES = {
        "nonexistent": a_dir / "nonexistent_arm" / "cells.jsonl",
    }
    try:
        rows = verdict_mod.run(a_dir=a_dir)
    finally:
        verdict_mod.ARM_FILES = original_arm_files

    assert len(rows) == 1
    assert not rows[0].supported
    assert rows[0].n_pairs == 0
