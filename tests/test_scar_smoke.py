"""Smoke artifact tests for M.4 SCAR vs CAA."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = ROOT / "experiments" / "scar_smoke"
SMOKE_PATH = SMOKE_DIR / "cells_smoke.jsonl"

pytestmark = pytest.mark.skipif(
    not SMOKE_PATH.exists(),
    reason=f"missing smoke output: {SMOKE_PATH}. Run `python experiments/scar_smoke/run.py --smoke` first.",
)

REQUIRED_FIELDS = {
    "cell_id", "model", "method", "alpha", "prompt_id", "drift", "max_abs_diff",
    "n_calib", "n_test", "inject_layer", "device", "dtype", "env_commit",
    "redline_violation",
}


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    with SMOKE_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_scar_smoke_schema():
    rows = _load_rows()
    assert len(rows) >= 10 * 5 * 3
    seen = {(r["method"], float(r["alpha"]), r["prompt_id"]) for r in rows}
    assert len(seen) == len(rows)

    methods = {r["method"] for r in rows}
    assert methods == {"none", "caa", "scar"}
    alphas = {float(r["alpha"]) for r in rows}
    assert alphas == {0.0, 0.5, 1.0, 1.5, 2.0}

    for row in rows:
        missing = REQUIRED_FIELDS - row.keys()
        assert not missing, f"row missing fields {missing}: {row}"
        assert isinstance(row["cell_id"], str) and len(row["cell_id"]) == 40
        assert row["model"] == "google/gemma-4-E2B"
        assert isinstance(row["prompt_id"], str) and row["prompt_id"]
        assert isinstance(row["drift"], (int, float))
        assert row["drift"] >= 0.0
        assert row["max_abs_diff"] == row["drift"]
        assert int(row["n_calib"]) == 16
        assert int(row["n_test"]) >= 10
        assert row["device"] in {"mps", "cpu", "cuda"}
        assert isinstance(row["redline_violation"], bool)


def test_scar_smoke_alpha0_bit_equal():
    rows = _load_rows()
    for method in ("caa", "scar"):
        alpha0 = [r for r in rows if r["method"] == method and float(r["alpha"]) == 0.0]
        assert alpha0, f"missing alpha=0 rows for {method}"
        max_abs = max(abs(float(r["drift"])) for r in alpha0)
        assert max_abs < 1e-4, f"{method} alpha=0 max |drift| = {max_abs:.3e}"
        print(f"[scar smoke] {method} alpha=0 max |drift| = {max_abs:.3e}")
