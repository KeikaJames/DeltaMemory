"""Smoke tests for W.6 (counter-prior Pareto)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
W6_DIR = ROOT / "experiments" / "W6_counter_prior"
SMOKE_PATH = W6_DIR / "cells_smoke.jsonl"

pytestmark = pytest.mark.skipif(
    not SMOKE_PATH.exists(),
    reason=(
        f"missing smoke output: {SMOKE_PATH}. "
        "Run `python experiments/W6_counter_prior/run.py --smoke` first. "
        "Skipped on fresh clones / CI without local smoke artifacts."
    ),
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_real(row: dict) -> bool:
    return (
        not row.get("method_unsupported", False)
        and not row.get("relation_template_missing", False)
        and row.get("alpha", -1) != -1
        and row.get("seed", -1) != -1
    )


def test_smoke_cells_alpha0_bit_equal():
    """cells_smoke.jsonl has >=10 real rows; alpha=0 max |drift| < 1e-4."""
    assert SMOKE_PATH.exists(), (
        f"missing smoke output: {SMOKE_PATH}. Run "
        "`python experiments/W6_counter_prior/run.py --smoke` first."
    )
    rows = _load_jsonl(SMOKE_PATH)
    real = [r for r in rows if _is_real(r)]
    assert len(real) >= 10, f"expected >=10 real cells, got {len(real)}"

    alpha0_inj = [
        r for r in real
        if float(r["alpha"]) == 0.0 and r.get("method") != "none"
    ]
    assert alpha0_inj, "no alpha=0 injected cells found"
    drifts = []
    for r in alpha0_inj:
        d = r.get("drift")
        assert d is not None and d == d, f"drift NaN at alpha=0: {r}"
        drifts.append(abs(float(d)))
    max_drift = max(drifts)
    assert max_drift < 1e-4, (
        f"alpha=0 bit-equality red-line violated: max |drift| = "
        f"{max_drift:.3e}"
    )
    # Stash for human inspection; the test passes regardless.
    print(f"[W6 smoke] alpha=0 max |drift| = {max_drift:.3e}")


def test_aggregate_runs_on_smoke(tmp_path):
    """aggregate.py runs end-to-end on cells_smoke.jsonl and produces
    summary.json and pareto.json without exceptions."""
    assert SMOKE_PATH.exists()
    out_dir = tmp_path / "agg"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        str(W6_DIR / "aggregate.py"),
        "--cells", str(SMOKE_PATH),
        "--out", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, (
        f"aggregate failed: stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert (out_dir / "summary.json").exists(), "summary.json missing"
    assert (out_dir / "pareto.json").exists(), "pareto.json missing"

    with open(out_dir / "summary.json") as f:
        summary = json.load(f)
    assert "h6a" in summary and "h6b" in summary
    with open(out_dir / "pareto.json") as f:
        pareto = json.load(f)
    assert "models" in pareto
