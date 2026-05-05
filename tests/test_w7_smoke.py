"""Smoke tests for W.7 (long-context degradation)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
W7_DIR = ROOT / "experiments" / "W7_longctx"
SMOKE_PATH = W7_DIR / "cells_smoke.jsonl"
ENV_PATH = W7_DIR / "env.json"

pytestmark = pytest.mark.skipif(
    not SMOKE_PATH.exists(),
    reason="W.7 smoke artefacts not generated; run experiments/W7_longctx/run.py --smoke first.",
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
        row.get("status") == "ok"
        and not row.get("method_unsupported", False)
        and row.get("alpha", -1) != -1
        and row.get("seed", -1) != -1
    )


def test_smoke_cells_exist_and_have_schema():
    assert SMOKE_PATH.exists(), (
        f"missing smoke output: {SMOKE_PATH}. Run "
        "`python experiments/W7_longctx/run.py --smoke` first."
    )
    rows = _load_jsonl(SMOKE_PATH)
    real = [r for r in rows if _is_real(r)]
    assert len(real) >= 10, f"expected >=10 real cells, got {len(real)}"
    required = {
        "cell_id", "model", "method", "alpha", "seed", "length",
        "prompt_id", "nll_target", "top1_match_frac", "drift",
        "redline_violation", "status",
    }
    missing = required - set(real[0].keys())
    assert not missing, f"missing fields: {missing}"


def test_smoke_alpha0_bit_equal():
    rows = _load_jsonl(SMOKE_PATH)
    real = [r for r in rows if _is_real(r)]
    a0_inj = [r for r in real
              if float(r["alpha"]) == 0.0 and r["method"] != "none"]
    assert a0_inj, "no alpha=0 injected cells in smoke output"
    drifts = []
    for r in a0_inj:
        d = r.get("drift")
        assert d is not None, f"drift missing at alpha=0: {r}"
        drifts.append(abs(float(d)))
    max_drift = max(drifts)
    assert max_drift < 1e-4, (
        f"W.7 alpha=0 bit-equality red-line violated: "
        f"max |drift| = {max_drift:.3e}"
    )
    print(f"[W7 smoke] alpha=0 max |drift| = {max_drift:.3e}")


def test_env_json_records_provenance():
    assert ENV_PATH.exists(), f"missing {ENV_PATH}"
    with open(ENV_PATH) as f:
        env = json.load(f)
    for key in ("torch", "transformers", "git_commit", "dtype", "device",
                "models", "total_cells_planned", "prereg_version"):
        assert key in env, f"env.json missing {key!r}"
    assert env["total_cells_planned"] >= 1
    assert env["torch"], "torch version empty"


def test_aggregate_runs_on_smoke(tmp_path):
    out_dir = tmp_path / "agg"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        str(W7_DIR / "aggregate.py"),
        "--cells", str(SMOKE_PATH),
        "--out", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, (
        f"aggregate failed: stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    curve_path = out_dir / "length_curve.json"
    assert curve_path.exists(), "length_curve.json missing"
    with open(curve_path) as f:
        curve = json.load(f)
    assert "groups" in curve and isinstance(curve["groups"], list)
    assert curve["groups"], "empty length_curve.groups"
