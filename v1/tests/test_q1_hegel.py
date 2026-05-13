"""Tests for Q.1 Hegel generation runner (experiments/Q1_hegel/).

Covers:
  1. PREREG.md schema parses (version, hypotheses, grid present).
  2. prompts.jsonl loads and has 5 valid rows with required fields.
  3. aggregate.py runs on synthetic cells.jsonl (deterministic).
  4. cell_id determinism.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

Q1_DIR = ROOT / "experiments" / "Q1_hegel"
PREREG_PATH = Q1_DIR / "PREREG.md"
PROMPTS_PATH = Q1_DIR / "prompts.jsonl"


# ---------------------------------------------------------------------------
# 1. PREREG.md schema
# ---------------------------------------------------------------------------

def test_prereg_exists_and_parses():
    """PREREG.md must exist and contain key schema tokens."""
    assert PREREG_PATH.exists(), f"Missing {PREREG_PATH}"
    text = PREREG_PATH.read_text(encoding="utf-8")
    # version token
    assert "Q1.v1" in text, "PREREG.md must declare version Q1.v1"
    # hypotheses
    assert "H_Q1" in text, "PREREG.md must contain H_Q1 hypothesis"
    assert "Red-line" in text or "redline" in text.lower(), "PREREG.md must contain Red-line clause"
    # grid keys
    for key in ("gpt2-medium", "Qwen", "alpha", "seeds", "max_new_tokens"):
        assert key in text, f"PREREG.md missing grid key: {key}"
    # authenticity clause
    assert "authenticity" in text.lower() or "env.json" in text, "PREREG.md must reference authenticity contract"


# ---------------------------------------------------------------------------
# 2. prompts.jsonl validity
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"id", "prompt", "canonical", "counterfact", "subject", "relation", "aliases"}

def test_prompts_jsonl_loads_five_valid_rows():
    """prompts.jsonl must have exactly 5 rows with required fields."""
    assert PROMPTS_PATH.exists(), f"Missing {PROMPTS_PATH}"
    rows = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert len(rows) == 5, f"Expected 5 prompt rows, got {len(rows)}"
    for i, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row.keys())
        assert not missing, f"Row {i} missing fields: {missing}"
        assert isinstance(row["aliases"], list), f"Row {i} aliases must be a list"
        assert len(row["aliases"]) >= 1, f"Row {i} must have at least one alias"
        assert row["canonical"] != row["counterfact"], f"Row {i} canonical must differ from counterfact"
        assert row["id"].startswith("hegel_"), f"Row {i} id should start with hegel_"
        assert len(row["prompt"]) > 10, f"Row {i} prompt too short"


# ---------------------------------------------------------------------------
# 3. aggregate.py runs on synthetic cells.jsonl
# ---------------------------------------------------------------------------

def _make_synthetic_cells(n_cf_hits: int = 8, n_total: int = 10) -> list[dict]:
    """Build a synthetic cells.jsonl with controllable hit counts."""
    rows = []
    for i in range(n_total):
        row = {
            "cell_id": f"cell_{i:04d}",
            "model": "gpt2-medium",
            "prompt_id": f"hegel_00{(i % 5) + 1}",
            "seed": i % 3,
            "alpha": 1.0,
            "contains_counterfact": i < n_cf_hits,
            "contains_canonical": i >= n_cf_hits,
            "full_text_sha1": hashlib.sha1(f"text_{i}".encode()).hexdigest(),
            "transcript_path": f"transcripts/cell_{i:04d}.txt",
            "status": "ok",
            "dataset_sha1": "abc123",
        }
        rows.append(row)
    # Add alpha=0 rows with redline_ok
    for i in range(3):
        row = {
            "cell_id": f"cell_rl_{i}",
            "model": "gpt2-medium",
            "prompt_id": f"hegel_00{i + 1}",
            "seed": i,
            "alpha": 0.0,
            "contains_counterfact": False,
            "contains_canonical": True,
            "full_text_sha1": hashlib.sha1(f"baseline_{i}".encode()).hexdigest(),
            "transcript_path": f"transcripts/cell_rl_{i}.txt",
            "status": "ok",
            "redline_ok": True,
            "dataset_sha1": "abc123",
        }
        rows.append(row)
    return rows


def test_aggregate_runs_on_synthetic_cells(tmp_path):
    """aggregate.py must produce summary.json with correct hit-rate from synthetic data."""
    # Write cells.jsonl
    cells_path = tmp_path / "cells.jsonl"
    rows = _make_synthetic_cells(n_cf_hits=8, n_total=10)
    with cells_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Write dummy transcripts so aggregate doesn't fail
    (tmp_path / "transcripts").mkdir()

    # Run aggregate
    sys.path.insert(0, str(Q1_DIR.parent.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("aggregate_q1", Q1_DIR / "aggregate.py")
    agg = importlib.util.load_from_spec = None
    # Import via exec to avoid caching issues
    agg_code = (Q1_DIR / "aggregate.py").read_text()
    ns: dict = {}
    exec(compile(agg_code, str(Q1_DIR / "aggregate.py"), "exec"), ns)

    summary = ns["aggregate"](tmp_path)

    # Validate structure
    assert "results" in summary
    assert "h_q1_verdicts" in summary
    assert summary["ok_cells"] == len(rows)  # all rows are 'ok'

    # Check hit-rate at alpha=1.0
    alpha1_results = [r for r in summary["results"] if abs(r["alpha"] - 1.0) < 1e-9]
    assert len(alpha1_results) == 1
    r = alpha1_results[0]
    expected_rate = 8 / 10  # 8 hits out of 10
    assert abs(r["counterfact_hit_rate"] - expected_rate) < 1e-9

    # CI bounds should bracket the mean
    assert r["counterfact_ci95_lo"] <= r["counterfact_hit_rate"]
    assert r["counterfact_ci95_hi"] >= r["counterfact_hit_rate"]

    # H_Q1 verdict: 0.8 > 0.5 → PASS
    assert "gpt2-medium" in summary["h_q1_verdicts"]
    assert summary["h_q1_verdicts"]["gpt2-medium"]["pass"] is True


def test_aggregate_fail_verdict_below_threshold(tmp_path):
    """aggregate.py emits FAIL when hit-rate <= 0.50."""
    cells_path = tmp_path / "cells.jsonl"
    rows = _make_synthetic_cells(n_cf_hits=3, n_total=10)
    with cells_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (tmp_path / "transcripts").mkdir()

    agg_code = (Q1_DIR / "aggregate.py").read_text()
    ns: dict = {}
    exec(compile(agg_code, str(Q1_DIR / "aggregate.py"), "exec"), ns)
    summary = ns["aggregate"](tmp_path)

    alpha1_results = [r for r in summary["results"] if abs(r["alpha"] - 1.0) < 1e-9]
    assert len(alpha1_results) == 1
    assert summary["h_q1_verdicts"]["gpt2-medium"]["pass"] is False


# ---------------------------------------------------------------------------
# 4. cell_id determinism
# ---------------------------------------------------------------------------

def test_cell_id_determinism():
    """cell_id must be stable and collision-free for distinct inputs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_q1", str(Q1_DIR / "run.py"))
    mod = importlib.util.module_from_spec(spec)
    # Patch __file__ and sys.path before exec
    mod.__file__ = str(Q1_DIR / "run.py")
    # Use importlib directly
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    make_cell_id = mod.make_cell_id

    # Same inputs → same id
    id1 = make_cell_id("gpt2-medium", "hegel_001", 0, 1.0)
    id2 = make_cell_id("gpt2-medium", "hegel_001", 0, 1.0)
    assert id1 == id2, "cell_id must be deterministic"

    # Length = 16
    assert len(id1) == 16, f"cell_id must be 16 chars, got {len(id1)}"

    # Hex characters only
    assert all(c in "0123456789abcdef" for c in id1), "cell_id must be hex"

    # Different inputs → different ids
    distinct_inputs = [
        ("gpt2-medium", "hegel_001", 0, 0.0),
        ("gpt2-medium", "hegel_001", 0, 1.0),
        ("gpt2-medium", "hegel_001", 1, 0.0),
        ("gpt2-medium", "hegel_002", 0, 0.0),
        ("Qwen/Qwen2.5-0.5B", "hegel_001", 0, 0.0),
    ]
    ids = [make_cell_id(*inp) for inp in distinct_inputs]
    assert len(set(ids)) == len(ids), "Distinct inputs must produce distinct cell_ids"

    # Manual cross-check against expected sha1 prefix
    expected_key = "gpt2-medium|hegel_001|0|0.0"
    expected_id = hashlib.sha1(expected_key.encode()).hexdigest()[:16]
    assert make_cell_id("gpt2-medium", "hegel_001", 0, 0.0) == expected_id
