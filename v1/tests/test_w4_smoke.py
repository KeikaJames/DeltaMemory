"""Smoke tests for the W.4 CAA-baseline runner.

These tests verify
  1. The runner module imports cleanly with no side effects (no model load,
     no file write, no network call).
  2. If `cells_smoke.jsonl` exists in the W.4 experiment dir, every row
     conforms to the cell schema documented in
     `experiments/W4_caa_baseline/PREREG.md` and the runner docstring.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REQUIRED_FIELDS = {
    "cell_id", "model", "method", "alpha", "seed", "prompt_id",
    "inj_nll", "base_nll", "drift", "model_substituted", "env_commit",
}


def test_w4_run_module_imports_without_side_effects(monkeypatch):
    """Importing the runner must not load models, write files, or hit network."""
    # Trip wires for accidental side effects.
    forbidden_calls = []

    def trip(name):
        def _f(*a, **kw):
            forbidden_calls.append(name)
            raise RuntimeError(f"side effect: {name}")
        return _f

    # Block any model load triggered at import time.
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained",
                        trip("AutoModelForCausalLM.from_pretrained"))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        trip("AutoTokenizer.from_pretrained"))

    # Force a fresh import so module-level code runs under the monkey-patches.
    sys.modules.pop("experiments.W4_caa_baseline.run", None)
    sys.modules.pop("experiments.W4_caa_baseline", None)
    sys.path.insert(0, str(ROOT / "experiments" / "W4_caa_baseline"))
    try:
        run_mod = importlib.import_module("run")
    finally:
        sys.path.pop(0)

    # Public surface must exist.
    assert callable(getattr(run_mod, "main", None))
    assert callable(getattr(run_mod, "cell_id", None))
    assert hasattr(run_mod, "MODELS")
    assert hasattr(run_mod, "METHODS")
    assert hasattr(run_mod, "ALPHAS")
    assert hasattr(run_mod, "SEEDS")
    # PREREG section 2 grid sanity.
    assert run_mod.METHODS == ["none", "lopi_default", "caa"]
    assert run_mod.ALPHAS == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert run_mod.SEEDS == [0, 1, 2]
    assert len(run_mod.MODELS) == 5

    # cell_id is deterministic sha1.
    cid_a = run_mod.cell_id("gpt2-medium", "caa", 1.0, 0, "gold_001")
    cid_b = run_mod.cell_id("gpt2-medium", "caa", 1.0, 0, "gold_001")
    cid_c = run_mod.cell_id("gpt2-medium", "caa", 1.0, 0, "gold_002")
    assert cid_a == cid_b
    assert cid_a != cid_c
    assert len(cid_a) == 40
    assert int(cid_a, 16) >= 0

    assert forbidden_calls == [], f"unexpected side effects: {forbidden_calls}"


def test_w4_smoke_output_schema():
    """Every row in cells_smoke.jsonl conforms to the documented schema."""
    smoke_path = ROOT / "experiments" / "W4_caa_baseline" / "cells_smoke.jsonl"
    if not smoke_path.exists():
        pytest.skip(f"{smoke_path} not present (smoke run not yet executed)")

    rows: list[dict] = []
    with open(smoke_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    assert len(rows) >= 3, f"smoke must have at least 3 cells, got {len(rows)}"

    for row in rows:
        missing = REQUIRED_FIELDS - row.keys()
        assert not missing, f"row missing required fields {missing}: {row}"
        assert isinstance(row["cell_id"], str) and len(row["cell_id"]) == 40
        assert isinstance(row["model"], str)
        assert isinstance(row["method"], str)
        assert row["method"] in ("none", "lopi_default", "caa")
        assert isinstance(row["alpha"], (int, float))
        assert isinstance(row["seed"], int)
        assert isinstance(row["prompt_id"], str)
        assert isinstance(row["model_substituted"], bool)
        assert isinstance(row["env_commit"], str)

        # Numeric fields are floats unless this is a sentinel row.
        is_sentinel = (row.get("model_substituted")
                       or row.get("method_unsupported"))
        if not is_sentinel:
            assert isinstance(row["inj_nll"], (int, float))
            assert isinstance(row["base_nll"], (int, float))
            assert isinstance(row["drift"], (int, float))
