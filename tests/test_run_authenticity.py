"""CI authenticity check: ensure every committed run directory contains
``cells.jsonl`` and ``env.json``.

This test is part of the standard pytest suite and will fail if any run
directory that is NOT explicitly excluded (see EXEMPT_RUNS below) is missing
either required artifact.

The rationale: ``cells.jsonl`` contains raw cell data (the primary evidence
artifact) and ``env.json`` captures the full reproducibility context
(commit SHA, model path, GPU, dtype, library versions).  A committed run
without these files cannot be audited or reproduced.

Exemptions
----------
A handful of legacy / in-progress run directories that predate the requirement
are listed in ``EXEMPT_RUNS``.  New runs MUST NOT be added to this list.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# Run directories that predate the mandatory-artifact requirement OR are
# intentionally incomplete (stub / pending execution on GB10).
# DO NOT add new entries here — fix the run instead.
EXEMPT_RUNS = {
    # Legacy run with pre-v0.6 env.json schema (uses 'branch' key, not 'commit')
    "X1_full_v2",
    # Legacy aggregate/rollup directories — no cells by design
    "A_per_method_v1_qwen3",
    "L1_full_v1_gemma4_31B",
    "L1_gemma4_flagship_v1",
    "L1_qwen3_v1",
    # Pending-execution placeholders (will be populated when run on spark1)
    "D1_bit_equality_v1",
    "GB10_e2e_v1",
    # Figure-only aggregate run (PNGs + half_life.json from upstream L1 runs;
    # raw cells live under L1_*_s{0,1,2}_t500/ which themselves are exempt aggregates).
    "L2_figures_v1",
    # SKIPPED-stub runs documenting pre-execution decisions for unsupported
    # architectures on current hardware (see SKIPPED.md inside each).
    "X7NL_full_v1_deepseek_v4",
    "X7NL_full_v1_llama4_scout",
    # In-progress / partial sub-runs (aggregate has the data)
    "L1_gemma4_flagship_s0_t500",
    "L1_gemma4_flagship_s1_t500",
    "L1_gemma4_flagship_s2_t500",
    "L1_qwen3_s0_t500",
    "L1_qwen3_s1_t500",
    "L1_qwen3_s2_t500",
    "L_dryrun",
    "X7_smoke",
    "smoke_a3a5",
    "Q1_smoke",
}

# NOTE: some X7_mech runs don't have cells.jsonl because the mechanistic data
# is stored in JSON files, not JSONL.  Exempt those too.
EXEMPT_RUNS |= {"X7_mech_v1_b1", "X7_mech_v1_b2", "X7_mech_v1_b3"}


def _run_dirs():
    if not RUNS_DIR.exists():
        return []
    return [
        d for d in sorted(RUNS_DIR.iterdir())
        if d.is_dir() and not d.name.startswith(".") and d.name not in EXEMPT_RUNS
    ]


@pytest.mark.parametrize("run_dir", _run_dirs(), ids=lambda d: d.name)
def test_run_has_cells_jsonl(run_dir: Path):
    """Every non-exempt run directory must have ``cells.jsonl``."""
    cells = run_dir / "cells.jsonl"
    assert cells.exists(), (
        f"Run '{run_dir.name}' is missing cells.jsonl — raw data must be retained. "
        "Either produce the artifact or add the run to EXEMPT_RUNS in "
        "tests/test_run_authenticity.py with a documented reason."
    )
    # cells.jsonl must not be empty
    assert cells.stat().st_size > 0, (
        f"Run '{run_dir.name}' has an empty cells.jsonl — artifact is incomplete."
    )


@pytest.mark.parametrize("run_dir", _run_dirs(), ids=lambda d: d.name)
def test_run_has_env_json(run_dir: Path):
    """Every non-exempt run directory must have a valid ``env.json``."""
    env_path = run_dir / "env.json"
    assert env_path.exists(), (
        f"Run '{run_dir.name}' is missing env.json — reproducibility context required. "
        "Add it via deltamemory.utils.run_env.write_env_json() or tools/env_writer.py."
    )
    try:
        env = json.loads(env_path.read_text())
    except json.JSONDecodeError as exc:
        pytest.fail(f"Run '{run_dir.name}': env.json is invalid JSON: {exc}")

    # Check mandatory keys
    for key in ("commit", "dtype", "device"):
        assert key in env, (
            f"Run '{run_dir.name}': env.json missing key '{key}'. "
            "Regenerate with write_env_json()."
        )
