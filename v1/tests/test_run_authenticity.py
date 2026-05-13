"""CI authenticity check: ensure every tracked run directory contains
``cells.jsonl`` and ``env.json``.

This test is part of the standard pytest suite and will fail if any run
directory tracked by Git and NOT explicitly excluded (see EXEMPT_RUNS below)
is missing either required artifact. Local-only archives under ``runs/`` are
ignored so production CI is not coupled to private experiment dumps.

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
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

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


def _tracked_run_dirs() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "runs"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    names = {
        Path(line).parts[1]
        for line in result.stdout.splitlines()
        if line.startswith("runs/") and len(Path(line).parts) >= 2
    }
    return [
        RUNS_DIR / name
        for name in sorted(names)
        if name not in EXEMPT_RUNS
    ]


@pytest.mark.parametrize("run_dir", _tracked_run_dirs(), ids=lambda d: d.name)
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


@pytest.mark.parametrize("run_dir", _tracked_run_dirs(), ids=lambda d: d.name)
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
