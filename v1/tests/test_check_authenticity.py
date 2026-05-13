"""Unit tests for tools/check_authenticity.py.

Build temp experiment trees and assert the contract checker accepts /
rejects them as documented in docs/authenticity.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "check_authenticity.py"


def _good_env() -> dict:
    return {
        "commit": "0" * 40,
        "dirty": False,
        "dirty_diff_sha1": "0" * 40,
        "prereg_version": "TEST.v1",
        "dataset_sha1": "1" * 40,
        "torch": "2.11.0",
        "transformers": "5.7.0",
        "python": "3.14.0",
        "device": "cpu",
        "dtype": "float32",
        "started_at": "2026-05-06T00:00:00Z",
        "host": "test-host",
    }


def _write_experiment(root: Path, *, env: dict | None, cells: list[dict] | None, summary: dict | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if env is not None:
        (root / "env.json").write_text(json.dumps(env))
    if cells is not None:
        with (root / "cells.jsonl").open("w") as f:
            for row in cells:
                f.write(json.dumps(row) + "\n")
    if summary is not None:
        (root / "summary.json").write_text(json.dumps(summary))


def _run_check(*paths: Path, strict: bool = False, bit_equality: bool = False) -> tuple[int, str]:
    argv = [sys.executable, str(TOOL)]
    if strict:
        argv.append("--strict")
    if bit_equality:
        argv.append("--bit-equality")
    if paths:
        argv.append("--paths")
        argv.extend(str(p.relative_to(REPO)) for p in paths)
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=REPO)
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Create a fresh experiments root inside the repo so check_authenticity walks it."""
    root = REPO / "experiments" / "_authenticity_test_tmp"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    yield root
    if root.exists():
        import shutil

        shutil.rmtree(root)


def test_passes_with_complete_env_and_cells(sandbox):
    d = sandbox / "good"
    _write_experiment(
        d,
        env=_good_env(),
        cells=[{"cell_id": "x", "model": "m", "method": "caa", "alpha": 0.0, "drift": 0.0}],
        summary={"n": 1},
    )
    code, out = _run_check(d, strict=True)
    assert code == 0, out


def test_fails_when_env_missing(sandbox):
    d = sandbox / "no-env"
    _write_experiment(d, env=None, cells=[{"x": 1}], summary=None)
    code, out = _run_check(d, strict=True)
    assert code == 1
    assert "missing env.json" in out


def test_fails_when_env_incomplete(sandbox):
    d = sandbox / "thin-env"
    bad = _good_env()
    del bad["dataset_sha1"]
    del bad["host"]
    _write_experiment(d, env=bad, cells=[{"x": 1}], summary=None)
    code, out = _run_check(d, strict=True)
    assert code == 1
    assert "dataset_sha1" in out
    assert "host" in out


def test_fails_when_summary_without_cells(sandbox):
    d = sandbox / "agg-only"
    _write_experiment(d, env=_good_env(), cells=None, summary={"n": 0})
    code, out = _run_check(d, strict=True)
    assert code == 1
    assert "aggregate-only forbidden" in out


def test_bit_equality_witness_required(sandbox):
    d = sandbox / "no-witness"
    _write_experiment(
        d,
        env=_good_env(),
        cells=[
            {"cell_id": "a", "model": "m", "method": "caa", "alpha": 1.0, "drift": 0.5},
            {"cell_id": "b", "model": "m", "method": "caa", "alpha": 2.0, "drift": 0.9},
        ],
        summary=None,
    )
    code, out = _run_check(d, strict=True, bit_equality=True)
    assert code == 1
    assert "no alpha=0 drift==0 witness" in out


def test_bit_equality_witness_satisfied(sandbox):
    d = sandbox / "witness"
    _write_experiment(
        d,
        env=_good_env(),
        cells=[
            {"cell_id": "a", "model": "m", "method": "caa", "alpha": 0.0, "drift": 0.0},
            {"cell_id": "b", "model": "m", "method": "caa", "alpha": 1.0, "drift": 0.5},
        ],
        summary=None,
    )
    code, out = _run_check(d, strict=True, bit_equality=True)
    assert code == 0, out


def test_grandfathered_w4_skipped_by_default():
    """W4 has incomplete env.json on disk; default mode must still pass."""
    code, out = _run_check()
    assert code == 0, out
    assert "PASS" in out
