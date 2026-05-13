from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_BENCH_SMOKE") != "1",
    reason="slow benchmark smoke is opt-in; set RUN_BENCH_SMOKE=1",
)
def test_bench_smoke_subprocess_jsonl():
    out = Path("experiments/bench/_smoke_test_output.jsonl")
    if out.exists():
        out.unlink()
    try:
        cmd = [
            sys.executable,
            "experiments/bench/run_bench.py",
            "--smoke",
            "--inject",
            "caa",
            "--iters",
            "3",
            "--warmup",
            "1",
            "--batch",
            "1",
            "--seq",
            "8",
            "--out",
            str(out),
        ]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        assert "| tokens/sec |" in result.stdout
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 3
        assert {row["inject"] for row in rows} == {"caa"}
        assert all(row["smoke"] for row in rows)
        assert all(row["tokens_per_sec"] > 0 for row in rows)
    finally:
        if out.exists():
            out.unlink()
