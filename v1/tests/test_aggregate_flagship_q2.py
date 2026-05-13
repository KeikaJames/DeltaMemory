"""Regression test for ``aggregate_flagship_q2`` partial-sweep handling.

Codex P2: when a model has no shield-on rows yet (partial sweep), the H1
``max_drift`` value is ``None``; the formatting code must guard against that.
"""
from __future__ import annotations

from pathlib import Path

from scripts.legacy.aggregate_flagship_q2 import check_hypotheses, render_report


def test_partial_sweep_no_shield_on_does_not_raise(tmp_path: Path) -> None:
    rows = [
        {"model": "model_a", "alpha": 0.5, "shield": False, "seed": 0,
         "mean_lift": 1.0, "mean_drift": 0.1},
    ]
    hypotheses = check_hypotheses(rows)
    assert hypotheses["h1"]["model_a"]["max_drift"] is None
    out = tmp_path / "AGGREGATE.md"
    # Must not raise TypeError on None max_drift formatting.
    render_report({"model_a"}, rows, hypotheses, out)
    text = out.read_text()
    assert "model_a" in text
    assert "max drift=N/A" in text
