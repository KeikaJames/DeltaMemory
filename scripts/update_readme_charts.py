#!/usr/bin/env python3
"""Regenerate the AUTOGEN block in README.md from experiment summary files.

Reads every ``reports/experiments/*/delta_experiment_summary.json``, builds a
Stage 6 channel comparison table from each report's ``stage2_binding_summary``
section, and rewrites the block delimited by::

    <!-- BEGIN AUTOGEN: stage6 -->
    ...
    <!-- END AUTOGEN: stage6 -->

The script is idempotent. Running it multiple times produces the same output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CHANNELS_OF_INTEREST = [
    "no_memory",
    "delta_qv",
    "payload_probe",
    "logit_bias",
    "lm_head_lora",
    "oracle_logit_answer_embedding",
]
BEGIN_MARK = "<!-- BEGIN AUTOGEN: stage6 -->"
END_MARK = "<!-- END AUTOGEN: stage6 -->"


def collect_summaries(reports_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(reports_dir.rglob("delta_experiment_summary.json")):
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception:
            continue
        rows.append({"path": summary_path, "summary": summary})
    return rows


def _binding_metric(stage2: dict[str, Any], channel: str, key: str) -> Any:
    block = (stage2 or {}).get("eval_modes", {}).get(channel) or {}
    return block.get(key)


def _aggregate_metric(eval_block: dict[str, Any], channel: str, key: str) -> Any:
    return ((eval_block or {}).get("aggregate", {}).get(channel) or {}).get(key)


def _format_row(report_name: str, summary: dict[str, Any]) -> str | None:
    cfg = summary.get("config", {})
    if not cfg.get("oracle_span_writer"):
        return None
    final_eval = summary.get("final_eval", {})
    stage2 = summary.get("stage2_binding_summary", {})
    cells = [
        report_name,
        cfg.get("task_suite", ""),
        cfg.get("writer_pool", "mean"),
        str(cfg.get("train_samples", "")),
        str(cfg.get("eval_samples", "")),
        str(cfg.get("seed", "")),
    ]
    for channel in CHANNELS_OF_INTEREST:
        nll = _aggregate_metric(final_eval, channel, "answer_nll")
        top1 = _binding_metric(stage2, channel, "top1_correct_rate")
        if top1 is None:
            top1 = _aggregate_metric(final_eval, channel, "top1_correct_rate")
        nll_str = f"{nll:.3f}" if isinstance(nll, (int, float)) else "—"
        top1_str = f"{top1:.3f}" if isinstance(top1, (int, float)) else "—"
        cells.append(f"{nll_str} / {top1_str}")
    return "| " + " | ".join(cells) + " |"


def render_block(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Stage 6 live experiment summary (auto-generated)",
        "",
        "Each row is one Stage 6 oracle-span-writer run. Cells are `held-out NLL / top1`.",
        "",
        "| Report | Suite | Pool | Train | Eval | Seed | "
        + " | ".join(CHANNELS_OF_INTEREST)
        + " |",
        "| " + " | ".join(["---"] * (6 + len(CHANNELS_OF_INTEREST))) + " |",
    ]
    table_rows = []
    for row in rows:
        formatted = _format_row(row["path"].parent.name, row["summary"])
        if formatted is not None:
            table_rows.append(formatted)
    if not table_rows:
        table_rows.append("| _no Stage 6 oracle-span runs found_ | — | — | — | — | — | "
                          + " | ".join(["—"] * len(CHANNELS_OF_INTEREST)) + " |")
    lines.extend(table_rows)
    lines.extend([
        "",
        "> Pass gate (Stage 6 strict): held-out top1 >= 0.85 on at least one of "
        "`payload_probe`, `logit_bias`, `lm_head_lora`, while `delta_qv` stays < 0.5 "
        "(Story A negative reference).",
        "",
        "Regenerate with `python3 scripts/update_readme_charts.py`.",
    ])
    return "\n".join(lines)


def rewrite_readme(readme_path: Path, block: str) -> bool:
    text = readme_path.read_text(encoding="utf-8")
    new_block = f"{BEGIN_MARK}\n{block}\n{END_MARK}"
    if BEGIN_MARK in text and END_MARK in text:
        before = text.split(BEGIN_MARK)[0]
        after = text.split(END_MARK, 1)[1]
        new_text = before + new_block + after
    else:
        sep = "\n\n---\n\n" if not text.endswith("\n") else "\n---\n\n"
        new_text = text.rstrip() + sep + new_block + "\n"
    if new_text == text:
        return False
    readme_path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports/experiments")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = (repo_root / args.reports_dir).resolve()
    readme_path = (repo_root / args.readme).resolve()
    rows = collect_summaries(reports_dir)
    block = render_block(rows)
    changed = rewrite_readme(readme_path, block)
    print(json.dumps({
        "reports_dir": str(reports_dir),
        "readme": str(readme_path),
        "rows_total": len(rows),
        "readme_updated": changed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
