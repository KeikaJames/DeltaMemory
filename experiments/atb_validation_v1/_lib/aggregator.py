"""Aggregator: builds summary.csv, tables/result_table.tex and plots/*.png
from a results.jsonl produced by any of the six experiment runners."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CANONICAL_COLUMNS = [
    "experiment", "model", "dataset", "variant", "method", "alpha",
    "bank_size", "n",
    "recall_at_1", "mean_margin", "median_margin",
    "js_drift", "kl_drift",
    "bank_attention_mass", "max_bank_prob",
    "mean_target_rank",
    "torch_equal_all", "max_abs_diff_max",
]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _safe_median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def aggregate(
    results_path: Path,
    *,
    experiment: str,
    model: str,
    dataset: str,
    out_dir: Path,
) -> Path:
    """Read results.jsonl and write summary.csv. Return summary path."""
    rows = _read_jsonl(results_path)
    if not rows:
        raise RuntimeError(f"no rows in {results_path}")
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r.get("variant", "unknown")].append(r)
    summary_rows: list[dict] = []
    for variant, vrows in by_variant.items():
        first = vrows[0]
        margins = [r.get("margin") for r in vrows
                   if r.get("margin") is not None]
        recalls = [bool(r.get("recall_at_1")) for r in vrows
                   if "recall_at_1" in r]
        ranks = [r.get("target_rank") for r in vrows
                 if r.get("target_rank") is not None]
        js = [r.get("js_drift") for r in vrows
              if r.get("js_drift") is not None]
        kl = [r.get("kl_drift") for r in vrows
              if r.get("kl_drift") is not None]
        bam = [r.get("bank_attention_mass") for r in vrows
               if r.get("bank_attention_mass") is not None]
        mbp = [r.get("max_bank_prob") for r in vrows
               if r.get("max_bank_prob") is not None]
        torch_eq = [bool(r.get("torch_equal")) for r in vrows
                    if "torch_equal" in r]
        max_abs = [r.get("max_abs_diff") for r in vrows
                   if r.get("max_abs_diff") is not None]
        summary_rows.append({
            "experiment": experiment,
            "model": model,
            "dataset": dataset,
            "variant": variant,
            "method": first.get("method", ""),
            "alpha": first.get("alpha", ""),
            "bank_size": first.get("bank_size", ""),
            "n": len(vrows),
            "recall_at_1": (sum(recalls) / len(recalls)) if recalls else "",
            "mean_margin": _safe_mean(margins) if margins else "",
            "median_margin": _safe_median(margins) if margins else "",
            "js_drift": _safe_mean(js) if js else "",
            "kl_drift": _safe_mean(kl) if kl else "",
            "bank_attention_mass": _safe_mean(bam) if bam else "",
            "max_bank_prob": _safe_mean(mbp) if mbp else "",
            "mean_target_rank": _safe_mean(ranks) if ranks else "",
            "torch_equal_all": (all(torch_eq) if torch_eq else ""),
            "max_abs_diff_max": (max(max_abs) if max_abs else ""),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    _write_tex(summary_rows, out_dir / "tables" / "result_table.tex")
    return csv_path


def _write_tex(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "method", "alpha", "n", "recall_at_1",
            "mean_margin", "median_margin", "js_drift",
            "bank_attention_mass", "mean_target_rank"]
    headers = ["Variant", "Method", "$\\alpha$", "n",
               "Recall@1", "Margin (mean)", "Margin (med)",
               "JS drift", "Bank mass", "Mean rank"]
    lines = [
        "\\begin{tabular}{l l r r r r r r r r}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def append_to_global(summary_path: Path, global_path: Path) -> None:
    """Concatenate per-experiment summary.csv into atb_validation_v1/SUMMARY.csv."""
    rows: list[dict] = []
    if global_path.exists():
        with open(global_path) as f:
            r = csv.DictReader(f)
            rows = list(r)
    with open(summary_path) as f:
        r = csv.DictReader(f)
        rows.extend(list(r))
    global_path.parent.mkdir(parents=True, exist_ok=True)
    with open(global_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CANONICAL_COLUMNS})
