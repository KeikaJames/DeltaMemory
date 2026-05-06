"""A.3 ablation verdict — Wilcoxon + bootstrap analysis.

For each ablation arm (A3, A5, A6) in `runs/A_per_method_v1_qwen3/`,
computes a paired Wilcoxon signed-rank test and bootstrap 95% CI
of the mean difference (ablation − control) on `nll_new`.

Necessity claim is SUPPORTED if p < 0.01 AND the 95% CI excludes 0.

Writes `runs/A_per_method_v1_qwen3/VERDICT.md`.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy import stats

RUNS_ROOT = pathlib.Path(__file__).parent.parent.parent / "runs"
A_DIR = RUNS_ROOT / "A_per_method_v1_qwen3"

ARM_FILES = {
    "caa": A_DIR / "caa_arm" / "cells.jsonl",
    "lopi": A_DIR / "lopi_arm" / "cells.jsonl",
}

N_RESAMPLES = 10_000
ALPHA_LEVEL = 0.01  # necessity threshold


@dataclass
class VerdictRow:
    name: str
    method: str
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    supported: bool
    n_pairs: int
    note: str = ""


def load_cells(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_pairs(
    cells: list[dict],
    ablation_arm: str,
    metric: str = "nll_new",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (control_values, ablation_values) arrays paired by (prompt_id, seed).

    Pairs are matched on (prompt_id, seed, alpha) where alpha>0 for treatment.
    If there is no exact match the pair is dropped; the number of valid pairs is
    reported in the calling function.
    """
    control_map: dict[tuple, float] = {}
    ablation_map: dict[tuple, float] = {}

    for c in cells:
        if c.get("status") != "ok":
            continue
        key = (c["prompt_id"], c["seed"])
        if c["arm"] == "control":
            control_map[key] = c[metric]
        elif c["arm"] == ablation_arm:
            ablation_map[key] = c[metric]

    shared_keys = sorted(set(control_map) & set(ablation_map))
    ctrl = np.array([control_map[k] for k in shared_keys])
    abl = np.array([ablation_map[k] for k in shared_keys])
    return ctrl, abl


def bootstrap_ci(diffs: np.ndarray, n_resamples: int = N_RESAMPLES, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap (n_resamples) 95% CI of the mean diff."""
    rng = np.random.default_rng(42)
    means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_resamples)
    ])
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


def wilcoxon_paired(ctrl: np.ndarray, abl: np.ndarray) -> float:
    """Wilcoxon signed-rank test; returns p-value (two-sided)."""
    diffs = abl - ctrl
    if np.all(diffs == 0):
        return 1.0
    result = stats.wilcoxon(diffs, alternative="two-sided")
    return float(result.pvalue)


def analyse_arm_file(path: pathlib.Path) -> list[VerdictRow]:
    if not path.exists():
        return [VerdictRow(
            name="(file missing)",
            method=path.parent.name,
            mean_delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value=float("nan"),
            supported=False,
            n_pairs=0,
            note=f"Missing: {path}",
        )]

    cells = load_cells(path)
    ablation_arms = sorted({c["arm"] for c in cells if c["arm"] != "control"})
    method = cells[0]["method"] if cells else "unknown"
    rows: list[VerdictRow] = []

    for arm_name in ablation_arms:
        ctrl, abl = build_pairs(cells, arm_name)
        n_pairs = len(ctrl)
        if n_pairs < 2:
            rows.append(VerdictRow(
                name=arm_name, method=method,
                mean_delta=float("nan"), ci_low=float("nan"),
                ci_high=float("nan"), p_value=float("nan"),
                supported=False, n_pairs=n_pairs,
                note="Too few pairs for statistical test",
            ))
            continue

        diffs = abl - ctrl
        mean_delta = float(diffs.mean())
        ci_low, ci_high = bootstrap_ci(diffs)
        p_value = wilcoxon_paired(ctrl, abl)

        ci_excludes_zero = not (ci_low <= 0.0 <= ci_high)
        supported = p_value < ALPHA_LEVEL and ci_excludes_zero

        rows.append(VerdictRow(
            name=arm_name, method=method,
            mean_delta=mean_delta, ci_low=ci_low, ci_high=ci_high,
            p_value=p_value, supported=supported, n_pairs=n_pairs,
        ))

    return rows


def run(a_dir: pathlib.Path = A_DIR) -> list[VerdictRow]:
    all_rows: list[VerdictRow] = []
    for arm_file_key, path in ARM_FILES.items():
        rows = analyse_arm_file(path)
        all_rows.extend(rows)

    # Write VERDICT.md
    lines = [
        "# A.3 Ablation Verdict\n\n",
        "Paired Wilcoxon signed-rank test + bootstrap 95% CI on `nll_new` "
        "(ablation − control, paired by prompt_id × seed).\n\n",
        "**Necessity claim**: SUPPORTED iff `p < 0.01` AND 95% CI excludes 0.\n\n",
        "## Results\n\n",
        "| arm | method | n_pairs | mean_delta | ci_low | ci_high | p_value | supported |\n",
        "|-----|--------|--------:|-----------:|-------:|--------:|--------:|:---------:|\n",
    ]

    for r in all_rows:
        if r.n_pairs == 0:
            row_str = (
                f"| {r.name} | {r.method} | 0 | N/A | N/A | N/A | N/A | ❌ |\n"
            )
        else:
            supported_str = "✅ SUPPORTED" if r.supported else "❌ NOT SUPPORTED"
            row_str = (
                f"| {r.name} | {r.method} | {r.n_pairs} "
                f"| {r.mean_delta:+.4f} | {r.ci_low:+.4f} | {r.ci_high:+.4f} "
                f"| {r.p_value:.4f} | {supported_str} |\n"
            )
        lines.append(row_str)

    lines += [
        "\n## Notes\n\n",
        "- **A5** (CAA random steering): replaces target-mean vector with a seeded random unit vector.\n",
        "- **A3** (LOPI η_σ=1): force eta_sigma=1 (disable σ-shrink) in `lopi_default`.\n",
        "- **A6** (LOPI θ=0): force theta=0 in `lopi_default`.\n\n",
        "## Interpretation\n\n",
    ]

    for r in all_rows:
        if r.n_pairs < 2:
            lines.append(f"- **{r.name}**: insufficient data ({r.n_pairs} pairs). {r.note}\n")
        elif r.supported:
            lines.append(
                f"- **{r.name}**: NECESSARY — removing this component degrades `nll_new` by "
                f"{r.mean_delta:+.4f} nats on average "
                f"(95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}], p={r.p_value:.4f}).\n"
            )
        else:
            ci_note = "CI includes 0" if not (r.ci_low > 0 or r.ci_high < 0) else "p ≥ 0.01"
            lines.append(
                f"- **{r.name}**: NO-OP — ablation does not significantly affect `nll_new` "
                f"(mean Δ={r.mean_delta:+.4f}, p={r.p_value:.4f}, {ci_note}).\n"
            )

    lines += [
        "\n## Data gaps\n\n",
        "- `scar_arm` and `bank_arm` not available (methods not registered in dispatcher).\n",
        "- A1, A2, A4 verdicts pending extension of run.py (see REPORT.md §Next actions).\n",
    ]

    verdict_path = a_dir / "VERDICT.md"
    verdict_path.write_text("".join(lines))
    print(f"[verdict.py] Verdict written to {verdict_path}")
    return all_rows


if __name__ == "__main__":
    run()
