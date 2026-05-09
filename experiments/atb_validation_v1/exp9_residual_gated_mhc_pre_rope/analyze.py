"""Exp9 — Residual-Gated mHC AttnNativeBank — post-processing analysis.

Analyzes all phases produced by run.py:

  Phase A1: phase_a1/mode_MODE_beta_BETA/results.jsonl
  Phase A2: phase_a2/config_N_.../results.jsonl
  Phase B:  phase_b/results.jsonl
  Phase C:  phase_c/alpha_X_XX/results.jsonl

Run after rsyncing results from spark1:

    rsync -av spark1:~/projects/RCV-HC/experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope/ \\
        experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope/

Then:

    python experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope/analyze.py \\
        --run-dir experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope/run_YYYYMMDD_HHMMSS \\
        --out experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope/analysis

Or auto-detect latest run:

    python experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope/analyze.py \\
        --exp-dir experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope \\
        --out experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EXP_NAME = "exp9_residual_gated_mhc_pre_rope"

VARIANT_ORDER = [
    "correct_bank",
    "shuffled_bank",
    "random_kv",
    "correct_K_random_V",
    "random_K_correct_V",
]

CONTROL_VARIANTS = [v for v in VARIANT_ORDER if v != "correct_bank"]
A1_CONTROL_VARIANTS = ["random_kv", "random_K_correct_V"]


# ---------------------------------------------------------------------------
# Stats helpers

def _isnan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return True


def _safe_mean(xs: list) -> float:
    xs = [float(x) for x in xs if x is not None and not _isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _safe_median(xs: list) -> float:
    xs = sorted(float(x) for x in xs if x is not None and not _isnan(x))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def bootstrap_ci(vals: list, n_boot: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    vals = [float(v) for v in vals if not _isnan(v)]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    mean = sum(vals) / n
    boots = [sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return mean, lo, hi


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt(v: Any, d: int = 4) -> str:
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Latest run finder

def latest_run(exp_dir: Path) -> Path | None:
    candidates = sorted(
        [p.parent.parent.parent for p in exp_dir.glob("run_*/phase_a1/*/results.jsonl")],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Per-directory variant stats

def compute_variant_stats(rows: list[dict]) -> dict[str, dict]:
    by_v: dict[str, list] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)

    def stats(vname: str) -> dict:
        vrows = by_v.get(vname, [])
        margins = [r["margin"] for r in vrows if "margin" in r]
        recalls = [bool(r.get("recall_at_1")) for r in vrows]
        ranks = [r["target_rank"] for r in vrows if "target_rank" in r]
        js = [r["js_drift"] for r in vrows if r.get("js_drift") is not None]
        kl = [r["kl_drift"] for r in vrows if r.get("kl_drift") is not None]
        bam = [r["bank_attention_mass"] for r in vrows
               if r.get("bank_attention_mass") is not None]
        m, lo, hi = bootstrap_ci(margins)
        return {
            "n": len(vrows),
            "recall_at_1": sum(recalls) / max(len(recalls), 1),
            "mean_margin": m,
            "ci_lo": lo,
            "ci_hi": hi,
            "median_margin": _safe_median(margins),
            "js_drift": _safe_mean(js),
            "kl_drift": _safe_mean(kl),
            "mean_target_rank": _safe_mean(ranks),
            "mean_bank_attention_mass": _safe_mean(bam),
        }

    return {v: stats(v) for v in VARIANT_ORDER if v in by_v}


def gap_score(variant_stats: dict, control_names: list[str] | None = None) -> float:
    correct = variant_stats.get("correct_bank", {}).get("mean_margin", float("-inf"))
    cnames = control_names or CONTROL_VARIANTS
    controls = [variant_stats[v]["mean_margin"] for v in cnames if v in variant_stats]
    return correct - max(controls) if controls else float("nan")


def verdict_b(variant_stats: dict, exp8_drift: float | None = None) -> dict:
    correct = variant_stats.get("correct_bank", {})
    controls = [v for v in CONTROL_VARIANTS if v in variant_stats]

    gap = gap_score(variant_stats)
    correct_ci_lo = correct.get("ci_lo", float("-inf"))
    max_ctrl_mean = max(
        (variant_stats[v].get("mean_margin", float("-inf")) for v in controls),
        default=float("-inf"),
    )
    strict_ci = correct_ci_lo > max_ctrl_mean

    drift_ok = True
    if exp8_drift is not None:
        my_drift = correct.get("js_drift", float("nan"))
        if not _isnan(my_drift):
            drift_ok = my_drift <= exp8_drift * 1.05  # 5% tolerance

    rkcv = variant_stats.get("random_K_correct_V", {})
    v_dominates = (rkcv.get("mean_margin", -999) > correct.get("mean_margin", -999) + 0.05)
    controls_dropped = all(
        variant_stats[v].get("mean_margin", 0) < -0.20
        for v in ["random_kv", "random_K_correct_V"] if v in variant_stats
    )

    if gap > 0 and strict_ci and drift_ok:
        label = "PASS_STRONG"
    elif gap > 0:
        label = "PASS_DIRECTIONAL"
    elif controls_dropped:
        label = "STABILIZER_ONLY"
    else:
        label = "FAIL"

    return {
        "verdict": label,
        "gap": gap,
        "strict_ci_dominance": strict_ci,
        "pattern_v_dominates": v_dominates,
        "drift_ok": drift_ok,
    }


# ---------------------------------------------------------------------------
# Phase A1 analysis

def analyze_phase_a1(run_dir: Path) -> list[dict]:
    phase_dir = run_dir / "phase_a1"
    if not phase_dir.exists():
        return []
    results = []
    for sub in sorted(phase_dir.iterdir()):
        jsonl = sub / "results.jsonl"
        if not jsonl.exists():
            continue
        rows = _read_jsonl(jsonl)
        if not rows:
            continue
        mode = rows[0].get("bank_separate_softmax", False) and "sep_beta_mhc" or "merged_beta_mhc"
        # Re-derive mode from data
        sep = rows[0].get("bank_separate_softmax", False)
        mode = "sep_beta_mhc" if sep else "merged_beta_mhc"
        beta = float(rows[0].get("bank_merge_beta", 1.0))
        vstats = compute_variant_stats(rows)
        a1_gap = gap_score(vstats, A1_CONTROL_VARIANTS)
        full_gap = gap_score(vstats)
        results.append({
            "mode": mode,
            "beta": beta,
            "gap_a1": a1_gap,
            "gap_full": full_gap,
            "n_total": len(rows),
            "variants": vstats,
            "subdir": str(sub),
        })
    return results


def pick_best_a1_configs(phase_a1: list[dict], top_n: int = 2) -> list[dict]:
    return sorted(phase_a1, key=lambda x: x["gap_a1"], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Phase A2 analysis

def analyze_phase_a2(run_dir: Path) -> list[dict]:
    phase_dir = run_dir / "phase_a2"
    if not phase_dir.exists():
        return []
    results = []
    for sub in sorted(phase_dir.iterdir()):
        jsonl = sub / "results.jsonl"
        if not jsonl.exists():
            continue
        rows = _read_jsonl(jsonl)
        if not rows:
            continue
        sep = rows[0].get("bank_separate_softmax", False)
        mode = "sep_beta_mhc" if sep else "merged_beta_mhc"
        beta = float(rows[0].get("bank_merge_beta", 1.0))
        vstats = compute_variant_stats(rows)
        a2_gap = gap_score(vstats)
        results.append({
            "mode": mode,
            "beta": beta,
            "gap_a2": a2_gap,
            "n_total": len(rows),
            "variants": vstats,
            "subdir": str(sub),
        })
    return results


def pick_best_config(phase_a2: list[dict]) -> dict | None:
    if not phase_a2:
        return None
    return sorted(phase_a2, key=lambda x: x["gap_a2"], reverse=True)[0]


# ---------------------------------------------------------------------------
# Phase B analysis

def analyze_phase_b(run_dir: Path) -> dict | None:
    jsonl = run_dir / "phase_b" / "results.jsonl"
    if not jsonl.exists():
        return None
    rows = _read_jsonl(jsonl)
    if not rows:
        return None
    sep = rows[0].get("bank_separate_softmax", False)
    mode = "sep_beta_mhc" if sep else "merged_beta_mhc"
    beta = float(rows[0].get("bank_merge_beta", 1.0))
    kappa = float(rows[0].get("mhc_kappa", float("nan")))
    vstats = compute_variant_stats(rows)
    return {
        "mode": mode,
        "beta": beta,
        "kappa": kappa,
        "n_total": len(rows),
        "variants": vstats,
    }


# ---------------------------------------------------------------------------
# Phase C analysis

def analyze_phase_c(run_dir: Path) -> list[dict]:
    phase_dir = run_dir / "phase_c"
    if not phase_dir.exists():
        return []
    results = []
    for sub in sorted(phase_dir.iterdir()):
        jsonl = sub / "results.jsonl"
        if not jsonl.exists():
            continue
        rows = _read_jsonl(jsonl)
        if not rows:
            continue
        alpha = float(rows[0].get("alpha", float("nan")))
        vstats = compute_variant_stats(rows)
        results.append({
            "alpha": alpha,
            "n_total": len(rows),
            "variants": vstats,
            "subdir": str(sub),
        })
    return results


# ---------------------------------------------------------------------------
# CSV / LaTeX writers

def write_summary_csv(label: str, variant_stats: dict, path: Path,
                      extra_cols: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["phase", "variant", "n", "mean_margin", "ci_lo", "ci_hi",
                  "median_margin", "recall_at_1", "mean_target_rank",
                  "js_drift", "kl_drift", "mean_bank_attention_mass"]
    if extra_cols:
        fieldnames += list(extra_cols.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for vname in VARIANT_ORDER:
            if vname not in variant_stats:
                continue
            row = {"phase": label, "variant": vname, **variant_stats[vname]}
            if extra_cols:
                row.update(extra_cols)
            w.writerow(row)


def write_a1_comparison_csv(phase_a1: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "beta", "gap_a1", "gap_full",
                                          "n_total", "correct_margin",
                                          "max_control_margin"])
        w.writeheader()
        for entry in sorted(phase_a1, key=lambda x: (x["mode"], x["beta"])):
            correct = entry["variants"].get("correct_bank", {}).get("mean_margin", float("nan"))
            ctrls = [entry["variants"][v]["mean_margin"]
                     for v in CONTROL_VARIANTS if v in entry["variants"]]
            w.writerow({
                "mode": entry["mode"],
                "beta": entry["beta"],
                "gap_a1": _fmt(entry["gap_a1"]),
                "gap_full": _fmt(entry.get("gap_full", float("nan"))),
                "n_total": entry["n_total"],
                "correct_margin": _fmt(correct),
                "max_control_margin": _fmt(max(ctrls) if ctrls else float("nan")),
            })


def write_tex(variant_stats: dict, phase_label: str, mode: str | None,
              beta: float | None, alpha: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("% Auto-generated by analyze.py — do not edit.\n")
        header = r"\begin{tabular}{lrrrrrr}" + "\n"
        f.write(header)
        f.write(r"\toprule" + "\n")
        f.write(r"Variant & N & Mean Margin & 95\% CI & Recall@1 & JS Drift & KL Drift \\" + "\n")
        f.write(r"\midrule" + "\n")
        for vname in VARIANT_ORDER:
            s = variant_stats.get(vname)
            if s is None:
                continue
            label = vname.replace("_", r"\_")
            ci = f"[{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]"
            f.write(
                f"{label} & {s['n']} & {_fmt(s['mean_margin'])} & {ci} & "
                f"{_fmt(s['recall_at_1'])} & {_fmt(s['js_drift'])} & {_fmt(s['kl_drift'])} \\\\\n"
            )
        note = ""
        if mode:
            note += f"mode={mode}"
        if beta is not None:
            note += f", $\\beta$={beta}"
        note += f", $\\alpha$={alpha}"
        f.write(r"\midrule" + "\n")
        f.write(f"\\multicolumn{{7}}{{l}}{{{note}}} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")


def write_phase_b_plot(phase_b: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print(f"  [skip plot] matplotlib not available: {path}")
        return

    vstats = phase_b["variants"]
    names = [v for v in VARIANT_ORDER if v in vstats]
    means = [vstats[v]["mean_margin"] for v in names]
    errs_lo = [vstats[v]["mean_margin"] - vstats[v]["ci_lo"] for v in names]
    errs_hi = [vstats[v]["ci_hi"] - vstats[v]["mean_margin"] for v in names]
    colors = ["green" if n == "correct_bank" else "steelblue" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, means, color=colors, yerr=[errs_lo, errs_hi], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Margin (log P(target_new) − log P(target_true))")
    ax.set_title(f"Exp9 Phase B: mode={phase_b.get('mode')} β={phase_b.get('beta')}")
    ax.set_xticklabels([n.replace("_", "\n") for n in names], rotation=0, ha="center",
                       fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] Phase B bar chart → {path}")


def write_a1_plot(phase_a1: list[dict], exp8_gap: float | None, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print(f"  [skip plot] matplotlib not available: {path}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in ["merged_beta_mhc", "sep_beta_mhc"]:
        entries = sorted([e for e in phase_a1 if e["mode"] == mode], key=lambda x: x["beta"])
        betas = [e["beta"] for e in entries]
        gaps = [e["gap_a1"] for e in entries]
        ax.plot(betas, gaps, marker="o", label=mode)

    if exp8_gap is not None:
        ax.axhline(exp8_gap, color="red", linestyle="--",
                   label=f"Exp8 gap baseline ({exp8_gap:.3f})")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Beta (bank_merge_beta)")
    ax.set_ylabel("A1 Gap (correct_bank − max A1 control)")
    ax.set_title("Exp9 Phase A1: Beta × Mode Grid")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] A1 beta-gap chart → {path}")


def write_phase_c_plot(phase_c: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print(f"  [skip plot] matplotlib not available: {path}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    stress_variants = ["correct_bank", "random_kv", "random_K_correct_V"]
    for vname in stress_variants:
        alphas = [e["alpha"] for e in phase_c if vname in e["variants"]]
        margins = [e["variants"][vname]["mean_margin"]
                   for e in phase_c if vname in e["variants"]]
        if alphas:
            ax.plot(alphas, margins, marker="o", label=vname)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean Margin")
    ax.set_title("Exp9 Phase C: High-Alpha Stress")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] Phase C stress chart → {path}")


# ---------------------------------------------------------------------------
# README writer

def write_readme(run_dir: Path, out: Path, phase_a1: list[dict],
                 phase_a2: list[dict], phase_b: dict | None,
                 phase_c: list[dict], best_config: dict | None,
                 verdict: dict | None) -> None:
    lines = [
        f"# Exp9 Analysis: {run_dir.name}",
        "",
        "Generated by `analyze.py`. Do not edit.",
        "",
        "## Best Config",
    ]
    if best_config:
        lines += [
            f"- mode: `{best_config.get('mode')}`",
            f"- beta: `{best_config.get('beta')}`",
            f"- gap_a2: `{_fmt(best_config.get('gap_a2', best_config.get('gap_a1', '?')))}`",
        ]
    else:
        lines += ["(not yet determined — A2 may not have run)"]

    lines += ["", "## Phase A1 — Beta × Mode Grid (n=100, 3 variants)", ""]
    if phase_a1:
        lines += [f"| mode | beta | A1-gap |", "|------|------|--------|"]
        for e in sorted(phase_a1, key=lambda x: x["gap_a1"], reverse=True):
            lines.append(f"| {e['mode']} | {e['beta']} | {_fmt(e['gap_a1'])} |")
    else:
        lines += ["(not yet complete)"]

    lines += ["", "## Phase B — Full Validation (n=807)", ""]
    if phase_b:
        vstats = phase_b["variants"]
        lines += [f"| variant | mean_margin | 95% CI | recall@1 | JS drift |",
                  "|---------|-------------|--------|----------|----------|"]
        for v in VARIANT_ORDER:
            if v not in vstats:
                continue
            s = vstats[v]
            ci = f"[{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]"
            lines.append(
                f"| {v} | {_fmt(s['mean_margin'])} | {ci} | "
                f"{_fmt(s['recall_at_1'])} | {_fmt(s['js_drift'])} |"
            )
    else:
        lines += ["(not yet complete)"]

    if verdict:
        lines += ["", "## Phase B Verdict", "", f"**{verdict['verdict']}**",
                  f"- gap: {_fmt(verdict['gap'])}",
                  f"- strict_ci_dominance: {verdict['strict_ci_dominance']}",
                  f"- pattern_v_dominates: {verdict['pattern_v_dominates']}"]

    lines += ["", "## Phase C — High-Alpha Stress", ""]
    if phase_c:
        lines += [f"| alpha | correct_bank | random_kv | random_K_correct_V |",
                  "|-------|-------------|-----------|-------------------|"]
        for e in phase_c:
            vs = e["variants"]
            cb = _fmt(vs.get("correct_bank", {}).get("mean_margin", float("nan")))
            rkv = _fmt(vs.get("random_kv", {}).get("mean_margin", float("nan")))
            rkcv = _fmt(vs.get("random_K_correct_V", {}).get("mean_margin", float("nan")))
            lines.append(f"| {e['alpha']} | {cb} | {rkv} | {rkcv} |")
    else:
        lines += ["(not yet complete)"]

    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  [readme] {out / 'README.md'}")


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None, help="Path to specific run dir.")
    ap.add_argument("--exp-dir", default=None, help="Exp9 root; latest run auto-selected.")
    ap.add_argument("--out", required=True, help="Output directory for analysis artifacts.")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.exp_dir:
        run_dir = latest_run(Path(args.exp_dir))
        if run_dir is None:
            print("No run dirs found under", args.exp_dir, file=sys.stderr)
            return 1
    else:
        print("Provide --run-dir or --exp-dir", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing run_dir: {run_dir}")

    # Load Exp7/Exp8 baselines for comparison.
    exp8_gap: float | None = None
    exp8_json = (ROOT / "experiments" / "atb_validation_v1" /
                 "exp8_mhc_smoothed_pre_rope" / "analysis" / "analysis.json")
    if exp8_json.exists():
        try:
            data = json.loads(exp8_json.read_text())
            phase_b8 = data.get("phase_b") or {}
            vstats8 = phase_b8.get("variants") or {}
            if vstats8:
                correct8 = vstats8.get("correct_bank", {}).get("mean_margin", float("inf"))
                ctrls8 = [vstats8[v]["mean_margin"] for v in CONTROL_VARIANTS
                          if v in vstats8]
                if ctrls8:
                    exp8_gap = correct8 - max(ctrls8)
        except Exception:
            pass
    if exp8_gap is not None:
        print(f"Exp8 Phase B gap baseline: {exp8_gap:.4f}")

    # Analyze phases.
    phase_a1 = analyze_phase_a1(run_dir)
    phase_a2 = analyze_phase_a2(run_dir)
    phase_b = analyze_phase_b(run_dir)
    phase_c = analyze_phase_c(run_dir)

    # Load/recompute best config.
    best_config: dict | None = None
    bc_path = run_dir / "best_config.json"
    if bc_path.exists():
        best_config = json.loads(bc_path.read_text())
    elif phase_a2:
        best_config = pick_best_config(phase_a2)
    elif phase_a1:
        top2 = pick_best_a1_configs(phase_a1)
        if top2:
            best_config = top2[0]

    # Phase B verdict.
    verdict: dict | None = None
    if phase_b:
        exp8_drift: float | None = None
        # Get Exp8 Phase B correct_bank JS drift for comparison.
        if exp8_json.exists():
            try:
                data = json.loads(exp8_json.read_text())
                vstats8 = (data.get("phase_b") or {}).get("variants") or {}
                exp8_drift = vstats8.get("correct_bank", {}).get("js_drift")
            except Exception:
                pass
        verdict = verdict_b(phase_b["variants"], exp8_drift)
        print(f"Phase B verdict: {verdict['verdict']}  gap={verdict['gap']:.4f}")

    # Write A1 comparison CSV.
    if phase_a1:
        a1_csv = out / "phase_a1_comparison.csv"
        write_a1_comparison_csv(phase_a1, a1_csv)
        print(f"[csv] A1 comparison → {a1_csv}")

        # Per-config summary CSVs and LaTeX.
        tables_dir = out / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for entry in phase_a1:
            tag = f"mode_{entry['mode']}_beta_{entry['beta']:.2f}".replace(".", "_")
            write_summary_csv(f"A1_{tag}", entry["variants"],
                              out / f"a1_{tag}_summary.csv",
                              {"mode": entry["mode"], "beta": entry["beta"]})
            write_tex(entry["variants"], f"A1 mode={entry['mode']} β={entry['beta']}",
                      entry["mode"], entry["beta"], 0.05,
                      tables_dir / f"exp9_a1_{tag}.tex")

    # Phase B CSV and LaTeX.
    if phase_b:
        write_summary_csv("phase_b", phase_b["variants"],
                          out / "phase_b_summary.csv",
                          {"mode": phase_b.get("mode"), "beta": phase_b.get("beta")})
        write_tex(phase_b["variants"], "Phase B", phase_b.get("mode"),
                  phase_b.get("beta"), 0.05,
                  (out / "tables" / "exp9_phase_b.tex"))

    # Phase C CSV.
    if phase_c:
        c_path = out / "phase_c_summary.csv"
        with open(c_path, "w", newline="") as f:
            fieldnames = ["alpha", "variant", "n", "mean_margin", "ci_lo", "ci_hi",
                          "js_drift", "kl_drift"]
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for entry in phase_c:
                for vname, s in entry["variants"].items():
                    w.writerow({"alpha": entry["alpha"], "variant": vname, **s})
        print(f"[csv] Phase C summary → {c_path}")

    # Plots.
    plots_dir = out / "plots"
    if phase_a1:
        write_a1_plot(phase_a1, exp8_gap, plots_dir / "exp9_a1_beta_gap.png")
    if phase_b:
        write_phase_b_plot(phase_b, plots_dir / "exp9_phase_b.png")
    if phase_c:
        write_phase_c_plot(phase_c, plots_dir / "exp9_phase_c_stress.png")

    # Save analysis.json for finalize.py.
    analysis = {
        "run_dir": str(run_dir),
        "exp8_gap_baseline": exp8_gap,
        "best_config": best_config,
        "phase_a1": phase_a1,
        "phase_a2": phase_a2,
        "phase_b": phase_b,
        "phase_c": phase_c,
        "verdict": verdict,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    print(f"[json] analysis.json → {out / 'analysis.json'}")

    # README.
    write_readme(run_dir, out, phase_a1, phase_a2, phase_b, phase_c,
                 best_config, verdict)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
