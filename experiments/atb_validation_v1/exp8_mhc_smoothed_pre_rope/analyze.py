"""Exp8 — mHC-Smoothed Pre-RoPE Negative Controls — post-processing analysis.

Analyzes all three phases produced by run.py:

  Phase A: phase_a/kappa_X_XX/results.jsonl   (3 kappas × 200 prompts)
  Phase B: phase_b/results.jsonl               (807 prompts, best kappa)
  Phase C: phase_c/alpha_X_XX/results.jsonl    (4 alphas × 200 prompts)

Run after rsyncing results from spark1:

    rsync -av spark1:~/projects/RCV-HC/experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/ \\
        experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/

Then:

    python experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/analyze.py \\
        --run-dir experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/run_YYYYMMDD_HHMMSS \\
        --out experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/analysis

Or to find the latest run automatically:

    python experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/analyze.py \\
        --exp-dir experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope \\
        --out experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/analysis
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EXP_NAME = "exp8_mhc_smoothed_pre_rope"

VARIANT_ORDER = [
    "correct_bank",
    "shuffled_bank",
    "random_kv",
    "correct_K_random_V",
    "random_K_correct_V",
]

VARIANT_LABELS = {
    "correct_bank": "Correct K/V + mHC",
    "shuffled_bank": "Correct K, Shuffled V + mHC",
    "random_kv": "Random K+V + mHC",
    "correct_K_random_V": "Correct K, Random V + mHC",
    "random_K_correct_V": "Random K, Correct V + mHC",
}


# ---------------------------------------------------------------------------
# Stats helpers

def _isnan(x):
    try:
        return math.isnan(float(x))
    except Exception:
        return True


def _safe_mean(xs):
    xs = [float(x) for x in xs if x is not None and not _isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _safe_median(xs):
    xs = sorted(float(x) for x in xs if x is not None and not _isnan(x))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def bootstrap_ci(vals, n_boot=10_000, seed=0, alpha=0.05):
    vals = [v for v in vals if not _isnan(v)]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    mean = sum(vals) / n
    boots = [sum(vals[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(n_boot)]
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


# ---------------------------------------------------------------------------
# Latest run finder

def latest_run(exp_dir: Path) -> Path | None:
    runs = sorted(exp_dir.glob("run_*/phase_a/*/results.jsonl"),
                  key=lambda p: p.parents[2].name)
    return runs[-1].parents[2] if runs else None


# ---------------------------------------------------------------------------
# Per-directory analysis

def compute_variant_stats(rows: list[dict]) -> dict[str, dict]:
    by_v: dict[str, list] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)

    def stats(vname):
        vrows = by_v.get(vname, [])
        margins = [r["margin"] for r in vrows if "margin" in r]
        recalls = [bool(r.get("recall_at_1")) for r in vrows]
        ranks = [r["target_rank"] for r in vrows if "target_rank" in r]
        js = [r["js_drift"] for r in vrows if r.get("js_drift") is not None]
        kl = [r["kl_drift"] for r in vrows if r.get("kl_drift") is not None]
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
        }

    return {v: stats(v) for v in VARIANT_ORDER if v in by_v}


def verdict_for_stats(variant_stats: dict) -> dict:
    correct = variant_stats.get("correct_bank", {})
    controls = [v for v in VARIANT_ORDER if v != "correct_bank" and v in variant_stats]

    dominates = all(
        correct.get("mean_margin", float("-inf")) >
        variant_stats[v].get("mean_margin", float("inf"))
        for v in controls
    )
    strict_ci = all(
        correct.get("ci_lo", float("-inf")) >
        variant_stats[v].get("mean_margin", float("inf"))
        for v in controls
    )

    rk_cv = variant_stats.get("random_K_correct_V", {})
    v_dominates = (
        (rk_cv.get("mean_margin") or -999) >
        (correct.get("mean_margin") or -999) + 0.05
    )
    k_addressing = (
        (correct.get("mean_margin") or -999) >
        (rk_cv.get("mean_margin") or -999) + 0.05
    )
    return {
        "correct_dominates": dominates,
        "strict_ci_dominance": strict_ci,
        "pattern_v_dominates": v_dominates,
        "pattern_k_addressing": k_addressing,
    }


# ---------------------------------------------------------------------------
# Phase analyses

def analyze_phase_a(run_dir: Path) -> list[dict]:
    """Analyze all kappa sub-dirs under phase_a/."""
    phase_a = run_dir / "phase_a"
    if not phase_a.exists():
        return []

    results = []
    for kappa_dir in sorted(phase_a.iterdir()):
        jsonl = kappa_dir / "results.jsonl"
        if not jsonl.exists():
            continue
        rows = _read_jsonl(jsonl)
        kappa = rows[0].get("mhc_kappa", None) if rows else None
        variant_stats = compute_variant_stats(rows)
        vd = verdict_for_stats(variant_stats)
        results.append({
            "kappa_dir": str(kappa_dir),
            "kappa": kappa,
            "n_total": len(rows),
            "variants": variant_stats,
            **vd,
        })
    return results


def analyze_phase_b(run_dir: Path) -> dict | None:
    """Analyze phase_b/results.jsonl (full 807-prompt run)."""
    jsonl = run_dir / "phase_b" / "results.jsonl"
    if not jsonl.exists():
        return None
    rows = _read_jsonl(jsonl)
    kappa = rows[0].get("mhc_kappa", None) if rows else None
    variant_stats = compute_variant_stats(rows)
    vd = verdict_for_stats(variant_stats)
    return {"kappa": kappa, "n_total": len(rows), "variants": variant_stats, **vd}


def analyze_phase_c(run_dir: Path) -> list[dict]:
    """Analyze all alpha sub-dirs under phase_c/."""
    phase_c = run_dir / "phase_c"
    if not phase_c.exists():
        return []
    results = []
    for alpha_dir in sorted(phase_c.iterdir()):
        jsonl = alpha_dir / "results.jsonl"
        if not jsonl.exists():
            continue
        rows = _read_jsonl(jsonl)
        alpha_val = rows[0].get("alpha", None) if rows else None
        kappa = rows[0].get("mhc_kappa", None) if rows else None
        variant_stats = compute_variant_stats(rows)
        vd = verdict_for_stats(variant_stats)
        results.append({
            "alpha_dir": str(alpha_dir),
            "alpha": alpha_val,
            "kappa": kappa,
            "n_total": len(rows),
            "variants": variant_stats,
            **vd,
        })
    return results


def pick_best_kappa(phase_a_results: list[dict]) -> float | None:
    """Return kappa with highest correct_bank - max_control_margin gap from Phase A.

    Maximising the gap selects the kappa where correct_bank most clearly
    outperforms controls, not just the kappa where it's highest in isolation.
    Falls back to correct_bank margin only if all gaps are equal (e.g., 1 variant).
    """
    best_kappa = None
    best_score = float("-inf")
    for entry in phase_a_results:
        vs = entry["variants"]
        cb_m = vs.get("correct_bank", {}).get("mean_margin", float("-inf"))
        controls = [v for v in vs if v != "correct_bank"]
        max_ctrl = max((vs[v].get("mean_margin", float("-inf")) for v in controls),
                       default=float("-inf"))
        gap = cb_m - max_ctrl
        if gap > best_score:
            best_score = gap
            best_kappa = entry["kappa"]
    return best_kappa


# ---------------------------------------------------------------------------
# Output writers

def write_summary_csv(phase_label: str, variant_stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "phase", "variant", "n", "recall_at_1",
              "mean_margin", "ci_lo", "ci_hi", "median_margin",
              "js_drift", "kl_drift", "mean_target_rank"]
    mode = "a" if path.exists() else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        for vname, s in variant_stats.items():
            row = {"experiment": EXP_NAME, "phase": phase_label, "variant": vname, **s}
            row = {k: row.get(k, "") for k in fields}
            w.writerow(row)


def _fmt(v, d=4):
    if v is None or (isinstance(v, float) and _isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def write_kappa_comparison_csv(phase_a_results: list[dict], path: Path) -> None:
    """One row per kappa with correct_bank stats for quick comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kappa", "correct_bank_mean_margin", "correct_bank_ci_lo",
              "correct_bank_ci_hi", "correct_dominates", "strict_ci_dominance",
              "pattern_v_dominates", "pattern_k_addressing", "n_total"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for entry in phase_a_results:
            cb = entry["variants"].get("correct_bank", {})
            w.writerow({
                "kappa": entry.get("kappa"),
                "correct_bank_mean_margin": cb.get("mean_margin"),
                "correct_bank_ci_lo": cb.get("ci_lo"),
                "correct_bank_ci_hi": cb.get("ci_hi"),
                "correct_dominates": entry.get("correct_dominates"),
                "strict_ci_dominance": entry.get("strict_ci_dominance"),
                "pattern_v_dominates": entry.get("pattern_v_dominates"),
                "pattern_k_addressing": entry.get("pattern_k_addressing"),
                "n_total": entry.get("n_total"),
            })


def write_tex(variant_stats: dict, phase_label: str, kappa: float | None,
              alpha: float | None, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kappa_str = _fmt(kappa, 2) if kappa is not None else "?"
    alpha_str = _fmt(alpha, 2) if alpha is not None else "?"
    lines = [
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        (r"Variant & $n$ & Recall@1 & $\mu$ Margin & 95\% CI & "
         r"$\tilde{\mu}$ Margin & JS Drift \\"),
        rf"% phase={phase_label}, kappa={kappa_str}, alpha={alpha_str}",
        r"\midrule",
    ]
    for v in VARIANT_ORDER:
        s = variant_stats.get(v)
        if s is None:
            continue
        ci = f"[{_fmt(s.get('ci_lo'))}, {_fmt(s.get('ci_hi'))}]"
        lines.append(
            f"{VARIANT_LABELS.get(v, v).replace('_', r' ')} & "
            f"{s.get('n', '—')} & "
            f"{_fmt(s.get('recall_at_1'))} & "
            f"{_fmt(s.get('mean_margin'))} & "
            f"{ci} & "
            f"{_fmt(s.get('median_margin'))} & "
            f"{_fmt(s.get('js_drift'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_kappa_plot(phase_a_results: list[dict], exp7_baseline: dict | None,
                     path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot")
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    kappas = [e["kappa"] for e in phase_a_results]
    for variant_name, color in [
        ("correct_bank", "steelblue"),
        ("random_kv", "crimson"),
        ("random_K_correct_V", "green"),
        ("shuffled_bank", "darkorange"),
    ]:
        means = [e["variants"].get(variant_name, {}).get("mean_margin", float("nan"))
                 for e in phase_a_results]
        ci_lo = [e["variants"].get(variant_name, {}).get("ci_lo", m)
                 for e, m in zip(phase_a_results, means)]
        ci_hi = [e["variants"].get(variant_name, {}).get("ci_hi", m)
                 for e, m in zip(phase_a_results, means)]
        errs_lo = [m - lo for m, lo in zip(means, ci_lo)]
        errs_hi = [hi - m for m, hi in zip(means, ci_hi)]
        plt.errorbar(kappas, means, yerr=[errs_lo, errs_hi],
                     marker="o", color=color,
                     label=VARIANT_LABELS.get(variant_name, variant_name),
                     capsize=4)

    # Draw Exp7 no-mHC baseline for correct_bank if available
    if exp7_baseline is not None:
        baseline_m = exp7_baseline.get("correct_bank", {}).get("mean_margin")
        if baseline_m is not None and not _isnan(baseline_m):
            plt.axhline(baseline_m, color="steelblue", linestyle=":",
                        alpha=0.5, label=f"Exp7 correct_bank (no mHC) = {baseline_m:.4f}")

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.gca().invert_xaxis()
    plt.xlabel("kappa (↓ = stronger shield)")
    plt.ylabel("Mean Margin (nats, ↑ better)")
    plt.title("Exp8 Phase A: mHC kappa sweep\nModel: Qwen3-4B-Instruct-2507")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved → {path}")


def write_phase_b_plot(phase_b: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "correct_bank": "steelblue",
        "shuffled_bank": "darkorange",
        "random_kv": "crimson",
        "correct_K_random_V": "mediumpurple",
        "random_K_correct_V": "green",
    }
    names = [v for v in VARIANT_ORDER if v in phase_b["variants"]]
    stats_list = [phase_b["variants"][v] for v in names]
    means = [s["mean_margin"] for s in stats_list]
    errs_lo = [s["mean_margin"] - s.get("ci_lo", s["mean_margin"]) for s in stats_list]
    errs_hi = [s.get("ci_hi", s["mean_margin"]) - s["mean_margin"] for s in stats_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, vname in enumerate(names):
        ax.bar(i, means[i], color=colors.get(vname, "gray"),
               yerr=[[max(0.0, errs_lo[i])], [max(0.0, errs_hi[i])]],
               capsize=5, label=VARIANT_LABELS.get(vname, vname))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in names],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Margin (nats, ↑ better)")
    kappa = phase_b.get("kappa", "?")
    ax.set_title(
        f"Exp8 Phase B: mHC kappa={kappa} (full 807-prompt run)\n"
        f"correct_bank dominates = {phase_b.get('correct_dominates', '?')}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# README

def write_readme(run_dir: Path,
                 phase_a_results: list[dict],
                 phase_b: dict | None,
                 phase_c_results: list[dict],
                 best_kappa: float | None,
                 out: Path) -> None:

    def vrow(vname, s, extra=""):
        ci = f"[{_fmt(s.get('ci_lo'))}, {_fmt(s.get('ci_hi'))}]"
        return (
            f"| {VARIANT_LABELS.get(vname, vname)} | {s.get('n', 'N/A')} | "
            f"{_fmt(s.get('recall_at_1'))} | {_fmt(s.get('mean_margin'))} | "
            f"{ci} | {_fmt(s.get('median_margin'))} | {_fmt(s.get('js_drift'))} |{extra}\n"
        )

    # Phase A table
    pa_table = ""
    for entry in phase_a_results:
        kappa = entry.get("kappa", "?")
        cb = entry["variants"].get("correct_bank", {})
        rkv = entry["variants"].get("random_K_correct_V", {})
        pa_table += (
            f"| {kappa} | {_fmt(cb.get('mean_margin'))} "
            f"[{_fmt(cb.get('ci_lo'))}, {_fmt(cb.get('ci_hi'))}] | "
            f"{_fmt(rkv.get('mean_margin'))} | "
            f"{'✅' if entry.get('correct_dominates') else '❌'} | "
            f"{'✅' if entry.get('strict_ci_dominance') else '❌'} | "
            f"{'⚠️' if entry.get('pattern_v_dominates') else '—'} |\n"
        )

    # Phase B table
    pb_table = ""
    if phase_b is not None:
        for vname in VARIANT_ORDER:
            s = phase_b["variants"].get(vname)
            if s:
                pb_table += vrow(vname, s)

    # Phase C table
    pc_table = ""
    for entry in phase_c_results:
        alpha = entry.get("alpha", "?")
        cb = entry["variants"].get("correct_bank", {})
        rkv = entry["variants"].get("random_K_correct_V", {})
        rkv_m = _fmt(rkv.get("mean_margin")) if rkv else "N/A"
        pc_table += (
            f"| {alpha} | {_fmt(cb.get('mean_margin'))} | "
            f"{rkv_m} | "
            f"{'✅' if entry.get('correct_dominates') else '❌'} |\n"
        )

    # Overall verdict
    phase_b_pass = phase_b is not None and phase_b.get("correct_dominates", False)
    phase_b_strict = phase_b is not None and phase_b.get("strict_ci_dominance", False)
    if phase_b_strict:
        overall = "**STRONG PASS** — H8.1 and H8.4 both met (Phase B, full 807-prompt run)"
    elif phase_b_pass:
        overall = "**DIRECTIONAL PASS** — H8.1 met (correct_bank highest margin), H8.4 not met"
    elif phase_b is None:
        best_pa = next(
            (e for e in phase_a_results if e.get("correct_dominates")), None)
        if best_pa:
            overall = f"**PHASE A PASS** — correct_bank dominates at kappa={best_pa.get('kappa')} (Phase B pending)"
        else:
            overall = "**FAIL** — correct_bank does not dominate any tested kappa"
    else:
        overall = "**FAIL** — correct_bank does not dominate at any kappa on full run"

    readme = f"""# Exp8 — mHC-Smoothed Pre-RoPE Negative Controls

**Model:** Qwen3-4B-Instruct-2507  
**bank_key_mode:** pre_rope  
**bank_size:** 200  
**mHC shield:** enabled (kappa swept)  
**Baseline:** Exp7 raw ATB (mhc_shield=False) — `pattern_v_dominates=True` at α=0.05

## Overall Verdict

{overall}

**Best kappa from Phase A:** {best_kappa}

---

## Phase A — kappa sweep (n=200 per kappa)

| kappa | correct_bank margin [95% CI] | random_K_correct_V margin | dominates | strict CI | V-dominates |
|-------|------------------------------|---------------------------|-----------|-----------|-------------|
{pa_table}

---

## Phase B — full run (n=807, kappa={best_kappa})

| Variant | n | Recall@1 | Mean Margin | 95% CI | Median Margin | JS Drift |
|---------|---|----------|-------------|--------|---------------|----------|
{pb_table if pb_table else "_Phase B not yet run._\n"}

---

## Phase C — alpha stress (kappa={best_kappa})

| alpha | correct_bank margin | random_K_correct_V margin | correct_dominates |
|-------|---------------------|---------------------------|-------------------|
{pc_table if pc_table else "_Phase C not yet run._\n"}

---

## Files

- `phase_a/kappa_*/results.jsonl` — Phase A raw results
- `phase_b/results.jsonl` — Phase B raw results
- `phase_c/alpha_*/results.jsonl` — Phase C raw results
- `analysis/kappa_comparison.csv` — Phase A kappa summary
- `analysis/summary.csv` — Phase B variant summary
- `analysis/tables/` — LaTeX tables per phase
- `analysis/plots/` — kappa sweep + Phase B bar chart
- `PREREG.md` — pre-registered hypotheses
- `best_kappa.json` — best kappa selected from Phase A

*Generated by `exp8_mhc_smoothed_pre_rope/analyze.py`*
"""
    (out / "README.md").write_text(readme)
    print(f"[readme] → {out / 'README.md'}")


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None,
                    help="Path to a specific run directory.")
    ap.add_argument("--exp-dir", default=None,
                    help="Path to exp8 root; latest run is selected automatically.")
    ap.add_argument("--out", required=True,
                    help="Output directory for analysis artifacts.")
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

    phase_a = analyze_phase_a(run_dir)
    phase_b = analyze_phase_b(run_dir)
    phase_c = analyze_phase_c(run_dir)

    # Try to load Exp7 baseline for comparison
    exp7_baseline: dict | None = None
    exp7_json = ROOT / "experiments" / "atb_validation_v1" / \
                "exp7_non_gemma_pre_rope" / "analysis" / "analysis.json"
    if exp7_json.exists():
        try:
            exp7_data = json.loads(exp7_json.read_text())
            exp7_baseline = exp7_data.get("variants")
        except Exception:
            pass

    # Best kappa
    best_kappa_path = run_dir / "best_kappa.json"
    if best_kappa_path.exists():
        best_kappa = json.loads(best_kappa_path.read_text()).get("best_kappa")
    else:
        best_kappa = pick_best_kappa(phase_a)

    # Write kappa comparison CSV
    if phase_a:
        write_kappa_comparison_csv(phase_a, out / "kappa_comparison.csv")
        print(f"[csv] kappa comparison → {out / 'kappa_comparison.csv'}")

        # Write per-kappa summary CSVs and LaTeX tables
        tables_dir = out / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for entry in phase_a:
            kappa = entry.get("kappa", "?")
            kappa_tag = f"{kappa:.2f}".replace(".", "_")
            write_summary_csv(f"phase_a_kappa_{kappa_tag}", entry["variants"],
                              out / f"phase_a_kappa_{kappa_tag}_summary.csv")
            write_tex(entry["variants"], f"Phase A κ={kappa}", kappa, 0.05,
                      tables_dir / f"phase_a_kappa_{kappa_tag}.tex")

        write_kappa_plot(phase_a, exp7_baseline,
                         out / "plots" / "exp8_phase_a_kappa_sweep.png")

    if phase_b is not None:
        write_summary_csv("phase_b", phase_b["variants"], out / "summary.csv")
        print(f"[csv] phase_b summary → {out / 'summary.csv'}")
        write_tex(phase_b["variants"], "Phase B (full)", phase_b.get("kappa"), 0.05,
                  out / "tables" / "phase_b.tex")
        write_phase_b_plot(phase_b, out / "plots" / "exp8_phase_b.png")

    if phase_c:
        tables_dir = out / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        for entry in phase_c:
            alpha = entry.get("alpha", "?")
            alpha_tag = f"{alpha:.2f}".replace(".", "_")
            write_summary_csv(f"phase_c_alpha_{alpha_tag}", entry["variants"],
                              out / f"phase_c_alpha_{alpha_tag}_summary.csv")
            write_tex(entry["variants"], f"Phase C α={alpha}",
                      entry.get("kappa"), alpha,
                      tables_dir / f"phase_c_alpha_{alpha_tag}.tex")

    # Full analysis JSON
    analysis_out = {
        "run_dir": str(run_dir),
        "best_kappa": best_kappa,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
    }
    (out / "analysis.json").write_text(
        json.dumps(analysis_out, indent=2, default=str)
    )
    print(f"[json] analysis → {out / 'analysis.json'}")

    write_readme(run_dir, phase_a, phase_b, phase_c, best_kappa, out)

    # Summary
    if phase_b:
        verdict_str = "STRONG PASS" if phase_b.get("strict_ci_dominance") else \
                      ("DIRECTIONAL PASS" if phase_b.get("correct_dominates") else "FAIL")
        print(f"\nPhase B verdict: {verdict_str}")
        print(f"  correct_bank mean_margin = "
              f"{_fmt(phase_b['variants'].get('correct_bank', {}).get('mean_margin'))}")
    elif phase_a:
        best_entry = max(phase_a,
                         key=lambda e: e["variants"].get("correct_bank", {})
                                                     .get("mean_margin", float("-inf")))
        print(f"\nPhase A best kappa: {best_entry.get('kappa')} "
              f"(correct_bank margin="
              f"{_fmt(best_entry['variants'].get('correct_bank', {}).get('mean_margin'))})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
