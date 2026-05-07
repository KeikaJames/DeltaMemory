"""Exp 6b — standalone post-processing analysis.

Run after rsyncing exp6b results from spark1:

    rsync -av spark1:~/projects/RCV-HC/experiments/atb_validation_v1/exp6b_post_rope_negative_controls/ \
        experiments/atb_validation_v1/exp6b_post_rope_negative_controls/

Then:

    python experiments/atb_validation_v1/exp6b_post_rope_negative_controls/analyze.py \
        --run-dir experiments/atb_validation_v1/exp6b_post_rope_negative_controls/run_20260508_041338 \
        --out experiments/atb_validation_v1/exp6b_post_rope_negative_controls/analysis

Or to analyze the latest run automatically:

    python experiments/atb_validation_v1/exp6b_post_rope_negative_controls/analyze.py \
        --exp-dir experiments/atb_validation_v1/exp6b_post_rope_negative_controls \
        --out experiments/atb_validation_v1/exp6b_post_rope_negative_controls/analysis
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
    runs = sorted(exp_dir.glob("run_*/results.jsonl"),
                  key=lambda p: p.parent.name)
    return runs[-1].parent if runs else None


# ---------------------------------------------------------------------------
# Core analysis

VARIANT_ORDER = [
    "correct_bank",
    "shuffled_bank",
    "random_kv",
    "correct_K_random_V",
    "random_K_correct_V",
]

VARIANT_LABELS = {
    "correct_bank": "Correct K/V (post-RoPE)",
    "shuffled_bank": "Shuffled K/V",
    "random_kv": "Random K + V",
    "correct_K_random_V": "Correct K, Random V",
    "random_K_correct_V": "Random K, Correct V",
}


def analyze(run_dir: Path) -> dict:
    rows = _read_jsonl(run_dir / "results.jsonl")
    n_total = len(rows)
    bank_key_mode = rows[0].get("bank_key_mode", "unknown") if rows else "unknown"

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

    variant_stats = {v: stats(v) for v in VARIANT_ORDER if v in by_v}

    correct = variant_stats.get("correct_bank", {})
    verdict = all(
        correct.get("mean_margin", float("-inf")) > variant_stats.get(v, {}).get("mean_margin", float("inf"))
        for v in VARIANT_ORDER if v != "correct_bank" and v in variant_stats
    )

    # CI-based strict dominance: correct_bank ci_lo > every other mean_margin
    strict_dominance = all(
        correct.get("ci_lo", float("-inf")) > variant_stats.get(v, {}).get("mean_margin", float("inf"))
        for v in VARIANT_ORDER if v != "correct_bank" and v in variant_stats
    )

    return {
        "run_dir": str(run_dir),
        "n_total": n_total,
        "bank_key_mode": bank_key_mode,
        "variants": variant_stats,
        "verdict_correct_dominates": verdict,
        "verdict_strict_ci_dominance": strict_dominance,
    }


# ---------------------------------------------------------------------------
# Output writers

def write_summary_csv(analysis: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "variant", "n", "recall_at_1",
              "mean_margin", "ci_lo", "ci_hi", "median_margin",
              "js_drift", "kl_drift", "mean_target_rank"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for vname, s in analysis["variants"].items():
            row = {"experiment": "exp6b_post_rope_negative_controls",
                   "variant": vname, **s}
            # keep only known fields
            row = {k: row.get(k, "") for k in fields}
            w.writerow(row)


def write_tex(analysis: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v, d=4):
        if v is None or (isinstance(v, float) and _isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.{d}f}"
        return str(v)

    mode = analysis.get("bank_key_mode", "—")
    lines = [
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        (r"Variant & $n$ & Recall@1 & $\mu$ Margin & 95\% CI & "
         r"$\tilde{\mu}$ Margin & JS Drift \\"),
        rf"% bank\_key\_mode={mode}",
        r"\midrule",
    ]
    for v in VARIANT_ORDER:
        s = analysis["variants"].get(v)
        if s is None:
            continue
        ci = f"[{fmt(s.get('ci_lo'))}, {fmt(s.get('ci_hi'))}]"
        lines.append(
            f"{VARIANT_LABELS.get(v, v).replace('_', r' ')} & "
            f"{s.get('n', '—')} & "
            f"{fmt(s.get('recall_at_1'))} & "
            f"{fmt(s.get('mean_margin'))} & "
            f"{ci} & "
            f"{fmt(s.get('median_margin'))} & "
            f"{fmt(s.get('js_drift'))} \\\\"
        )
    lines += [
        r"\midrule",
        rf"\multicolumn{{7}}{{l}}{{correct\_bank dominates = "
        rf"{'Yes' if analysis['verdict_correct_dominates'] else 'No'} "
        rf"(strict CI = {'Yes' if analysis['verdict_strict_ci_dominance'] else 'No'})}} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_plot(analysis: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plot")
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "correct_bank": "steelblue",
        "shuffled_bank": "darkorange",
        "random_kv": "crimson",
        "correct_K_random_V": "mediumpurple",
        "random_K_correct_V": "green",
    }
    names = [v for v in VARIANT_ORDER if v in analysis["variants"]]
    stats = [analysis["variants"][v] for v in names]
    means = [s["mean_margin"] for s in stats]
    errs_lo = [s["mean_margin"] - s.get("ci_lo", s["mean_margin"]) for s in stats]
    errs_hi = [s.get("ci_hi", s["mean_margin"]) - s["mean_margin"] for s in stats]

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
    ax.set_title(
        f"Exp 6b: post-RoPE Negative Controls\n"
        f"correct_bank dominates = {analysis['verdict_correct_dominates']}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved → {path}")


def write_readme(analysis: dict, out: Path) -> None:
    def fmt(v, d=4):
        if v is None or (isinstance(v, float) and _isnan(v)):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{d}f}"
        return str(v)

    rows = ""
    for vname in VARIANT_ORDER:
        s = analysis["variants"].get(vname)
        if s is None:
            continue
        ci = f"[{fmt(s.get('ci_lo'))}, {fmt(s.get('ci_hi'))}]"
        rows += (
            f"| {VARIANT_LABELS.get(vname, vname)} | {s.get('n','N/A')} | "
            f"{fmt(s.get('recall_at_1'))} | {fmt(s.get('mean_margin'))} | "
            f"{ci} | {fmt(s.get('median_margin'))} | "
            f"{fmt(s.get('js_drift'))} |\n"
        )

    mode = analysis.get("bank_key_mode", "unknown")
    verdict_str = "✅ PASS" if analysis["verdict_correct_dominates"] else "❌ FAIL"
    strict_str = "✅ PASS" if analysis["verdict_strict_ci_dominance"] else "❌ FAIL"

    readme = f"""# Exp 6b — post-RoPE Negative Controls

**bank_key_mode:** `{mode}`  
**Motivation:** Exp 6 used `pre_rope` which is a near-no-op on Gemma-4-31B due to
native V-norm (`auto_rms_cap` → scale≈1.0). This rerun uses `post_rope`, the only
mode confirmed to produce positive margin on Gemma-4-31B (Exp 1: +0.088 mean margin).

**Total cells:** {analysis['n_total']}

## Results

| Variant | n | Recall@1 | Mean Margin | 95% CI | Median Margin | JS Drift |
|---------|---|----------|-------------|--------|---------------|----------|
{rows}

## Verdict

| Criterion | Result |
|-----------|--------|
| correct_bank dominates (mean) | {verdict_str} |
| correct_bank dominates (strict CI) | {strict_str} |

**Interpretation:**  
- If correct_bank mean_margin > all others → mechanism is real (correct K/V binding matters).  
- If correct_K_random_V has positive margin → K alone is sufficient for addressing
  (V content doesn't matter for margin, but may matter for factual recall).  
- If random_K_correct_V fails → K is necessary for addressing.  
- If shuffled_bank fails → fact-level binding (not just "any bank") drives the effect.

## Files

- `results.jsonl` — raw per-prompt results
- `analysis/summary.csv` — per-variant aggregated stats
- `analysis/tables/exp6b.tex` — LaTeX table
- `analysis/plots/exp6b_post_rope_negative_controls.png` — bar chart
- `manifest.yaml` — run protocol

*Generated by `exp6b_post_rope_negative_controls/analyze.py`*
"""
    (out / "README.md").write_text(readme)
    print(f"[readme] → {out / 'README.md'}")


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run-dir", help="Path to specific run_* directory")
    grp.add_argument("--exp-dir", help="Path to exp6b dir (uses latest run)")
    ap.add_argument("--out", required=True, help="Output directory for analysis")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run(Path(args.exp_dir))
        if run_dir is None:
            print(f"[error] No run_*/results.jsonl found in {args.exp_dir}")
            return 1

    print(f"[analyze] run_dir = {run_dir}")
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        print(f"[error] results.jsonl not found: {results_path}")
        return 1

    n = sum(1 for _ in open(results_path))
    print(f"[analyze] {n} result rows")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    analysis = analyze(run_dir)

    # Print summary
    print("\n=== Exp 6b: post-RoPE Negative Controls ===")
    print(f"bank_key_mode: {analysis['bank_key_mode']}")
    print(f"n_total: {analysis['n_total']}")
    print()
    for vname in VARIANT_ORDER:
        s = analysis["variants"].get(vname)
        if s is None:
            continue
        print(f"  {vname:30s}  n={s['n']:5d}  "
              f"mean_margin={s['mean_margin']:+.4f}  "
              f"ci=[{s.get('ci_lo', float('nan')):+.4f}, {s.get('ci_hi', float('nan')):+.4f}]  "
              f"median={s['median_margin']:+.4f}  "
              f"recall@1={s['recall_at_1']:.4f}")

    print()
    print(f"verdict_correct_dominates:   {analysis['verdict_correct_dominates']}")
    print(f"verdict_strict_ci_dominance: {analysis['verdict_strict_ci_dominance']}")

    # Write outputs
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    write_summary_csv(analysis, out / "summary.csv")
    write_tex(analysis, out / "tables" / "exp6b.tex")
    write_plot(analysis, out / "plots" / "exp6b_post_rope_negative_controls.png")
    write_readme(analysis, out)

    print(f"\n[done] analysis → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
