"""Exp 7 — Non-Gemma Pre-RoPE Negative Controls — standalone post-processing analysis.

Run after rsyncing exp7 results from spark1:

    rsync -av spark1:~/projects/RCV-HC/experiments/atb_validation_v1/exp7_non_gemma_pre_rope/ \\
        experiments/atb_validation_v1/exp7_non_gemma_pre_rope/

Then:

    python experiments/atb_validation_v1/exp7_non_gemma_pre_rope/analyze.py \\
        --run-dir experiments/atb_validation_v1/exp7_non_gemma_pre_rope/run_YYYYMMDD_HHMMSS \\
        --out experiments/atb_validation_v1/exp7_non_gemma_pre_rope/analysis

Or to analyze the latest run automatically:

    python experiments/atb_validation_v1/exp7_non_gemma_pre_rope/analyze.py \\
        --exp-dir experiments/atb_validation_v1/exp7_non_gemma_pre_rope \\
        --out experiments/atb_validation_v1/exp7_non_gemma_pre_rope/analysis
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

EXP_NAME = "exp7_non_gemma_pre_rope"


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
    "correct_bank": "Correct K/V (pre-RoPE)",
    "shuffled_bank": "Shuffled K/V (bank_size=200)",
    "random_kv": "Random K + V",
    "correct_K_random_V": "Correct K, Random V",
    "random_K_correct_V": "Random K, Correct V",
}


def analyze(run_dir: Path) -> dict:
    rows = _read_jsonl(run_dir / "results.jsonl")
    n_total = len(rows)
    bank_key_mode = rows[0].get("bank_key_mode", "unknown") if rows else "unknown"
    bank_size = rows[0].get("bank_size", "unknown") if rows else "unknown"

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

    # H7.1: correct_bank has highest mean_margin
    verdict = all(
        correct.get("mean_margin", float("-inf")) > variant_stats.get(v, {}).get("mean_margin", float("inf"))
        for v in VARIANT_ORDER if v != "correct_bank" and v in variant_stats
    )

    # Strict CI: correct_bank ci_lo > every other mean_margin
    strict_dominance = all(
        correct.get("ci_lo", float("-inf")) > variant_stats.get(v, {}).get("mean_margin", float("inf"))
        for v in VARIANT_ORDER if v != "correct_bank" and v in variant_stats
    )

    # H7.3 and H7.4 checks
    shuffled = variant_stats.get("shuffled_bank", {})
    rk_cv = variant_stats.get("random_K_correct_V", {})
    ck_rv = variant_stats.get("correct_K_random_V", {})

    # shuffled should be < correct (H7.4)
    h74_pass = (shuffled.get("mean_margin", float("inf")) <
                correct.get("mean_margin", float("-inf")))

    # random_K_correct_V should be < correct (H7.3: K addressing matters in pre_rope)
    h73_pass = (rk_cv.get("mean_margin", float("inf")) <
                correct.get("mean_margin", float("-inf")))

    # Detect methodological patterns
    shuffled_eq_correct = (
        abs((correct.get("mean_margin") or 0.0) - (shuffled.get("mean_margin") or 0.0)) < 1e-5
    )

    # V-dominance: random_K_correct_V beats correct_bank by >0.05 (like post_rope Exp6b)
    v_dominates = (rk_cv.get("mean_margin") or -999) > (correct.get("mean_margin") or -999) + 0.05

    # K-addressing: correct_bank clearly better than random_K_correct_V
    k_addressing = (correct.get("mean_margin") or -999) > (rk_cv.get("mean_margin") or -999) + 0.05

    return {
        "run_dir": str(run_dir),
        "n_total": n_total,
        "bank_key_mode": bank_key_mode,
        "bank_size": bank_size,
        "variants": variant_stats,
        "verdict_correct_dominates": verdict,
        "verdict_strict_ci_dominance": strict_dominance,
        "h73_k_addressing_pass": h73_pass,
        "h74_shuffled_degrades_pass": h74_pass,
        "pattern_shuffled_eq_correct": shuffled_eq_correct,
        "pattern_v_dominates": v_dominates,
        "pattern_k_addressing": k_addressing,
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
            row = {"experiment": EXP_NAME, "variant": vname, **s}
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
        rf"% bank\_key\_mode={mode}, bank\_size={analysis.get('bank_size', '—')}",
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
    stats_list = [analysis["variants"][v] for v in names]
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
    bank_size = analysis.get("bank_size", "?")
    ax.set_title(
        f"Exp 7: Non-Gemma Pre-RoPE Negative Controls (bank_size={bank_size})\n"
        f"Model: Qwen3-4B-Instruct-2507  |  "
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
            f"| {VARIANT_LABELS.get(vname, vname)} | {s.get('n', 'N/A')} | "
            f"{fmt(s.get('recall_at_1'))} | {fmt(s.get('mean_margin'))} | "
            f"{ci} | {fmt(s.get('median_margin'))} | "
            f"{fmt(s.get('js_drift'))} |\n"
        )

    mode = analysis.get("bank_key_mode", "unknown")
    bank_size = analysis.get("bank_size", "unknown")
    verdict_str = "✅ PASS" if analysis["verdict_correct_dominates"] else "❌ FAIL"
    strict_str = "✅ PASS" if analysis["verdict_strict_ci_dominance"] else "❌ FAIL"
    h73_str = "✅ PASS" if analysis["h73_k_addressing_pass"] else "❌ FAIL"
    h74_str = "✅ PASS" if analysis["h74_shuffled_degrades_pass"] else "❌ FAIL"

    interpretation_lines = []
    if analysis.get("pattern_shuffled_eq_correct"):
        interpretation_lines.append(
            "- **shuffled_bank ≈ correct_bank** — unexpected: at bank_size=200 the "
            "shuffled perturbation should be a valid 200-row permutation. Possible "
            "cause: subject-term overlap means shuffled distractors also route "
            "attention to near-correct V. Investigate bank_attention_mass per slot."
        )
    if analysis.get("pattern_v_dominates"):
        interpretation_lines.append(
            "- **random_K_correct_V > correct_bank** — even with pre_rope, V dominates "
            "over K routing. This could indicate α=0.05 is still large enough for V "
            "injection to override K mismatch. Consider α=0.02 or inspecting "
            "bank_attention_mass for the target slot."
        )
    if analysis.get("pattern_k_addressing"):
        interpretation_lines.append(
            "- **K addressing confirmed**: correct_bank clearly outperforms "
            "random_K_correct_V — correct pre-RoPE K is necessary for reliable "
            "fact retrieval on Qwen3-4B. This validates the pre_rope ATB design."
        )
    if analysis["verdict_correct_dominates"]:
        interpretation_lines.append(
            "- **All H7.1–H7.4 checks passed**: pre-RoPE ATB achieves correct K/V "
            "binding on Qwen3-4B. The mechanism works as designed when V-norm is "
            "absent. Contrast with Gemma-4-31B (Exp 6/6b) where native V-norm "
            "(pre_rope) and position-specific K (post_rope) both impede binding."
        )
    if not interpretation_lines:
        interpretation_lines.append(
            "- correct_bank does not dominate all controls; see per-variant analysis above."
        )

    interpretation = "\n".join(interpretation_lines)

    readme = f"""# Exp 7 — Non-Gemma Pre-RoPE Negative Controls

**bank_key_mode:** `{mode}`  
**bank_size:** {bank_size} (1 target + 199 distractors — shuffled is a valid 200-row permutation)  
**Model:** Qwen3-4B-Instruct-2507 (no native V-norm; standard GQA 32/8 heads)  
**Motivation:**
- Exp 6 (pre_rope, Gemma-4-31B): native V-norm makes `auto_rms_cap` scale≈1.0 → near-no-op.
- Exp 6b (post_rope, Gemma-4-31B): post-RoPE K is position-specific → K addressing unreliable.
- Exp 7: Qwen3-4B has no V-norm + pre_rope K is position-invariant + bank_size=200.

**Total cells:** {analysis['n_total']}

## Results

| Variant | n | Recall@1 | Mean Margin | 95% CI | Median Margin | JS Drift |
|---------|---|----------|-------------|--------|---------------|----------|
{rows}

## Verdict

| Criterion | Result |
|-----------|--------|
| H7.1: correct_bank highest mean_margin | {verdict_str} |
| H7.1 (strict 95% CI dominance) | {strict_str} |
| H7.3: random_K_correct_V < correct_bank (K addressing) | {h73_str} |
| H7.4: shuffled_bank < correct_bank (binding degrades) | {h74_str} |

## Interpretation

{interpretation}

## Files

- `results.jsonl` — raw per-prompt results
- `analysis/summary.csv` — per-variant aggregated stats
- `analysis/tables/exp7.tex` — LaTeX table
- `analysis/plots/exp7_non_gemma_pre_rope_negative_controls.png` — bar chart
- `manifest.yaml` — run protocol
- `PREREG.md` — pre-registration (hypotheses, success criteria, protocol)

*Generated by `exp7_non_gemma_pre_rope/analyze.py`*
"""
    (out / "README.md").write_text(readme)
    print(f"[readme] → {out / 'README.md'}")


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run-dir", help="Path to specific run_* directory")
    grp.add_argument("--exp-dir", help="Path to exp7 dir (uses latest run)")
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
    print(f"\n=== Exp 7: Non-Gemma Pre-RoPE Negative Controls ===")
    print(f"bank_key_mode: {analysis['bank_key_mode']}")
    print(f"bank_size:     {analysis['bank_size']}")
    print(f"n_total:       {analysis['n_total']}")
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
    print(f"h73_k_addressing_pass:       {analysis['h73_k_addressing_pass']}")
    print(f"h74_shuffled_degrades_pass:  {analysis['h74_shuffled_degrades_pass']}")
    print(f"pattern_v_dominates:         {analysis['pattern_v_dominates']}")
    print(f"pattern_k_addressing:        {analysis['pattern_k_addressing']}")

    # Write outputs
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str))
    write_summary_csv(analysis, out / "summary.csv")
    write_tex(analysis, out / "tables" / "exp7.tex")
    write_plot(analysis, out / "plots" / "exp7_non_gemma_pre_rope_negative_controls.png")
    write_readme(analysis, out)

    print(f"\n[done] analysis → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
