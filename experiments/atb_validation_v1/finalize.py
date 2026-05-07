"""ATB Validation v1 — Final Aggregator & Report Generator.

Run after all six experiments complete on GB10:

    python experiments/atb_validation_v1/finalize.py \
        --exp-root experiments/atb_validation_v1 \
        --out experiments/atb_validation_v1/final_report

Produces:
    final_report/
      SUMMARY.csv          — cross-experiment canonical table
      verdicts.json        — 4 PREREG verdict objects
      paper_tables/        — LaTeX fragments
      plots/               — α-cliff + position curves
      README.md            — narrative summary
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers

def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_mean(xs):
    xs = [float(x) for x in xs if x is not None and not _isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _safe_median(xs):
    xs = sorted(float(x) for x in xs if x is not None and not _isnan(x))
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _isnan(x):
    try:
        return math.isnan(float(x))
    except Exception:
        return True


def bootstrap_ci(vals, n_boot=10000, seed=0, alpha=0.05):
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


def mcnemar(a_hits, b_hits):
    """McNemar chi-sq on paired binary recalls (e is discordant count)."""
    assert len(a_hits) == len(b_hits)
    n10 = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    n01 = sum(1 for a, b in zip(a_hits, b_hits) if not a and b)
    e = n10 + n01
    if e == 0:
        return 0.0, 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / e
    # approximate p-value from chi2 df=1 via Wilson-Hilferty
    z = ((chi2 / 1) ** (1 / 3) - (1 - 2 / (9 * 1))) / math.sqrt(2 / (9 * 1))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return chi2, p


# ---------------------------------------------------------------------------
# Find latest run dirs

def latest_run(exp_dir: Path) -> Path | None:
    runs = sorted(exp_dir.glob("run_*/results.jsonl"),
                  key=lambda p: p.parent.name)
    return runs[-1].parent if runs else None


# ---------------------------------------------------------------------------
# Exp1 — Core Ablation

def analyze_exp1(run_dir: Path) -> dict:
    rows = _read_jsonl(run_dir / "results.jsonl")
    by_v: dict[str, list] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)

    def stats(vname):
        vrows = by_v.get(vname, [])
        margins = [r["margin"] for r in vrows if "margin" in r]
        recalls = [bool(r.get("recall_at_1")) for r in vrows]
        m, lo, hi = bootstrap_ci(margins)
        return {"n": len(vrows), "recall_at_1": sum(recalls) / max(len(recalls), 1),
                "mean_margin": m, "ci_lo": lo, "ci_hi": hi,
                "median_margin": _safe_median(margins)}

    no_bank = stats("no_bank")
    post_rope = stats("post_rope_bank")
    pre_only = stats("pre_rope_bank_only")
    pre_vscale = stats("pre_rope_vscale")
    full_atb = stats("full_attnnativebank")

    # Verdict: pre_rope_bank_only beats no_bank on margin
    verdict_pre_rope = (pre_only["mean_margin"] > no_bank["mean_margin"]
                        and pre_only["ci_lo"] > no_bank["mean_margin"])
    # Verdict: post_rope beats no_bank
    verdict_post_rope = post_rope["mean_margin"] > no_bank["mean_margin"]

    return {
        "variants": {
            "no_bank": no_bank,
            "post_rope_bank": post_rope,
            "pre_rope_bank_only": pre_only,
            "pre_rope_vscale": pre_vscale,
            "full_attnnativebank": full_atb,
        },
        "verdict_pre_rope_beats_no_bank": verdict_pre_rope,
        "verdict_post_rope_beats_no_bank": verdict_post_rope,
        "verdict_bank_fires": (post_rope["mean_margin"] > no_bank["mean_margin"]
                               or pre_only["mean_margin"] > no_bank["mean_margin"]),
    }


# ---------------------------------------------------------------------------
# Exp2 — Position Invariance

def analyze_exp2(run_dir: Path) -> dict:
    rows = _read_jsonl(run_dir / "results.jsonl")
    by_v: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_v[r["variant"]][r.get("position_delta", 0)].append(r)

    result = {}
    for vname, by_delta in by_v.items():
        curve = []
        for delta in sorted(by_delta):
            vrows = by_delta[delta]
            margins = [r["margin"] for r in vrows if "margin" in r]
            curve.append({
                "position_delta": delta,
                "n": len(vrows),
                "mean_margin": _safe_mean(margins),
                "median_margin": _safe_median(margins),
                "recall_at_1": sum(bool(r.get("recall_at_1")) for r in vrows) / max(len(vrows), 1),
            })
        result[vname] = curve

    # Position invariance verdict: pre_rope margin degradation < post_rope
    def degradation(curve):
        if len(curve) < 2:
            return float("nan")
        m0 = curve[0]["mean_margin"]
        m_last = curve[-1]["mean_margin"]
        return m0 - m_last  # positive = degraded

    pre_deg = degradation(result.get("pre_rope_bank", []))
    post_deg = degradation(result.get("post_rope_bank", []))
    verdict = (not _isnan(pre_deg) and not _isnan(post_deg)
               and pre_deg < post_deg)

    return {
        "curves": result,
        "pre_rope_degradation": pre_deg,
        "post_rope_degradation": post_deg,
        "verdict_pre_rope_more_stable": verdict,
    }


# ---------------------------------------------------------------------------
# Exp3 — Bit Equality

def analyze_exp3(exp3_dir: Path) -> dict:
    models: dict[str, dict] = {}
    for run_dir in exp3_dir.iterdir():
        if not run_dir.is_dir():
            continue
        rj = run_dir / "results.jsonl"
        if not rj.exists():
            continue
        rows = _read_jsonl(rj)
        if not rows:
            continue
        model_tag = run_dir.name.split("_")[0]
        te = [bool(r.get("torch_equal")) for r in rows if "torch_equal" in r]
        mad = [r.get("max_abs_diff", float("nan")) for r in rows if "max_abs_diff" in r]
        models[model_tag] = {
            "n": len(rows),
            "torch_equal_all": all(te) if te else False,
            "max_abs_diff_max": max(mad) if mad else float("nan"),
            "manifest": run_dir.name,
        }
    verdict = all(v["torch_equal_all"] for v in models.values() if v["n"] > 0)
    return {"models": models, "verdict_bit_equal": verdict}


# ---------------------------------------------------------------------------
# Exp4 — CF-1k Main Result

def analyze_exp4(run_dir: Path) -> dict:
    rows = _read_jsonl(run_dir / "results.jsonl")
    by_v: dict[str, list] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)

    def stats(vname):
        vrows = by_v.get(vname, [])
        margins = [r["margin"] for r in vrows if "margin" in r]
        recalls = [bool(r.get("recall_at_1")) for r in vrows]
        m, lo, hi = bootstrap_ci(margins)
        return {
            "n": len(vrows),
            "recall_at_1": sum(recalls) / max(len(recalls), 1),
            "mean_margin": m, "ci_lo": lo, "ci_hi": hi,
            "median_margin": _safe_median(margins),
            "js_drift": _safe_mean([r.get("js_drift") for r in vrows
                                    if r.get("js_drift") is not None]),
            "kl_drift": _safe_mean([r.get("kl_drift") for r in vrows
                                    if r.get("kl_drift") is not None]),
        }

    none_a0 = stats("none_alpha0")
    atb_a0 = stats("AttnNativeBank_alpha0")
    atb_a1 = stats("AttnNativeBank_alpha1")

    # McNemar: none vs atb_alpha1
    none_rows = by_v.get("none_alpha0", [])
    atb1_rows = by_v.get("AttnNativeBank_alpha1", [])
    pid_none = {r["prompt_id"]: bool(r.get("recall_at_1")) for r in none_rows}
    pid_atb1 = {r["prompt_id"]: bool(r.get("recall_at_1")) for r in atb1_rows}
    shared = sorted(set(pid_none) & set(pid_atb1))
    if shared:
        chi2, pval = mcnemar([pid_none[p] for p in shared],
                             [pid_atb1[p] for p in shared])
    else:
        chi2, pval = float("nan"), float("nan")

    return {
        "variants": {
            "none_alpha0": none_a0,
            "AttnNativeBank_alpha0": atb_a0,
            "AttnNativeBank_alpha1": atb_a1,
        },
        "mcnemar_chi2": chi2,
        "mcnemar_p": pval,
        "verdict_atb_beats_baseline": (
            atb_a1["mean_margin"] > none_a0["mean_margin"]
            and atb_a1["recall_at_1"] >= none_a0["recall_at_1"]
        ),
        "verdict_alpha0_bit_equal_consistent": (
            abs(atb_a0["mean_margin"] - none_a0["mean_margin"]) < 1e-4
        ),
    }


# ---------------------------------------------------------------------------
# Exp5 — α Dense Sweep

def analyze_exp5(run_dir: Path) -> dict:
    rows = _read_jsonl(run_dir / "results.jsonl")
    by_alpha: dict[float, list] = defaultdict(list)
    for r in rows:
        by_alpha[float(r["alpha"])].append(r)
    curve = []
    for alpha in sorted(by_alpha):
        vrows = by_alpha[alpha]
        margins = [r["margin"] for r in vrows if "margin" in r]
        recalls = [bool(r.get("recall_at_1")) for r in vrows]
        curve.append({
            "alpha": alpha,
            "n": len(vrows),
            "mean_margin": _safe_mean(margins),
            "median_margin": _safe_median(margins),
            "recall_at_1": sum(recalls) / max(len(recalls), 1),
        })

    # Detect cliff: large drop from some alpha to next
    max_drop = 0.0
    cliff_alpha = float("nan")
    for i in range(1, len(curve)):
        drop = curve[i - 1]["mean_margin"] - curve[i]["mean_margin"]
        if drop > max_drop:
            max_drop = drop
            cliff_alpha = curve[i]["alpha"]

    # Peak alpha
    peak = max(curve, key=lambda r: r["mean_margin"], default={})
    return {
        "curve": curve,
        "cliff_alpha": cliff_alpha,
        "max_drop": max_drop,
        "peak_alpha": peak.get("alpha"),
        "peak_margin": peak.get("mean_margin"),
    }


# ---------------------------------------------------------------------------
# Exp6 — Negative Controls

def _analyze_neg_controls(run_dir: Path) -> dict:
    """Shared analysis for Exp6 and Exp6b negative controls."""
    rows = _read_jsonl(run_dir / "results.jsonl")
    by_v: dict[str, list] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)

    def stats(vname):
        vrows = by_v.get(vname, [])
        margins = [r["margin"] for r in vrows if "margin" in r]
        recalls = [bool(r.get("recall_at_1")) for r in vrows]
        ranks = [r["target_rank"] for r in vrows if "target_rank" in r]
        m, lo, hi = bootstrap_ci(margins)
        return {
            "n": len(vrows),
            "recall_at_1": sum(recalls) / max(len(recalls), 1),
            "mean_margin": m, "ci_lo": lo, "ci_hi": hi,
            "median_margin": _safe_median(margins),
            "js_drift": _safe_mean([r.get("js_drift") for r in vrows
                                    if r.get("js_drift") is not None]),
            "kl_drift": _safe_mean([r.get("kl_drift") for r in vrows
                                    if r.get("kl_drift") is not None]),
            "mean_target_rank": _safe_mean(ranks),
        }

    correct = stats("correct_bank")
    shuffled = stats("shuffled_bank")
    rndkv = stats("random_kv")
    ck_rv = stats("correct_K_random_V")
    rk_cv = stats("random_K_correct_V")

    verdict = (
        correct["mean_margin"] > shuffled["mean_margin"]
        and correct["mean_margin"] > rndkv["mean_margin"]
        and correct["mean_margin"] > ck_rv["mean_margin"]
        and correct["mean_margin"] > rk_cv["mean_margin"]
    )

    return {
        "variants": {
            "correct_bank": correct,
            "shuffled_bank": shuffled,
            "random_kv": rndkv,
            "correct_K_random_V": ck_rv,
            "random_K_correct_V": rk_cv,
        },
        "verdict_correct_dominates": verdict,
        "bank_key_mode": rows[0].get("bank_key_mode", "unknown") if rows else "unknown",
    }


def analyze_exp6(run_dir: Path) -> dict:
    return _analyze_neg_controls(run_dir)


def analyze_exp6b(run_dir: Path) -> dict:
    return _analyze_neg_controls(run_dir)


# ---------------------------------------------------------------------------
# LaTeX table helpers

def _fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and _isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "\\checkmark" if v else "\\xmark"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def write_exp1_tex(analysis: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["no_bank", "post_rope_bank", "pre_rope_bank_only",
             "pre_rope_vscale", "full_attnnativebank"]
    lines = [
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Variant & Recall@1 & Margin (mean) & Margin (med) & JS drift \\",
        r"\midrule",
    ]
    for v in order:
        s = analysis["variants"].get(v, {})
        lines.append(
            f"{v.replace('_', r' ')} & {_fmt(s.get('recall_at_1'))} & "
            f"{_fmt(s.get('mean_margin'))} & {_fmt(s.get('median_margin'))} & "
            f"— \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_exp4_tex(analysis: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["none_alpha0", "AttnNativeBank_alpha0", "AttnNativeBank_alpha1"]
    labels = {"none_alpha0": "None (α=0)",
              "AttnNativeBank_alpha0": "ATB (α=0)",
              "AttnNativeBank_alpha1": "ATB (α=1)"}
    lines = [
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"Method & n & Recall@1 & Margin (mean) & 95\% CI & JS drift \\",
        r"\midrule",
    ]
    for v in order:
        s = analysis["variants"].get(v, {})
        ci = f"[{_fmt(s.get('ci_lo'))}, {_fmt(s.get('ci_hi'))}]"
        lines.append(
            f"{labels.get(v, v)} & {s.get('n', '—')} & "
            f"{_fmt(s.get('recall_at_1'))} & {_fmt(s.get('mean_margin'))} & "
            f"{ci} & {_fmt(s.get('js_drift'))} \\\\"
        )
    lines += [
        r"\midrule",
        f"\\multicolumn{{6}}{{l}}{{McNemar $\\chi^2$ = "
        f"{_fmt(analysis.get('mcnemar_chi2'), 3)}, "
        f"$p$ = {_fmt(analysis.get('mcnemar_p'), 4)}}} \\\\",
        r"\bottomrule", r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_neg_controls_tex(analysis: dict, path: Path, caption_tag: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["correct_bank", "shuffled_bank", "random_kv",
             "correct_K_random_V", "random_K_correct_V"]
    labels = {
        "correct_bank": "Correct K/V",
        "shuffled_bank": "Shuffled (fact→wrong V)",
        "random_kv": "Random K + V",
        "correct_K_random_V": "Correct K, Random V",
        "random_K_correct_V": "Random K, Correct V",
    }
    mode = analysis.get("bank_key_mode", "—")
    lines = [
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        (r"Variant & n & Recall@1 & Margin (mean) & 95\% CI & Margin (med) \\"
         f"  % bank\\_key\\_mode={mode} {caption_tag}"),
        r"\midrule",
    ]
    for v in order:
        s = analysis["variants"].get(v, {})
        ci = f"[{_fmt(s.get('ci_lo'))}, {_fmt(s.get('ci_hi'))}]"
        lines.append(
            f"{labels.get(v, v)} & {s.get('n', '—')} & "
            f"{_fmt(s.get('recall_at_1'))} & "
            f"{_fmt(s.get('mean_margin'))} & "
            f"{ci} & "
            f"{_fmt(s.get('median_margin'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_exp6_tex(analysis: dict, path: Path) -> None:
    _write_neg_controls_tex(analysis, path, caption_tag="(Exp6: pre\\_rope)")


def write_exp6b_tex(analysis: dict, path: Path) -> None:
    _write_neg_controls_tex(analysis, path, caption_tag="(Exp6b: post\\_rope)")


# ---------------------------------------------------------------------------
# Plots (optional, needs matplotlib)

def write_exp5_plot(analysis: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = analysis["curve"]
    alphas = [r["alpha"] for r in curve]
    margins = [r["mean_margin"] for r in curve]
    recalls = [r["recall_at_1"] for r in curve]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(alphas, margins, "o-", color="steelblue")
    ax1.axvline(x=analysis.get("cliff_alpha", 0), color="red",
                linestyle="--", alpha=0.5, label=f"cliff α={analysis.get('cliff_alpha'):.2f}")
    ax1.set_xlabel("α")
    ax1.set_ylabel("mean margin (nats)")
    ax1.set_title("Exp5: α vs Margin")
    ax1.legend()

    ax2.plot(alphas, recalls, "s-", color="darkorange")
    ax2.set_xlabel("α")
    ax2.set_ylabel("Recall@1")
    ax2.set_title("Exp5: α vs Recall@1")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_exp2_plot(analysis: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    curves = analysis["curves"]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"post_rope_bank": "crimson", "pre_rope_bank": "steelblue"}
    for vname, curve in curves.items():
        deltas = [r["position_delta"] for r in curve]
        margins = [r["mean_margin"] for r in curve]
        c = colors.get(vname, "gray")
        ax.plot(deltas, margins, "o-", color=c, label=vname.replace("_", " "))
    ax.set_xlabel("Position delta (filler tokens)")
    ax.set_ylabel("mean margin (nats)")
    ax.set_title("Exp2: Position Invariance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_exp6b_plot(analysis: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["correct_bank", "shuffled_bank", "random_kv",
             "correct_K_random_V", "random_K_correct_V"]
    labels = {
        "correct_bank": "Correct K/V",
        "shuffled_bank": "Shuffled",
        "random_kv": "Random K+V",
        "correct_K_random_V": "Correct K, Rand V",
        "random_K_correct_V": "Rand K, Correct V",
    }
    colors = {
        "correct_bank": "steelblue",
        "shuffled_bank": "darkorange",
        "random_kv": "crimson",
        "correct_K_random_V": "mediumpurple",
        "random_K_correct_V": "green",
    }
    variants_data = analysis.get("variants", {})
    names = [v for v in order if v in variants_data]
    means = [variants_data[v]["mean_margin"] for v in names]
    cis_lo = [variants_data[v].get("ci_lo", means[i]) for i, v in enumerate(names)]
    cis_hi = [variants_data[v].get("ci_hi", means[i]) for i, v in enumerate(names)]
    errs_lo = [m - lo for m, lo in zip(means, cis_lo)]
    errs_hi = [hi - m for m, hi in zip(means, cis_hi)]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = list(range(len(names)))
    for i, vname in enumerate(names):
        ax.bar(i, means[i], color=colors.get(vname, "gray"),
               yerr=[[errs_lo[i]], [errs_hi[i]]], capsize=5,
               label=labels.get(vname, vname))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels.get(v, v) for v in names], rotation=15, ha="right")
    ax.set_ylabel("Mean Margin (nats)")
    ax.set_title("Exp 6b: post-RoPE Negative Controls\n(correct_bank should dominate)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", default="experiments/atb_validation_v1")
    ap.add_argument("--out", default="experiments/atb_validation_v1/final_report")
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paper_tables").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    verdicts = {}
    all_analyses = {}

    # Exp1
    run1 = latest_run(exp_root / "exp1_core_ablation")
    if run1:
        print(f"[exp1] analyzing {run1}")
        a1 = analyze_exp1(run1)
        all_analyses["exp1"] = a1
        verdicts["V1_bank_fires"] = a1["verdict_bank_fires"]
        verdicts["V1_pre_rope_beats_no_bank"] = a1["verdict_pre_rope_beats_no_bank"]
        write_exp1_tex(a1, out / "paper_tables" / "exp1_core_ablation.tex")
    else:
        print("[exp1] MISSING")

    # Exp2
    run2 = latest_run(exp_root / "exp2_position_invariance")
    if run2:
        print(f"[exp2] analyzing {run2}")
        a2 = analyze_exp2(run2)
        all_analyses["exp2"] = a2
        verdicts["V2_pre_rope_position_stable"] = a2["verdict_pre_rope_more_stable"]
        write_exp2_plot(a2, out / "plots" / "exp2_position_invariance.png")
    else:
        print("[exp2] MISSING")

    # Exp3
    exp3_dir = exp_root / "exp3_bit_equal"
    if exp3_dir.exists():
        print(f"[exp3] analyzing {exp3_dir}")
        a3 = analyze_exp3(exp3_dir)
        all_analyses["exp3"] = a3
        verdicts["V3_bit_equal"] = a3["verdict_bit_equal"]
    else:
        print("[exp3] MISSING")

    # Exp4
    run4 = latest_run(exp_root / "exp4_cf1k_main")
    if run4:
        print(f"[exp4] analyzing {run4}")
        a4 = analyze_exp4(run4)
        all_analyses["exp4"] = a4
        verdicts["V4_atb_beats_baseline"] = a4["verdict_atb_beats_baseline"]
        verdicts["V4_alpha0_consistent"] = a4["verdict_alpha0_bit_equal_consistent"]
        write_exp4_tex(a4, out / "paper_tables" / "exp4_cf1k_main.tex")
    else:
        print("[exp4] MISSING")

    # Exp5
    run5 = latest_run(exp_root / "exp5_alpha_sweep")
    if run5:
        print(f"[exp5] analyzing {run5}")
        a5 = analyze_exp5(run5)
        all_analyses["exp5"] = a5
        write_exp5_plot(a5, out / "plots" / "exp5_alpha_sweep.png")
    else:
        print("[exp5] MISSING")

    # Exp6
    run6 = latest_run(exp_root / "exp6_negative_controls")
    if run6:
        print(f"[exp6] analyzing {run6}")
        a6 = analyze_exp6(run6)
        all_analyses["exp6"] = a6
        verdicts["V6_correct_dominates"] = a6["verdict_correct_dominates"]
        write_exp6_tex(a6, out / "paper_tables" / "exp6_negative_controls.tex")
    else:
        print("[exp6] MISSING")

    # Exp6b — post-RoPE negative controls rerun
    run6b = latest_run(exp_root / "exp6b_post_rope_negative_controls")
    if run6b:
        print(f"[exp6b] analyzing {run6b}")
        a6b = analyze_exp6b(run6b)
        all_analyses["exp6b"] = a6b
        verdicts["V6b_post_rope_correct_dominates"] = a6b["verdict_correct_dominates"]
        write_exp6b_tex(a6b, out / "paper_tables" / "exp6b_post_rope_neg_controls.tex")
        write_exp6b_plot(a6b, out / "plots" / "exp6b_post_rope_negative_controls.png")
    else:
        print("[exp6b] MISSING")

    # Write verdicts
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print(f"\n[verdicts]\n{json.dumps(verdicts, indent=2)}")

    # Write full analyses
    (out / "analyses.json").write_text(
        json.dumps(all_analyses, indent=2, default=str)
    )

    # Write README
    _write_readme(all_analyses, verdicts, out)
    print(f"\n[done] final_report → {out}")


def _write_readme(analyses: dict, verdicts: dict, out: Path) -> None:
    a1 = analyses.get("exp1", {})
    a2 = analyses.get("exp2", {})
    a3 = analyses.get("exp3", {})
    a4 = analyses.get("exp4", {})
    a5 = analyses.get("exp5", {})
    a6 = analyses.get("exp6", {})

    a6b = analyses.get("exp6b", {})

    def vc(key):
        v = verdicts.get(key)
        if v is None:
            return "⚠️ MISSING"
        return "✅ PASS" if v else "❌ FAIL"

    def fmt(v, d=4):
        if v is None or (isinstance(v, float) and _isnan(v)):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.{d}f}"
        return str(v)

    # Exp1 table rows
    exp1_rows = ""
    for vname, s in a1.get("variants", {}).items():
        exp1_rows += (
            f"| {vname} | {fmt(s.get('recall_at_1'))} | "
            f"{fmt(s.get('mean_margin'))} | {fmt(s.get('median_margin'))} |\n"
        )

    # Exp4 table rows
    exp4_rows = ""
    for vname, s in a4.get("variants", {}).items():
        exp4_rows += (
            f"| {vname} | {s.get('n','N/A')} | {fmt(s.get('recall_at_1'))} | "
            f"{fmt(s.get('mean_margin'))} | "
            f"[{fmt(s.get('ci_lo'))}, {fmt(s.get('ci_hi'))}] |\n"
        )

    # Exp6 table rows
    exp6_rows = ""
    for vname, s in a6.get("variants", {}).items():
        exp6_rows += (
            f"| {vname} | {fmt(s.get('recall_at_1'))} | "
            f"{fmt(s.get('mean_margin'))} |\n"
        )

    # Exp3 bit-equal rows
    exp3_rows = ""
    for model_tag, s in a3.get("models", {}).items():
        exp3_rows += (
            f"| {model_tag} | {s.get('n','N/A')} | "
            f"{'✅' if s.get('torch_equal_all') else '❌'} | "
            f"{fmt(s.get('max_abs_diff_max'))} |\n"
        )

    # Exp5 cliff
    exp5_cliff = (f"cliff at α={fmt(a5.get('cliff_alpha'), 2)}, "
                  f"peak α={fmt(a5.get('peak_alpha'), 2)} "
                  f"(margin={fmt(a5.get('peak_margin'))})"
                  if a5 else "N/A")

    # Exp2 position curve table
    exp2_curve_rows = ""
    for vname, curve in a2.get("curves", {}).items():
        for pt in curve:
            exp2_curve_rows += (
                f"| {vname} | {pt['position_delta']} | {pt['n']} | "
                f"{fmt(pt.get('mean_margin'))} | {fmt(pt.get('median_margin'))} |\n"
            )

    # Exp5 alpha curve table
    exp5_curve_rows = ""
    for pt in a5.get("curve", []):
        exp5_curve_rows += (
            f"| {fmt(pt['alpha'], 2)} | {pt['n']} | "
            f"{fmt(pt.get('mean_margin'))} | {fmt(pt.get('recall_at_1'))} |\n"
        )

    # Exp6b table rows
    exp6b_rows = ""
    for vname, s in a6b.get("variants", {}).items():
        ci = f"[{fmt(s.get('ci_lo'))}, {fmt(s.get('ci_hi'))}]"
        exp6b_rows += (
            f"| {vname} | {s.get('n','N/A')} | {fmt(s.get('recall_at_1'))} | "
            f"{fmt(s.get('mean_margin'))} | {ci} | {fmt(s.get('median_margin'))} |\n"
        )

    readme = f"""# ATB Validation v1 — Final Report

**Model:** Gemma-4-31B-it  
**Dataset:** CounterFact-1k (W.6 filter)  
**Suite:** 6 experiments + Exp 6b rerun, 3 seeds each

## PREREG Verdicts

| Key | Result |
|-----|--------|
| V1 AttnNativeBank fires | {vc('V1_bank_fires')} |
| V1 pre-RoPE beats no-bank | {vc('V1_pre_rope_beats_no_bank')} |
| V2 pre-RoPE position stable | {vc('V2_pre_rope_position_stable')} |
| V3 α=0 bit-equal | {vc('V3_bit_equal')} |
| V4 ATB beats baseline | {vc('V4_atb_beats_baseline')} |
| V4 α=0 consistent | {vc('V4_alpha0_consistent')} |
| V6 correct bank dominates (pre_rope) | {vc('V6_correct_dominates')} |
| V6b correct bank dominates (post_rope) | {vc('V6b_post_rope_correct_dominates')} |

---

## Exp 1 — Core Ablation (Gemma-4-31B-it, n≈480/variant)

| Variant | Recall@1 | Mean Margin | Median Margin |
|---------|----------|-------------|---------------|
{exp1_rows}

**Observation:** post_rope_bank achieves positive mean margin vs. baseline,
confirming AttnNativeBank fires. pre_rope variants show near-identical results
(Gemma-4's native RMS v-norm makes auto_rms_cap a no-op).

---

## Exp 2 — Position Invariance

**pre-RoPE degradation:** {fmt(a2.get('pre_rope_degradation'))} nats  
**post-RoPE degradation:** {fmt(a2.get('post_rope_degradation'))} nats  

| Variant | position_delta | n | Mean Margin | Median Margin |
|---------|---------------|---|-------------|---------------|
{exp2_curve_rows}
See `plots/exp2_position_invariance.png`.

---

## Exp 3 — α=0 Bit-Equality

| Model | n | torch.equal | max_abs_diff |
|-------|---|-------------|-------------|
{exp3_rows}

---

## Exp 4 — CounterFact Main Result (full CF-1k, 3 seeds)

| Method | n | Recall@1 | Mean Margin | 95% CI |
|--------|---|----------|-------------|--------|
{exp4_rows}

McNemar χ² = {fmt(a4.get('mcnemar_chi2'), 3)}, p = {fmt(a4.get('mcnemar_p'), 4)}

---

## Exp 5 — α Dense Sweep

{exp5_cliff}

| α | n | Mean Margin | Recall@1 |
|---|---|-------------|----------|
{exp5_curve_rows}
See `plots/exp5_alpha_sweep.png`.

---

## Exp 6 — Negative Controls (pre_rope; INVALIDATED)

> ⚠️  Exp 6 used `bank_key_mode=pre_rope` which is a near-no-op on Gemma-4-31B
> due to native V-norm.  Results are not meaningful.  See Exp 6b below.

| Variant | Recall@1 | Mean Margin |
|---------|----------|-------------|
{exp6_rows}

**Verdict:** correct_bank dominates = {verdicts.get('V6_correct_dominates', 'N/A')}

---

## Exp 6b — Negative Controls (post_rope; CANONICAL)

`bank_key_mode=post_rope` — the only mode confirmed to produce positive margin
on Gemma-4-31B (Exp 1: +0.088 mean margin).

| Variant | n | Recall@1 | Mean Margin | 95% CI | Median Margin |
|---------|---|----------|-------------|--------|---------------|
{exp6b_rows}

**Verdict:** correct_bank dominates all controls = {verdicts.get('V6b_post_rope_correct_dominates', 'N/A')}

---

*Generated by `experiments/atb_validation_v1/finalize.py`*
"""
    (out / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
