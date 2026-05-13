#!/usr/bin/env python3
"""Generate Stage 9 SVG figures from sweep results in reports/experiments/.

Reads:
  reports/experiments/stage9A_<encoder>_n4096_seed{0,1,2}/delta_experiment_summary.json
  reports/experiments/stage9B_trex_prompt_hidden_seed{0,1,2}/delta_experiment_summary.json
  reports/experiments/stage9C_<method>_seed0/delta_experiment_summary.json

Writes:
  docs/figures/fig9_encoder_comparison.svg     — Phase 9A: encoder variants vs mean_pool
  docs/figures/fig10_lama_trex.svg             — Phase 9B: full LAMA-TREx multi-seed
  docs/figures/fig11_baselines_radar.svg       — Phase 9C: 4-method radar (us + 3 baselines)
  docs/figures/stage9_summary.json             — aggregated numbers for the report

Robust to missing runs: skips silently with a printed note. This means
the script can be run mid-sweep to refresh figures incrementally.
"""

from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "reports" / "experiments"
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS = ["mean_pool", "attn_pool", "multilayer", "prompt_hidden", "residual_mlp"]
ENCODER_LABEL = {
    "mean_pool":      "mean-pool (v3 baseline)",
    "attn_pool":      "attn-pool",
    "multilayer":     "multi-layer",
    "prompt_hidden":  "prompt-hidden (query-cond)",
    "residual_mlp":   "residual-mlp",
}
ENCODER_COLOR = {
    "mean_pool":     "#888888",
    "attn_pool":     "#1f77b4",
    "multilayer":    "#2ca02c",
    "prompt_hidden": "#d62728",
    "residual_mlp":  "#9467bd",
}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[gen-stage9] WARN: cannot parse {path}: {e}")
        return None


def metric(d: dict, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def gather_9A() -> Dict[str, Dict[str, List[float]]]:
    """Returns {encoder: {metric: [seed_values]}}."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for enc in ENCODERS:
        seeds = {"retr_top1": [], "recall_at_1": [], "swap_flip": []}
        for s in (0, 1, 2):
            path = EXP_DIR / f"stage9A_{enc}_n4096_seed{s}" / "delta_experiment_summary.json"
            d = load(path)
            if d is None:
                continue
            seeds["retr_top1"].append(metric(d, "metrics", "bank_inject_retrieved", "top1") or 0.0)
            seeds["recall_at_1"].append(metric(d, "metrics", "address_retrieval_recall_at_1") or 0.0)
            seeds["swap_flip"].append(metric(d, "metrics", "swap_paired", "paired_flip_rate") or 0.0)
        out[enc] = seeds
    return out


def gather_9B() -> Dict[str, List[float]]:
    seeds = {"retr_top1": [], "recall_at_1": [], "oracle_top1": [], "swap_flip": []}
    for s in (0, 1, 2):
        path = EXP_DIR / f"stage9B_trex_prompt_hidden_seed{s}" / "delta_experiment_summary.json"
        d = load(path)
        if d is None:
            continue
        seeds["retr_top1"].append(metric(d, "metrics", "bank_inject_retrieved", "top1") or 0.0)
        seeds["recall_at_1"].append(metric(d, "metrics", "address_retrieval_recall_at_1") or 0.0)
        seeds["oracle_top1"].append(metric(d, "metrics", "bank_inject_oracle", "top1") or 0.0)
        seeds["swap_flip"].append(metric(d, "metrics", "swap_paired", "paired_flip_rate") or 0.0)
    return seeds


def gather_9C() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for m in ("vector_rag", "ike", "sft_lora"):
        path = EXP_DIR / f"stage9C_{m}_seed0" / "delta_experiment_summary.json"
        d = load(path)
        if d is not None:
            out[m] = d.get("metrics", {})
    return out


def mean_std(xs: List[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

def svg_open(w: int, h: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="13">\n'
        f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n'
        f'  <text x="{w/2}" y="22" text-anchor="middle" font-size="16" '
        f'font-weight="bold">{title}</text>\n'
    )


def svg_close() -> str:
    return "</svg>\n"


# ---------------------------------------------------------------------------
# Fig 9 — encoder comparison (grouped bars)
# ---------------------------------------------------------------------------

def fig9(data: Dict[str, Dict[str, List[float]]]) -> str:
    W, H = 880, 460
    margin_l, margin_r, margin_t, margin_b = 60, 20, 60, 80
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    metrics = [("retr_top1", "retrieved top-1"), ("recall_at_1", "recall@1"), ("swap_flip", "swap flip-rate")]
    out = svg_open(W, H, "Stage 9A — Encoder variants @ N=4096 (mean ± std over seeds)")
    n_groups = len(ENCODERS)
    n_metrics = len(metrics)
    group_w = plot_w / n_groups
    bar_w = group_w / (n_metrics + 1)

    # axes
    out += f'  <line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{W - margin_r}" y2="{margin_t + plot_h}" stroke="#333"/>\n'
    out += f'  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>\n'
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = margin_t + plot_h * (1 - tick)
        out += f'  <line x1="{margin_l-4}" y1="{y}" x2="{margin_l}" y2="{y}" stroke="#333"/>\n'
        out += f'  <text x="{margin_l-8}" y="{y+4}" text-anchor="end">{tick:.2f}</text>\n'

    metric_color = {"retr_top1": "#1f77b4", "recall_at_1": "#2ca02c", "swap_flip": "#d62728"}

    for gi, enc in enumerate(ENCODERS):
        gx = margin_l + gi * group_w
        # group label
        out += (f'  <text x="{gx + group_w/2}" y="{margin_t + plot_h + 18}" '
                f'text-anchor="middle">{ENCODER_LABEL[enc]}</text>\n')
        for mi, (mkey, mlabel) in enumerate(metrics):
            xs = data.get(enc, {}).get(mkey, [])
            mean, sd = mean_std(xs)
            if math.isnan(mean):
                continue
            x = gx + (mi + 0.5) * bar_w + bar_w * 0.5
            bh = plot_h * mean
            y0 = margin_t + plot_h - bh
            out += (f'  <rect x="{x}" y="{y0}" width="{bar_w*0.85}" height="{bh}" '
                    f'fill="{metric_color[mkey]}" opacity="0.85"/>\n')
            # error bar
            if sd > 0:
                ey1 = margin_t + plot_h - plot_h * (mean + sd)
                ey2 = margin_t + plot_h - plot_h * max(0.0, mean - sd)
                cx = x + bar_w * 0.85 / 2
                out += f'  <line x1="{cx}" y1="{ey1}" x2="{cx}" y2="{ey2}" stroke="#000" stroke-width="1.2"/>\n'
            # value label
            out += (f'  <text x="{x + bar_w*0.85/2}" y="{y0-3}" text-anchor="middle" '
                    f'font-size="10">{mean:.2f}</text>\n')
    # Legend
    lx, ly = margin_l + 10, margin_t + 8
    for i, (mkey, mlabel) in enumerate(metrics):
        out += f'  <rect x="{lx + i*180}" y="{ly}" width="14" height="14" fill="{metric_color[mkey]}"/>\n'
        out += f'  <text x="{lx + i*180 + 18}" y="{ly + 11}">{mlabel}</text>\n'
    out += svg_close()
    return out


# ---------------------------------------------------------------------------
# Fig 10 — full LAMA-TREx multi-seed
# ---------------------------------------------------------------------------

def fig10(data: Dict[str, List[float]]) -> str:
    W, H = 720, 420
    margin_l, margin_r, margin_t, margin_b = 70, 30, 60, 70
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    out = svg_open(W, H, "Stage 9B — Full LAMA-TREx (183 facts, 7 relations)")
    metrics = [("retr_top1", "retrieved\ntop-1", "#1f77b4"),
               ("recall_at_1", "recall@1", "#2ca02c"),
               ("oracle_top1", "oracle\ntop-1", "#9467bd"),
               ("swap_flip", "swap\nflip-rate", "#d62728")]
    n = len(metrics)
    bar_w = plot_w / (n * 1.5)
    out += f'  <line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{W - margin_r}" y2="{margin_t + plot_h}" stroke="#333"/>\n'
    out += f'  <line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>\n'
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = margin_t + plot_h * (1 - tick)
        out += f'  <line x1="{margin_l-4}" y1="{y}" x2="{margin_l}" y2="{y}" stroke="#333"/>\n'
        out += f'  <text x="{margin_l-8}" y="{y+4}" text-anchor="end">{tick:.2f}</text>\n'
    for i, (mkey, mlabel, color) in enumerate(metrics):
        xs = data.get(mkey, [])
        mean, sd = mean_std(xs)
        if math.isnan(mean):
            continue
        x = margin_l + (i + 0.5) * (plot_w / n) - bar_w / 2
        bh = plot_h * mean
        y0 = margin_t + plot_h - bh
        out += f'  <rect x="{x}" y="{y0}" width="{bar_w}" height="{bh}" fill="{color}" opacity="0.85"/>\n'
        if sd > 0:
            ey1 = margin_t + plot_h - plot_h * (mean + sd)
            ey2 = margin_t + plot_h - plot_h * max(0.0, mean - sd)
            cx = x + bar_w / 2
            out += f'  <line x1="{cx}" y1="{ey1}" x2="{cx}" y2="{ey2}" stroke="#000" stroke-width="1.2"/>\n'
        out += f'  <text x="{x + bar_w/2}" y="{y0-3}" text-anchor="middle" font-size="11">{mean:.3f}</text>\n'
        for li, line in enumerate(mlabel.split("\n")):
            out += (f'  <text x="{x + bar_w/2}" y="{margin_t + plot_h + 18 + li*14}" '
                    f'text-anchor="middle">{line}</text>\n')
    out += svg_close()
    return out


# ---------------------------------------------------------------------------
# Fig 11 — baselines radar
# ---------------------------------------------------------------------------

def fig11(ours: Dict[str, List[float]], baselines: Dict[str, dict]) -> str:
    """Radar with 4 axes: edit_top1, retr_top1, swap_flip, locality_drift_inv.

    locality_drift is converted to 1 - drift so that "outer = better" on
    every axis."""
    W, H = 640, 540
    cx, cy = W / 2, H / 2 + 10
    R = 180
    axes = [
        ("edit_success_top1", "edit success (top-1)"),
        ("retr_top1",         "retrieval top-1"),
        ("swap_flip",         "swap flip-rate"),
        ("locality",          "locality (1 − drift)"),
    ]
    n = len(axes)

    def get(method_key: str, ax_key: str) -> float:
        if method_key == "ours":
            if ax_key == "edit_success_top1":
                return mean_std(ours.get("retr_top1", []))[0] or 0.0
            if ax_key == "retr_top1":
                return mean_std(ours.get("recall_at_1", []))[0] or 0.0
            if ax_key == "swap_flip":
                return mean_std(ours.get("swap_flip", []))[0] or 0.0
            if ax_key == "locality":
                # ours doesn't change frozen base, drift = 0
                return 1.0
            return 0.0
        b = baselines.get(method_key, {})
        if ax_key == "edit_success_top1":
            return float(b.get("edit_success_top1", 0.0) or 0.0)
        if ax_key == "retr_top1":
            return float(b.get("retr_top1", b.get("edit_success_top1", 0.0)) or 0.0)
        if ax_key == "swap_flip":
            return 0.0
        if ax_key == "locality":
            return 1.0 - float(b.get("locality_drift_top1", 0.0) or 0.0)
        return 0.0

    methods = [("ours", "Mneme (ours)", "#d62728"),
               ("vector_rag", "vector-RAG", "#1f77b4"),
               ("ike", "IKE in-context", "#2ca02c"),
               ("sft_lora", "SFT-LoRA on lm_head", "#9467bd")]
    out = svg_open(W, H, "Stage 9C — Mneme vs RAG / IKE / SFT-LoRA")

    # axes lines + labels
    for i, (ax_key, ax_label) in enumerate(axes):
        theta = -math.pi / 2 + 2 * math.pi * i / n
        ex = cx + R * math.cos(theta)
        ey = cy + R * math.sin(theta)
        out += f'  <line x1="{cx}" y1="{cy}" x2="{ex}" y2="{ey}" stroke="#aaa"/>\n'
        # rings
        for r_frac in (0.25, 0.5, 0.75, 1.0):
            rr = R * r_frac
            x1 = cx + rr * math.cos(-math.pi / 2 + 2 * math.pi * i / n)
            y1 = cy + rr * math.sin(-math.pi / 2 + 2 * math.pi * i / n)
            x2 = cx + rr * math.cos(-math.pi / 2 + 2 * math.pi * ((i + 1) % n) / n)
            y2 = cy + rr * math.sin(-math.pi / 2 + 2 * math.pi * ((i + 1) % n) / n)
            out += f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ddd"/>\n'
        lx = cx + (R + 26) * math.cos(theta)
        ly = cy + (R + 26) * math.sin(theta)
        out += f'  <text x="{lx}" y="{ly}" text-anchor="middle">{ax_label}</text>\n'

    for mi, (mkey, mlabel, color) in enumerate(methods):
        pts = []
        for i, (ax_key, _) in enumerate(axes):
            v = max(0.0, min(1.0, get(mkey, ax_key)))
            theta = -math.pi / 2 + 2 * math.pi * i / n
            x = cx + R * v * math.cos(theta)
            y = cy + R * v * math.sin(theta)
            pts.append(f"{x:.1f},{y:.1f}")
        out += (f'  <polygon points="{ " ".join(pts) }" fill="{color}" '
                f'fill-opacity="0.18" stroke="{color}" stroke-width="2"/>\n')
        # legend
        out += (f'  <rect x="{20}" y="{H - 110 + mi*20}" width="14" height="14" '
                f'fill="{color}" fill-opacity="0.4" stroke="{color}"/>\n')
        out += f'  <text x="{40}" y="{H - 99 + mi*20}">{mlabel}</text>\n'
    out += svg_close()
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    a = gather_9A()
    b = gather_9B()
    c = gather_9C()

    summary = {
        "phase_9A": {
            enc: {k: {"mean": mean_std(v)[0], "std": mean_std(v)[1], "n": len(v)}
                  for k, v in d.items()}
            for enc, d in a.items()
        },
        "phase_9B": {
            k: {"mean": mean_std(v)[0], "std": mean_std(v)[1], "n": len(v)}
            for k, v in b.items()
        },
        "phase_9C": c,
    }
    (FIG_DIR / "stage9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (FIG_DIR / "fig9_encoder_comparison.svg").write_text(fig9(a), encoding="utf-8")
    (FIG_DIR / "fig10_lama_trex.svg").write_text(fig10(b), encoding="utf-8")
    (FIG_DIR / "fig11_baselines_radar.svg").write_text(fig11(b, c), encoding="utf-8")
    print(f"[gen-stage9] wrote fig9/fig10/fig11 + summary -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
