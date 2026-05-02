#!/usr/bin/env python3
"""Generate Stage 8 closed-book capacity figure (SVG) from real summaries.

Reads ``reports/experiments/stage8_v2_n*_seed0/delta_experiment_summary.json``
and emits ``docs/figures/fig6_stage8_capacity.svg``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports" / "experiments"
FIGS = REPO / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

FONT = "ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def load_runs():
    runs = []
    for p in sorted(REPORTS.glob("stage8_v2_n*_seed0/delta_experiment_summary.json")):
        d = json.loads(p.read_text())
        n = d["n_facts"]
        m = d["metrics"]
        runs.append({
            "n": n,
            "retrieved_top1": m["bank_inject_retrieved"]["top1"],
            "oracle_top1": m["bank_inject_oracle"]["top1"],
            "retrieval_at_1": m["address_retrieval_recall_at_1"],
            "no_memory_top1": m["no_memory"]["top1"],
            "swap_flip": m["swap_paired"].get("paired_flip_rate", 0.0),
        })
    runs.sort(key=lambda r: r["n"])
    return runs


def make_fig(runs):
    W, H = 760, 420
    pad_l, pad_r, pad_t, pad_b = 70, 180, 50, 60
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    ns = [r["n"] for r in runs]
    log_ns = [math.log10(n) for n in ns]
    x_min, x_max = min(log_ns) - 0.2, max(log_ns) + 0.2

    def x_of(logn):
        return pad_l + (logn - x_min) / (x_max - x_min) * pw

    def y_of(v):
        return pad_t + (1.0 - v) * ph

    series = [
        ("oracle_top1", "Bank inject (oracle slot)", "#0a7"),
        ("retrieved_top1", "Bank inject (retrieved slot)", "#06c"),
        ("retrieval_at_1", "Address retrieval recall@1", "#a26"),
        ("swap_flip", "Swap-paired flip rate", "#c63"),
        ("no_memory_top1", "No-memory baseline", "#999"),
    ]

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}" font-size="13">')
    out.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    out.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16" font-weight="600">Stage 8 closed-book recall: capacity curve (Gemma-4-E2B, frozen, GB10/CUDA)</text>')

    # Axes
    out.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#333"/>')
    out.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="#333"/>')
    # Y ticks 0..1
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_of(v)
        out.append(f'<line x1="{pad_l-4}" y1="{y}" x2="{pad_l}" y2="{y}" stroke="#333"/>')
        out.append(f'<text x="{pad_l-8}" y="{y+4}" text-anchor="end">{v:.2f}</text>')
        out.append(f'<line x1="{pad_l}" y1="{y}" x2="{pad_l+pw}" y2="{y}" stroke="#eee"/>')
    out.append(f'<text x="20" y="{pad_t+ph/2}" text-anchor="middle" transform="rotate(-90 20 {pad_t+ph/2})">top-1 / recall / flip rate</text>')
    # X ticks (log10)
    for n in ns:
        x = x_of(math.log10(n))
        y = pad_t + ph
        out.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+4}" stroke="#333"/>')
        out.append(f'<text x="{x}" y="{y+18}" text-anchor="middle">N={n}</text>')
    out.append(f'<text x="{pad_l+pw/2}" y="{pad_t+ph+44}" text-anchor="middle">Number of stored facts (log scale)</text>')

    # 0.80 gate line
    yg = y_of(0.80)
    out.append(f'<line x1="{pad_l}" y1="{yg}" x2="{pad_l+pw}" y2="{yg}" stroke="#d33" stroke-dasharray="4 3" opacity="0.6"/>')
    out.append(f'<text x="{pad_l+pw-4}" y="{yg-4}" text-anchor="end" fill="#d33" font-size="11">G1 gate = 0.80</text>')

    # Series
    for key, label, color in series:
        pts = [(x_of(math.log10(r["n"])), y_of(r[key])) for r in runs]
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')

    # Legend
    lx = pad_l + pw + 20
    ly = pad_t + 10
    out.append(f'<text x="{lx}" y="{ly}" font-weight="600">Channel</text>')
    for i, (_, label, color) in enumerate(series):
        y = ly + 22 + i * 22
        out.append(f'<line x1="{lx}" y1="{y-4}" x2="{lx+24}" y2="{y-4}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<circle cx="{lx+12}" cy="{y-4}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')
        out.append(f'<text x="{lx+30}" y="{y}" font-size="11">{label}</text>')

    out.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="11" fill="#555">3 seeds pending; seed 0 reported. value-token absent at read time. Address selected by cosine over learned key projector.</text>')
    out.append('</svg>')
    return "\n".join(out)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("no stage8_v2 runs found")
    svg = make_fig(runs)
    out = FIGS / "fig6_stage8_capacity.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out}  ({len(runs)} runs)")
    summary = {
        "runs": runs,
        "gates_passed_at_n": {
            r["n"]: {
                "G1_retrieved_top1>=0.80": r["retrieved_top1"] >= 0.80,
                "G5_swap_flip>=0.80": r["swap_flip"] >= 0.80,
                "G6_no_memory<=0.05": r["no_memory_top1"] <= 0.05,
                "GR_retrieval@1>=0.95": r["retrieval_at_1"] >= 0.95,
            }
            for r in runs
        },
    }
    (FIGS / "stage8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
