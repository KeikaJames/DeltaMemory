#!/usr/bin/env python3
"""Generate Stage 8 v3 figures (capacity, interference, LAMA) from real summaries.

Reads ``reports/experiments/stage8_v3_n*_seed{0,1,2}/delta_experiment_summary.json``
plus interference/lama summaries and emits:
  docs/figures/fig6_stage8_capacity.svg     (mean ± std across seeds)
  docs/figures/fig7_stage8_interference.svg (sequential-write retention)
  docs/figures/fig8_stage8_lama.svg         (synthetic vs LAMA bars)
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports" / "experiments"
FIGS = REPO / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
FONT = "ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def load_capacity_runs():
    by_n = defaultdict(list)
    for p in sorted(REPORTS.glob("stage8_v3_n*_seed*/delta_experiment_summary.json")):
        d = json.loads(p.read_text())
        n = d["n_facts"]
        m = d["metrics"]
        by_n[n].append({
            "retrieved_top1": m["bank_inject_retrieved"]["top1"],
            "oracle_top1": m["bank_inject_oracle"]["top1"],
            "retrieval_at_1": m["address_retrieval_recall_at_1"],
            "no_memory_top1": m["no_memory"]["top1"],
            "swap_flip": m["swap_paired"]["paired_flip_rate"],
        })
    runs = []
    for n in sorted(by_n.keys()):
        seeds = by_n[n]
        agg = {"n": n, "n_seeds": len(seeds)}
        for k in ["retrieved_top1","oracle_top1","retrieval_at_1","no_memory_top1","swap_flip"]:
            vals = [s[k] for s in seeds]
            agg[k] = statistics.mean(vals)
            agg[k+"_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        runs.append(agg)
    return runs


def make_capacity_fig(runs):
    W, H = 800, 440
    pad_l, pad_r, pad_t, pad_b = 70, 200, 50, 60
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    ns = [r["n"] for r in runs]
    log_ns = [math.log10(n) for n in ns]
    x_min, x_max = min(log_ns) - 0.2, max(log_ns) + 0.2
    x_of = lambda lg: pad_l + (lg - x_min) / (x_max - x_min) * pw
    y_of = lambda v: pad_t + (1.0 - v) * ph
    series = [
        ("oracle_top1", "Bank inject (oracle slot)", "#0a7"),
        ("retrieved_top1", "Bank inject (retrieved slot)", "#06c"),
        ("retrieval_at_1", "Address retrieval recall@1", "#a26"),
        ("swap_flip", "Swap-paired flip rate", "#c63"),
        ("no_memory_top1", "No-memory baseline", "#999"),
    ]
    n_seeds = runs[0]["n_seeds"] if runs else 0
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16" font-weight="600">Stage 8 v3 closed-book capacity (Gemma-4-E2B, frozen, NVIDIA GB10/CUDA, mean ± std over {n_seeds} seeds)</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="#333"/>',
    ]
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_of(v)
        out += [
            f'<line x1="{pad_l-4}" y1="{y}" x2="{pad_l}" y2="{y}" stroke="#333"/>',
            f'<text x="{pad_l-8}" y="{y+4}" text-anchor="end">{v:.2f}</text>',
            f'<line x1="{pad_l}" y1="{y}" x2="{pad_l+pw}" y2="{y}" stroke="#eee"/>',
        ]
    out.append(f'<text x="20" y="{pad_t+ph/2}" text-anchor="middle" transform="rotate(-90 20 {pad_t+ph/2})">top-1 / recall / flip rate</text>')
    for n in ns:
        x = x_of(math.log10(n)); y = pad_t + ph
        out += [
            f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+4}" stroke="#333"/>',
            f'<text x="{x}" y="{y+18}" text-anchor="middle">N={n}</text>',
        ]
    out.append(f'<text x="{pad_l+pw/2}" y="{pad_t+ph+44}" text-anchor="middle">Number of stored facts (log scale)</text>')
    yg = y_of(0.80)
    out += [
        f'<line x1="{pad_l}" y1="{yg}" x2="{pad_l+pw}" y2="{yg}" stroke="#d33" stroke-dasharray="4 3" opacity="0.6"/>',
        f'<text x="{pad_l+pw-4}" y="{yg-4}" text-anchor="end" fill="#d33" font-size="11">G1 gate = 0.80</text>',
    ]
    yg2 = y_of(0.95)
    out += [
        f'<line x1="{pad_l}" y1="{yg2}" x2="{pad_l+pw}" y2="{yg2}" stroke="#7a3" stroke-dasharray="2 2" opacity="0.5"/>',
        f'<text x="{pad_l+pw-4}" y="{yg2-4}" text-anchor="end" fill="#7a3" font-size="11">GR gate = 0.95</text>',
    ]
    for key, label, color in series:
        pts = [(x_of(math.log10(r["n"])), y_of(r[key]), r[key+"_std"]) for r in runs]
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x,y,_ in pts)
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, std in pts:
            if std > 0:
                y_lo = y_of(max(0, (1.0 - (y - pad_t)/ph) - std))
                y_hi = y_of(min(1, (1.0 - (y - pad_t)/ph) + std))
                out += [
                    f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1.2"/>',
                    f'<line x1="{x-3:.1f}" y1="{y_hi:.1f}" x2="{x+3:.1f}" y2="{y_hi:.1f}" stroke="{color}"/>',
                    f'<line x1="{x-3:.1f}" y1="{y_lo:.1f}" x2="{x+3:.1f}" y2="{y_lo:.1f}" stroke="{color}"/>',
                ]
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')
    lx = pad_l + pw + 20; ly = pad_t + 10
    out.append(f'<text x="{lx}" y="{ly}" font-weight="600">Channel</text>')
    for i,(_,label,color) in enumerate(series):
        y = ly + 22 + i*22
        out += [
            f'<line x1="{lx}" y1="{y-4}" x2="{lx+24}" y2="{y-4}" stroke="{color}" stroke-width="2"/>',
            f'<circle cx="{lx+12}" cy="{y-4}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>',
            f'<text x="{lx+30}" y="{y}" font-size="11">{label}</text>',
        ]
    out.append(f'<text x="{W/2}" y="{H-12}" text-anchor="middle" font-size="11" fill="#555">value-token absent at read time. KeyProjector saturates above N≈1024 (in-batch InfoNCE structural ceiling). Oracle slot perfect throughout.</text>')
    out.append('</svg>')
    return "\n".join(out)


def make_interference_fig():
    p = REPORTS / "stage8_v3_interference_n1024_seed0" / "delta_experiment_summary.json"
    if not p.exists(): return None
    d = json.loads(p.read_text())
    hist = d["history"]
    W, H = 720, 360
    pad_l, pad_r, pad_t, pad_b = 70, 200, 50, 60
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    xs = [h["checkpoint_slots_written"] for h in hist]
    x_min, x_max = 0, max(xs)*1.05
    x_of = lambda v: pad_l + (v - x_min)/(x_max - x_min)*pw
    y_of = lambda v: pad_t + (1.0 - v) * ph
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16" font-weight="600">Stage 8.3 sequential-write interference (N=1024, frozen, GB10/CUDA)</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="#333"/>',
    ]
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_of(v)
        out += [f'<line x1="{pad_l-4}" y1="{y}" x2="{pad_l}" y2="{y}" stroke="#333"/>',
                f'<text x="{pad_l-8}" y="{y+4}" text-anchor="end">{v:.2f}</text>',
                f'<line x1="{pad_l}" y1="{y}" x2="{pad_l+pw}" y2="{y}" stroke="#eee"/>']
    for v in xs:
        x = x_of(v)
        out += [f'<line x1="{x}" y1="{pad_t+ph}" x2="{x}" y2="{pad_t+ph+4}" stroke="#333"/>',
                f'<text x="{x}" y="{pad_t+ph+18}" text-anchor="middle">{v}</text>']
    out += [f'<text x="20" y="{pad_t+ph/2}" text-anchor="middle" transform="rotate(-90 20 {pad_t+ph/2})">top-1 / recall@1</text>',
            f'<text x="{pad_l+pw/2}" y="{pad_t+ph+44}" text-anchor="middle">Slots written so far</text>']
    yg = y_of(0.80)
    out += [f'<line x1="{pad_l}" y1="{yg}" x2="{pad_l+pw}" y2="{yg}" stroke="#d33" stroke-dasharray="4 3" opacity="0.6"/>',
            f'<text x="{pad_l+pw-4}" y="{yg-4}" text-anchor="end" fill="#d33" font-size="11">G3 gate = 0.80</text>']
    series = [
        ("earliest_top1", "Earliest 128 slots, top-1 (retention)", "#06c"),
        ("all_written_top1", "All slots written so far, top-1", "#a26"),
    ]
    for key, label, color in series:
        pts = [(x_of(h["checkpoint_slots_written"]), y_of(h[key])) for h in hist]
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')
    lx = pad_l + pw + 20; ly = pad_t + 10
    out.append(f'<text x="{lx}" y="{ly}" font-weight="600">Series</text>')
    for i,(_,label,color) in enumerate(series):
        y = ly + 22 + i*22
        out += [f'<line x1="{lx}" y1="{y-4}" x2="{lx+24}" y2="{y-4}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{lx+30}" y="{y}" font-size="11">{label}</text>']
    out.append(f'<text x="{W/2}" y="{H-10}" text-anchor="middle" font-size="11" fill="#555">earliest 128 slots retain top-1=0.969 throughout. graceful decay on full pool as bank fills.</text>')
    out.append('</svg>')
    return "\n".join(out)


def make_lama_fig():
    by_seed = []
    for seed in [0,1,2]:
        p = REPORTS / f"stage8_v3_lama_seed{seed}" / "delta_experiment_summary.json"
        if not p.exists(): continue
        d = json.loads(p.read_text())
        m = d["metrics"]
        by_seed.append({
            "retrieved": m["bank_inject_retrieved"]["top1"],
            "oracle": m["bank_inject_oracle"]["top1"],
            "recall": m["address_retrieval_recall_at_1"],
            "no_mem": m["no_memory"]["top1"],
            "swap": m["swap_paired"]["paired_flip_rate"],
        })
    if not by_seed: return None
    syn_p = REPORTS / "stage8_v3_n128_seed0" / "delta_experiment_summary.json"
    syn_runs = []
    for s in [0,1,2]:
        sp = REPORTS / f"stage8_v3_n128_seed{s}" / "delta_experiment_summary.json"
        if sp.exists():
            sd = json.loads(sp.read_text())["metrics"]
            syn_runs.append({
                "retrieved": sd["bank_inject_retrieved"]["top1"],
                "oracle": sd["bank_inject_oracle"]["top1"],
                "recall": sd["address_retrieval_recall_at_1"],
                "no_mem": sd["no_memory"]["top1"],
                "swap": sd["swap_paired"]["paired_flip_rate"],
            })
    def agg(rs, k):
        vs = [r[k] for r in rs]
        return statistics.mean(vs), (statistics.stdev(vs) if len(vs)>1 else 0.0)
    keys = [("retrieved","retrieved top1"),("recall","recall@1"),("oracle","oracle slot top1"),("swap","swap-paired flip"),("no_mem","no-memory baseline")]
    W, H = 720, 380
    pad_l, pad_r, pad_t, pad_b = 70, 30, 60, 80
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    n_groups = len(keys)
    group_w = pw / n_groups
    bar_w = group_w * 0.32
    y_of = lambda v: pad_t + (1.0 - v) * ph
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}" font-size="13">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16" font-weight="600">Stage 8 closed-book: synthetic colors (N=128) vs LAMA curated (3 seeds, GB10/CUDA)</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="#333"/>',
    ]
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_of(v)
        out += [f'<line x1="{pad_l-4}" y1="{y}" x2="{pad_l}" y2="{y}" stroke="#333"/>',
                f'<text x="{pad_l-8}" y="{y+4}" text-anchor="end">{v:.2f}</text>',
                f'<line x1="{pad_l}" y1="{y}" x2="{pad_l+pw}" y2="{y}" stroke="#eee"/>']
    for i,(k,label) in enumerate(keys):
        cx = pad_l + i*group_w + group_w/2
        m_syn, s_syn = agg(syn_runs, k)
        m_lam, s_lam = agg(by_seed, k)
        x1 = cx - bar_w - 2
        x2 = cx + 2
        y1 = y_of(m_syn); y2 = y_of(m_lam)
        out += [
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{(pad_t+ph-y1):.1f}" fill="#06c" opacity="0.85"/>',
            f'<rect x="{x2:.1f}" y="{y2:.1f}" width="{bar_w:.1f}" height="{(pad_t+ph-y2):.1f}" fill="#0a7" opacity="0.85"/>',
            f'<text x="{x1+bar_w/2:.1f}" y="{y1-4:.1f}" text-anchor="middle" font-size="10">{m_syn:.3f}</text>',
            f'<text x="{x2+bar_w/2:.1f}" y="{y2-4:.1f}" text-anchor="middle" font-size="10">{m_lam:.3f}</text>',
            f'<text x="{cx:.1f}" y="{pad_t+ph+18}" text-anchor="middle" font-size="11">{label}</text>',
        ]
    lx = pad_l + 12; ly = H - 30
    out += [
        f'<rect x="{lx}" y="{ly-12}" width="14" height="14" fill="#06c" opacity="0.85"/>',
        f'<text x="{lx+20}" y="{ly}" font-size="12">Synthetic colors (N=128)</text>',
        f'<rect x="{lx+200}" y="{ly-12}" width="14" height="14" fill="#0a7" opacity="0.85"/>',
        f'<text x="{lx+220}" y="{ly}" font-size="12">LAMA curated (N=135 single-token facts)</text>',
        '</svg>',
    ]
    return "\n".join(out)


def main():
    runs = load_capacity_runs()
    if runs:
        (FIGS / "fig6_stage8_capacity.svg").write_text(make_capacity_fig(runs), encoding="utf-8")
        print(f"wrote fig6 ({len(runs)} N points, {runs[0]['n_seeds']} seeds)")
    inter = make_interference_fig()
    if inter:
        (FIGS / "fig7_stage8_interference.svg").write_text(inter, encoding="utf-8")
        print("wrote fig7")
    lama = make_lama_fig()
    if lama:
        (FIGS / "fig8_stage8_lama.svg").write_text(lama, encoding="utf-8")
        print("wrote fig8")
    summary = {
        "capacity_runs": runs,
        "gates_passed_at_n": {
            r["n"]: {
                "G1_retrieved_top1>=0.80": r["retrieved_top1"] >= 0.80,
                "G5_swap_flip>=0.80": r["swap_flip"] >= 0.80,
                "G6_no_memory<=0.05": r["no_memory_top1"] <= 0.05,
                "GR_retrieval@1>=0.95": r["retrieval_at_1"] >= 0.95,
            } for r in runs
        },
    }
    (FIGS / "stage8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote stage8_summary.json")


if __name__ == "__main__":
    main()
