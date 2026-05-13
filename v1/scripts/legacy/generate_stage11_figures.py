#!/usr/bin/env python3
"""Stage 11 SVG figures (paper-style, no matplotlib dependency at the
output level — pure SVG so Gits/web renderers handle them)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "reports/experiments/stage11_grand_evaluation/stage11_summary.json"
OUT_DIR = REPO / "docs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bar_chart(title, items, gate_value=None, ymax=1.0, gate_label="gate", w=720, h=320):
    """items: list of (label, value, ci_low, ci_high, pass_bool|None)."""
    pad_l, pad_r, pad_t, pad_b = 90, 30, 50, 100
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    n = len(items)
    bar_w = plot_w / max(n, 1) * 0.6
    gap = plot_w / max(n, 1) * 0.4
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    svg.append(f'<text x="{w/2}" y="24" font-size="15" font-weight="600" text-anchor="middle">{title}</text>')
    # Y axis
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h * (1 - frac)
        svg.append(f'<line x1="{pad_l}" x2="{pad_l+plot_w}" y1="{y}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(f'<text x="{pad_l-8}" y="{y+4}" font-size="10" text-anchor="end" fill="#6b7280">{frac*ymax:.2f}</text>')
    # Gate line
    if gate_value is not None:
        gy = pad_t + plot_h * (1 - gate_value/ymax)
        svg.append(f'<line x1="{pad_l}" x2="{pad_l+plot_w}" y1="{gy}" y2="{gy}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3"/>')
        svg.append(f'<text x="{pad_l+plot_w-4}" y="{gy-4}" font-size="10" text-anchor="end" fill="#dc2626">{gate_label} {gate_value:.2f}</text>')
    # Bars
    for i, (lab, v, lo, hi, ok) in enumerate(items):
        x = pad_l + (gap/2) + i * (bar_w + gap)
        bh = plot_h * (v / ymax)
        by = pad_t + plot_h - bh
        color = "#10b981" if ok else "#ef4444" if ok is False else "#3b82f6"
        svg.append(f'<rect x="{x}" y="{by}" width="{bar_w}" height="{bh}" fill="{color}" opacity="0.85"/>')
        # CI whiskers
        if hi > lo:
            cx = x + bar_w/2
            ly = pad_t + plot_h * (1 - lo/ymax)
            hy = pad_t + plot_h * (1 - hi/ymax)
            svg.append(f'<line x1="{cx}" x2="{cx}" y1="{ly}" y2="{hy}" stroke="#1f2937" stroke-width="1.5"/>')
            svg.append(f'<line x1="{cx-5}" x2="{cx+5}" y1="{ly}" y2="{ly}" stroke="#1f2937" stroke-width="1.5"/>')
            svg.append(f'<line x1="{cx-5}" x2="{cx+5}" y1="{hy}" y2="{hy}" stroke="#1f2937" stroke-width="1.5"/>')
        # Bar label
        svg.append(f'<text x="{x+bar_w/2}" y="{by-6}" font-size="11" text-anchor="middle" font-weight="600">{v:.3f}</text>')
        # X label (rotated)
        lx = x + bar_w/2
        ly = pad_t + plot_h + 14
        svg.append(f'<text x="{lx}" y="{ly}" font-size="11" text-anchor="end" transform="rotate(-30 {lx} {ly})">{lab}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    if not SUMMARY.exists():
        print(f"missing {SUMMARY}; run aggregate_stage11.py first")
        return 1
    s = json.loads(SUMMARY.read_text())

    # Fig 12: paraphrase recall held-out
    items = []
    for enc, e in s["11A_paraphrase_holdout_recall_at_1"].items():
        items.append((enc, e["mean"], e["ci95_low"], e["ci95_high"], e.get("pass")))
    (OUT_DIR / "stage11_fig12_paraphrase_holdout.svg").write_text(
        _bar_chart("Stage 11A — held-out paraphrase recall@1 (gate G11A ≥ 0.85)",
                   items, gate_value=0.85))

    # Fig 13: LORO bind by relation
    items = [(rel, e["mean"], e["ci95_low"], e["ci95_high"], e.get("pass"))
             for rel, e in s["11B_loro_bind_top1_by_relation"].items()]
    (OUT_DIR / "stage11_fig13_loro_bind.svg").write_text(
        _bar_chart("Stage 11B — train-time LORO bind top-1 by held-out relation (gate ≥ 0.50)",
                   items, gate_value=0.50))

    # Fig 14: ConvQA recall vs k
    items = []
    for k in (1, 3, 5, 10):
        e = s["11D_convqa"][f"k_{k}_recall"]
        items.append((f"k={k}", e["mean"], e["ci95_low"], e["ci95_high"], e.get("pass")))
    (OUT_DIR / "stage11_fig14_convqa_vs_k.svg").write_text(
        _bar_chart("Stage 11D-1 — ConvQA recall@1 vs filler turns (gate k=10 ≥ 0.85)",
                   items, gate_value=0.85))

    # Fig 15: chat-API vs RAG
    items = [
        ("DM (chat-API)", s["11D_chat_api_dm_top1"]["mean"], s["11D_chat_api_dm_top1"]["ci95_low"], s["11D_chat_api_dm_top1"]["ci95_high"], None),
        ("RAG baseline",  s["11D_chat_api_rag_top1"]["mean"], s["11D_chat_api_rag_top1"]["ci95_low"], s["11D_chat_api_rag_top1"]["ci95_high"], None),
    ]
    (OUT_DIR / "stage11_fig15_chat_api_vs_rag.svg").write_text(
        _bar_chart("Stage 11D-2 — chat-as-write-API top-1 (DM vs RAG, equal-budget)", items))

    # Fig 16: poisoning
    items = [
        ("overwrite rate", s["11D_poisoning_overwrite_rate"]["mean"], s["11D_poisoning_overwrite_rate"]["ci95_low"], s["11D_poisoning_overwrite_rate"]["ci95_high"], s["11D_poisoning_overwrite_rate"].get("pass")),
        ("benign accept",  s["11D_poisoning_benign_accept"]["mean"], s["11D_poisoning_benign_accept"]["ci95_low"], s["11D_poisoning_benign_accept"]["ci95_high"], s["11D_poisoning_benign_accept"].get("pass")),
        ("recall after attack", s["11D_poisoning_original_recall"]["mean"], s["11D_poisoning_original_recall"]["ci95_low"], s["11D_poisoning_original_recall"]["ci95_high"], s["11D_poisoning_original_recall"].get("pass")),
    ]
    (OUT_DIR / "stage11_fig16_poisoning.svg").write_text(
        _bar_chart("Stage 11D-3 — prompt-injection / poisoning resistance (3 gates)", items))

    print(f"wrote 5 SVGs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
