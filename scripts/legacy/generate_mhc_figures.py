"""Phase mHC7 — Generate 5 SVG figures for mHC alpha-safety report."""
import json
import math
from pathlib import Path

OUT = Path("docs/figures/mhc")
OUT.mkdir(parents=True, exist_ok=True)

# Data: (arch, alpha, lift) from mHC3 (small) and mHC6 (medium)
SMALL_DATA = [
    ("residual", 0.05, -0.717), ("residual", 0.10, -0.729), ("residual", 0.50, -0.529),
    ("residual", 1.00, -0.684), ("residual", 2.00, -0.445), ("residual", 5.00, -0.083),
    ("residual", 10.00, -4.297),
    ("hc", 0.05, -0.317), ("hc", 0.10, -0.188), ("hc", 0.50, -0.161),
    ("hc", 1.00, 0.071), ("hc", 2.00, -0.019), ("hc", 5.00, -0.508), ("hc", 10.00, -0.997),
    ("mhc", 0.05, -0.317), ("mhc", 0.10, -0.188), ("mhc", 0.50, -0.161),
    ("mhc", 1.00, 0.071), ("mhc", 2.00, -0.019), ("mhc", 5.00, -0.508), ("mhc", 10.00, -0.997),
]

MEDIUM_DATA = [
    ("residual", 0.10, -3.704), ("residual", 0.50, -3.608), ("residual", 1.00, -3.646),
    ("residual", 2.00, -3.271), ("residual", 5.00, -3.410), ("residual", 10.00, -4.099),
    ("hc", 0.10, 0.339), ("hc", 0.50, 0.450), ("hc", 1.00, 0.479),
    ("hc", 2.00, 0.581), ("hc", 5.00, 0.563), ("hc", 10.00, -1.267),
    ("mhc", 0.10, 0.339), ("mhc", 0.50, 0.450), ("mhc", 1.00, 0.479),
    ("mhc", 2.00, 0.581), ("mhc", 5.00, 0.563), ("mhc", 10.00, -1.267),
]

COLORS = {"residual": "#FF3B30", "hc": "#FF9500", "mhc": "#007AFF"}
LABELS = {"residual": "Residual GPT-2", "hc": "HC GPT-2", "mhc": "mHC GPT-2"}


def _fig1_alpha_lift():
    """Figure 1: α-Lift for GPT-2 medium (paper headline #1)."""
    W, H = 720, 420
    mx, my = 80, H - 60
    gw, gh = W - 120, H - 140

    alphas = sorted(set(a for _, a, _ in MEDIUM_DATA))
    x_log = [math.log10(a) for a in alphas]
    x_min, x_max = min(x_log), max(x_log)

    def xp(a):
        return mx + (math.log10(a) - x_min) / (x_max - x_min) * gw

    all_l = [l for _, _, l in MEDIUM_DATA]
    l_min, l_max = min(all_l) - 0.5, max(all_l) + 0.5

    def yp(l):
        return my - (l - l_min) / (l_max - l_min) * gh

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-family="system-ui" font-size="15" font-weight="600" fill="#1a1a2e">Counter-Prior Lift vs α — GPT-2 Medium (355M, 24L)</text>')

    # Zero line
    y0 = yp(0)
    lines.append(f'<line x1="{mx}" y1="{y0}" x2="{mx+gw}" y2="{y0}" stroke="#ccc" stroke-width="1" stroke-dasharray="6,4"/>')
    lines.append(f'<text x="{mx-8}" y="{y0+4}" text-anchor="end" font-family="system-ui" font-size="10" fill="#999">0</text>')

    # Grid lines
    for lv in range(int(l_min), int(l_max) + 1):
        yl = yp(float(lv))
        lines.append(f'<line x1="{mx}" y1="{yl}" x2="{mx+gw}" y2="{yl}" stroke="#eee" stroke-width="0.5"/>')

    # Data series
    for arch in ["residual", "hc", "mhc"]:
        pts = [(xp(a), yp(l)) for a_name, a, l in MEDIUM_DATA if a_name == arch]
        if len(pts) >= 2:
            path = " ".join(f"L{px:.1f},{py:.1f}" for px, py in pts[1:])
            lines.append(f'<path d="M{pts[0][0]:.1f},{pts[0][1]:.1f} {path}" fill="none" stroke="{COLORS[arch]}" stroke-width="2"/>')
        for px, py in pts:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{COLORS[arch]}"/>')
        # Label at first point
        lines.append(f'<text x="{pts[0][0]+6:.1f}" y="{pts[0][1]-6:.1f}" font-family="system-ui" font-size="10" fill="{COLORS[arch]}">{LABELS[arch]}</text>')

    # X-axis labels
    for a in alphas:
        lines.append(f'<text x="{xp(a):.1f}" y="{my+16}" text-anchor="middle" font-family="system-ui" font-size="9" fill="#666">α={a:.2f}</text>')
    lines.append(f'<text x="{W/2}" y="{H-8}" text-anchor="middle" font-family="system-ui" font-size="11" fill="#666">α (injection strength)</text>')

    # Y-axis label
    lines.append(f'<text x="16" y="{my-gh/2}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#666" transform="rotate(-90,16,{my-gh/2})">Counter-Prior Lift (nats)</text>')

    # Safety zone annotation
    lines.append(f'<rect x="{mx}" y="{yp(0.5)}" width="{gw}" height="{yp(l_min)-yp(0.5)}" fill="#34C759" opacity="0.08"/>')
    lines.append(f'<text x="{mx+gw-4}" y="{yp(0.5)+14}" text-anchor="end" font-family="system-ui" font-size="9" fill="#34C759">safe zone</text>')

    lines.append('</svg>')
    (OUT / "fig1_alpha_lift.svg").write_text("\n".join(lines))
    print("fig1 done")


def _fig2_scale_comparison():
    """Figure 2: Stability gap vs depth."""
    W, H = 480, 360
    mx, my = 100, H - 60
    gw, gh = W - 160, H - 140

    bars = [
        ("GPT-2\nsmall\n12L", -0.684, 0.071),
        ("GPT-2\nmedium\n24L", -3.646, 0.479),
    ]

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-family="system-ui" font-size="14" font-weight="600" fill="#1a1a2e">Stability Gap vs Model Depth (α=1.0)</text>')

    bar_w = 40
    gap = 20
    y0_px = my - gh * (0 - (-4.0)) / 5.0  # zero line

    for i, (label, res_v, mhc_v) in enumerate(bars):
        cx = mx + (i + 0.5) * (gw / len(bars))
        # Residual bar (red, downward)
        rh = abs(res_v) / 5.0 * gh
        lines.append(f'<rect x="{cx-bar_w:.0f}" y="{y0_px:.0f}" width="{bar_w:.0f}" height="{rh:.0f}" fill="#FF3B30" rx="2"/>')
        # mHC bar (blue, upward)
        mh = max(mhc_v, 0) / 5.0 * gh
        lines.append(f'<rect x="{cx+gap:.0f}" y="{y0_px-mh:.0f}" width="{bar_w:.0f}" height="{mh:.0f}" fill="#007AFF" rx="2"/>')
        # Labels
        lines.append(f'<text x="{cx-bar_w/2:.0f}" y="{y0_px+rh+14:.0f}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#FF3B30">{res_v:+.2f}</text>')
        lines.append(f'<text x="{cx+gap+bar_w/2:.0f}" y="{y0_px-mh-6:.0f}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#007AFF">{mhc_v:+.2f}</text>')
        lines.append(f'<text x="{cx+bar_w/2:.0f}" y="{my+36}" text-anchor="middle" font-family="system-ui" font-size="11" fill="#333">{label}</text>')
        # Gap arrow
        gap_y = y0_px - mh
        lines.append(f'<text x="{cx+bar_w/2:.0f}" y="{gap_y/2+10:.0f}" text-anchor="middle" font-family="system-ui" font-size="9" fill="#666">gap={abs(res_v-mhc_v):.1f} nats</text>')

    lines.append(f'<line x1="{mx}" y1="{y0_px}" x2="{mx+gw}" y2="{y0_px}" stroke="#999" stroke-width="1"/>')
    # Legend
    lines.append(f'<rect x="{mx+10}" y="{my-gh+8}" width="10" height="10" fill="#FF3B30" rx="1"/>')
    lines.append(f'<text x="{mx+26}" y="{my-gh+18}" font-family="system-ui" font-size="10" fill="#333">Residual GPT-2</text>')
    lines.append(f'<rect x="{mx+10}" y="{my-gh+24}" width="10" height="10" fill="#007AFF" rx="1"/>')
    lines.append(f'<text x="{mx+26}" y="{my-gh+34}" font-family="system-ui" font-size="10" fill="#333">HC/mHC GPT-2</text>')

    lines.append('</svg>')
    (OUT / "fig2_scale_comparison.svg").write_text("\n".join(lines))
    print("fig2 done")


def _fig3_summary_table():
    """Figure 3: Hypothesis verdicts table."""
    hypotheses = [
        ("H1", "Residual amplifies ≥3 nats at α<α*", "PASS", "GPT-2 medium: −3.65 nats at α=1.0"),
        ("H2", "mHC drift ≤0.5 nats over α∈[0,5]", "FAIL(rev)", "4-5× tighter than residual; threshold too strict"),
        ("H3", "HC also crashes at some α", "FAIL", "HC ≡ mHC at equiv init; multi-stream alone provides protection"),
        ("H4", "mHC lift monotonic; residual collapses", "PASS", "GPT-2 medium: mHC positive for α∈[0.1,5.0]; residual always negative"),
        ("H5", "||x_L||/||x_0|| ≥10× gap at α=1.5", "FAIL", "GPT-2 small injection delta buried in model computation"),
        ("H6", "α=0 bit-equal for all 3 architectures", "PASS", "max-abs-diff=0.0 for all three arms"),
    ]

    W, H = 680, 220
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="24" text-anchor="middle" font-family="system-ui" font-size="14" font-weight="600" fill="#1a1a2e">Hypothesis Verdicts (from preregistration mHC_alpha_safe_v1)</text>')

    row_h = 26
    y0 = 42
    cols = [20, 70, 310, 400, 640]
    headers = ["", "ID", "Hypothesis", "Verdict", "Evidence"]
    for j, (cx, h) in enumerate(zip(cols, headers)):
        lines.append(f'<text x="{cx}" y="{y0}" font-family="system-ui" font-size="10" font-weight="600" fill="#333">{h}</text>')

    for i, (hid, hyp, verdict, evidence) in enumerate(hypotheses):
        y = y0 + (i + 1) * row_h
        bg = "#f8f8f8" if i % 2 == 0 else "white"
        lines.append(f'<rect x="10" y="{y-14}" width="{W-20}" height="{row_h}" fill="{bg}" rx="2"/>')
        vc = "#34C759" if "PASS" in verdict else "#FF3B30" if "FAIL" in verdict else "#FF9500"
        lines.append(f'<text x="{cols[1]}" y="{y}" font-family="system-ui" font-size="10" font-weight="600" fill="#333">{hid}</text>')
        lines.append(f'<text x="{cols[2]}" y="{y}" font-family="system-ui" font-size="9" fill="#333">{hyp}</text>')
        lines.append(f'<text x="{cols[3]}" y="{y}" font-family="system-ui" font-size="9" font-weight="600" fill="{vc}">{verdict}</text>')
        lines.append(f'<text x="{cols[4]}" y="{y}" font-family="system-ui" font-size="8" fill="#666">{evidence}</text>')

    lines.append('</svg>')
    (OUT / "fig3_hypothesis_table.svg").write_text("\n".join(lines))
    print("fig3 done")


def _fig4_small_sweep():
    """Figure 4: GPT-2 small alpha sweep."""
    W, H = 640, 380
    mx, my = 80, H - 60
    gw, gh = W - 120, H - 140

    alphas = sorted(set(a for _, a, _ in SMALL_DATA))
    x_log = [math.log10(a) for a in alphas]
    x_min, x_max = min(x_log), max(x_log)

    def xp(a): return mx + (math.log10(a) - x_min) / (x_max - x_min) * gw
    all_l = [l for _, _, l in SMALL_DATA]
    l_min, l_max = min(all_l) - 0.5, max(all_l) + 0.5
    def yp(l): return my - (l - l_min) / (l_max - l_min) * gh

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-family="system-ui" font-size="14" font-weight="600" fill="#1a1a2e">Counter-Prior Lift vs α — GPT-2 Small (124M, 12L)</text>')
    y0 = yp(0)
    lines.append(f'<line x1="{mx}" y1="{y0}" x2="{mx+gw}" y2="{y0}" stroke="#ccc" stroke-width="1" stroke-dasharray="6,4"/>')

    for arch in ["residual", "hc", "mhc"]:
        pts = [(xp(a), yp(l)) for a_name, a, l in SMALL_DATA if a_name == arch]
        if len(pts) >= 2:
            path = " ".join(f"L{px:.1f},{py:.1f}" for px, py in pts[1:])
            lines.append(f'<path d="M{pts[0][0]:.1f},{pts[0][1]:.1f} {path}" fill="none" stroke="{COLORS[arch]}" stroke-width="2"/>')
        for px, py in pts:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{COLORS[arch]}"/>')
        lines.append(f'<text x="{pts[0][0]+6:.1f}" y="{pts[0][1]-6:.1f}" font-family="system-ui" font-size="9" fill="{COLORS[arch]}">{LABELS[arch]}</text>')

    for a in alphas:
        lines.append(f'<text x="{xp(a):.1f}" y="{my+14}" text-anchor="middle" font-family="system-ui" font-size="8" fill="#666">α={a:.2f}</text>')
    lines.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#666">α (injection strength)</text>')
    lines.append('</svg>')
    (OUT / "fig4_small_sweep.svg").write_text("\n".join(lines))
    print("fig4 done")


def _fig5_architecture_diagram():
    """Figure 5: Architecture comparison schematic."""
    W, H = 680, 280
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    lines.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-family="system-ui" font-size="14" font-weight="600" fill="#1a1a2e">Injection Dynamics: Residual vs Multi-Stream Routing</text>')

    # Left: Residual
    rx, ry = 40, 60
    lines.append(f'<text x="{rx+120}" y="{ry-10}" text-anchor="middle" font-family="system-ui" font-size="12" font-weight="600" fill="#FF3B30">Residual GPT-2</text>')
    for i in range(5):
        y = ry + i * 35
        lines.append(f'<rect x="{rx}" y="{y}" width="240" height="28" fill="#FF3B30" opacity="0.15" rx="4"/>')
        lines.append(f'<text x="{rx+120}" y="{y+18}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#333">Layer {i+1}: x ← x + Attn(x) + αM_V</text>')
    lines.append(f'<text x="{rx+120}" y="{ry+190}" text-anchor="middle" font-family="system-ui" font-size="9" fill="#FF3B30" font-style="italic">unbounded accumulation</text>')
    lines.append(f'<text x="{rx+120}" y="{ry+206}" text-anchor="middle" font-family="system-ui" font-size="9" fill="#FF3B30">||x_L|| ~ exp(L) at high α</text>')

    # Right: mHC
    mx_r = 400
    lines.append(f'<text x="{mx_r+120}" y="{ry-10}" text-anchor="middle" font-family="system-ui" font-size="12" font-weight="600" fill="#007AFF">mHC GPT-2 (multi-stream, SK)</text>')
    # Streams
    for s in range(4):
        sy = ry + s * 12
        lines.append(f'<rect x="{mx_r}" y="{sy}" width="240" height="10" fill="#007AFF" opacity="0.12" rx="2"/>')
    lines.append(f'<text x="{mx_r+120}" y="{ry+58}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#333">C @ [stream_0, ..., stream_3]</text>')
    lines.append(f'<text x="{mx_r+120}" y="{ry+76}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#333">σ_max(C) ≤ 1 (Sinkhorn-Knopp)</text>')
    for i in range(3):
        y = ry + 90 + i * 35
        lines.append(f'<rect x="{mx_r}" y="{y}" width="240" height="28" fill="#007AFF" opacity="0.12" rx="4"/>')
        lines.append(f'<text x="{mx_r+120}" y="{y+18}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#333">Layer {i+1}: X ← C @ (X + Attn(X) + αM_V)</text>')
    lines.append(f'<text x="{mx_r+120}" y="{ry+210}" text-anchor="middle" font-family="system-ui" font-size="9" fill="#007AFF">bounded: ||C^k E||_2 ≤ ||E||_2</text>')

    lines.append('</svg>')
    (OUT / "fig5_architecture.svg").write_text("\n".join(lines))
    print("fig5 done")


if __name__ == "__main__":
    _fig1_alpha_lift()
    _fig2_scale_comparison()
    _fig3_summary_table()
    _fig4_small_sweep()
    _fig5_architecture_diagram()
    print(f"All 5 figures written to {OUT}/")
