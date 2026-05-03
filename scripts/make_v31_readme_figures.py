"""Generate README figures for the v3.1 counter-prior evidence.

The figures are intentionally pure-SVG and dependency-free so the README can be
regenerated anywhere the committed JSON transcripts are available.
"""
from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "transcripts" / "v31_intervention"
REPORTS = ROOT / "reports" / "cleanroom"
OUT = ROOT / "docs" / "figures" / "v31"

COLORS = {
    "apple_blue": "#007AFF",
    "apple_green": "#34C759",
    "apple_orange": "#FF9500",
    "apple_red": "#FF3B30",
    "google_blue": "#4285F4",
    "google_green": "#34A853",
    "google_yellow": "#FBBC05",
    "google_red": "#EA4335",
    "ink": "#1D1D1F",
    "muted": "#6E6E73",
    "grid": "#E8EAED",
    "panel": "#F5F5F7",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_svg(name: str, width: int, height: int, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <style>
    .title {{ font: 700 28px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {COLORS['ink']}; }}
    .subtitle {{ font: 500 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {COLORS['muted']}; }}
    .label {{ font: 600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {COLORS['ink']}; }}
    .small {{ font: 500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {COLORS['muted']}; }}
    .tiny {{ font: 500 10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {COLORS['muted']}; }}
    .value {{ font: 700 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {COLORS['ink']}; }}
  </style>
{body}
</svg>
""",
        encoding="utf-8",
    )


def false_deltas(run_dir: str) -> list[tuple[str, float]]:
    data = load_json(TRANSCRIPTS / run_dir / "demo.json")
    rows = []
    for row in data["results"]:
        target = row["target_token"].strip()
        b0 = row["B0_no_memory"]["target_logprob"]
        v3 = row["v3_attn_bank"]["target_logprob"]
        rows.append((target, v3 - b0))
    return rows


def true_means() -> dict[str, float]:
    data = load_json(REPORTS / "stage15_dev_v31" / "summary.json")

    def mean(key: str) -> float:
        rows = data[key]
        return sum(r["recall_at_1_mean"] for r in rows) / len(rows)

    return {
        "B0": mean("B0_no_memory"),
        "B1": mean("B1_prompt_insertion"),
        "RAG": mean("B2_rag_oracle"),
        "v2": mean("v2_period_no_kproj"),
        "v3.1": mean("v3_period_kproj"),
    }


def draw_header(title: str, subtitle: str) -> str:
    return (
        f'  <text x="40" y="48" class="title">{esc(title)}</text>\n'
        f'  <text x="40" y="75" class="subtitle">{esc(subtitle)}</text>\n'
    )


def fig_architecture() -> None:
    boxes = [
        (54, 145, 180, 86, "Write prompt", "False or true fact"),
        (292, 145, 210, 86, "Frozen LLM", "one write forward"),
        (560, 145, 210, 86, "Per-layer K/V bank", "captured at attention"),
        (828, 145, 210, 86, "Read prompt", "question only"),
        (292, 325, 478, 100, "Attention merge", "Attn(Q, [K; MK], [V; α·MV]) in every supported layer"),
        (828, 325, 210, 100, "LM head", "target log-prob shifts"),
    ]
    body = draw_header(
        "DeltaMemory v3.1 forward path",
        "External K/V memory is merged into frozen attention; LLM weights are never edited.",
    )
    body += f'  <rect x="24" y="104" width="1040" height="368" rx="34" fill="{COLORS["panel"]}"/>\n'
    for x, y, w, h, title, subtitle in boxes:
        body += (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="white" stroke="{COLORS["grid"]}" stroke-width="1.4"/>\n'
            f'  <text x="{x + 22}" y="{y + 36}" class="label">{esc(title)}</text>\n'
            f'  <text x="{x + 22}" y="{y + 62}" class="small">{esc(subtitle)}</text>\n'
        )
    arrows = [
        (240, 188, 284, 188, COLORS["apple_blue"]),
        (508, 188, 552, 188, COLORS["apple_blue"]),
        (776, 188, 820, 188, COLORS["apple_blue"]),
        (665, 235, 665, 315, COLORS["google_green"]),
        (932, 235, 932, 315, COLORS["google_blue"]),
        (776, 375, 820, 375, COLORS["google_red"]),
    ]
    for x1, y1, x2, y2, color in arrows:
        body += (
            f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>\n'
            f'  <circle cx="{x2}" cy="{y2}" r="5" fill="{color}"/>\n'
        )
    body += (
        f'  <text x="54" y="520" class="small">Invariant: α=0 or empty bank ⇒ bit-equal to the unpatched base model.</text>\n'
        f'  <text x="54" y="546" class="small">Trainable surface: bank-side K-projector only; no LoRA, no MEMIT/ROME weight edit, no prompt insertion at read time.</text>\n'
    )
    write_svg("v31_architecture.svg", 1088, 580, body)


def fig_false_fact_lift() -> None:
    runs = [
        ("Gemma GB10", "gemma-4-e2b-gb10-FALSE", COLORS["apple_blue"]),
        ("Gemma Mac", "gemma-4-e2b-mac-FALSE", COLORS["apple_green"]),
        ("Qwen3 GB10", "qwen3-4b-gb10-FALSE", COLORS["google_blue"]),
        ("Qwen3 Mac", "qwen3-4b-mac-FALSE", COLORS["google_green"]),
    ]
    series = [(name, false_deltas(run), color) for name, run, color in runs]
    labels = [target for target, _ in series[0][1]]
    width, height = 1088, 600
    left, top, chart_w, chart_h = 82, 126, 888, 350
    ymax = 3.0
    body = draw_header(
        "Counter-prior target lift",
        "Δ log-prob = DeltaMemory attn-bank − no-memory baseline; every bar is positive on Gemma-4 and Qwen3.",
    )
    body += f'  <rect x="24" y="94" width="1040" height="470" rx="34" fill="{COLORS["panel"]}"/>\n'
    for i in range(4):
        y = top + chart_h - i * chart_h / 3
        value = i * ymax / 3
        body += f'  <line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        body += f'  <text x="38" y="{y + 4:.1f}" class="tiny">+{value:.1f}</text>\n'
    group_w = chart_w / len(labels)
    bar_w = 28
    for gi, label in enumerate(labels):
        gx = left + gi * group_w + 18
        for si, (_, rows, color) in enumerate(series):
            delta = rows[gi][1]
            h = delta / ymax * chart_h
            x = gx + si * (bar_w + 4)
            y = top + chart_h - h
            body += f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="7" fill="{color}"/>\n'
            body += f'  <text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="tiny">+{delta:.2f}</text>\n'
        body += f'  <text x="{gx + 61:.1f}" y="{top + chart_h + 34}" text-anchor="middle" class="label">{esc(label)}</text>\n'
    lx = 760
    for i, (name, _, color) in enumerate(series):
        y = 115 + i * 24
        body += f'  <rect x="{lx}" y="{y}" width="14" height="14" rx="4" fill="{color}"/>\n'
        body += f'  <text x="{lx + 22}" y="{y + 12}" class="small">{esc(name)}</text>\n'
    body += f'  <text x="82" y="532" class="small">Source: transcripts/v31_intervention/*-FALSE/demo.json. Base LLM frozen; identity-init K-projector.</text>\n'
    write_svg("v31_false_fact_lift.svg", width, height, body)


def fig_cross_hardware() -> None:
    pairs = [
        ("Gemma-4", false_deltas("gemma-4-e2b-gb10-FALSE"), false_deltas("gemma-4-e2b-mac-FALSE"), COLORS["apple_blue"]),
        ("Qwen3", false_deltas("qwen3-4b-gb10-FALSE"), false_deltas("qwen3-4b-mac-FALSE"), COLORS["google_blue"]),
    ]
    width, height = 1088, 600
    left, top, size = 160, 128, 360
    body = draw_header(
        "Cross-hardware reproducibility",
        "Mac MPS reproduces GB10 CUDA on the same five counter-prior writes.",
    )
    body += f'  <rect x="24" y="94" width="1040" height="470" rx="34" fill="{COLORS["panel"]}"/>\n'
    for i in range(4):
        xy = left + i * size / 3
        value = i * 3.0 / 3
        y = top + size - i * size / 3
        body += f'  <line x1="{xy:.1f}" y1="{top}" x2="{xy:.1f}" y2="{top + size}" stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        body += f'  <line x1="{left}" y1="{y:.1f}" x2="{left + size}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        body += f'  <text x="{xy:.1f}" y="{top + size + 24}" text-anchor="middle" class="tiny">+{value:.0f}</text>\n'
        body += f'  <text x="{left - 18}" y="{y + 4:.1f}" text-anchor="end" class="tiny">+{value:.0f}</text>\n'
    body += f'  <line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke="{COLORS["muted"]}" stroke-width="2" stroke-dasharray="7 8"/>\n'
    for model, gb10, mac, color in pairs:
        for (target, xdelta), (_, ydelta) in zip(gb10, mac):
            x = left + xdelta / 3.0 * size
            y = top + size - ydelta / 3.0 * size
            body += f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" opacity="0.86"/>\n'
            body += f'  <text x="{x + 11:.1f}" y="{y - 9:.1f}" class="tiny">{esc(target)}</text>\n'
    body += f'  <text x="{left + size / 2}" y="{top + size + 54}" text-anchor="middle" class="label">GB10 CUDA Δ log-prob</text>\n'
    body += f'  <text x="64" y="{top + size / 2}" transform="rotate(-90 64 {top + size / 2})" text-anchor="middle" class="label">Mac MPS Δ log-prob</text>\n'
    body += f'  <rect x="650" y="180" width="300" height="156" rx="24" fill="white" stroke="{COLORS["grid"]}"/>\n'
    body += f'  <text x="680" y="222" class="label">Result</text>\n'
    body += f'  <text x="680" y="254" class="small">Gemma-4: 5/5 positive on both machines</text>\n'
    body += f'  <text x="680" y="282" class="small">Qwen3: 5/5 positive on both machines</text>\n'
    body += f'  <text x="680" y="310" class="small">bf16 differences stay in the same band</text>\n'
    body += f'  <circle cx="682" cy="378" r="7" fill="{COLORS["apple_blue"]}"/><text x="700" y="382" class="small">Gemma-4</text>\n'
    body += f'  <circle cx="805" cy="378" r="7" fill="{COLORS["google_blue"]}"/><text x="823" y="382" class="small">Qwen3</text>\n'
    write_svg("v31_cross_hardware.svg", width, height, body)


def fig_recall_evolution() -> None:
    data = true_means()
    order = [
        ("B0", "No memory", COLORS["muted"]),
        ("v2", "Raw bank", COLORS["google_yellow"]),
        ("v3.1", "K-projector", COLORS["apple_blue"]),
        ("B1", "Prompt", COLORS["google_green"]),
        ("RAG", "RAG oracle", COLORS["google_blue"]),
    ]
    width, height = 1088, 560
    left, top, chart_w, chart_h = 96, 132, 850, 300
    body = draw_header(
        "Held-out recall context",
        "Gemma-4 dev_v31 recall@1: v3.1 materially lifts the raw bank, but prompt/RAG remain the upper bar.",
    )
    body += f'  <rect x="24" y="94" width="1040" height="424" rx="34" fill="{COLORS["panel"]}"/>\n'
    for i in range(4):
        y = top + chart_h - i * chart_h / 3
        value = i / 3
        body += f'  <line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        body += f'  <text x="50" y="{y + 4:.1f}" class="tiny">{value:.2f}</text>\n'
    bar_gap = chart_w / len(order)
    for i, (key, label, color) in enumerate(order):
        value = data[key]
        h = value * chart_h
        x = left + i * bar_gap + 48
        y = top + chart_h - h
        body += f'  <rect x="{x:.1f}" y="{y:.1f}" width="82" height="{h:.1f}" rx="15" fill="{color}"/>\n'
        body += f'  <text x="{x + 41:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="value">{value:.3f}</text>\n'
        body += f'  <text x="{x + 41:.1f}" y="{top + chart_h + 34}" text-anchor="middle" class="label">{esc(label)}</text>\n'
    body += f'  <text x="96" y="486" class="small">Source: reports/cleanroom/stage15_dev_v31/summary.json.</text>\n'
    write_svg("v31_recall_context.svg", width, height, body)


def fig_deepseek_alpha() -> None:
    runs = [
        ("0.05", "deepseek-r1-distill-qwen-32b-gb10-FALSE"),
        ("0.10", "deepseek-r1-distill-qwen-32b-gb10-FALSE-a0.1"),
        ("0.20", "deepseek-r1-distill-qwen-32b-gb10-FALSE-a0.2"),
        ("0.30", "deepseek-r1-distill-qwen-32b-gb10-FALSE-a0.3"),
    ]
    colors = [COLORS["muted"], COLORS["google_yellow"], COLORS["apple_orange"], COLORS["google_red"]]
    series = [(alpha, false_deltas(run), color) for (alpha, run), color in zip(runs, colors)]
    labels = [target for target, _ in series[0][1]]
    width, height = 1088, 600
    left, top, chart_w, chart_h = 92, 136, 870, 330
    ymin, ymax = -2.0, 2.0

    def y_of(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * chart_h

    body = draw_header(
        "DeepSeek-32B counter-prior α sweep",
        "A stronger 32B prior is not fully overridden by the identity-init bank; this remains the main limitation.",
    )
    body += f'  <rect x="24" y="94" width="1040" height="470" rx="34" fill="{COLORS["panel"]}"/>\n'
    for value in [-2, -1, 0, 1, 2]:
        y = y_of(value)
        stroke = COLORS["ink"] if value == 0 else COLORS["grid"]
        body += f'  <line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1"/>\n'
        body += f'  <text x="44" y="{y + 4:.1f}" class="tiny">{value:+.0f}</text>\n'
    group_w = chart_w / len(labels)
    bar_w = 27
    zero_y = y_of(0)
    for gi, label in enumerate(labels):
        gx = left + gi * group_w + 20
        for si, (_, rows, color) in enumerate(series):
            delta = rows[gi][1]
            y = y_of(max(delta, 0))
            h = abs(y_of(delta) - zero_y)
            if delta < 0:
                y = zero_y
            x = gx + si * (bar_w + 4)
            body += f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="7" fill="{color}"/>\n'
        body += f'  <text x="{gx + 61:.1f}" y="{top + chart_h + 34}" text-anchor="middle" class="label">{esc(label)}</text>\n'
    lx = 750
    for i, (alpha, _, color) in enumerate(series):
        y = 115 + i * 24
        body += f'  <rect x="{lx}" y="{y}" width="14" height="14" rx="4" fill="{color}"/>\n'
        body += f'  <text x="{lx + 22}" y="{y + 12}" class="small">α={esc(alpha)}</text>\n'
    body += f'  <text x="92" y="532" class="small">Source: transcripts/v31_intervention/deepseek-r1-distill-qwen-32b-gb10-FALSE*/demo.json.</text>\n'
    write_svg("v31_deepseek_alpha.svg", width, height, body)


def main() -> None:
    fig_architecture()
    fig_false_fact_lift()
    fig_cross_hardware()
    fig_recall_evolution()
    fig_deepseek_alpha()
    print(f"Wrote SVG figures to {OUT}")


if __name__ == "__main__":
    main()
