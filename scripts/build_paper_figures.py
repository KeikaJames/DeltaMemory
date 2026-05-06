#!/usr/bin/env python3
"""Generate the three arXiv-blocker figures for the Mnemosyne paper.

Outputs:
  papers/figures/fig1_u_curve.{png,pdf}        — log-margin vs |M| (gemma-4-31B,
                                                  Qwen3-4B replication overlay if
                                                  present)
  papers/figures/fig2_alpha_cliff.{png,pdf}    — log-margin vs α (gemma-4-31B,
                                                  cliff region [0.05, 0.4]
                                                  highlighted)
  papers/figures/fig3_scar_vs_caa.{png,pdf}    — drift vs α grouped by arch
                                                  (Gemma-4-E2B / Qwen3-4B /
                                                  GLM-4-9B), CAA + SCAR

Data sources:
  runs/X7NL_full_v1_gemma4_31B/cells.jsonl  (sub=A, sub=B)
  runs/X7NL_full_v1_qwen3_4B/cells.jsonl    (sub=A, optional)
  /tmp/scar_data/{gemma4,qwen3_4b,glm4_9b}_summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "papers" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_cells(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok" and "log_margin" not in r:
                continue
            rows.append(r)
    return rows


def aggregate_by(rows: list[dict], key: str, sub: str, value: str = "log_margin"):
    """Returns sorted_keys, means, stds (population stdev or 0 if n=1)."""
    bucket: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("sub") != sub:
            continue
        v = r.get(value)
        if v is None:
            continue
        bucket[float(r[key])].append(float(v))
    keys = sorted(bucket.keys())
    ms = [mean(bucket[k]) for k in keys]
    ss = [stdev(bucket[k]) if len(bucket[k]) > 1 else 0.0 for k in keys]
    ns = [len(bucket[k]) for k in keys]
    return keys, ms, ss, ns


def fig1_u_curve():
    gemma_rows = load_cells(ROOT / "runs/X7NL_full_v1_gemma4_31B/cells.jsonl")
    qwen_rows = load_cells(ROOT / "runs/X7NL_full_v1_qwen3_4B/cells.jsonl")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    if gemma_rows:
        xs, ms, ss, ns = aggregate_by(gemma_rows, "bank_size", "A")
        if xs:
            ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, lw=1.6,
                        label=f"Gemma-4-31B (3 seeds, |M|∈{{{','.join(str(int(x)) for x in xs)}}})",
                        color="#1f4f8b")

    if qwen_rows:
        xs, ms, ss, ns = aggregate_by(qwen_rows, "bank_size", "A")
        if xs:
            ax.errorbar(xs, ms, yerr=ss, marker="s", capsize=3, lw=1.6,
                        label=f"Qwen3-4B (3 seeds, |M|∈{{{','.join(str(int(x)) for x in xs)}}})",
                        color="#c0392b")

    ax.set_xscale("log")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Bank size |M|  (log scale)")
    ax.set_ylabel("Mean log-margin   (target − distractor)")
    ax.set_title("Figure 1 — Bank-size U-curve  (X.7-NL sub-A, α=1.0, native AttnNativeBank)")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig1_u_curve.{ext}", dpi=200)
    plt.close(fig)
    print("[fig1] wrote", FIG / "fig1_u_curve.png")


def fig2_alpha_cliff():
    rows = load_cells(ROOT / "runs/X7NL_full_v1_gemma4_31B/cells.jsonl")
    if not rows:
        print("[fig2] no gemma cells.jsonl; skipping")
        return
    xs, ms, ss, ns = aggregate_by(rows, "alpha", "B")
    if not xs:
        print("[fig2] no sub=B rows; skipping")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axvspan(0.05, 0.4, color="#fde0dc", alpha=0.6, label="Cliff region α∈[0.05, 0.4]")
    ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, lw=1.6,
                color="#1f4f8b", label="Gemma-4-31B (3 seeds, |M|=200)")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Injection scale α")
    ax.set_ylabel("Mean log-margin")
    ax.set_title("Figure 2 — α-cliff at small bank  (X.7-NL sub-B)")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_alpha_cliff.{ext}", dpi=200)
    plt.close(fig)
    print("[fig2] wrote", FIG / "fig2_alpha_cliff.png")


def fig3_scar_vs_caa():
    archs = [
        ("gemma4",   "Gemma-4-E2B"),
        ("qwen3_4b", "Qwen3-4B"),
        ("glm4_9b",  "GLM-4-9B"),
    ]
    data = {}
    for tag, name in archs:
        p = Path(f"/tmp/scar_data/{tag}_summary.json")
        if not p.exists():
            p = ROOT / f"reports/cleanroom/scar_smoke_gb10/{tag}_summary.json"
        if not p.exists():
            print(f"[fig3] missing {tag}; skipping that arch")
            continue
        d = json.load(open(p))
        data[name] = (d["alphas"], d["results"])
    if not data:
        print("[fig3] no SCAR data; skipping")
        return

    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.8),
                             sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (name, (alphas, res)) in zip(axes, data.items()):
        x = np.arange(len(alphas))
        w = 0.38
        caa = [res["caa"][f"{a}"] for a in alphas]
        scar = [res["scar"][f"{a}"] for a in alphas]
        ax.bar(x - w/2, caa, w, label="CAA", color="#c0392b")
        ax.bar(x + w/2, scar, w, label="SCAR", color="#1f4f8b")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}" for a in alphas])
        ax.set_xlabel("α")
        ax.set_title(name, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("Drift  (mean max |Δ logits| on unrelated prompts)")
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Figure 3 — SCAR vs CAA drift across architectures  "
                 "(SCAR smoke, n_test=20 per α)", y=1.02, fontsize=11)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig3_scar_vs_caa.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("[fig3] wrote", FIG / "fig3_scar_vs_caa.png")


def main():
    fig1_u_curve()
    fig2_alpha_cliff()
    fig3_scar_vs_caa()


if __name__ == "__main__":
    main()
