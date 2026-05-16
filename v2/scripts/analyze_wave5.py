#!/usr/bin/env python3
"""Wave5 analyzer.

Three questions, three tables:
  A. e11 noise variants at L21 — does random ≈ real replicate?
  B. e02 scale sweep — where exactly does B2 break?
  C. e13 multi-task — is Δ uniform across tasks (capacity) or task-specific (retrieval)?

Also dumps e17 negation, e18 2-hop, and e11 n2/n4 leftovers.

Outputs both stdout and v2/scripts/wave5_summary.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E = REPO / "v2" / "experiments"


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def delta(d: dict | None) -> float | None:
    if not d:
        return None
    for k in ("delta_signed", "delta_real", "delta"):
        v = d.get("verdict", {}).get(k) if isinstance(d.get("verdict"), dict) else None
        if isinstance(v, (int, float)):
            return float(v)
    b = d.get("before", {}); a = d.get("after", {})
    if "real" in a and "base" in b:
        return float(a["real"]) - float(b["base"])
    return None


def get(d: dict | None, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def fmt(v, w=8):
    if v is None:
        return "   —   "
    if isinstance(v, float):
        return f"{v:+{w}.3f}"
    return f"{v:>{w}}"


def section_a() -> list[str]:
    out = ["## A. e11 noise at L21 (replication test)\n"]
    out.append("| variant | layer | Δ | post.real | post.off | base | rule |")
    out.append("|---|---:|---:|---:|---:|---:|---|")
    # L9 baseline (already known) + L21
    cells = [
        ("n1_iid_gaussian", 9), ("n1_iid_gaussian", 21),
        ("n3_single_row_replicated", 9), ("n3_single_row_replicated", 21),
        ("n5_constant_vector", 9), ("n5_constant_vector", 21),
        ("n6_real_bank_K1", 9),
        ("n2_uniform_sphere", 9), ("n4_single_random_replicated", 9),
    ]
    for v, L in cells:
        suffix = f"_L{L}" if L != 9 else ""
        p = E / "e11_noise_robustness" / f"e11_{v}{suffix}_seed0.json"
        d = load(p)
        out.append(
            f"| {v} | {L} | {fmt(delta(d))} | {fmt(get(d,'after','real'))} | "
            f"{fmt(get(d,'after','off'))} | {fmt(get(d,'before','base'))} | "
            f"{(get(d,'verdict','rule') or '—')[:40]} |"
        )
    # Also include real-bank reference rows from e01
    out.append("")
    out.append("Reference (real bank, same config):")
    out.append("| variant | layer | Δ |")
    out.append("|---|---:|---:|")
    for L in (9, 21):
        for s in (0, 1, 2):
            tag = "canonical" if L == 9 else "h6_layer21"
            p = E / "e01_anticheat_b2" / f"e01_{tag}_seed{s}.json"
            d = load(p)
            if d:
                out.append(f"| e01 {tag} s{s} | {L} | {fmt(delta(d))} |")
    return out


def section_b() -> list[str]:
    out = ["\n## B. e02 scale sweep — find the breakpoint\n"]
    out.append("| n_preload | n_train | steps | Δ | post.real | base |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    cells_dir = E / "e02_scale_matrix" / "cells"
    rows = []
    if cells_dir.exists():
        for p in sorted(cells_dir.glob("*.json")):
            d = load(p)
            cfg = d.get("config", {}) if d and isinstance(d.get("config"), dict) else (d or {})
            rows.append((
                cfg.get("n_preload"), cfg.get("n_train"), cfg.get("steps"),
                delta(d), get(d, "after", "real"), get(d, "before", "base"), p.name,
            ))
    rows.sort(key=lambda r: (r[0] or 0, r[1] or 0, r[2] or 0))
    for n, t, s, dl, pr, b, name in rows:
        out.append(f"| {fmt(n,5)} | {fmt(t,5)} | {fmt(s,5)} | {fmt(dl)} | {fmt(pr)} | {fmt(b)} |")
    if not rows:
        out.append("| _no cells_ |  |  |  |  |  |")
    return out


def section_c() -> list[str]:
    out = ["\n## C. e13 multi-task — capacity vs retrieval discriminator\n"]
    p = E / "e13_multi_task_capability" / "e13_seed0.json"
    if not p.exists():
        cands = list((E / "e13_multi_task_capability").glob("*.json"))
        p = cands[0] if cands else p
    d = load(p)
    if not d:
        out.append("_e13 not done yet_")
        return out
    benchmarks = d.get("benchmarks", {}) or {}
    if not benchmarks:
        out.append(f"_unrecognized shape; keys: {list(d.keys())}_")
        return out
    out.append("| task | metric | base | bank_off | bank_on | Δ (on−base) |")
    out.append("|---|---|---:|---:|---:|---:|")
    deltas = []
    for name, blk in benchmarks.items():
        base = blk.get("base", {}); off = blk.get("bank_off", {}); on = blk.get("bank_on", {})
        metric = "nll" if "nll" in base else ("acc" if "acc" in base else None)
        if metric is None:
            out.append(f"| {name} | ? | — | — | — | — |")
            continue
        bv, ov, nv = base.get(metric), off.get(metric), on.get(metric)
        dl = (nv - bv) if isinstance(nv,(int,float)) and isinstance(bv,(int,float)) else None
        if dl is not None and metric == "nll":
            deltas.append(("nll", name, dl))
        elif dl is not None and metric == "acc":
            deltas.append(("acc", name, dl))
        out.append(f"| {name} | {metric} | {fmt(bv)} | {fmt(ov)} | {fmt(nv)} | {fmt(dl)} |")
    nll_helps = [(n, x) for k, n, x in deltas if k == "nll" and x <= -0.5]
    nll_hurts = [(n, x) for k, n, x in deltas if k == "nll" and x >  0.5]
    acc_helps = [(n, x) for k, n, x in deltas if k == "acc" and x >=  0.03]
    acc_hurts = [(n, x) for k, n, x in deltas if k == "acc" and x <= -0.03]
    out.append("")
    out.append(f"NLL tasks helped (Δ≤−0.5): {nll_helps or 'none'}")
    out.append(f"NLL tasks hurt  (Δ≥+0.5): {nll_hurts or 'none'}")
    out.append(f"Acc tasks helped (Δ≥+0.03): {acc_helps or 'none'}")
    out.append(f"Acc tasks hurt  (Δ≤−0.03): {acc_hurts or 'none'}")
    out.append("")
    out.append("**Interpretation:**")
    out.append("- ≥2 unrelated tasks helped → **capacity / adapter** reading confirmed.")
    out.append("- Only the train-aligned task helped (others ≈0 or hurt) → **retrieval-specific** reading survives.")
    return out


def section_others() -> list[str]:
    out = ["\n## D. e17 negation / e18 2-hop / e08 interrupt\n"]
    paths = [
        ("e17 negation", E / "e17_negation_robustness" / "e17_seed0.json"),
        ("e18 2-hop",    E / "e18_2hop" / "e18_seed0.json"),
        ("e08 interrupt",E / "e08_interrupt_api_demo" / "e08_seed0.json"),
    ]
    out.append("| exp | Δ | pass | rule |")
    out.append("|---|---:|:-:|---|")
    for name, p in paths:
        d = load(p)
        if not d:
            out.append(f"| {name} | — | — | _not done_ |")
            continue
        out.append(
            f"| {name} | {fmt(delta(d))} | "
            f"{get(d,'verdict','pass')} | "
            f"{(get(d,'verdict','rule') or '—')[:50]} |"
        )
    return out


def main() -> int:
    lines = ["# Wave5 summary\n"]
    lines += section_a()
    lines += section_b()
    lines += section_c()
    lines += section_others()
    text = "\n".join(lines) + "\n"
    out = REPO / "v2" / "scripts" / "wave5_summary.md"
    out.write_text(text)
    sys.stdout.write(text)
    sys.stderr.write(f"\nWrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
