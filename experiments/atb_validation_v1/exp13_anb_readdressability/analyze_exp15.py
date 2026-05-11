"""Exp15 analyzer — K/V binding 2×2 verdict."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

CELLS = ("Kc_Vc", "Kc_Vr", "Kr_Vc", "Kr_Vr")
B_BOOT = 2000


def bootstrap_ci(d, B=B_BOOT, alpha=0.05, seed=12345):
    rng = random.Random(seed); n = len(d)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = [sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)]
    means.sort()
    return (statistics.fmean(d),
            means[int(alpha / 2 * B)],
            means[int((1 - alpha / 2) * B) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    rows = [json.loads(ln) for ln in (in_dir / "cells.jsonl").read_text().splitlines() if ln.strip()]
    idx = {}
    for r in rows:
        if r.get("variant") == "skipped" or "margin" not in r:
            continue
        idx[(r["seed"], r["fact_id"], r["variant"])] = float(r["margin"])
    seeds = sorted({r["seed"] for r in rows})
    fids = sorted({r["fact_id"] for r in rows})

    means = {}
    for v in ("base",) + CELLS:
        ms = [idx[(s, f, v)] for s in seeds for f in fids if (s, f, v) in idx]
        means[v] = statistics.fmean(ms) if ms else float("nan")

    # Paired contrasts
    contrasts = {
        "Kc_Vc_vs_Kc_Vr": ("Kc_Vc", "Kc_Vr"),
        "Kc_Vc_vs_Kr_Vc": ("Kc_Vc", "Kr_Vc"),
        "Kc_Vc_vs_Kr_Vr": ("Kc_Vc", "Kr_Vr"),
        "Kc_Vr_vs_Kr_Vr": ("Kc_Vr", "Kr_Vr"),  # K-only effect
        "Kr_Vc_vs_Kr_Vr": ("Kr_Vc", "Kr_Vr"),  # V-only effect
    }
    cis = {}
    for name, (a, b) in contrasts.items():
        d = [idx[(s, f, a)] - idx[(s, f, b)]
             for s in seeds for f in fids
             if (s, f, a) in idx and (s, f, b) in idx]
        m, lo, hi = bootstrap_ci(d)
        cis[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(d)}

    # Binding verdict: Kc_Vc must beat both Kc_Vr and Kr_Vc with CI>0.
    binding_pass = (cis["Kc_Vc_vs_Kc_Vr"]["ci_lo"] > 0
                    and cis["Kc_Vc_vs_Kr_Vc"]["ci_lo"] > 0)
    if binding_pass:
        verdict = "BINDING_PASS"
    elif (cis["Kc_Vc_vs_Kc_Vr"]["mean"] > 0
          and cis["Kc_Vc_vs_Kr_Vc"]["mean"] > 0):
        verdict = "BINDING_DIRECTIONAL"
    else:
        verdict = "BINDING_FAIL"

    analysis = {
        "experiment": "exp15_kv_binding",
        "n_seeds": len(seeds), "n_facts": len(fids),
        "mean_margin": means,
        "contrasts": cis,
        "verdict": verdict,
        "notes": ("Kc_Vc_vs_Kc_Vr captures V-causality given correct K; "
                  "Kc_Vc_vs_Kr_Vc captures K-causality given correct V. "
                  "BINDING_PASS requires both CIs > 0."),
    }
    (in_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (in_dir / "VERDICT.txt").write_text(verdict + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
