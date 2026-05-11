"""Exp16 analyzer — layer-quartile site map of K/V causality."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

CELLS = ("Kc_Vc", "Kc_Vr", "Kr_Vc", "Kr_Vr")
B_BOOT = 2000


def boot(d, B=B_BOOT, alpha=0.05, seed=12345):
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
        key = (r["seed"], r["fact_id"], r.get("quartile", "none"), r["variant"])
        idx[key] = float(r["margin"])
    seeds = sorted({r["seed"] for r in rows})
    fids = sorted({r["fact_id"] for r in rows})
    quartiles = sorted({r.get("quartile", "none") for r in rows
                        if r.get("quartile") and r["quartile"] != "none"})

    per_q = {}
    for q in quartiles:
        means = {}
        for v in CELLS:
            ms = [idx[(s, f, q, v)] for s in seeds for f in fids if (s, f, q, v) in idx]
            means[v] = statistics.fmean(ms) if ms else float("nan")
        # Key K-causality contrast: Kc_Vc - Kr_Vc (does correct K add when V correct?)
        d_K = [idx[(s, f, q, "Kc_Vc")] - idx[(s, f, q, "Kr_Vc")]
               for s in seeds for f in fids
               if (s, f, q, "Kc_Vc") in idx and (s, f, q, "Kr_Vc") in idx]
        # V-causality contrast: Kc_Vc - Kc_Vr.
        d_V = [idx[(s, f, q, "Kc_Vc")] - idx[(s, f, q, "Kc_Vr")]
               for s in seeds for f in fids
               if (s, f, q, "Kc_Vc") in idx and (s, f, q, "Kc_Vr") in idx]
        # Joint vs baseline noise.
        d_J = [idx[(s, f, q, "Kc_Vc")] - idx[(s, f, q, "Kr_Vr")]
               for s in seeds for f in fids
               if (s, f, q, "Kc_Vc") in idx and (s, f, q, "Kr_Vr") in idx]
        K_m, K_lo, K_hi = boot(d_K)
        V_m, V_lo, V_hi = boot(d_V)
        J_m, J_lo, J_hi = boot(d_J)
        per_q[q] = {
            "mean_margin": means,
            "n": len(d_K),
            "K_causality_Kc_Vc_minus_Kr_Vc": {"mean": K_m, "ci_lo": K_lo, "ci_hi": K_hi},
            "V_causality_Kc_Vc_minus_Kc_Vr": {"mean": V_m, "ci_lo": V_lo, "ci_hi": V_hi},
            "joint_Kc_Vc_minus_Kr_Vr":      {"mean": J_m, "ci_lo": J_lo, "ci_hi": J_hi},
        }

    # Locate best K-causality quartile (highest CI_lo).
    by_K = sorted(quartiles, key=lambda q: per_q[q]["K_causality_Kc_Vc_minus_Kr_Vc"]["ci_lo"], reverse=True)
    best_K_q = by_K[0]
    by_V = sorted(quartiles, key=lambda q: per_q[q]["V_causality_Kc_Vc_minus_Kc_Vr"]["ci_lo"], reverse=True)
    best_V_q = by_V[0]

    K_pass = per_q[best_K_q]["K_causality_Kc_Vc_minus_Kr_Vc"]["ci_lo"] > 0
    V_pass = per_q[best_V_q]["V_causality_Kc_Vc_minus_Kc_Vr"]["ci_lo"] > 0
    if K_pass and V_pass:
        verdict = "SITE_PASS"
    elif K_pass or V_pass:
        verdict = "SITE_DIRECTIONAL"
    else:
        verdict = "SITE_FAIL"

    analysis = {
        "experiment": "exp16_site_map",
        "n_seeds": len(seeds), "n_facts": len(fids),
        "per_quartile": per_q,
        "best_K_quartile": best_K_q,
        "best_V_quartile": best_V_q,
        "verdict": verdict,
        "notes": ("SITE_PASS requires both K-causality and V-causality CIs > 0 "
                  "in at least one quartile each. SITE_DIRECTIONAL if only one."),
    }
    (in_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (in_dir / "VERDICT.txt").write_text(verdict + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
