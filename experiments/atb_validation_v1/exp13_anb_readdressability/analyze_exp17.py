"""Exp17 analyzer — per-site K/V binding verdicts."""
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
        if "margin" not in r:
            continue
        idx[(r["site"], r["seed"], r["fact_id"], r["variant"])] = float(r["margin"])
    sites = sorted({r["site"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    fids = sorted({r["fact_id"] for r in rows})

    per_site = {}
    site_pass = []
    site_dir = []
    for site in sites:
        means = {}
        for v in ("base",) + CELLS:
            ms = [idx[(site, s, f, v)] for s in seeds for f in fids if (site, s, f, v) in idx]
            means[v] = statistics.fmean(ms) if ms else float("nan")

        contrasts = {
            "K_causality_Kc_Vc_minus_Kr_Vc": ("Kc_Vc", "Kr_Vc"),
            "V_causality_Kc_Vc_minus_Kc_Vr": ("Kc_Vc", "Kc_Vr"),
            "Joint_Kc_Vc_minus_Kr_Vr":      ("Kc_Vc", "Kr_Vr"),
        }
        cis = {}
        for name, (a, b) in contrasts.items():
            d = [idx[(site, s, f, a)] - idx[(site, s, f, b)]
                 for s in seeds for f in fids
                 if (site, s, f, a) in idx and (site, s, f, b) in idx]
            m, lo, hi = boot(d)
            cis[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(d)}

        K_pass = cis["K_causality_Kc_Vc_minus_Kr_Vc"]["ci_lo"] > 0
        V_pass = cis["V_causality_Kc_Vc_minus_Kc_Vr"]["ci_lo"] > 0
        if K_pass and V_pass:
            verdict = "BINDING_PASS"
            site_pass.append(site)
        elif K_pass or V_pass:
            verdict = "BINDING_DIRECTIONAL"
            site_dir.append(site)
        else:
            verdict = "BINDING_FAIL"
        per_site[site] = {
            "mean_margin": means,
            "contrasts": cis,
            "verdict": verdict,
        }

    overall = "SWEEP_PASS" if site_pass else ("SWEEP_DIRECTIONAL" if site_dir else "SWEEP_FAIL")
    analysis = {
        "experiment": "exp17_capture_sweep",
        "n_seeds": len(seeds), "n_facts": len(fids), "sites": sites,
        "sites_BINDING_PASS": site_pass,
        "sites_BINDING_DIRECTIONAL": site_dir,
        "per_site": per_site,
        "verdict": overall,
        "notes": ("SWEEP_PASS if any site reaches BINDING_PASS (both K & V CIs > 0). "
                  "Per-site verdict matches Exp15 ladder."),
    }
    (in_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (in_dir / "VERDICT.txt").write_text(overall + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
