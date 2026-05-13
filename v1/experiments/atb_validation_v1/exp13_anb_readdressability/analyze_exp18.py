"""Exp18 analyzer — natural addressed ANB."""
from __future__ import annotations

import argparse, json, random, statistics
from pathlib import Path

B = 2000
VARIANTS = ("base", "full_bank", "full_bank_shuffled_L",
            "full_bank_random_K", "minus_correct_bank")


def boot(d, B=B, alpha=0.05, seed=12345):
    rng = random.Random(seed); n = len(d)
    if n == 0: return (float("nan"),)*3
    means = [sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(B)]
    means.sort()
    return (statistics.fmean(d), means[int(alpha/2*B)], means[int((1-alpha/2)*B)-1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in_dir", required=True)
    a = ap.parse_args()
    in_dir = Path(a.in_dir)
    rows = [json.loads(l) for l in (in_dir/"cells.jsonl").read_text().splitlines() if l.strip()]
    idx = {(r["site"], r["seed"], r["fact_id"], r["variant"]): float(r["margin"])
           for r in rows if "margin" in r}
    sites = sorted({r["site"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    fids = sorted({r["fact_id"] for r in rows})

    per = {}
    overall_pass, overall_dir = [], []
    for site in sites:
        means = {}
        for v in VARIANTS:
            ms = [idx[(site,s,f,v)] for s in seeds for f in fids if (site,s,f,v) in idx]
            means[v] = statistics.fmean(ms) if ms else float("nan")
        contrasts = {
            "full_vs_base":         ("full_bank", "base"),
            "full_vs_shuffled_L":   ("full_bank", "full_bank_shuffled_L"),
            "full_vs_random_K":     ("full_bank", "full_bank_random_K"),
            "full_vs_minus_correct":("full_bank", "minus_correct_bank"),
        }
        cis = {}
        for nm, (xa, xb) in contrasts.items():
            d = [idx[(site,s,f,xa)] - idx[(site,s,f,xb)]
                 for s in seeds for f in fids
                 if (site,s,f,xa) in idx and (site,s,f,xb) in idx]
            m, lo, hi = boot(d)
            cis[nm] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(d)}

        # Natural-addressing test: full vs random_K AND full vs minus_correct
        # both CI > 0 means the correct slot IS getting routed.
        natural_pass = (cis["full_vs_random_K"]["ci_lo"] > 0
                        and cis["full_vs_minus_correct"]["ci_lo"] > 0)
        natural_dir = (cis["full_vs_random_K"]["mean"] > 0
                       and cis["full_vs_minus_correct"]["mean"] > 0)
        verdict = ("NATURAL_PASS" if natural_pass
                   else ("NATURAL_DIRECTIONAL" if natural_dir else "NATURAL_FAIL"))
        per[site] = {"mean_margin": means, "contrasts": cis, "verdict": verdict}
        if natural_pass: overall_pass.append(site)
        elif natural_dir: overall_dir.append(site)

    overall = ("PASS" if overall_pass else ("DIRECTIONAL" if overall_dir else "FAIL"))
    analysis = {
        "experiment": "exp18_natural_addressed",
        "n_seeds": len(seeds), "n_facts": len(fids),
        "per_site": per,
        "sites_PASS": overall_pass,
        "sites_DIRECTIONAL": overall_dir,
        "verdict": "NATURAL_" + overall,
    }
    (in_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (in_dir / "VERDICT.txt").write_text(analysis["verdict"] + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
