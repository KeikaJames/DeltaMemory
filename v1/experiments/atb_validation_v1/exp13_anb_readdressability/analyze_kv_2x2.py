"""Generic 2x2 K/V binding analyzer (reusable for Exp15/Exp19/Exp22)."""
from __future__ import annotations
import argparse, json, random, statistics
from pathlib import Path

B = 2000
VARIANTS = ("base", "Kc_Vc", "Kc_Vr", "Kr_Vc", "Kr_Vr")


def boot(d, B=B, alpha=0.05, seed=12345):
    rng = random.Random(seed); n = len(d)
    if n == 0: return (float("nan"),)*3
    means = [sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(B)]
    means.sort()
    return (statistics.fmean(d), means[int(alpha/2*B)], means[int((1-alpha/2)*B)-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--tag", default="exp")
    a = ap.parse_args()
    in_dir = Path(a.in_dir)
    rows = [json.loads(l) for l in (in_dir/"cells.jsonl").read_text().splitlines() if l.strip()]
    idx = {(r["seed"], r["fact_id"], r["variant"]): float(r["margin"]) for r in rows if "margin" in r}
    seeds = sorted({r["seed"] for r in rows})
    fids = sorted({r["fact_id"] for r in rows})

    means = {}
    for v in VARIANTS:
        ms = [idx[(s,f,v)] for s in seeds for f in fids if (s,f,v) in idx]
        means[v] = statistics.fmean(ms) if ms else float("nan")
    pairs = {
        "Kc_Vc_vs_base":   ("Kc_Vc", "base"),
        "V_causality":     ("Kc_Vc", "Kc_Vr"),
        "K_causality":     ("Kc_Vc", "Kr_Vc"),
        "Kc_Vc_vs_Kr_Vr":  ("Kc_Vc", "Kr_Vr"),
    }
    cis = {}
    for nm, (xa, xb) in pairs.items():
        d = [idx[(s,f,xa)] - idx[(s,f,xb)] for s in seeds for f in fids if (s,f,xa) in idx and (s,f,xb) in idx]
        m, lo, hi = boot(d)
        cis[nm] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(d)}

    Vpass = cis["V_causality"]["ci_lo"] > 0
    Kpass = cis["K_causality"]["ci_lo"] > 0
    Vdir = cis["V_causality"]["mean"] > 0
    Kdir = cis["K_causality"]["mean"] > 0
    if Vpass and Kpass: verdict = "BINDING_PASS"
    elif Vpass or Kpass: verdict = "BINDING_DIRECTIONAL"
    elif Vdir or Kdir: verdict = "BINDING_WEAK_DIRECTIONAL"
    else: verdict = "BINDING_FAIL"

    out = {"experiment": a.tag, "n_seeds": len(seeds), "n_facts": len(fids),
           "mean_margin": means, "contrasts": cis, "verdict": verdict}
    (in_dir / "analysis.json").write_text(json.dumps(out, indent=2))
    (in_dir / "VERDICT.txt").write_text(verdict + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
