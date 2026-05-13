"""Exp20 analyzer — K-causality as a function of bank size k."""
from __future__ import annotations
import argparse, json, random, statistics
from pathlib import Path

B = 2000


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
    idx = {(r["seed"], r["fact_id"], r["k"], r["variant"]): float(r["margin"]) for r in rows if "margin" in r}
    seeds = sorted({r["seed"] for r in rows})
    fids = sorted({r["fact_id"] for r in rows})
    ks = sorted({r["k"] for r in rows if r["k"] > 0})

    per_k = {}
    K_dir_count = 0
    K_pass_count = 0
    for k in ks:
        means_Kc = [idx[(s,f,k,"Kc_Vc")] for s in seeds for f in fids if (s,f,k,"Kc_Vc") in idx]
        means_Kr = [idx[(s,f,k,"Kr_Vc")] for s in seeds for f in fids if (s,f,k,"Kr_Vc") in idx]
        means_base = [idx[(s,f,0,"base")] for s in seeds for f in fids if (s,f,0,"base") in idx]

        d_vs_base = [idx[(s,f,k,"Kc_Vc")] - idx[(s,f,0,"base")]
                     for s in seeds for f in fids
                     if (s,f,k,"Kc_Vc") in idx and (s,f,0,"base") in idx]
        d_K = [idx[(s,f,k,"Kc_Vc")] - idx[(s,f,k,"Kr_Vc")]
               for s in seeds for f in fids
               if (s,f,k,"Kc_Vc") in idx and (s,f,k,"Kr_Vc") in idx]

        m_vb, lo_vb, hi_vb = boot(d_vs_base)
        m_K, lo_K, hi_K = boot(d_K)
        per_k[k] = {
            "mean_Kc_Vc": statistics.fmean(means_Kc) if means_Kc else float("nan"),
            "mean_Kr_Vc": statistics.fmean(means_Kr) if means_Kr else float("nan"),
            "Kc_Vc_vs_base": {"mean": m_vb, "ci_lo": lo_vb, "ci_hi": hi_vb, "n": len(d_vs_base)},
            "K_causality":   {"mean": m_K,  "ci_lo": lo_K,  "ci_hi": hi_K,  "n": len(d_K)},
        }
        if lo_K > 0: K_pass_count += 1
        if m_K > 0:  K_dir_count += 1

    if K_pass_count == len(ks): verdict = "SCALE_K_PASS"
    elif K_pass_count >= 1:     verdict = "SCALE_K_PARTIAL"
    elif K_dir_count >= 1:      verdict = "SCALE_K_DIRECTIONAL"
    else:                       verdict = "SCALE_K_FAIL"

    base_mean = statistics.fmean([idx[(s,f,0,"base")] for s in seeds for f in fids if (s,f,0,"base") in idx])
    out = {"experiment": "exp20_bank_size_scaling",
           "n_seeds": len(seeds), "n_facts": len(fids),
           "base_mean_margin": base_mean,
           "per_k": per_k, "verdict": verdict}
    (in_dir / "analysis.json").write_text(json.dumps(out, indent=2))
    (in_dir / "VERDICT.txt").write_text(verdict + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
