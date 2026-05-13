"""Exp23 analyzer — paired bootstrap CI on 6 PASS contrasts.

PASS if all 6 contrasts have 95% CI lower bound > 0:
  oracle_RS > base
  oracle_RS > old_full_bank
  oracle_RS > random_relationK_subjectV
  oracle_RS > relationK_randomV
  oracle_RS > randomK_subjectV
  oracle_RS > minus_correct
"""
import argparse, json
from pathlib import Path
import numpy as np

PAIRS = [
    ("oracle_relationK_subjectV", "base"),
    ("oracle_relationK_subjectV", "old_full_bank"),
    ("oracle_relationK_subjectV", "random_relationK_subjectV"),
    ("oracle_relationK_subjectV", "relationK_randomV"),
    ("oracle_relationK_subjectV", "randomK_subjectV"),
    ("oracle_relationK_subjectV", "minus_correct"),
]

def bootstrap_diff(a, b, n_boot=10000, seed=0):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_key = {}
    for line in open(args.cells):
        d = json.loads(line)
        key = (d["variant"], d["seed"], d["fact_id"])
        by_key[key] = d["margin"]

    variants = sorted({k[0] for k in by_key})
    pairs_ix = sorted({(k[1], k[2]) for k in by_key})

    def vec(v):
        return [by_key[(v, s, f)] for s, f in pairs_ix if (v, s, f) in by_key]

    summary = {"n_pairs": len(pairs_ix), "variants": {}}
    for v in variants:
        xs = vec(v)
        summary["variants"][v] = {
            "mean_margin": float(np.mean(xs)),
            "sem": float(np.std(xs, ddof=1) / len(xs) ** 0.5),
            "n": len(xs),
        }

    contrasts = []
    all_pass = True
    for hi_name, lo_name in PAIRS:
        a = vec(hi_name); b = vec(lo_name)
        m = min(len(a), len(b))
        diff_mean, ci_lo, ci_hi = bootstrap_diff(a[:m], b[:m])
        passes = ci_lo > 0
        if not passes: all_pass = False
        contrasts.append({
            "hi": hi_name, "lo": lo_name,
            "diff_mean": diff_mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "passes": passes,
        })

    if all_pass:
        verdict = "DUAL_ORACLE_PASS"
    elif all(c["diff_mean"] > 0 for c in contrasts):
        verdict = "DUAL_ORACLE_DIRECTIONAL"
    else:
        verdict = "DUAL_ORACLE_FAIL"

    summary["contrasts"] = contrasts
    summary["verdict"] = verdict
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(summary, indent=2))
    (out / "verdict.txt").write_text(verdict + "\n")
    print(f"[exp23] verdict = {verdict}")
    for c in contrasts:
        flag = "✓" if c["passes"] else "✗"
        print(f"  {flag} {c['hi']} - {c['lo']}: mean={c['diff_mean']:+.3f} CI=[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]")


if __name__ == "__main__":
    main()
