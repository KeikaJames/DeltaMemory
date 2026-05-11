"""Exp25 analyzer — K-routing gap stabilization.

Computes per-α:
  - Mean margin per variant.
  - Paired bootstrap CI for the K-routing primary contrasts.
  - Retrieval accuracy per variant.
  - PASS gates A/B/C/D.

Gates:
  A. topk1 − minus_correct       CI lower bound > +0.10 at ≥2 α values
  B. retrieval_accuracy(topk1)   > 1/N_bank + 3·SE_binomial
  C. topk1 − meanV               CI lower bound > 0       (K-routing > V-bias-only)
  D. topk1 − shuffled_factids    CI lower bound > 0       (K↔V identity matters)
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

PRIMARY_CONTRASTS = [
    ("full_bank_topk1", "full_bank_topk1_minus_correct", "A"),
    ("full_bank_topk1", "full_bank_topk1_meanV",         "C"),
    ("full_bank_topk1", "full_bank_topk1_shuffled_factids", "D"),
]

def bootstrap_diff(a, b, n_boot=10000, seed=0):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = a - b
    if len(d) == 0:
        return 0.0, 0.0, 0.0
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

    # Load all rows.
    rows = []
    for line in open(args.cells):
        if line.strip():
            rows.append(json.loads(line))

    alphas = sorted({float(r["alpha"]) for r in rows if r.get("variant") != "base"})
    variants = sorted({r["variant"] for r in rows})
    bank_size = max(int(r.get("bank_size", 0) or 0) for r in rows) or 100

    # Index by (variant, alpha, seed, fact_id) → row.
    by_key = {}
    for r in rows:
        a = float(r.get("alpha", 0.0))
        by_key[(r["variant"], a, r["seed"], r["fact_id"])] = r

    # Per-alpha analysis.
    out_summary = {"n_total_cells": len(rows),
                   "alphas": alphas, "bank_size": bank_size,
                   "per_alpha": {}}

    base_keys = sorted({(r["seed"], r["fact_id"]) for r in rows if r["variant"] == "base"})

    for a in alphas:
        keys = sorted({(s, f) for (v, aa, s, f) in by_key if aa == a})
        per_alpha = {"variants": {}, "contrasts": [], "n_pairs": len(keys)}
        for v in variants:
            margins, rets = [], []
            mass, prob, gap = [], [], []
            for s, f in keys:
                k = (v, a, s, f) if v != "base" else ("base", 0.0, s, f)
                r = by_key.get(k)
                if r is None:
                    continue
                margins.append(r["margin"])
                rc = r.get("retrieval_correct")
                if rc is not None and rc != -1:
                    rets.append(int(rc))
                if r.get("bank_attention_mass") is not None:
                    mass.append(r["bank_attention_mass"])
                    prob.append(r.get("max_bank_prob", 0))
                    gap.append(r.get("top1_top2_gap", 0))
            if not margins:
                continue
            entry = {
                "n": len(margins),
                "mean_margin": float(np.mean(margins)),
                "sem_margin": float(np.std(margins, ddof=1) / math.sqrt(len(margins))) if len(margins) > 1 else 0.0,
            }
            if rets:
                p = float(np.mean(rets))
                se = math.sqrt(p * (1 - p) / len(rets)) if len(rets) > 1 else 0.0
                chance = 1.0 / max(bank_size, 1)
                entry.update({
                    "retrieval_accuracy": p,
                    "retrieval_se": se,
                    "chance": chance,
                    "accuracy_above_chance_3se": p > chance + 3 * se,
                    "n_retrieval": len(rets),
                })
            if mass:
                entry["mean_bank_mass"] = float(np.mean(mass))
                entry["mean_max_bank_prob"] = float(np.mean(prob))
                entry["mean_top1_top2_gap"] = float(np.mean(gap))
            per_alpha["variants"][v] = entry

        # Contrasts.
        for hi, lo, gate in PRIMARY_CONTRASTS:
            a_vec, b_vec = [], []
            for s, f in keys:
                rh = by_key.get((hi, a, s, f))
                rl = by_key.get((lo, a, s, f))
                if rh is None or rl is None: continue
                a_vec.append(rh["margin"]); b_vec.append(rl["margin"])
            if not a_vec:
                continue
            d, lo_ci, hi_ci = bootstrap_diff(a_vec, b_vec)
            per_alpha["contrasts"].append({
                "gate": gate, "hi": hi, "lo": lo, "n": len(a_vec),
                "diff_mean": d, "ci_lo": lo_ci, "ci_hi": hi_ci,
                "passes": lo_ci > (0.10 if gate == "A" else 0.0),
            })
        out_summary["per_alpha"][f"{a:.4f}"] = per_alpha

    # Aggregate verdict.
    gate_A_pass_count = 0   # strict (> +0.10)
    gate_A_ci_pos_count = 0  # CI lower bound > 0
    gate_C_any = False
    gate_D_any = False
    gate_B_any = False
    for ka, pa in out_summary["per_alpha"].items():
        for c in pa["contrasts"]:
            if c["gate"] == "A":
                if c["passes"]: gate_A_pass_count += 1
                if c["ci_lo"] > 0: gate_A_ci_pos_count += 1
            if c["gate"] == "C" and c["ci_lo"] > 0: gate_C_any = True
            if c["gate"] == "D" and c["ci_lo"] > 0: gate_D_any = True
        topk1 = pa["variants"].get("full_bank_topk1", {})
        if topk1.get("accuracy_above_chance_3se"):
            gate_B_any = True

    if gate_A_pass_count >= 2 and gate_B_any and gate_C_any and gate_D_any:
        verdict = "EXP25_PASS_STRONG"
    elif gate_A_pass_count >= 2 and gate_B_any:
        verdict = "EXP25_PASS_K_ROUTING_STRONG"
    elif gate_A_ci_pos_count >= 2:
        verdict = "EXP25_DIRECTIONAL_K_ROUTING"   # Exp24 replication only
    elif gate_A_ci_pos_count >= 1 or gate_B_any:
        verdict = "EXP25_DIRECTIONAL"
    else:
        verdict = "EXP25_FAIL"

    out_summary["gates"] = {
        "A_strict_passes_count": gate_A_pass_count,
        "A_ci_pos_count": gate_A_ci_pos_count,
        "B_any_alpha": gate_B_any,
        "C_any_alpha": gate_C_any,
        "D_any_alpha": gate_D_any,
    }
    out_summary["verdict"] = verdict

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(out_summary, indent=2))
    (out / "verdict.txt").write_text(verdict + "\n")

    # Print summary.
    print(f"[exp25] verdict = {verdict}")
    print(f"  gates: A_passes_count={gate_A_pass_count}/{len(alphas)}  B={gate_B_any}  C={gate_C_any}  D={gate_D_any}")
    print(f"  bank_size={bank_size}")
    print()
    for ka, pa in out_summary["per_alpha"].items():
        print(f"--- α = {ka} (n_pairs={pa['n_pairs']}) ---")
        tk1 = pa["variants"].get("full_bank_topk1", {})
        if "retrieval_accuracy" in tk1:
            print(f"  topk1 retrieval_accuracy = {tk1['retrieval_accuracy']:.3f} "
                  f"(chance {tk1['chance']:.3f}, SE {tk1['retrieval_se']:.3f})  "
                  f"bank_mass={tk1.get('mean_bank_mass',0):.3f}")
        for c in pa["contrasts"]:
            flag = "✓" if c["passes"] else "✗"
            print(f"  {flag} [{c['gate']}] {c['hi']} − {c['lo']}: "
                  f"diff={c['diff_mean']:+.3f}  CI=[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]  n={c['n']}")
        print()


if __name__ == "__main__":
    main()
