"""Exp38 Phase B — G2 ablation matrix.

Runs after Phase A. Three sub-studies:

B1 — fine k_r sweep on G2: k_r ∈ {2, 3, 7, 10, 50} (in addition to 1,5,20,100
     already in Phase 1) → 5 runs

B2 — score variant ablation:
     G2cos k_r=5 (cosine instead of raw dot)
     G2l2  k_r=5 (L2-normalize x before scoring)
     → 2 runs

Each run: Φ1 + 37.C only (these don't involve negation, the failure mode is
constant across panels). HellaSwag/negation skipped to save time.

Total ≈ 7 runs × 25min ≈ 3h.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def run(name, cmd, log):
    print(f"\n========== {name} ==========", flush=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"[{name}] rc={rc} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    if rc != 0:
        try:
            print(open(log).read()[-1500:])
        except Exception:
            pass
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    logs = HERE / "logs"; logs.mkdir(exist_ok=True)
    rel = lambda p: str(p.relative_to(ROOT))
    PY = "python3"

    n_phi1 = 100 if args.fast else 1500
    seeds = "0" if args.fast else "0 1 2"
    n_probes = 20 if args.fast else 100
    n_sets = 10 if args.fast else 50

    common = ["--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
              "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
              "--skip", "neg", "hellaswag"]

    # B1 — fine k_r sweep
    for k_r in [2, 3, 7, 10, 50]:
        run(f"B1 G2 k_r={k_r}",
            [PY, rel(HERE / "05_eval_variant.py"), "--variant", "G2",
             "--k-r", str(k_r), *common],
            logs / f"B1_G2_kr{k_r}.log")

    # B2 — score variants at k_r=5
    for variant in ["G2cos", "G2l2"]:
        run(f"B2 {variant} k_r=5",
            [PY, rel(HERE / "05_eval_variant.py"), "--variant", variant,
             "--k-r", "5", *common],
            logs / f"B2_{variant}.log")

    print("\n[Phase B done]")


if __name__ == "__main__":
    main()
