"""Exp38 — full pipeline orchestrator.

Runs the complete cascade serially on one MPS device:
  1. G0 baseline eval
  2. G1 theta calibration → G1 eval
  3. G2 retrieval eval (sweep k_r ∈ {1,5,20,100})
  4. G3 training → G3 eval
  5. G4 training → G4 eval
  6. G5 eval (reuses G3 heads)
  7. Anti-cheat audits
  8. Verdict

Usage: python 99_run_all.py [--fast]    # --fast = small-scale smoke
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def step(name, cmd, log):
    print(f"\n========== {name} ==========", flush=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    print(f"log: {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE.parents[3]))
    print(f"[{name}] rc={rc} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    if rc != 0:
        # tail log
        try:
            print("--- tail log ---")
            print(open(log).read()[-2000:])
        except Exception:
            pass
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="quick smoke (50 facts each)")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["G0", "G1", "G2", "G3", "G4", "G5", "audits"])
    args = ap.parse_args()

    logs = HERE / "logs"; logs.mkdir(exist_ok=True)
    PY = "python3"
    rel = lambda p: str(p.relative_to(HERE.parents[3]))

    n_phi1 = 50 if args.fast else 1500
    seeds = "0" if args.fast else "0 1 2"
    n_neg = 50 if args.fast else 600
    n_probes = 20 if args.fast else 100
    n_sets = 10 if args.fast else 50
    n_hella = 100 if args.fast else 1000

    if "G0" not in args.skip:
        step("G0 baseline",
             [PY, rel(HERE / "05_eval_variant.py"), "--variant", "G0",
              "--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
              "--n-neg-facts", str(n_neg),
              "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
              "--n-hellaswag", str(n_hella)],
             logs / "G0.log")

    if "G1" not in args.skip:
        step("G1 calibrate theta",
             [PY, rel(HERE / "02_calibrate_g1.py"),
              "--n-negatives", "300" if not args.fast else "100"],
             logs / "G1_calibrate.log")
        step("G1 eval",
             [PY, rel(HERE / "05_eval_variant.py"), "--variant", "G1",
              "--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
              "--n-neg-facts", str(n_neg),
              "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
              "--n-hellaswag", str(n_hella)],
             logs / "G1_eval.log")

    if "G2" not in args.skip:
        for k_r in ([5] if args.fast else [1, 5, 20, 100]):
            step(f"G2 eval k_r={k_r}",
                 [PY, rel(HERE / "05_eval_variant.py"), "--variant", "G2",
                  "--k-r", str(k_r),
                  "--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
                  "--n-neg-facts", str(n_neg),
                  "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
                  "--n-hellaswag", str(n_hella)],
                 logs / f"G2_kr{k_r}.log")

    if "G3" not in args.skip:
        step("G3 train heads",
             [PY, rel(HERE / "04_train_g3_g4.py"), "--variant", "G3",
              "--n-facts", "500" if args.fast else "2000",
              "--n-neg-per-fact", "30" if args.fast else "50",
              "--steps", "100" if args.fast else "200"],
             logs / "G3_train.log")
        step("G3 eval",
             [PY, rel(HERE / "05_eval_variant.py"), "--variant", "G3",
              "--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
              "--n-neg-facts", str(n_neg),
              "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
              "--n-hellaswag", str(n_hella)],
             logs / "G3_eval.log")

    if "G4" not in args.skip:
        step("G4 train heads",
             [PY, rel(HERE / "04_train_g3_g4.py"), "--variant", "G4",
              "--n-facts", "500" if args.fast else "2000",
              "--n-neg-per-fact", "30" if args.fast else "50",
              "--steps", "100" if args.fast else "200"],
             logs / "G4_train.log")
        step("G4 eval",
             [PY, rel(HERE / "05_eval_variant.py"), "--variant", "G4",
              "--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
              "--n-neg-facts", str(n_neg),
              "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
              "--n-hellaswag", str(n_hella)],
             logs / "G4_eval.log")

    if "G5" not in args.skip:
        step("G5 eval (G3 heads × top-5)",
             [PY, rel(HERE / "05_eval_variant.py"), "--variant", "G5",
              "--k-r", "5",
              "--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
              "--n-neg-facts", str(n_neg),
              "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
              "--n-hellaswag", str(n_hella)],
             logs / "G5_eval.log")

    if "audits" not in args.skip:
        step("Anti-cheat audits",
             [PY, rel(HERE / "07_anti_cheat.py")],
             logs / "audits.log")

    step("Verdict",
         [PY, rel(HERE / "08_verdict.py")],
         logs / "verdict.log")
    print("\n[all done]")


if __name__ == "__main__":
    main()
