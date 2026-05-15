"""Exp38 — run extended anti-cheat suite (AC9, AC10) after main pipeline.

Sequential MPS jobs:
  1. AC10 build random-vector control bank
  2. AC10 eval Φ1+37.C with G0 and G2 k_r=5 gates on the random bank
  3. AC9 train G3-shuffled head (same hyperparams + --shuffle-labels-sanity)
  4. AC9 eval G3-shuffled on Φ1 (lightweight)
  5. Aggregate with 10_ac_audits_extended.py
  6. Also run original 07_anti_cheat.py for AC1-AC8 view
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
    print(f"log: {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"[{name}] rc={rc} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    if rc != 0:
        try:
            print("--- tail log ---")
            print(open(log).read()[-2000:])
        except Exception:
            pass
        return rc
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-ac10", action="store_true")
    ap.add_argument("--skip-ac9", action="store_true")
    args = ap.parse_args()

    logs = HERE / "logs"; logs.mkdir(exist_ok=True)
    rel = lambda p: str(p.relative_to(ROOT))
    PY = "python3"

    n_phi1 = 100 if args.fast else 800       # smaller than main; just need signal
    seeds = "0" if args.fast else "0 1"
    n_neg = 50 if args.fast else 400
    n_probes = 20 if args.fast else 60
    n_sets = 10 if args.fast else 30

    rand_bank = HERE / "data" / "bank_random.pt"

    if not args.skip_ac10:
        run("AC10 build random bank",
            [PY, rel(HERE / "09_ac10_build_random_bank.py")],
            logs / "AC10_build.log")

        for variant, extra in [("G0", []), ("G2", ["--k-r", "5"])]:
            tag = f"{variant}_random" if variant == "G0" else "G2_kr5_random"
            run(f"AC10 eval {tag}",
                [PY, rel(HERE / "05_eval_variant.py"), "--variant", variant,
                 "--bank", rel(rand_bank),
                 *extra,
                 "--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
                 "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
                 "--skip", "neg", "hellaswag",
                 "--out", rel(HERE / "run_qwen_exp38" / tag)],
                logs / f"AC10_{tag}.log")

    if not args.skip_ac9:
        run("AC9 train shuffled-labels G3",
            [PY, rel(HERE / "04_train_g3_g4.py"), "--variant", "G3",
             "--n-facts", "500" if args.fast else "1000",
             "--n-neg-per-fact", "30" if args.fast else "50",
             "--steps", "100" if args.fast else "200",
             "--shuffle-labels-sanity",
             "--out", rel(HERE / "data" / "G3_shuffled_heads.pt")],
            logs / "AC9_train.log")

    run("AC1-AC8 audits",
        [PY, rel(HERE / "07_anti_cheat.py")],
        logs / "audits_basic.log")

    run("AC9-AC14 aggregate",
        [PY, rel(HERE / "10_ac_audits_extended.py")],
        logs / "audits_extended.log")


if __name__ == "__main__":
    main()
