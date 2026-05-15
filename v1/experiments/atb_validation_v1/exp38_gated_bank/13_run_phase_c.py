"""Exp38 Phase C — async parallel verifications of the G2 direction.

These verify the G2 retrieval gating result is robust to:
  C1 — bank size scaling (N ∈ {1k, 3k, 10k}; new builds for 1k/3k)
  C2 — different model (Qwen2.5-0.5B with N=3k bank)  [DEFERRED — requires
        rebuilding bank on smaller model; documented but not auto-run here]
  C3 — relation subset (geography only sub-bank from existing 10k bank)
  C4 — random-bank baseline (already covered by AC10 in 11_run_ac_suite.py)

We can do C1 (sub-sampling) and C3 (relation filter) on the existing bank by
masking — no new bank build needed. C2 is much more expensive and is left
to a separate manual launch.

Usage: python3 13_run_phase_c.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP35B = HERE.parent / "exp35b_memit_bank"


def build_subbank(src: Path, dst: Path, fact_ids_keep: set):
    d = torch.load(src, map_location="cpu", weights_only=False)
    entries = d["entries"]
    kept = {fid: e for fid, e in entries.items() if fid in fact_ids_keep}
    d2 = {"entries": kept,
          "summary": dict(d.get("summary", {}),
                          subbank_n=len(kept), subbank_src=str(src))}
    torch.save(d2, dst)
    print(f"[subbank] {dst} — {len(kept)} entries")


def run(name, cmd, log):
    print(f"\n========== {name} ==========", flush=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"[{name}] rc={rc} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-c1", action="store_true")
    ap.add_argument("--skip-c3", action="store_true")
    args = ap.parse_args()

    logs = HERE / "logs"; logs.mkdir(exist_ok=True)
    rel = lambda p: str(p.relative_to(ROOT))
    PY = "python3"

    src_bank = EXP35B / "data" / "bank.pt"
    src_d = torch.load(src_bank, map_location="cpu", weights_only=False)
    all_ids = list(src_d["entries"].keys())
    print(f"[load] source bank: {len(all_ids)} entries")

    n_phi1 = 100 if args.fast else 1500
    seeds = "0" if args.fast else "0 1 2"
    n_probes = 20 if args.fast else 100
    n_sets = 10 if args.fast else 50

    eval_args = ["--n-phi1", str(n_phi1), "--phi1-seeds", *seeds.split(),
                 "--n-cross-probes", str(n_probes), "--n-cross-sets", str(n_sets),
                 "--skip", "neg", "hellaswag"]

    # ---- C1: bank size scaling ----
    if not args.skip_c1:
        rng = torch.Generator().manual_seed(38001)
        for n_bank in [1000, 3000]:  # 10000 already in Phase 1
            perm = torch.randperm(len(all_ids), generator=rng).tolist()
            keep = set(all_ids[i] for i in perm[:n_bank])
            sub_path = HERE / "data" / f"bank_n{n_bank}.pt"
            sub_path.parent.mkdir(parents=True, exist_ok=True)
            build_subbank(src_bank, sub_path, keep)
            for variant, k_r_arg in [("G0", []), ("G2", ["--k-r", "5"])]:
                tag = f"C1_{variant}_n{n_bank}" if variant == "G0" else f"C1_G2_kr5_n{n_bank}"
                run(f"C1 {tag}",
                    [PY, rel(HERE / "05_eval_variant.py"), "--variant", variant,
                     "--bank", rel(sub_path), *k_r_arg, *eval_args,
                     "--out", rel(HERE / "run_qwen_exp38" / tag)],
                    logs / f"{tag}.log")

    # ---- C3: relation subset (geography) ----
    if not args.skip_c3:
        # Pick a single relation that is large enough to give signal.
        # P190 (twin city), P17 (country), P30 (continent), P36 (capital), P276 (location)
        # are common geography relations in CounterFact.
        rel_target = "P17"  # country
        rel_keep = set()
        for fid, e in src_d["entries"].items():
            if e.get("relation") == rel_target:
                rel_keep.add(fid)
        print(f"[C3] relation={rel_target} → {len(rel_keep)} facts")
        if len(rel_keep) >= 200:
            sub_path = HERE / "data" / f"bank_rel_{rel_target}.pt"
            build_subbank(src_bank, sub_path, rel_keep)
            for variant, k_r_arg in [("G0", []), ("G2", ["--k-r", "5"])]:
                tag = f"C3_{variant}_{rel_target}" if variant == "G0" else f"C3_G2_kr5_{rel_target}"
                run(f"C3 {tag}",
                    [PY, rel(HERE / "05_eval_variant.py"), "--variant", variant,
                     "--bank", rel(sub_path), *k_r_arg, *eval_args,
                     "--out", rel(HERE / "run_qwen_exp38" / tag)],
                    logs / f"{tag}.log")
        else:
            print(f"[C3] skipped: only {len(rel_keep)} facts for {rel_target}")

    print("\n[Phase C done]")


if __name__ == "__main__":
    main()
