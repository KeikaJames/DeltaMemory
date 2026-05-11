"""Exp23 alpha-sweep: test whether oracle wins at any alpha.

Restricts to the 4 critical variants for cheap MPS sweep:
  base, old_full_bank, oracle_relationK_subjectV, minus_correct
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import torch

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank
from deltamemory.memory.anb_dual_site import write_fact_dual_site
from deltamemory.memory.anb_addressed import subbank_select

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (load_model, load_counterfact,
    filter_cf_for_tokenizer, seed_everything, continuation_logp, first_token_rank)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt
from experiments.atb_validation_v1.exp13_anb_readdressability.run_dual_oracle import (
    build_dual_bank, subbank_correct, subbank_minus_correct, eval_with_bank)
from tools.env_writer import write_env_json, sha1_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact", default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--alphas", default="0.0005,0.001,0.005,0.02")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    cf = load_counterfact(Path(args.counterfact))
    kept_rows, _ = filter_cf_for_tokenizer(cf, tok)
    rows = []
    for r in kept_rows:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None: continue
        r = dict(r); r["write_prompt"] = wp
        rows.append(r)
    rows = rows[: args.n]
    write_env_json(out_dir, prereg_version="exp23.alpha.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"experiment": "exp23_alpha_sweep", "n": len(rows),
                          "alphas": alphas})

    patcher = AttnNativePatcher(model); patcher.install()
    fout = (out_dir / "cells.jsonl").open("a")
    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        b_RS, kept = build_dual_bank(patcher, tok, rows, "relation_last", "subject_last")
        kept_set = set(kept)
        print(f"[exp23a] seed={seed} kept {len(kept_set)}/{len(rows)}", flush=True)
        for row in rows:
            fid = str(row["id"])
            if fid not in kept_set: continue
            tn = row["target_new"]; tt = row.get("target_true") or ""
            if not tt: continue
            q = render_query(row)
            for a in alphas:
                cells = {
                    "base": (None, 0.0),
                    "old_full_bank": (b_RS, a),
                    "oracle_relationK_subjectV": (subbank_correct(b_RS, fid), a),
                    "minus_correct": (subbank_minus_correct(b_RS, fid), a),
                }
                for v, (bnk, aa) in cells.items():
                    m = eval_with_bank(model, tok, patcher, q, tn, tt, args.device, bnk, aa)
                    fout.write(json.dumps({"seed": seed, "fact_id": fid,
                                           "alpha": a, "variant": v, **m}) + "\n")
                gc.collect()
            fout.flush()
        del b_RS; gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    fout.close(); patcher.remove()
    print(f"[exp23a] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
