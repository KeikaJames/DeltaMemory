"""Exp31 Φ2/Φ3 — Model-attached evaluation of trained K-adapter.

For each test fact + alpha + variant, compute target_new vs target_true
log-prob margin with the bank attached. The five-gate panel is computed
offline from cells.jsonl.

Variants
--------
  base                          no bank
  full_bank_no_adapter          identity projector (Exp24 baseline)
  full_bank_learned_adapter     trained K-adapter   <- MAIN
  full_bank_topk1_no_adapter    identity + topk=1
  full_bank_topk1_learned       trained + topk=1    <- MAIN-T1
  minus_correct                 bank minus correct fact
  meanV                         bank with V replaced by mean
  shuffled_factids              K/V identity scrambled
  shuffled_adapter              SAME projector but trained on *shuffled* fact-ids
                                (Gate E: rules out fact-content leakage)

Usage
-----
  python3 .../eval_k_adapter.py \\
      --model Qwen/Qwen3-4B-Instruct-2507 \\
      --projector run_mps_exp31_qwen_smoke/seed0/k_projector_seed0.pt \\
      --projector-shuffled run_mps_exp31_qwen_smoke/seed0_shuffled/k_projector_seed0.pt \\
      --bank-size 200 --n 100 \\
      --out run_mps_exp31_qwen_smoke/eval_seed0 \\
      --alphas 0.005,0.01,0.02 --seeds 0
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank, write_fact  # noqa: E402
from deltamemory.memory.anb_dual_site_multi import write_fact_dual_site_multi_v  # noqa: E402
from deltamemory.memory.anb_addressed import subbank_select, subbank_swap_KV  # noqa: E402
from deltamemory.memory.k_projector import KProjectorBank  # noqa: E402

from atb_validation_v1._lib import (  # noqa: E402
    load_model, seed_everything, continuation_logp, first_token_rank,
)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt  # noqa: E402

from experiments.atb_validation_v1.exp13_anb_readdressability.exp25_metrics import subbank_meanV  # noqa: E402
from experiments.atb_validation_v1.exp13_anb_readdressability.run_dual_oracle import (  # noqa: E402
    resolve_pos, subbank_minus_correct, subbank_shuffle_fact_ids,
)
from experiments.atb_validation_v1.exp13_anb_readdressability.run_exp26b_multi_v import (  # noqa: E402
    resolve_v_span,
)


SPLITS_DIR = Path(__file__).parent / "data" / "splits"


def build_dual_bank(patcher, tok, rows, site_K="relation_last", v_span="subj_to_obj"):
    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"
    bank.bank_key_mode = "pre_rope"
    kept = []
    for row in rows:
        wp = row["write_prompt"]
        pK = resolve_pos(site_K, row, tok, wp)
        vspan = resolve_v_span(row, tok, wp, v_span)
        if pK is None or vspan is None:
            continue
        write_fact_dual_site_multi_v(
            patcher, bank, tok,
            write_prompt=wp, fact_id=str(row["id"]),
            address=row.get("subject"),
            capture_pos_K=pK, capture_pos_V_list=vspan,
        )
        kept.append(str(row["id"]))
    return bank, kept


@torch.no_grad()
def eval_with_bank(model, tok, patcher, prompt, tn, tt, device, bank, alpha, topk=0,
                   projector=None):
    use = bank is not None and alpha > 0 and not getattr(bank, "empty", False)
    if use and len(bank.fact_ids) == 0:
        use = False
    if use:
        bank.k_projector = projector  # may be None
        if topk > 0:
            bank.bank_topk = int(topk)
            bank.bank_topk_per_layer_separate = False
        else:
            bank.bank_topk = 0
        try:
            with patcher.patched(), patcher.injecting(bank=bank, alpha=float(alpha)):
                lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
                lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
                tnf = ids_new[0] if ids_new else -1
                rank, _ = first_token_rank(model, tok, prompt, tnf, device)
        finally:
            bank.k_projector = None
            bank.bank_topk = 0
    else:
        lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
        lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
        tnf = ids_new[0] if ids_new else -1
        rank, _ = first_token_rank(model, tok, prompt, tnf, device)
    return {
        "log_p_new": float(lp_new),
        "log_p_true": float(lp_true),
        "margin": float(lp_new - lp_true),
        "target_rank": int(rank),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projector", required=True, help="trained K-projector .pt path")
    ap.add_argument("--projector-shuffled", default=None,
                    help="optional 'shuffled fact-id' adapter for Gate E")
    ap.add_argument("--n", type=int, default=100, help="eval queries per seed")
    ap.add_argument("--bank-size", type=int, default=200)
    ap.add_argument("--alphas", default="0.005,0.01,0.02")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--v-span", default="subj_to_obj")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model} ...", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    patcher = AttnNativePatcher(model)
    patcher.install()

    # Load test facts + distractors (full records from X1 jsonl)
    test_facts = json.loads((SPLITS_DIR / "test.json").read_text())
    dist_ids = set(json.loads((SPLITS_DIR / "distractors.json").read_text()))
    dist_path = REPO / "experiments" / "X1_bank_scaling" / "distractors.jsonl"
    dists: list[dict] = []
    with open(dist_path) as f:
        for line in f:
            r = json.loads(line)
            if r["fact_id"] in dist_ids:
                # normalize: distractors use 'fact_id'; bank build expects 'id'
                r["id"] = r["fact_id"]
                # distractors have no target_true; use empty string for eval skip-safety
                r.setdefault("target_true", "")
                dists.append(r)
    print(f"test_facts: {len(test_facts)}  distractors loaded: {len(dists)}", flush=True)

    # Build augmented rows: test facts (already cf format) + distractor format (already has write_prompt)
    test_rows = []
    for r in test_facts:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        rr = dict(r); rr["write_prompt"] = wp
        test_rows.append(rr)

    eval_rows = test_rows[: args.n]

    bank_pool_extra: list[dict] = []
    # bank is test_rows + distractors up to bank_size
    n_dist_needed = max(0, args.bank_size - len(test_rows))
    if n_dist_needed > 0:
        for d in dists[: n_dist_needed]:
            rr = dict(d)
            # distractor entries already have write_prompt + subject + target_new
            bank_pool_extra.append(rr)

    print(f"eval rows={len(eval_rows)}  bank pool: {len(test_rows)} test + {len(bank_pool_extra)} distractors", flush=True)

    # Load projector
    proj = KProjectorBank.load(args.projector)
    proj.eval()
    # Note: KProjectorBank.forward moves params to the requested device/dtype lazily,
    # so we leave it on cpu and it will autocast on first call inside the model.
    print(f"loaded projector: rank={proj.rank}", flush=True)

    proj_shuf = None
    if args.projector_shuffled:
        proj_shuf = KProjectorBank.load(args.projector_shuffled)
        proj_shuf.eval()
        print(f"loaded shuffled projector: rank={proj_shuf.rank}", flush=True)

    # write env
    (out_dir / "env.json").write_text(json.dumps({
        "model": args.model, "dtype": args.dtype, "device": args.device,
        "n": args.n, "bank_size": args.bank_size, "alphas": alphas, "seeds": seeds,
        "v_span": args.v_span,
        "projector": str(args.projector),
        "projector_shuffled": str(args.projector_shuffled) if args.projector_shuffled else None,
        "split": "test",
    }, indent=2))

    fout = (out_dir / "cells.jsonl").open("w")
    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)

        # Build bank: all eval test rows (so we can score correct retrieval) + distractors
        bank_input_rows = test_rows + bank_pool_extra
        bank_input_rows = bank_input_rows[: args.bank_size]
        bank, kept = build_dual_bank(patcher, tok, bank_input_rows, v_span=args.v_span)
        kept_set = set(kept)
        print(f"[seed={seed}] bank built: {len(kept)}/{len(bank_input_rows)} slots", flush=True)

        # Precompute sub-banks for controls
        meanV_bank = subbank_meanV(bank)
        shuf_facts_bank = subbank_shuffle_fact_ids(bank, rng)

        n_done = 0
        for row in eval_rows:
            fid = str(row["id"])
            if fid not in kept_set:
                continue
            tn = row["target_new"]
            tt = row.get("target_true") or ""
            if not tt:
                continue
            q = render_query(row)

            # base (no bank)
            m = eval_with_bank(model, tok, patcher, q, tn, tt, args.device, None, 0.0)
            fout.write(json.dumps({"seed": seed, "fact_id": fid, "alpha": 0.0,
                                   "variant": "base", **m}) + "\n")

            minus = subbank_minus_correct(bank, fid)
            cells = [
                # (name, bank, topk, projector)
                ("full_bank_no_adapter",       bank,            0, None),
                ("full_bank_learned_adapter",  bank,            0, proj),
                ("full_bank_topk1_no_adapter", bank,            1, None),
                ("full_bank_topk1_learned",    bank,            1, proj),
                ("minus_correct_no_adapter",   minus,           0, None),
                ("minus_correct_learned",      minus,           0, proj),
                ("meanV_no_adapter",           meanV_bank,      0, None),
                ("meanV_learned",              meanV_bank,      0, proj),
                ("shuffled_factids_no_adapter",shuf_facts_bank, 0, None),
                ("shuffled_factids_learned",   shuf_facts_bank, 0, proj),
            ]
            if proj_shuf is not None:
                cells.append(("full_bank_shuffled_adapter", bank, 0, proj_shuf))

            for a in alphas:
                for (vname, bnk, tk, pj) in cells:
                    m = eval_with_bank(model, tok, patcher, q, tn, tt, args.device,
                                       bnk, a, topk=tk, projector=pj)
                    fout.write(json.dumps({
                        "seed": seed, "fact_id": fid, "alpha": float(a),
                        "variant": vname, "topk": tk, **m,
                    }) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                dt = time.time() - t0
                rate = n_done / max(dt, 1e-6)
                eta = (len(eval_rows) - n_done) / max(rate, 1e-6)
                print(f"  seed={seed} {n_done}/{len(eval_rows)} done  rate={rate:.2f}/s eta={eta:.0f}s", flush=True)

        del bank, meanV_bank, shuf_facts_bank
        gc.collect()
        if args.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    fout.close()
    print(f"done in {time.time()-t0:.0f}s -> {out_dir/'cells.jsonl'}")


if __name__ == "__main__":
    main()
