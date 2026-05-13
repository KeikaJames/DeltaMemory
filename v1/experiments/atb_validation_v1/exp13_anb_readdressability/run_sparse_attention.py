"""Exp24 — Sparse-Attention Readout on full bank.

The existing injector already does concat-softmax attn(q, [K_seq; M_K],
[V_seq; alpha*M_V]). Two knobs control "sparseness":

  - bank_topk = K:  keep only top-K bank slots before softmax (Stage 15A).
                    K=1 = natural top-1 routing via softmax score.
  - alpha:          V output scale. At alpha=1 there is NO V damping;
                    bank V participates at full magnitude.

Hypothesis: if relation_last K-causality survives at scale, full_bank with
bank_topk=1 should approach single-slot oracle. If correct fact wins via
softmax routing, natural beats random/minus_correct/shuffled.

Variants (12) at α ∈ {0.005, 0.05, 1.0}:
  base
  full_bank_concat                  (default, no topk)
  full_bank_topk1                   (natural top-1 via softmax)
  full_bank_topk3                   (natural top-3)
  full_bank_topk1_minus_correct
  full_bank_topk1_shuffled_factids  (K/V identity mismatch in same bank)
  full_bank_topk1_random_bank       (independently sampled facts, no correct in bank)
  bank_separate_softmax_topk1       (bank gets its own softmax, beta=1)
  bank_separate_softmax_alpha1      (bank softmax + alpha=1)
  oracle_relK_subjV_alpha1          (Exp23 oracle at alpha=1)
  oracle_relK_subjV_concat          (Exp23 oracle at this alpha)
  full_bank_topk1_shuffled_layers
"""
from __future__ import annotations
import argparse, copy, gc, json, sys, time
from pathlib import Path
import torch

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank
from deltamemory.memory.anb_dual_site import write_fact_dual_site
from deltamemory.memory.anb_addressed import (
    subbank_select, subbank_random,
    subbank_shuffle_layer,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (load_model, load_counterfact,
    filter_cf_for_tokenizer, seed_everything, continuation_logp, first_token_rank)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt
from experiments.atb_validation_v1.exp13_anb_readdressability.run_dual_oracle import (
    build_dual_bank, subbank_correct, subbank_minus_correct,
    subbank_shuffle_fact_ids)
from tools.env_writer import write_env_json, sha1_of


def configure_bank(bank, *, topk=0, separate=False, beta=1.0):
    """Mutate bank knobs for variant; restore via configure_bank(bank, defaults)."""
    bank.bank_topk = int(topk)
    bank.bank_separate_softmax = bool(separate)
    bank.bank_merge_beta = float(beta)
    return bank


@torch.no_grad()
def eval_with_bank_cfg(model, tok, patcher, prompt, tn, tt, device, bank, alpha,
                       *, topk=0, separate=False, beta=1.0):
    if bank is None or alpha <= 0 or getattr(bank, "empty", False) or len(bank.fact_ids) == 0:
        lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
        lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
        tnf = ids_new[0] if ids_new else -1
        rank, _ = first_token_rank(model, tok, prompt, tnf, device)
    else:
        configure_bank(bank, topk=topk, separate=separate, beta=beta)
        with patcher.patched(), patcher.injecting(bank=bank, alpha=float(alpha)):
            lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
            lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
            tnf = ids_new[0] if ids_new else -1
            rank, _ = first_token_rank(model, tok, prompt, tnf, device)
        configure_bank(bank, topk=0, separate=False, beta=1.0)
    return {"target_new_logprob": float(lp_new),
            "target_true_logprob": float(lp_true),
            "margin": float(lp_new - lp_true),
            "target_rank": int(rank), "recall_at_1": bool(rank == 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact", default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--alphas", default="0.005,0.05,1.0")
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
    n_eligible = len(rows)
    print(f"[exp24] eligible rows = {n_eligible}", flush=True)

    write_env_json(out_dir, prereg_version="exp24.sparse.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"experiment": "exp24_sparse_attention_readout",
                          "n": n_eligible, "alphas": alphas})
    (out_dir / "manifest.json").write_text(json.dumps({
        "experiment": "exp24_sparse_attention_readout",
        "model": args.model, "n_eligible": n_eligible,
        "alphas": alphas, "seeds": seeds}, indent=2))

    patcher = AttnNativePatcher(model); patcher.install()
    fout = (out_dir / "cells.jsonl").open("a")

    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)
        bank_RS, kept = build_dual_bank(patcher, tok, rows, "relation_last", "subject_last")
        kept_set = set(kept)
        print(f"[exp24] seed={seed} kept {len(kept_set)}/{n_eligible} built bank", flush=True)
        shuf_layers = subbank_shuffle_layer(bank_RS, rng)
        shuf_facts = subbank_shuffle_fact_ids(bank_RS, rng)

        for row in rows:
            fid = str(row["id"])
            if fid not in kept_set: continue
            tn = row["target_new"]; tt = row.get("target_true") or ""
            if not tt: continue
            q = render_query(row)

            # base only once per row (alpha-independent)
            m = eval_with_bank_cfg(model, tok, patcher, q, tn, tt, args.device, None, 0.0)
            fout.write(json.dumps({"seed": seed, "fact_id": fid,
                                   "alpha": 0.0, "variant": "base", **m}) + "\n")

            try:
                rand_RS = subbank_random(bank_RS, rng, k=1, exclude=(fid,))
            except ValueError:
                continue
            minus = subbank_minus_correct(bank_RS, fid)
            oracle = subbank_correct(bank_RS, fid)

            for a in alphas:
                cells = []
                cells.append(("full_bank_concat", bank_RS, a, 0, False))
                cells.append(("full_bank_topk1", bank_RS, a, 1, False))
                cells.append(("full_bank_topk3", bank_RS, a, 3, False))
                cells.append(("full_bank_topk1_minus_correct", minus, a, 1, False))
                cells.append(("full_bank_topk1_shuffled_factids", shuf_facts, a, 1, False))
                cells.append(("full_bank_topk1_shuffled_layers", shuf_layers, a, 1, False))
                cells.append(("bank_separate_softmax_topk1", bank_RS, a, 1, True))
                cells.append(("bank_separate_softmax_concat", bank_RS, a, 0, True))
                cells.append(("oracle_concat", oracle, a, 0, False))
                cells.append(("oracle_topk1", oracle, a, 1, False))  # vacuous topk on 1-slot
                cells.append(("random_topk1_concat", rand_RS, a, 0, False))
                cells.append(("random_topk1", rand_RS, a, 1, False))

                for vname, bnk, aa, tk, sep in cells:
                    m = eval_with_bank_cfg(model, tok, patcher, q, tn, tt, args.device,
                                           bnk, aa, topk=tk, separate=sep)
                    fout.write(json.dumps({"seed": seed, "fact_id": fid,
                                           "alpha": aa, "variant": vname,
                                           "topk": tk, "separate": sep, **m}) + "\n")
                gc.collect()
            fout.flush()

        del bank_RS, shuf_layers, shuf_facts
        gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    fout.close(); patcher.remove()
    print(f"[exp24] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
