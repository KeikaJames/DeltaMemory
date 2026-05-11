"""Exp25 — K-routing gap stabilization.

Goal: push the Exp24 +0.18 nat retrieval signal to a robust, reproducible
+0.10 minimum across α, bank-size, seed. Also log the *retrieval_accuracy*
metric (which slot K-routing actually picks) that Exp24 was missing.

Phases (entry point chosen by --phase):

  alpha:    α ∈ {0.003,0.005,0.007,0.010,0.015,0.020,0.030} × n × 3 seeds
            × 6 core variants. Bank N = --bank-size (default 100).
  banksize: N ∈ {32,64,100,200,400,807} × n × 3 seeds × 4 variants at
            --alpha (use top-2 from `alpha`).
  hardneg:  Bank composed of hard negatives.

Variants:
  base                                (no bank)
  full_bank_concat                    (default native attention, full bank)
  full_bank_topk1                     (sparse routing — primary)
  full_bank_topk1_minus_correct       (correct slot removed — primary control)
  full_bank_topk1_meanV               (V replaced by bank-mean V — NEW)
  full_bank_topk1_shuffled_factids    (K/V identity mismatch)

New per-cell fields:
  selected_slot_id        which slot K-routing picked at last token (mode over layers)
  retrieval_correct       int(selected_slot_id == correct_slot_id_in_bank)
  bank_attention_mass     mean bank-weight sum at last token, over layers
  max_bank_prob           mean top-1 bank weight at last token, over layers
  top1_top2_gap           proxy for routing confidence
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import torch

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (  # noqa: E402
    load_model, load_counterfact, filter_cf_for_tokenizer,
    seed_everything, continuation_logp, first_token_rank,
)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt  # noqa: E402
from experiments.atb_validation_v1.exp13_anb_readdressability.run_dual_oracle import (  # noqa: E402
    build_dual_bank, subbank_minus_correct, subbank_shuffle_fact_ids,
)
from experiments.atb_validation_v1.exp13_anb_readdressability.exp25_metrics import (  # noqa: E402
    SlotRecorder, subbank_meanV,
)
from deltamemory.memory.anb_addressed import subbank_select  # noqa: E402
from tools.env_writer import write_env_json, sha1_of  # noqa: E402


def configure_bank(bank, *, topk=0, separate=False, beta=1.0):
    bank.bank_topk = int(topk)
    bank.bank_separate_softmax = bool(separate)
    bank.bank_merge_beta = float(beta)
    return bank


@torch.no_grad()
def eval_cell(model, tok, patcher, prompt, tn, tt, device,
              bank, alpha, *, topk=0, separate=False,
              record: bool = False, correct_slot: int = -1):
    """Run one cell evaluation. If ``record`` and bank/alpha are active,
    also run an extra single-pass forward over ``prompt`` with SlotRecorder
    enabled and merge retrieval metrics.
    """
    use_bank = bank is not None and alpha > 0 and not getattr(bank, "empty", False) and len(bank.fact_ids) > 0
    if use_bank:
        configure_bank(bank, topk=topk, separate=separate)
        with patcher.patched(), patcher.injecting(bank=bank, alpha=float(alpha)):
            lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
            lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
            tnf = ids_new[0] if ids_new else -1
            rank, _ = first_token_rank(model, tok, prompt, tnf, device)
            metrics_extra = {}
            if record:
                with SlotRecorder() as rec:
                    _ = continuation_logp(model, tok, prompt, tn, device)
                agg = rec.aggregate()
                sel = int(agg["selected_slot_id"])
                metrics_extra = {
                    "selected_slot_id": sel,
                    "retrieval_correct": int(sel == correct_slot) if correct_slot >= 0 else -1,
                    "bank_attention_mass": agg["bank_attention_mass"],
                    "max_bank_prob": agg["max_bank_prob"],
                    "top1_top2_gap": agg["top1_top2_gap"],
                }
        configure_bank(bank, topk=0, separate=False)
    else:
        lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
        lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
        tnf = ids_new[0] if ids_new else -1
        rank, _ = first_token_rank(model, tok, prompt, tnf, device)
        metrics_extra = {}
    out = {"target_new_logprob": float(lp_new),
           "target_true_logprob": float(lp_true),
           "margin": float(lp_new - lp_true),
           "target_rank": int(rank), "recall_at_1": bool(rank == 0)}
    out.update(metrics_extra)
    return out


def variants_for_phase(phase: str, bank, fact_id, rng):
    """Return list of (variant_name, bank_obj, topk, separate, record_flag, correct_slot)."""
    cells = []
    minus = subbank_minus_correct(bank, fact_id)
    meanV = subbank_meanV(bank)
    shuf_facts = subbank_shuffle_fact_ids(bank, rng)
    # correct slot index for original full bank
    correct_idx = bank.fact_ids.index(fact_id)
    cells.append(("full_bank_concat",                bank,       0, False, False, -1))
    cells.append(("full_bank_topk1",                 bank,       1, False, True,  correct_idx))
    cells.append(("full_bank_topk1_minus_correct",   minus,      1, False, True,  -1))  # correct absent
    cells.append(("full_bank_topk1_meanV",           meanV,      1, False, True,  correct_idx))
    cells.append(("full_bank_topk1_shuffled_factids",shuf_facts, 1, False, True,  correct_idx))
    return cells


def run_alpha_phase(args, model, tok, patcher, rows, seeds, alphas, out_dir, bank_size):
    fout = (out_dir / "cells.jsonl").open("a")
    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)
        bank_rows = rows[:bank_size]
        bank, kept = build_dual_bank(patcher, tok, bank_rows, args.site_k, args.site_v)
        kept_set = set(kept)
        n_bank = len(kept_set)
        print(f"[exp25.alpha] seed={seed} bank_kept={n_bank}/{bank_size}", flush=True)

        # Pre-build bank-level variants (fact-id-independent).
        meanV_bank = subbank_meanV(bank)
        shuf_facts_bank = subbank_shuffle_fact_ids(bank, rng)

        for row in rows[:args.n]:
            fid = str(row["id"])
            if fid not in kept_set: continue
            tn = row["target_new"]; tt = row.get("target_true") or ""
            if not tt: continue
            q = render_query(row)

            m = eval_cell(model, tok, patcher, q, tn, tt, args.device, None, 0.0)
            fout.write(json.dumps({"seed": seed, "fact_id": fid, "alpha": 0.0,
                                   "bank_size": n_bank, "variant": "base", **m}) + "\n")

            minus = subbank_minus_correct(bank, fid)
            correct_idx = bank.fact_ids.index(fid)

            cells = [
                ("full_bank_concat",                bank,            0, False, False, -1),
                ("full_bank_topk1",                 bank,            1, False, True,  correct_idx),
                ("full_bank_topk1_minus_correct",   minus,           1, False, True,  -1),
                ("full_bank_topk1_meanV",           meanV_bank,      1, False, True,  correct_idx),
                ("full_bank_topk1_shuffled_factids",shuf_facts_bank, 1, False, True,  correct_idx),
            ]

            for a in alphas:
                for (vname, bnk, tk, sep, rec, cslot) in cells:
                    m = eval_cell(model, tok, patcher, q, tn, tt, args.device,
                                  bnk, a, topk=tk, separate=sep,
                                  record=rec, correct_slot=cslot)
                    fout.write(json.dumps({
                        "seed": seed, "fact_id": fid, "alpha": float(a),
                        "bank_size": n_bank, "variant": vname,
                        "topk": tk, "separate": sep, **m}) + "\n")
            fout.flush()
            gc.collect()

        del bank, meanV_bank, shuf_facts_bank
        gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass
    fout.close()
    print(f"[exp25.alpha] done in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["alpha", "banksize", "hardneg"], default="alpha")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact", default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alphas", default="0.003,0.005,0.007,0.010,0.015,0.020,0.030")
    ap.add_argument("--bank-size", type=int, default=100)
    ap.add_argument("--site-k", default="relation_last")
    ap.add_argument("--site-v", default="subject_last")
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
    # need enough rows to cover bank + eval
    needed = max(args.n, args.bank_size)
    rows = rows[: max(needed, args.n)]
    print(f"[exp25] eligible rows = {len(rows)} phase={args.phase}", flush=True)

    write_env_json(out_dir, prereg_version=f"exp25.{args.phase}.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"experiment": f"exp25_{args.phase}",
                          "n": args.n, "alphas": alphas,
                          "bank_size": args.bank_size, "seeds": seeds,
                          "site_k": args.site_k, "site_v": args.site_v})
    (out_dir / "manifest.json").write_text(json.dumps({
        "experiment": f"exp25_{args.phase}", "model": args.model,
        "n": args.n, "bank_size": args.bank_size,
        "alphas": alphas, "seeds": seeds}, indent=2))

    patcher = AttnNativePatcher(model); patcher.install()
    try:
        if args.phase == "alpha":
            run_alpha_phase(args, model, tok, patcher, rows, seeds, alphas, out_dir, args.bank_size)
        elif args.phase == "banksize":
            # Loop N over alphas (single alpha)
            assert len(alphas) >= 1, "banksize phase: provide one --alphas value"
            for N in [int(x) for x in args.bank_size.split(",")] if isinstance(args.bank_size, str) else [args.bank_size]:
                run_alpha_phase(args, model, tok, patcher, rows, seeds, alphas[:1], out_dir, N)
        else:  # hardneg
            run_alpha_phase(args, model, tok, patcher, rows, seeds, alphas, out_dir, args.bank_size)
    finally:
        patcher.remove()


if __name__ == "__main__":
    main()
