"""Exp26b — Multi-token V capture at object span.

K@relation_last (single token, retained from Exp26).
V averaged over [object_first .. object_last] (multi-token span).

Hypothesis: V@object_last single-token under-captures object semantics;
mean-pooled object-span V gives more discriminable bank slots, lifting
retrieval_accuracy above the 2× chance ceiling.
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path
import torch

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank
from deltamemory.memory.anb_dual_site_multi import write_fact_dual_site_multi_v
from deltamemory.memory.anb_addressed import subbank_select
from deltamemory.memory.anb_capture_sweep import (
    resolve_extended_capture, derive_relation_phrase,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (  # noqa: E402
    load_model, load_counterfact, filter_cf_for_tokenizer,
    seed_everything, continuation_logp, first_token_rank,
)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt  # noqa: E402
from tools.env_writer import write_env_json, sha1_of  # noqa: E402

from experiments.atb_validation_v1.exp13_anb_readdressability.exp25_metrics import (  # noqa: E402
    SlotRecorder, subbank_meanV,
)
from experiments.atb_validation_v1.exp13_anb_readdressability.run_dual_oracle import (  # noqa: E402
    resolve_pos, subbank_minus_correct, subbank_shuffle_fact_ids,
)


def resolve_v_span(row, tok, write_prompt, span_kind: str) -> list[int] | None:
    """Return token positions for V capture span.

    span_kind:
      'object'        -> [object_first .. object_last]  (usually 1 tok)
      'subject'       -> [subject_first .. subject_last] (2-5 tok typical)
      'relation'      -> [relation_first .. relation_last]
      'subj_to_obj'   -> [subject_first .. object_last]  (full fact span, 5-10 tok)
      'rel_to_obj'    -> [relation_first .. object_last]
    """
    kind_map = {
        "object":      ("object_first", "object_last"),
        "subject":     ("subject_first", "subject_last"),
        "relation":    ("relation_first", "relation_last"),
        "subj_to_obj": ("subject_first", "object_last"),
        "rel_to_obj":  ("relation_first", "object_last"),
    }
    if span_kind not in kind_map: return None
    a_site, b_site = kind_map[span_kind]
    enc = tok(write_prompt, return_tensors="pt", add_special_tokens=True)
    am = enc["attention_mask"][0]
    rel = derive_relation_phrase(row.get("prompt", ""))
    pf = resolve_extended_capture(
        site=a_site, write_prompt=write_prompt,
        subject=row.get("subject", ""), relation_phrase=rel,
        object_str=row.get("target_new", ""),
        tokenizer=tok, attention_mask_row=am, add_special_tokens=True,
    )
    pl = resolve_extended_capture(
        site=b_site, write_prompt=write_prompt,
        subject=row.get("subject", ""), relation_phrase=rel,
        object_str=row.get("target_new", ""),
        tokenizer=tok, attention_mask_row=am, add_special_tokens=True,
    )
    if not pf or not pl or not pf.token_positions or not pl.token_positions:
        return None
    s = pf.token_positions[-1]
    e = pl.token_positions[-1]
    if s > e: return None
    return list(range(s, e + 1))


def build_multi_v_bank(patcher, tok, rows, site_K="relation_last", v_span="subj_to_obj"):
    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"
    bank.bank_key_mode = "pre_rope"
    kept = []
    span_lens = []
    for row in rows:
        wp = row["write_prompt"]
        pK = resolve_pos(site_K, row, tok, wp)
        vspan = resolve_v_span(row, tok, wp, v_span)
        if pK is None or vspan is None: continue
        span_lens.append(len(vspan))
        write_fact_dual_site_multi_v(
            patcher, bank, tok,
            write_prompt=wp, fact_id=str(row["id"]),
            address=row.get("subject"),
            capture_pos_K=pK, capture_pos_V_list=vspan,
        )
        kept.append(str(row["id"]))
    return bank, kept, span_lens


@torch.no_grad()
def eval_cell(model, tok, patcher, prompt, tn, tt, device, bank, alpha,
              topk=0, record=False, correct_slot=-1):
    use = bank is not None and alpha > 0 and not getattr(bank, "empty", False)
    if use and len(bank.fact_ids) == 0: use = False

    extras = {}
    if use:
        if topk > 0:
            bank.bank_topk = int(topk); bank.bank_topk_per_layer_separate = False
        else:
            bank.bank_topk = 0
        if record:
            with SlotRecorder() as rec, patcher.patched(), \
                 patcher.injecting(bank=bank, alpha=float(alpha)):
                lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
                lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
                tnf = ids_new[0] if ids_new else -1
                rank, _ = first_token_rank(model, tok, prompt, tnf, device)
            agg = rec.aggregate()
            slot = agg["selected_slot_id"]
            bm = agg["bank_attention_mass"]
            extras = {"selected_slot_id": int(slot),
                      "retrieval_correct": int(slot == correct_slot) if correct_slot >= 0 else None,
                      "bank_attention_mass": float(bm)}
        else:
            with patcher.patched(), patcher.injecting(bank=bank, alpha=float(alpha)):
                lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
                lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
                tnf = ids_new[0] if ids_new else -1
                rank, _ = first_token_rank(model, tok, prompt, tnf, device)
    else:
        lp_new, ids_new = continuation_logp(model, tok, prompt, tn, device)
        lp_true, _ = continuation_logp(model, tok, prompt, tt, device)
        tnf = ids_new[0] if ids_new else -1
        rank, _ = first_token_rank(model, tok, prompt, tnf, device)

    out = {"log_p_new": float(lp_new), "log_p_true": float(lp_true),
           "margin": float(lp_new - lp_true),
           "target_rank": int(rank), "recall_at_1": bool(rank == 0)}
    out.update(extras)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact", default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alphas", default="0.005,0.010,0.020")
    ap.add_argument("--bank-size", type=int, default=100)
    ap.add_argument("--v-span", default="subj_to_obj",
                    choices=["object","subject","relation","subj_to_obj","rel_to_obj"])
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
    needed = max(args.n, args.bank_size)
    rows = rows[:needed]
    print(f"[exp26b] eligible rows = {len(rows)}", flush=True)

    write_env_json(out_dir, prereg_version="exp26b.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"experiment": "exp26b_multi_v",
                          "v_span": args.v_span,
                          "n": args.n, "alphas": alphas,
                          "bank_size": args.bank_size, "seeds": seeds})
    patcher = AttnNativePatcher(model); patcher.install()
    fout = (out_dir / "cells.jsonl").open("a")
    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)
        bank_rows = rows[: args.bank_size]
        bank, kept, span_lens = build_multi_v_bank(patcher, tok, bank_rows, v_span=args.v_span)
        kept_set = set(kept)
        n_bank = len(kept_set)
        mean_span = sum(span_lens)/len(span_lens) if span_lens else 0
        print(f"[exp26b] seed={seed} bank={n_bank}/{args.bank_size} mean_V_span={mean_span:.2f}", flush=True)
        meanV_bank = subbank_meanV(bank)
        shuf_facts_bank = subbank_shuffle_fact_ids(bank, rng)

        for row in rows[: args.n]:
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
                ("full_bank_concat",                bank,            0, False, -1),
                ("full_bank_topk1",                 bank,            1, True,  correct_idx),
                ("full_bank_topk1_minus_correct",   minus,           1, True,  -1),
                ("full_bank_topk1_meanV",           meanV_bank,      1, True,  correct_idx),
                ("full_bank_topk1_shuffled_factids",shuf_facts_bank, 1, True,  correct_idx),
            ]
            for a in alphas:
                for (vname, bnk, tk, rec, cslot) in cells:
                    m = eval_cell(model, tok, patcher, q, tn, tt, args.device,
                                  bnk, a, topk=tk, record=rec, correct_slot=cslot)
                    fout.write(json.dumps({"seed": seed, "fact_id": fid, "alpha": float(a),
                                           "bank_size": n_bank, "variant": vname,
                                           "topk": tk, **m}) + "\n")
            fout.flush(); gc.collect()
        del bank, meanV_bank, shuf_facts_bank
        gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass
    fout.close()
    print(f"[exp26b] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
