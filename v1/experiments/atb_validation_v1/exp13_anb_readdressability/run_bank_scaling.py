"""Exp20 — bank-size scaling for K-causal routing.

For each k in {1, 3, 7, 21}, build a bank of size k:
  correct slot + (k-1) random distractors
and measure:
  Kc_Vc - base                (does oracle still beat base?)
  Kc_Vc - Kr_Vc               (K-causality, paired)
  Kc_Vc - random_Kc_Vc        (vs correct slot from a random subbank)

If K-causality CI shrinks toward zero as k grows, routing breaks under
distractors — confirms NATURAL_FAIL from Exp18.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import torch

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank, write_fact
from deltamemory.memory.anb_addressed import subbank_correct, subbank_random, subbank_select, subbank_swap_KV
from deltamemory.memory.anb_capture_sweep import resolve_extended_capture, derive_relation_phrase

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (  # noqa: E402
    load_model, load_counterfact, filter_cf_for_tokenizer,
    seed_everything, continuation_logp, first_token_rank,
)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt  # noqa: E402
from tools.env_writer import write_env_json, sha1_of  # noqa: E402


def resolve_capture_pos(site, row, tok, write_prompt, add_special=True):
    enc = tok(write_prompt, return_tensors="pt", add_special_tokens=add_special)
    am = enc["attention_mask"][0]
    rel = derive_relation_phrase(row.get("prompt", ""))
    spec = resolve_extended_capture(
        site=site, write_prompt=write_prompt,
        subject=row.get("subject", ""), relation_phrase=rel,
        object_str=row.get("target_new", ""),
        tokenizer=tok, attention_mask_row=am, add_special_tokens=add_special,
    )
    if spec is None or not spec.token_positions:
        return None
    return spec.token_positions[-1]


def build_full_bank(patcher, tok, rows, site):
    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"
    bank.bank_key_mode = "pre_rope"
    kept = []
    for row in rows:
        wp = row["write_prompt"]
        cap_pos = resolve_capture_pos(site, row, tok, wp)
        if cap_pos is None:
            continue
        write_fact(patcher, bank, tok,
                   write_prompt=wp, fact_id=str(row["id"]),
                   address=row.get("subject"), capture_pos=cap_pos)
        kept.append(str(row["id"]))
    return bank, kept


@torch.no_grad()
def eval_with_bank(model, tok, patcher, prompt, target_new, target_true, device, bank, alpha):
    ctx = patcher.injecting(bank=bank, alpha=float(alpha)) if (bank is not None and alpha > 0) else None
    if ctx is not None:
        with patcher.patched(), ctx:
            logp_new, ids_new = continuation_logp(model, tok, prompt, target_new, device)
            logp_true, _ = continuation_logp(model, tok, prompt, target_true, device)
            tnf = ids_new[0] if ids_new else -1
            rank, _ = first_token_rank(model, tok, prompt, tnf, device)
    else:
        logp_new, ids_new = continuation_logp(model, tok, prompt, target_new, device)
        logp_true, _ = continuation_logp(model, tok, prompt, target_true, device)
        tnf = ids_new[0] if ids_new else -1
        rank, _ = first_token_rank(model, tok, prompt, tnf, device)
    return {
        "target_new_logprob": float(logp_new),
        "target_true_logprob": float(logp_true),
        "margin": float(logp_new - logp_true),
        "target_rank": int(rank),
        "recall_at_1": bool(rank == 0),
    }


def make_k_bank(full, fid, k, rng):
    """correct slot + k-1 random distractors, returned as size-k bank."""
    if k == 1:
        return subbank_correct(full, fid)
    distractors = subbank_random(full, rng, k=k - 1, exclude=(fid,))
    # concat correct + distractors via subbank_select
    correct_idx = full.fact_ids.index(fid)
    distract_idx = [full.fact_ids.index(f) for f in distractors.fact_ids]
    return subbank_select(full, [correct_idx] + distract_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact", default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=21)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--site", default="relation_last")
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--ks", default="1,3,7,21")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    alpha = float(args.alpha)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    cf = load_counterfact(Path(args.counterfact))
    kept_rows, dropped = filter_cf_for_tokenizer(cf, tok)
    rows = []
    for r in kept_rows:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None: continue
        r = dict(r); r["write_prompt"] = wp
        rows.append(r)
    if args.n > 0:
        rows = rows[: args.n]
    n_eligible = len(rows)
    print(f"[exp20] eligible rows = {n_eligible}, ks={ks}")

    manifest = {"experiment": "exp20_bank_size_scaling",
                "model": args.model, "site": args.site, "alpha": alpha,
                "seeds": seeds, "ks": ks, "n_eligible": n_eligible}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_env_json(out_dir, prereg_version="exp13.prereg.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"n_eligible": n_eligible, "experiment": "exp20_bank_size_scaling",
                          "site": args.site, "alpha": alpha, "ks": ks})

    patcher = AttnNativePatcher(model)
    patcher.install()
    rows_path = out_dir / "cells.jsonl"
    rows_path.unlink(missing_ok=True)
    fout = rows_path.open("a")

    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)
        full, kept_ids = build_full_bank(patcher, tok, rows, args.site)
        kept_set = set(kept_ids)
        print(f"[exp20] seed={seed} kept {len(kept_ids)}/{n_eligible}", flush=True)

        for row in rows:
            fid = str(row["id"])
            if fid not in kept_set: continue
            tn = row["target_new"]; tt = row.get("target_true") or ""
            if not tt: continue
            query = render_query(row)

            base = eval_with_bank(model, tok, patcher, query, tn, tt, args.device, None, 0.0)
            fout.write(json.dumps({"seed": seed, "fact_id": fid, "k": 0,
                                   "variant": "base", **base}) + "\n")

            for k in ks:
                try:
                    bk = make_k_bank(full, fid, k, rng)
                    # Kr_Vc surrogate: same-size bank with random K (swap K with another random bank)
                    rand_K_bank = subbank_random(full, rng, k=k, exclude=(fid,))
                    KrVc = subbank_swap_KV(bank_K_source=rand_K_bank, bank_V_source=bk)
                except ValueError:
                    continue
                for vname, b in [("Kc_Vc", bk), ("Kr_Vc", KrVc)]:
                    m = eval_with_bank(model, tok, patcher, query, tn, tt, args.device, b, alpha)
                    fout.write(json.dumps({"seed": seed, "fact_id": fid, "k": k,
                                           "variant": vname, **m}) + "\n")
                del bk, rand_K_bank, KrVc
            gc.collect(); fout.flush()
        del full; gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    fout.close(); patcher.remove()
    print(f"[exp20] done in {time.time()-t0:.1f}s -> {rows_path}")


if __name__ == "__main__":
    main()
