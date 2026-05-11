"""Exp23 — Site-Stratified Oracle Ceiling.

Test whether dual-site capture (K@relation_last, V@subject_last) with
oracle slot selection beats every control. This sets the upper bound for
Exp24 natural routing.

Variants (12):
  base, old_full_bank,
  oracle_relationK_subjectV, oracle_relationK_relationV,
  oracle_subjectK_subjectV, oracle_subjectK_relationV,
  random_relationK_subjectV, relationK_randomV, randomK_subjectV,
  minus_correct, shuffled_fact_ids, shuffled_layers.

Triage: α=0.005 only at n=100 seeds {0,1}. Full α sweep gated on PASS.
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path
import torch

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher, fresh_bank, write_fact,
)
from deltamemory.memory.anb_dual_site import write_fact_dual_site
from deltamemory.memory.anb_addressed import (
    subbank_select, subbank_random, subbank_swap_KV,
    subbank_shuffle_layer,
)
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


def resolve_pos(site: str, row, tok, write_prompt, add_special=True) -> int | None:
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


def build_dual_bank(patcher, tok, rows, site_K: str, site_V: str):
    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"
    bank.bank_key_mode = "pre_rope"
    kept = []
    for row in rows:
        wp = row["write_prompt"]
        pK = resolve_pos(site_K, row, tok, wp)
        pV = resolve_pos(site_V, row, tok, wp)
        if pK is None or pV is None:
            continue
        write_fact_dual_site(patcher, bank, tok,
                             write_prompt=wp, fact_id=str(row["id"]),
                             address=row.get("subject"),
                             capture_pos_K=pK, capture_pos_V=pV)
        kept.append(str(row["id"]))
    return bank, kept


def subbank_correct(bank, fact_id):
    """Single-slot oracle."""
    i = bank.fact_ids.index(fact_id)
    return subbank_select(bank, [i])


def subbank_minus_correct(bank, fact_id):
    """All slots EXCEPT fact_id."""
    keep = [i for i, f in enumerate(bank.fact_ids) if f != fact_id]
    return subbank_select(bank, keep)


def subbank_shuffle_fact_ids(bank, rng):
    """Per-layer: keep K at slot order, V permuted (mismatch K/V identities)."""
    n = len(bank.fact_ids)
    if n <= 1:
        return bank
    perm = torch.randperm(n, generator=rng).tolist()
    while perm == list(range(n)):
        perm = torch.randperm(n, generator=rng).tolist()
    out_K = bank
    # build a "V permuted" partner
    perm_bank = subbank_select(bank, perm)
    return subbank_swap_KV(bank_K_source=out_K, bank_V_source=perm_bank)


@torch.no_grad()
def eval_with_bank(model, tok, patcher, prompt, tn, tt, device, bank, alpha):
    use = bank is not None and alpha > 0 and not getattr(bank, "empty", False)
    if use and len(bank.fact_ids) == 0:
        use = False
    if use:
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
    return {"target_new_logprob": float(lp_new),
            "target_true_logprob": float(lp_true),
            "margin": float(lp_new - lp_true),
            "target_rank": int(rank), "recall_at_1": bool(rank == 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact", default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--site_K", default="relation_last")
    ap.add_argument("--site_V", default="subject_last")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alpha = float(args.alpha)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[exp23] eligible rows = {n_eligible} (dropped {dropped})")

    manifest = {"experiment": "exp23_dual_oracle_ceiling",
                "model": args.model, "site_K": args.site_K, "site_V": args.site_V,
                "alpha": alpha, "seeds": seeds, "n_eligible": n_eligible}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_env_json(out_dir, prereg_version="exp23.prereg.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"n_eligible": n_eligible, "experiment": "exp23_dual_oracle_ceiling",
                          "site_K": args.site_K, "site_V": args.site_V, "alpha": alpha})

    patcher = AttnNativePatcher(model); patcher.install()
    rows_path = out_dir / "cells.jsonl"; rows_path.unlink(missing_ok=True)
    fout = rows_path.open("a")

    t0 = time.time()
    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)

        # Build 4 banks: (relK, subjV), (relK, relV), (subjK, subjV), (subjK, relV)
        # Reuse same row order across banks for paired contrast.
        print(f"[exp23] seed={seed} building 4 dual-site banks...", flush=True)
        b_RS, kept_RS = build_dual_bank(patcher, tok, rows, "relation_last", "subject_last")
        b_RR, kept_RR = build_dual_bank(patcher, tok, rows, "relation_last", "relation_last")
        b_SS, kept_SS = build_dual_bank(patcher, tok, rows, "subject_last", "subject_last")
        b_SR, kept_SR = build_dual_bank(patcher, tok, rows, "subject_last", "relation_last")
        b_oldfull = b_SS  # old single-site bank == subj-only, alias
        kept = set(kept_RS) & set(kept_RR) & set(kept_SS) & set(kept_SR)
        print(f"[exp23] seed={seed} kept {len(kept)}/{n_eligible}", flush=True)

        shuf_RS_layers = subbank_shuffle_layer(b_RS, rng)
        shuf_RS_facts  = subbank_shuffle_fact_ids(b_RS, rng)

        for row in rows:
            fid = str(row["id"])
            if fid not in kept: continue
            tn = row["target_new"]; tt = row.get("target_true") or ""
            if not tt: continue
            query = render_query(row)

            # variants
            cells = {}
            cells["base"] = (None, 0.0)
            cells["old_full_bank"] = (b_oldfull, alpha)
            cells["oracle_relationK_subjectV"] = (subbank_correct(b_RS, fid), alpha)
            cells["oracle_relationK_relationV"] = (subbank_correct(b_RR, fid), alpha)
            cells["oracle_subjectK_subjectV"] = (subbank_correct(b_SS, fid), alpha)
            cells["oracle_subjectK_relationV"] = (subbank_correct(b_SR, fid), alpha)

            # random fact for swap controls
            try:
                rand_RS = subbank_random(b_RS, rng, k=1, exclude=(fid,))
            except ValueError:
                continue
            cells["random_relationK_subjectV"] = (rand_RS, alpha)
            cells["relationK_randomV"] = (subbank_swap_KV(
                bank_K_source=subbank_correct(b_RS, fid),
                bank_V_source=rand_RS), alpha)
            cells["randomK_subjectV"] = (subbank_swap_KV(
                bank_K_source=rand_RS,
                bank_V_source=subbank_correct(b_RS, fid)), alpha)
            cells["minus_correct"] = (subbank_minus_correct(b_RS, fid), alpha)
            cells["shuffled_fact_ids"] = (shuf_RS_facts, alpha)
            cells["shuffled_layers"] = (shuf_RS_layers, alpha)

            for vname, (bnk, a) in cells.items():
                m = eval_with_bank(model, tok, patcher, query, tn, tt, args.device, bnk, a)
                fout.write(json.dumps({"seed": seed, "fact_id": fid,
                                       "site_K": args.site_K, "site_V": args.site_V,
                                       "alpha": a, "variant": vname, **m}) + "\n")
            del cells, rand_RS
            gc.collect(); fout.flush()

        del b_RS, b_RR, b_SS, b_SR, shuf_RS_layers, shuf_RS_facts
        gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    fout.close(); patcher.remove()
    print(f"[exp23] done in {time.time()-t0:.1f}s -> {rows_path}")


if __name__ == "__main__":
    main()
