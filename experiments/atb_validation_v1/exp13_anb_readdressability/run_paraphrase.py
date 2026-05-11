"""Exp19 — paraphrase robustness (K/V binding 2×2 on paraphrase queries).

After Exp14 showed ORACLE_DIRECTIONAL at alpha=0.005 with the strongest separation
from shuffled_layer and a positive K-only-V-random contrast, Exp15 zooms in on
the K/V binding hypothesis with a clean 2×2:

  (K_correct,  V_correct)   ← oracle
  (K_correct,  V_random )   ← K binds the address; V wrong
  (K_random,   V_correct)   ← V holds the content; K cannot address
  (K_random,   V_random )   ← negative control

If the "correct K + correct V" cell beats BOTH off-diagonal cells (K-only and
V-only) with paired CI > 0, K/V binding causality is established.

We run at the best alpha from Exp14 (default 0.005) and on the QK-best site
(subject_last).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.anb_addressed import (
    subbank_correct,
    subbank_random,
    subbank_swap_KV,
)
from deltamemory.memory.anb_capture_sweep import (
    resolve_extended_capture,
    derive_relation_phrase,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (  # noqa: E402
    load_model,
    load_counterfact,
    filter_cf_for_tokenizer,
    seed_everything,
    continuation_logp,
    first_token_rank,
)
from atb_validation_v1._lib.cf_runner import (  # noqa: E402
    render_query,
    build_write_prompt,
)
from tools.env_writer import write_env_json, sha1_of  # noqa: E402


def resolve_capture_pos(site, row, tok, write_prompt, add_special=True):
    if site == "period":
        return None
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
        if site != "period" and cap_pos is None:
            continue
        write_fact(patcher, bank, tok,
                   write_prompt=wp, fact_id=str(row["id"]),
                   address=row.get("subject"), capture_pos=cap_pos)
        kept.append(str(row["id"]))
    return bank, kept


@torch.no_grad()
def eval_with_bank(model, tok, patcher, prompt, target_new, target_true,
                   device, bank, alpha):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact",
                    default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=21)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--site", default="subject_last")
    ap.add_argument("--alpha", type=float, default=0.005)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alpha = float(args.alpha)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    cf = load_counterfact(Path(args.counterfact))
    kept_rows, dropped = filter_cf_for_tokenizer(cf, tok)
    rows = []
    for r in kept_rows:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        r = dict(r); r["write_prompt"] = wp
        rows.append(r)
    if args.n > 0:
        rows = rows[: args.n]
    n_eligible = len(rows)
    print(f"[exp19] eligible rows = {n_eligible} (dropped {dropped} pre-filter)")

    manifest = {
        "experiment": "exp19_paraphrase",
        "model": args.model, "dtype": args.dtype, "device": args.device,
        "site": args.site, "alpha": alpha, "seeds": seeds,
        "n_eligible": n_eligible, "counterfact": args.counterfact,
        "torch_version": torch.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_env_json(out_dir,
                   prereg_version="exp13.prereg.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"n_eligible": n_eligible,
                          "experiment": "exp19_paraphrase",
                          "site": args.site, "alpha": alpha})

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
        print(f"[exp19] seed={seed} kept {len(kept_ids)}/{n_eligible}", flush=True)
        kept_set = set(kept_ids)

        for row in rows:
            fid = str(row["id"])
            if fid not in kept_set:
                continue
            target_new = row["target_new"]
            target_true = row.get("target_true") or row.get("target_old") or ""
            if not target_true:
                continue
            query = render_query(row)
            paras = row.get("paraphrase_prompts") or []
            if not paras:
                continue
            para_query = paras[0]

            b_K = subbank_correct(full, fid)            # correct K & V
            try:
                b_rand_a = subbank_random(full, rng, k=1, exclude=(fid,))
                b_rand_b = subbank_random(full, rng, k=1, exclude=(fid,))
            except ValueError:
                continue
            # 2x2 cells:
            cells = {
                "Kc_Vc": b_K,
                "Kc_Vr": subbank_swap_KV(bank_K_source=b_K, bank_V_source=b_rand_a),
                "Kr_Vc": subbank_swap_KV(bank_K_source=b_rand_a, bank_V_source=b_K),
                "Kr_Vr": b_rand_b,
            }

            # base
            base = eval_with_bank(model, tok, patcher, para_query, target_new,
                                  target_true, args.device, None, 0.0)
            fout.write(json.dumps({"seed": seed, "fact_id": fid, "site": args.site,
                                   "alpha": 0.0, "variant": "base", "query_type": "paraphrase", **base}) + "\n")
            for vname, bnk in cells.items():
                m = eval_with_bank(model, tok, patcher, para_query, target_new,
                                   target_true, args.device, bnk, alpha)
                fout.write(json.dumps({"seed": seed, "fact_id": fid, "site": args.site,
                                       "alpha": alpha, "variant": vname, "query_type": "paraphrase", **m}) + "\n")
            del b_K, b_rand_a, b_rand_b, cells
            gc.collect()
            fout.flush()
        del full
        gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    fout.close()
    patcher.remove()
    dt = time.time() - t0
    print(f"[exp19] done in {dt:.1f}s -> {rows_path}")


if __name__ == "__main__":
    main()
