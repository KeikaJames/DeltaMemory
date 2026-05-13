"""Exp18 — Natural addressed ANB (full-bank softmax routing).

Up through Exp17 we always injected a 1-slot oracle bank carved from the
correct fact. Exp18 asks the deployment-relevant question: when we attach the
FULL multi-slot bank and let the bank-softmax route naturally, does the
correct fact's slot get up-weighted on its own — i.e. does the trained
attention naturally pick the right address?

Variants per row (alpha=0.005, sites relation_last and subject_last):

  * base                   — no bank
  * full_bank              — all 21 slots, correct fact present
  * full_bank_shuffled_L   — same bank, layer indices permuted
  * full_bank_random_K     — K replaced by RMS-matched noise on all slots
  * minus_correct_bank     — full bank with correct fact's slot zeroed
                              (does removing the addressee hurt?)
"""
from __future__ import annotations

import argparse
import copy as _copy
import gc
import json
import sys
import time
from pathlib import Path

import torch

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher, fresh_bank, write_fact,
)
from deltamemory.memory.anb_addressed import (
    subbank_shuffle_layer,
)
from deltamemory.memory.anb_capture_sweep import (
    resolve_extended_capture, derive_relation_phrase,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (  # noqa: E402
    load_model, load_counterfact, filter_cf_for_tokenizer,
    seed_everything, continuation_logp, first_token_rank,
)
from atb_validation_v1._lib.cf_runner import render_query, build_write_prompt  # noqa: E402
from tools.env_writer import write_env_json, sha1_of  # noqa: E402


def resolve_capture_pos(site, row, tok, wp, add_special=True):
    if site == "period":
        return None
    enc = tok(wp, return_tensors="pt", add_special_tokens=add_special)
    am = enc["attention_mask"][0]
    rel = derive_relation_phrase(row.get("prompt", ""))
    spec = resolve_extended_capture(
        site=site, write_prompt=wp,
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


def make_random_k_bank(src):
    out = _copy.copy(src)
    out.M_K = []
    out.M_V = [v.clone() for v in src.M_V]
    out.fact_ids = list(src.fact_ids)
    out.address_strs = list(src.address_strs)
    gen = torch.Generator(device="cpu").manual_seed(0xBEEF)
    for k in src.M_K:
        if k.numel() == 0:
            out.M_K.append(k.clone()); continue
        rms = float(k.float().pow(2).mean().sqrt().item()) or 1e-3
        noise = torch.randn(k.shape, generator=gen, dtype=torch.float32)
        noise = noise / noise.float().pow(2).mean().sqrt().clamp_min(1e-8)
        noise = noise * rms
        out.M_K.append(noise.to(k.device, dtype=k.dtype).contiguous())
    return out


def make_minus_correct_bank(src, fid):
    """Clone bank; zero only the correct-fact slot at all layers."""
    if fid not in src.fact_ids:
        return None
    i = src.fact_ids.index(fid)
    out = _copy.copy(src)
    out.M_K = [k.clone() for k in src.M_K]
    out.M_V = [v.clone() for v in src.M_V]
    out.fact_ids = list(src.fact_ids)
    out.address_strs = list(src.address_strs)
    for layer in range(out.num_layers):
        if out.M_K[layer].numel() == 0:
            continue
        out.M_K[layer][i, ...] = 0
        out.M_V[layer][i, ...] = 0
    return out


@torch.no_grad()
def eval_with_bank(model, tok, patcher, prompt, target_new, target_true,
                   device, bank, alpha):
    ctx = patcher.injecting(bank=bank, alpha=float(alpha)) if (bank is not None and alpha > 0) else None
    if ctx is not None:
        with patcher.patched(), ctx:
            lpn, idn = continuation_logp(model, tok, prompt, target_new, device)
            lpt, _ = continuation_logp(model, tok, prompt, target_true, device)
            tnf = idn[0] if idn else -1
            rk, _ = first_token_rank(model, tok, prompt, tnf, device)
    else:
        lpn, idn = continuation_logp(model, tok, prompt, target_new, device)
        lpt, _ = continuation_logp(model, tok, prompt, target_true, device)
        tnf = idn[0] if idn else -1
        rk, _ = first_token_rank(model, tok, prompt, tnf, device)
    return {"target_new_logprob": float(lpn),
            "target_true_logprob": float(lpt),
            "margin": float(lpn - lpt),
            "target_rank": int(rk),
            "recall_at_1": bool(rk == 0)}


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
    ap.add_argument("--sites", default="relation_last,subject_last")
    ap.add_argument("--alpha", type=float, default=0.005)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    alpha = float(args.alpha)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    cf = load_counterfact(Path(args.counterfact))
    kept_rows, dropped = filter_cf_for_tokenizer(cf, tok)
    rows = []
    for r in kept_rows:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None: continue
        r = dict(r); r["write_prompt"] = wp; rows.append(r)
    if args.n > 0: rows = rows[: args.n]
    n_eligible = len(rows)
    print(f"[exp18] eligible rows = {n_eligible} (dropped {dropped})")

    manifest = {"experiment": "exp18_natural_addressed",
                "model": args.model, "dtype": args.dtype, "device": args.device,
                "sites": sites, "alpha": alpha, "seeds": seeds,
                "n_eligible": n_eligible, "counterfact": args.counterfact,
                "torch_version": torch.__version__}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_env_json(out_dir, prereg_version="exp13.prereg.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"n_eligible": n_eligible,
                          "experiment": "exp18_natural_addressed",
                          "sites": sites, "alpha": alpha})

    patcher = AttnNativePatcher(model); patcher.install()
    rows_path = out_dir / "cells.jsonl"
    rows_path.unlink(missing_ok=True)
    fout = rows_path.open("a")
    t0 = time.time()

    for site in sites:
        for seed in seeds:
            seed_everything(seed)
            rng = torch.Generator().manual_seed(seed)
            full, kept_ids = build_full_bank(patcher, tok, rows, site)
            print(f"[exp18] site={site} seed={seed} kept {len(kept_ids)}/{n_eligible}", flush=True)
            kept_set = set(kept_ids)
            bank_shufL = subbank_shuffle_layer(full, rng)
            bank_randK = make_random_k_bank(full)

            for row in rows:
                fid = str(row["id"])
                if fid not in kept_set: continue
                target_new = row["target_new"]
                target_true = row.get("target_true") or row.get("target_old") or ""
                if not target_true: continue
                query = render_query(row)

                bank_minus = make_minus_correct_bank(full, fid)

                base = eval_with_bank(model, tok, patcher, query, target_new,
                                      target_true, args.device, None, 0.0)
                fout.write(json.dumps({"site": site, "seed": seed,
                                       "fact_id": fid, "alpha": 0.0,
                                       "variant": "base", **base}) + "\n")
                for vname, bnk in [("full_bank", full),
                                   ("full_bank_shuffled_L", bank_shufL),
                                   ("full_bank_random_K", bank_randK),
                                   ("minus_correct_bank", bank_minus)]:
                    if bnk is None: continue
                    m = eval_with_bank(model, tok, patcher, query, target_new,
                                       target_true, args.device, bnk, alpha)
                    fout.write(json.dumps({"site": site, "seed": seed,
                                           "fact_id": fid, "alpha": alpha,
                                           "variant": vname, **m}) + "\n")
                del bank_minus
                gc.collect()
                fout.flush()
            del full, bank_shufL, bank_randK
            gc.collect()
            if args.device == "mps":
                try: torch.mps.empty_cache()
                except Exception: pass

    fout.close(); patcher.remove()
    dt = time.time() - t0
    print(f"[exp18] done in {dt:.1f}s -> {rows_path}")


if __name__ == "__main__":
    main()
