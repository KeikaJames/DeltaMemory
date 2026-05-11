"""Exp16 — Layer-group site map.

Exp15 showed BINDING_DIRECTIONAL: V dominates, K-gating doesn't show at the
whole-stack injection. Exp16 asks: is there a layer subset where K-gating
recovers — i.e., where Kc_Vc clearly beats Kr_Vc?

Procedure: split the Qwen3-4B stack into 4 quartiles (Q1..Q4). For each
quartile, mask the oracle bank to inject only at those layers, and evaluate
Kc_Vc / Kc_Vr / Kr_Vc / Kr_Vr at the best alpha from Exp15 (default 0.005).

Output: cells.jsonl with one row per (seed, fact, quartile, variant).
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
    AttnNativePatcher, fresh_bank, write_fact,
)
from deltamemory.memory.anb_addressed import (
    subbank_correct, subbank_random, subbank_swap_KV, subbank_mask_layers,
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


def quartile_layers(num_layers):
    # Returns dict name->list[int]. Q1 lowest, Q4 highest.
    q = num_layers // 4
    spans = {
        "Q1": list(range(0, q)),
        "Q2": list(range(q, 2 * q)),
        "Q3": list(range(2 * q, 3 * q)),
        "Q4": list(range(3 * q, num_layers)),
    }
    return spans


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
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    cf = load_counterfact(Path(args.counterfact))
    kept_rows, dropped = filter_cf_for_tokenizer(cf, tok)
    rows = []
    for r in kept_rows:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        r = dict(r); r["write_prompt"] = wp; rows.append(r)
    if args.n > 0:
        rows = rows[: args.n]
    n_eligible = len(rows)
    print(f"[exp16] eligible rows = {n_eligible} (dropped {dropped} pre-filter)")

    patcher = AttnNativePatcher(model); patcher.install()
    num_layers = patcher.num_layers
    spans = quartile_layers(num_layers)
    print(f"[exp16] num_layers={num_layers} quartiles={ {k: f'{v[0]}-{v[-1]}' for k,v in spans.items()} }")

    manifest = {
        "experiment": "exp16_site_map",
        "model": args.model, "dtype": args.dtype, "device": args.device,
        "site": args.site, "alpha": alpha, "seeds": seeds,
        "n_eligible": n_eligible, "counterfact": args.counterfact,
        "num_layers": num_layers, "quartiles": spans,
        "torch_version": torch.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_env_json(out_dir,
                   prereg_version="exp13.prereg.v1",
                   dataset_sha1=sha1_of(Path(args.counterfact)),
                   device=args.device, dtype=args.dtype,
                   extra={"n_eligible": n_eligible,
                          "experiment": "exp16_site_map",
                          "site": args.site, "alpha": alpha,
                          "num_layers": num_layers})

    rows_path = out_dir / "cells.jsonl"
    rows_path.unlink(missing_ok=True)
    fout = rows_path.open("a")
    t0 = time.time()

    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)
        full, kept_ids = build_full_bank(patcher, tok, rows, args.site)
        print(f"[exp16] seed={seed} kept {len(kept_ids)}/{n_eligible}", flush=True)
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

            b_Kc = subbank_correct(full, fid)
            try:
                b_rand = subbank_random(full, rng, k=1, exclude=(fid,))
            except ValueError:
                continue

            # Pre-build the 4 KV combos at full layers.
            full_cells = {
                "Kc_Vc": b_Kc,
                "Kc_Vr": subbank_swap_KV(bank_K_source=b_Kc, bank_V_source=b_rand),
                "Kr_Vc": subbank_swap_KV(bank_K_source=b_rand, bank_V_source=b_Kc),
                "Kr_Vr": b_rand,
            }

            # base once per row.
            base = eval_with_bank(model, tok, patcher, query, target_new,
                                  target_true, args.device, None, 0.0)
            fout.write(json.dumps({"seed": seed, "fact_id": fid,
                                   "site": args.site, "quartile": "none",
                                   "alpha": 0.0, "variant": "base",
                                   **base}) + "\n")

            for qname, qlayers in spans.items():
                for vname, bnk in full_cells.items():
                    masked = subbank_mask_layers(bnk, qlayers)
                    m = eval_with_bank(model, tok, patcher, query, target_new,
                                       target_true, args.device, masked, alpha)
                    fout.write(json.dumps({"seed": seed, "fact_id": fid,
                                           "site": args.site, "quartile": qname,
                                           "alpha": alpha, "variant": vname,
                                           **m}) + "\n")
                    del masked
                gc.collect()
            del b_Kc, b_rand, full_cells
            gc.collect()
            fout.flush()
        del full; gc.collect()
        if args.device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    fout.close(); patcher.remove()
    dt = time.time() - t0
    print(f"[exp16] done in {dt:.1f}s -> {rows_path}")


if __name__ == "__main__":
    main()
