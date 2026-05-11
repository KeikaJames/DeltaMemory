"""Exp14 — Oracle KV injection causality.

Exp13 established QK addressability (correct M_K is naturally rankable by Q at
the subject_last capture site). Exp14 asks the next question: when we inject
the *value* `M_V` along with the correct `M_K` at small alpha, does the model's
prediction shift toward `target_new` more than it does under matched controls?

Per row, build a single-slot oracle bank captured at the QK-best site
(default: subject_last). Evaluate the read query under five variants:

  * base                       — alpha = 0 (sanity)
  * oracle_correct_KV          — K and V from the correct fact
  * oracle_random_KV           — K and V from a different (random) fact
  * oracle_shuffled_layer_KV   — correct K/V but with layer indices permuted
  * oracle_KcorrectVrandom     — correct K, V from another fact (K/V binding probe)

Sweep alpha ∈ {0.005, 0.01, 0.02, 0.05}.

Verdict gate (PREREG; implemented in analyze.py):
  If ``margin(oracle_correct_KV) ≤ max(margin over all controls)`` for every
  alpha in the sweep, mark Exp14 FAIL and skip Exp15/Exp18.
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
    forward_with_bank,
    write_fact,
)
from deltamemory.memory.anb_addressed import (
    subbank_correct,
    subbank_random,
    subbank_shuffle_layer,
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
        site=site,
        write_prompt=write_prompt,
        subject=row.get("subject", ""),
        relation_phrase=rel,
        object_str=row.get("target_new", ""),
        tokenizer=tok,
        attention_mask_row=am,
        add_special_tokens=add_special,
    )
    if spec is None or not spec.token_positions:
        return None
    return spec.token_positions[-1]


def build_full_bank(patcher, tok, rows, site):
    """Write every row into one bank at `site`; return (bank, kept_ids)."""
    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"
    bank.bank_key_mode = "pre_rope"
    kept = []
    for row in rows:
        wp = row["write_prompt"]
        cap_pos = resolve_capture_pos(site, row, tok, wp)
        if site != "period" and cap_pos is None:
            continue
        write_fact(
            patcher, bank, tok,
            write_prompt=wp,
            fact_id=str(row["id"]),
            address=row.get("subject"),
            capture_pos=cap_pos,
        )
        kept.append(str(row["id"]))
    return bank, kept


@torch.no_grad()
def eval_with_bank(model, tok, patcher, prompt, target_new, target_true,
                   device, bank, alpha):
    """Margin under a (possibly empty / alpha=0) injected bank."""
    if bank is None or alpha == 0.0:
        ctx = None
    else:
        ctx = patcher.injecting(bank=bank, alpha=float(alpha))
    if ctx is not None:
        with patcher.patched(), ctx:
            logp_new, ids_new = continuation_logp(model, tok, prompt,
                                                  target_new, device)
            logp_true, _ = continuation_logp(model, tok, prompt,
                                             target_true, device)
            tnf = ids_new[0] if ids_new else -1
            rank, _ = first_token_rank(model, tok, prompt, tnf, device)
    else:
        logp_new, ids_new = continuation_logp(model, tok, prompt,
                                              target_new, device)
        logp_true, _ = continuation_logp(model, tok, prompt,
                                         target_true, device)
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
    ap.add_argument("--site", default="subject_last",
                    help="capture site (QK-best from Exp13 = subject_last)")
    ap.add_argument("--alphas", default="0.005,0.01,0.02,0.05")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    cf_rows = load_counterfact(Path(args.counterfact))
    kept_rows, dropped = filter_cf_for_tokenizer(cf_rows, tok)
    rows = []
    for r in kept_rows:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        r = dict(r)
        r["write_prompt"] = wp
        rows.append(r)
    if args.n > 0:
        rows = rows[: args.n]
    n_eligible = len(rows)
    print(f"[exp14] eligible rows = {n_eligible} (dropped {dropped} pre-filter)")

    manifest = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "site": args.site,
        "alphas": alphas,
        "seeds": seeds,
        "n_eligible": n_eligible,
        "counterfact": args.counterfact,
        "torch_version": torch.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_env_json(
        out_dir,
        prereg_version="exp13.prereg.v1",
        dataset_sha1=sha1_of(Path(args.counterfact)),
        device=args.device,
        dtype=args.dtype,
        extra={"n_eligible": n_eligible,
               "experiment": "exp14_oracle_addressed",
               "site": args.site, "alphas": alphas},
    )

    patcher = AttnNativePatcher(model)
    patcher.install()

    rows_path = out_dir / "cells.jsonl"
    rows_path.unlink(missing_ok=True)
    fout = rows_path.open("a")

    t0 = time.time()

    for seed in seeds:
        seed_everything(seed)
        rng = torch.Generator().manual_seed(seed)

        # Single multi-slot bank at the chosen site; carved per-row below.
        print(f"[exp14] seed={seed} building bank @ site={args.site} ...",
              flush=True)
        full_bank, kept_ids = build_full_bank(patcher, tok, rows, args.site)
        print(f"          kept {len(kept_ids)}/{n_eligible} slots", flush=True)
        kept_set = set(kept_ids)

        for row in rows:
            fid = str(row["id"])
            if fid not in kept_set:
                fout.write(json.dumps({
                    "seed": seed, "fact_id": fid, "site": args.site,
                    "variant": "skipped", "reason": "fid_not_in_bank",
                }) + "\n")
                continue

            target_new = row["target_new"]
            target_true = row.get("target_true") or row.get("target_old") or ""
            if not target_true:
                fout.write(json.dumps({
                    "seed": seed, "fact_id": fid, "site": args.site,
                    "variant": "skipped", "reason": "no_target_true",
                }) + "\n")
                continue
            query = render_query(row)

            # Build oracle banks once per row.
            b_correct = subbank_correct(full_bank, fid)
            try:
                b_random = subbank_random(full_bank, rng, k=1, exclude=(fid,))
            except ValueError:
                b_random = None
            b_shuf_L = subbank_shuffle_layer(b_correct, rng)
            # K-correct, V-random binding probe.
            if b_random is not None:
                b_kcvr = subbank_swap_KV(bank_K_source=b_correct,
                                         bank_V_source=b_random)
            else:
                b_kcvr = None

            # base (alpha=0): bank-free.
            mb = eval_with_bank(model, tok, patcher, query, target_new,
                                target_true, args.device, None, 0.0)
            fout.write(json.dumps({
                "seed": seed, "fact_id": fid, "site": args.site,
                "alpha": 0.0, "variant": "base", **mb,
            }) + "\n")

            for alpha in alphas:
                variants = [
                    ("oracle_correct_KV", b_correct),
                    ("oracle_random_KV", b_random),
                    ("oracle_shuffled_layer_KV", b_shuf_L),
                    ("oracle_KcorrectVrandom", b_kcvr),
                ]
                for vname, bnk in variants:
                    if bnk is None:
                        fout.write(json.dumps({
                            "seed": seed, "fact_id": fid, "site": args.site,
                            "alpha": alpha, "variant": vname,
                            "reason": "no_eligible_random",
                        }) + "\n")
                        continue
                    m = eval_with_bank(model, tok, patcher, query, target_new,
                                       target_true, args.device, bnk, alpha)
                    fout.write(json.dumps({
                        "seed": seed, "fact_id": fid, "site": args.site,
                        "alpha": alpha, "variant": vname, **m,
                    }) + "\n")

            del b_correct, b_random, b_shuf_L, b_kcvr
            gc.collect()
            fout.flush()

        del full_bank
        gc.collect()
        if args.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    fout.close()
    patcher.remove()
    dt = time.time() - t0
    print(f"[exp14] done in {dt:.1f}s -> {rows_path}")


if __name__ == "__main__":
    main()
