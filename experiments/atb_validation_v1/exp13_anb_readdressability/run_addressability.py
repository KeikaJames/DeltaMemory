"""Exp13 — ANB QK addressability smoke runner.

Goal: prove (or disprove) that the per-layer write-time key M_K[l, ·, fact_i]
is naturally re-addressable by the per-layer read-time query Q.

This runner records Q with NO value injection (alpha=0; the bank is never
attached to the patcher), then scores Q against M_K offline. It compares
the correct fact against three negative-control banks:

  * shuffle_layer  — M_K and M_V layer indices permuted
  * shuffle_V      — K unchanged; V slot order permuted (sanity that QK is K-only)
  * random_K_only  — M_K replaced by RMS-matched Gaussian noise; M_V kept

Output: rows.jsonl, summary.json (call analyze.py to add bootstrap + verdict).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.anb_diagnostics import (
    record_read_queries,
    score_query_against_bank,
    rank_correct,
    hard_negative_win_rate,
)
from deltamemory.memory.anb_addressed import (
    subbank_shuffle_layer,
    subbank_shuffle_V,
)
from deltamemory.memory.anb_capture_sweep import (
    resolve_extended_capture,
    derive_relation_phrase,
)

# Add repo root + experiments lib to path.
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from atb_validation_v1._lib import (  # noqa: E402
    load_model,
    load_counterfact,
    filter_cf_for_tokenizer,
    seed_everything,
)
from atb_validation_v1._lib.cf_runner import (  # noqa: E402
    render_query,
    relation_phrase,
    build_write_prompt,
)
from atb_validation_v1._lib.hard_negatives import build_hard_negatives  # noqa: E402
from tools.env_writer import write_env_json, sha1_of  # noqa: E402


def make_random_k_bank(src):
    """Clone src bank; replace each layer's M_K with RMS-matched Gaussian noise.

    M_V untouched. Slot order / fact_ids preserved so scoring stays comparable.
    """
    import copy as _copy
    out = _copy.copy(src)
    out.M_K = []
    out.M_V = [v.clone() for v in src.M_V]
    out.fact_ids = list(src.fact_ids)
    out.address_strs = list(src.address_strs)
    gen = torch.Generator(device="cpu").manual_seed(0xBEEF)
    for k in src.M_K:
        if k.numel() == 0:
            out.M_K.append(k.clone())
            continue
        rms = float(k.float().pow(2).mean().sqrt().item()) or 1e-3
        noise = torch.randn(k.shape, generator=gen, dtype=torch.float32)
        noise = noise / noise.float().pow(2).mean().sqrt().clamp_min(1e-8)
        noise = noise * rms
        out.M_K.append(noise.to(k.device, dtype=k.dtype).contiguous())
    return out


def resolve_capture_pos(site, row, tok, write_prompt, add_special=True):
    """Return integer token position for `site`, or None if unresolvable."""
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


def build_bank_at_site(patcher, tok, rows, site):
    """Write all rows into a fresh bank, capturing at `site` per row.

    Returns (bank, kept_fact_ids).  Rows for which the site is unresolvable
    are skipped (their fact_id will not appear in this bank).
    """
    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"
    bank.bank_key_mode = "pre_rope"
    kept = []
    for row in rows:
        wp = row["write_prompt"]
        if site == "period":
            cap_pos = None  # write_fact defaults to last real token
        else:
            cap_pos = resolve_capture_pos(site, row, tok, wp)
            if cap_pos is None:
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


def build_read_queries(row):
    """Three read-query variants for one CF row."""
    query = render_query(row)
    subject = row.get("subject") or ""
    paraphrase = None
    pps = row.get("paraphrase_prompts") or []
    if pps:
        paraphrase = pps[0]
    return {
        "query": query,
        "subject_only": subject,
        "paraphrase": paraphrase or query,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--counterfact",
                    default="experiments/datasets/counterfact_1k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=21,
                    help="max number of eligible rows to use (after filter)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sites", default="period",
                    help="comma-separated capture sites (period,subject_last,object_last,...)")
    ap.add_argument("--queries", default="query,subject_only,paraphrase")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    qkinds = [s.strip() for s in args.queries.split(",") if s.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    cf_rows = load_counterfact(Path(args.counterfact))
    kept, dropped = filter_cf_for_tokenizer(cf_rows, tok)
    # Attach write prompts; drop rows where build_write_prompt returns None.
    rows = []
    for r in kept:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        r = dict(r)
        r["write_prompt"] = wp
        rows.append(r)
    if args.n > 0:
        rows = rows[: args.n]
    n_eligible = len(rows)
    print(f"[exp13] eligible rows = {n_eligible} (dropped {dropped} pre-filter)")

    hn_index = build_hard_negatives(rows)

    manifest = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "counterfact": args.counterfact,
        "n_eligible": n_eligible,
        "seeds": seeds,
        "sites": sites,
        "queries": qkinds,
        "torch_version": torch.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Authenticity contract — see docs/authenticity.md.
    write_env_json(
        out_dir,
        prereg_version="exp13.prereg.v1",
        dataset_sha1=sha1_of(Path(args.counterfact)),
        device=args.device,
        dtype=args.dtype,
        extra={"n_eligible": n_eligible, "experiment": "exp13_anb_readdressability"},
    )

    patcher = AttnNativePatcher(model)
    patcher.install()

    rows_path = out_dir / "cells.jsonl"
    rows_path.unlink(missing_ok=True)
    fout = rows_path.open("a")

    t0 = time.time()

    for seed in seeds:
        seed_everything(seed)
        seed_rng = torch.Generator().manual_seed(seed)

        # Build correct bank at each requested site.
        site_banks = {}
        for site in sites:
            print(f"[exp13] seed={seed} building bank @ site={site} ...",
                  flush=True)
            bank, kept_ids = build_bank_at_site(patcher, tok, rows, site)
            site_banks[site] = (bank, kept_ids)
            print(f"          kept {len(kept_ids)}/{n_eligible} slots", flush=True)

        # For each test row + read-query, record Q once, score against all banks.
        for row in rows:
            fid = str(row["id"])
            read_queries = build_read_queries(row)
            for qkind in qkinds:
                read_prompt = read_queries.get(qkind)
                if not read_prompt:
                    continue
                # Record Q once for this (row, qkind).
                probe = record_read_queries(patcher, tok, read_prompt,
                                            capture_pos=None)

                for site, (bank, kept_ids) in site_banks.items():
                    if fid not in kept_ids:
                        # Correct fact wasn't captured at this site.
                        out_row = {
                            "seed": seed, "fact_id": fid, "site": site,
                            "query_kind": qkind, "variant": "skipped",
                            "reason": "fid_not_in_bank",
                        }
                        fout.write(json.dumps(out_row) + "\n")
                        continue

                    # ----- correct_in_bank
                    scores_correct = score_query_against_bank(
                        probe, bank, patcher, mode="pre_rope")
                    rk_correct = rank_correct(scores_correct, kept_ids, fid,
                                              reduce="max_layer")
                    # Hard negatives only meaningful on the correct bank.
                    hn_subj = hard_negative_win_rate(
                        scores_correct, kept_ids, fid,
                        hn_index.neighbors(row, "same_subject_wrong_object"))
                    hn_rel = hard_negative_win_rate(
                        scores_correct, kept_ids, fid,
                        hn_index.neighbors(row, "same_relation_wrong_subject"))
                    hn_obj = hard_negative_win_rate(
                        scores_correct, kept_ids, fid,
                        hn_index.neighbors(row, "same_object_wrong_subject"))
                    out_row = {
                        "seed": seed, "fact_id": fid, "site": site,
                        "query_kind": qkind, "variant": "correct_in_bank",
                        **rk_correct,
                        "hn_same_subject": hn_subj,
                        "hn_same_relation": hn_rel,
                        "hn_same_object": hn_obj,
                        "bank_size": len(kept_ids),
                    }
                    fout.write(json.dumps(out_row) + "\n")

                    # ----- shuffle_layer (permutation depends on seed_rng so
                    # we re-create here to keep per-row determinism).
                    bank_sl = subbank_shuffle_layer(bank, seed_rng)
                    s_sl = score_query_against_bank(probe, bank_sl, patcher,
                                                    mode="pre_rope")
                    rk_sl = rank_correct(s_sl, kept_ids, fid,
                                         reduce="max_layer")
                    fout.write(json.dumps({
                        "seed": seed, "fact_id": fid, "site": site,
                        "query_kind": qkind, "variant": "shuffle_layer",
                        **rk_sl, "bank_size": len(kept_ids),
                    }) + "\n")

                    # ----- shuffle_V (V is irrelevant for QK so this should
                    # match correct_in_bank; sanity check).
                    bank_sv = subbank_shuffle_V(bank, seed_rng)
                    s_sv = score_query_against_bank(probe, bank_sv, patcher,
                                                    mode="pre_rope")
                    rk_sv = rank_correct(s_sv, kept_ids, fid,
                                         reduce="max_layer")
                    fout.write(json.dumps({
                        "seed": seed, "fact_id": fid, "site": site,
                        "query_kind": qkind, "variant": "shuffle_V",
                        **rk_sv, "bank_size": len(kept_ids),
                    }) + "\n")

                    # ----- random_K_only
                    bank_rk = make_random_k_bank(bank)
                    s_rk = score_query_against_bank(probe, bank_rk, patcher,
                                                    mode="pre_rope")
                    rk_rk = rank_correct(s_rk, kept_ids, fid,
                                         reduce="max_layer")
                    fout.write(json.dumps({
                        "seed": seed, "fact_id": fid, "site": site,
                        "query_kind": qkind, "variant": "random_K_only",
                        **rk_rk, "bank_size": len(kept_ids),
                    }) + "\n")

                    del bank_sl, bank_sv, bank_rk
                    gc.collect()

                fout.flush()

        # Free banks before next seed.
        for site, (bank, _) in site_banks.items():
            del bank
        site_banks.clear()
        gc.collect()
        if args.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    fout.close()
    patcher.remove()
    dt = time.time() - t0
    print(f"[exp13] done in {dt:.1f}s -> {rows_path}")


if __name__ == "__main__":
    main()
