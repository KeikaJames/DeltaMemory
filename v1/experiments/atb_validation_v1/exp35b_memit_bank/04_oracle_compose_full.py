"""Exp35b — 04: Oracle composition Φ1 + D5 permuted-bank + D11 stratified distractors.

For each test fact i (full test set, not solo-pass-filtered: D1), patch a budget
of k facts and measure:
  target_margin   — i's read prompts (Gate B)
  shuffled_margin — i's read prompts when i is replaced by random rank-1 (Gate D)
  permuted_margin — i's read prompts when i is replaced by bank[π(i)] — D5
  distractor_margins — locality (C5/D11)

Sweeps k ∈ {1, 10, 100, 1000}, seeds {0, 1, 2}.

D11: distractor sampling stratified by relation, drawn from train+val excluding
the target's own relation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
sys.path.insert(0, str(EXP35))
import importlib.util as _u
_spec = _u.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = _u.module_from_spec(_spec); _spec.loader.exec_module(_bb)

first_target_id = _bb.first_target_id
apply_factors = _bb.apply_factors
restore = _bb.restore
margin_at_last = _bb.margin_at_last
assert_bit_equal = _bb.assert_bit_equal


def margins_for_fact(model, tok, row, t_new, t_true):
    canon = row["prompt"].format(row["subject"])
    paraphrases = list(row.get("paraphrase_prompts", []))[:2]
    read_prompts = [canon] + paraphrases
    margins = [margin_at_last(model, tok, p, t_new, t_true) for p in read_prompts]
    return margins, sum(margins) / len(margins)


def random_unit_rank1(d_out, d_in, dtype, device, gen, target_norm):
    b = torch.randn(d_out, generator=gen, dtype=torch.float32)
    a = torch.randn(d_in, generator=gen, dtype=torch.float32)
    cur = float((b.norm() * a.norm()).item())
    s = (target_norm / max(cur, 1e-8)) ** 0.5
    return (b * s).to(dtype).to(device), (a * s).to(dtype).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(DATA / "bank.pt"))
    ap.add_argument("--n-test-eval", type=int, default=1500)
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 10, 100, 1000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-locality-per-k", type=int, default=5)
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35b"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[load] bank {args.bank}", flush=True)
    bank_blob = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank_blob["entries"]
    print(f"[load] {len(entries)} facts in bank", flush=True)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def get_factor(fid):
        e = entries[fid]
        return (e["b"].to(device, dtype=dtype),
                e["a"].to(device, dtype=dtype))

    # Splits
    train = json.load(open(DATA / "splits" / "train.json"))
    val = json.load(open(DATA / "splits" / "val.json"))
    test = json.load(open(DATA / "splits" / "test.json"))
    all_rows = {r["id"]: r for r in train + val + test}

    pool_fids = [r["id"] for r in train + val if r["id"] in entries
                 and not entries[r["id"]].get("norm_outlier", False)]
    test_fids = [r["id"] for r in test if r["id"] in entries
                 and not entries[r["id"]].get("norm_outlier", False)][: args.n_test_eval]

    # D11: relation buckets in pool
    pool_by_rel = defaultdict(list)
    for fid in pool_fids:
        pool_by_rel[entries[fid]["relation"]].append(fid)
    relations = sorted(pool_by_rel.keys())
    print(f"[D11] {len(pool_fids)} pool facts across {len(relations)} relations", flush=True)
    print(f"[eval] {len(test_fids)} test facts (full test, NOT solo-pass filtered = D1)", flush=True)

    W_ref = model.model.layers[args.edit_layer].mlp.down_proj.weight.data.clone()

    # --- base margins precompute ---
    base_fids = list(set(test_fids) | set(pool_fids))
    base = {}
    base_path = out / "base_margins.pt"
    if base_path.exists():
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        print(f"[base] resumed {len(base)} cached", flush=True)
    print(f"[base] precomputing {len(base_fids)} base margins ...", flush=True)
    t0 = time.time()
    for n, fid in enumerate(base_fids):
        if fid in base:
            continue
        e = entries[fid]; row = all_rows[fid]
        t_new = first_target_id(tok, e["target_new"])
        t_true = first_target_id(tok, e["target_true"])
        margins, mean = margins_for_fact(model, tok, row, t_new, t_true)
        base[fid] = {"margins": margins, "mean": mean,
                     "t_new": t_new, "t_true": t_true}
        if (n + 1) % 500 == 0:
            torch.save(base, base_path)
            print(f"  base {n+1}/{len(base_fids)} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(base, base_path)
    assert_bit_equal(model, args.edit_layer, W_ref)

    # --- main sweep ---
    rows_path = out / "phi1_cells.jsonl"
    rows = []
    done_keys: set = set()
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            rows.append(r)
            done_keys.add((r["seed"], r["k"], r["fact_id"]))
        print(f"[sweep] resumed {len(rows)} cells", flush=True)

    rows_fp = open(rows_path, "a")

    def sample_distractors(target_rel, k, gen):
        """Relation-stratified: pick from all relations != target, then top-up."""
        other_rels = [r for r in relations if r != target_rel]
        per_rel = max(1, (k - 1) // max(1, len(other_rels)))
        out_ids: list = []
        for r in other_rels:
            bucket = pool_by_rel[r]
            idx = torch.randperm(len(bucket), generator=gen).tolist()[:per_rel]
            out_ids.extend(bucket[j] for j in idx)
            if len(out_ids) >= k - 1:
                break
        # Top-up if short (e.g., few small relations)
        while len(out_ids) < k - 1:
            r = relations[torch.randint(0, len(relations), (1,), generator=gen).item()]
            bucket = pool_by_rel[r]
            j = torch.randint(0, len(bucket), (1,), generator=gen).item()
            cand = bucket[j]
            if cand not in out_ids:
                out_ids.append(cand)
        return out_ids[: k - 1]

    for seed in args.seeds:
        for k in args.k_values:
            print(f"\n[Φ1] seed={seed}  k={k}", flush=True)
            gen = torch.Generator().manual_seed(1000 + seed * 100 + k)
            # Permutation π for D5 (bank-permuted control)
            perm_idx = torch.randperm(len(test_fids), generator=gen).tolist()
            perm_map = {test_fids[i]: test_fids[perm_idx[i]] for i in range(len(test_fids))}

            t0 = time.time()
            for ti, fid in enumerate(test_fids):
                if (seed, k, fid) in done_keys:
                    continue
                e = entries[fid]; row = all_rows[fid]
                t_new = base[fid]["t_new"]; t_true = base[fid]["t_true"]

                distractor_fids = sample_distractors(e["relation"], k, gen)
                factors_correct = [get_factor(fid)] + [get_factor(d) for d in distractor_fids]

                # variant A: correct
                W_old = apply_factors(model, args.edit_layer, factors_correct)
                try:
                    _, target_mean = margins_for_fact(model, tok, row, t_new, t_true)
                    locality_shifts = []
                    n_loc = min(args.n_locality_per_k, len(distractor_fids))
                    for dj in distractor_fids[:n_loc]:
                        drow = all_rows[dj]
                        dt_new = base[dj]["t_new"]; dt_true = base[dj]["t_true"]
                        _, dj_mean = margins_for_fact(model, tok, drow, dt_new, dt_true)
                        locality_shifts.append({
                            "id": dj,
                            "shift": dj_mean - base[dj]["mean"],
                        })
                finally:
                    restore(model, args.edit_layer, W_old)
                assert_bit_equal(model, args.edit_layer, W_ref)

                # variant B: shuffled (random rank-1 in place of i)
                med_norm = max(1e-3, e["delta_norm"])
                d_out, d_in = W_ref.shape
                b_r, a_r = random_unit_rank1(d_out, d_in, dtype, device, gen, med_norm)
                shuf_factors = [(b_r, a_r)] + [get_factor(d) for d in distractor_fids]
                W_old = apply_factors(model, args.edit_layer, shuf_factors)
                try:
                    _, shuffled_mean = margins_for_fact(model, tok, row, t_new, t_true)
                finally:
                    restore(model, args.edit_layer, W_old)
                assert_bit_equal(model, args.edit_layer, W_ref)

                # variant C: permuted (bank-permuted — D5)
                # replace i's factor with another test-fact's factor
                perm_fid = perm_map[fid]
                if perm_fid == fid and len(test_fids) > 1:
                    perm_fid = test_fids[(test_fids.index(fid) + 1) % len(test_fids)]
                perm_factors = [get_factor(perm_fid)] + [get_factor(d) for d in distractor_fids]
                W_old = apply_factors(model, args.edit_layer, perm_factors)
                try:
                    _, permuted_mean = margins_for_fact(model, tok, row, t_new, t_true)
                finally:
                    restore(model, args.edit_layer, W_old)
                assert_bit_equal(model, args.edit_layer, W_ref)

                r = {
                    "seed": seed, "k": k, "fact_id": fid,
                    "base_mean": base[fid]["mean"],
                    "target_mean": target_mean,
                    "shuffled_mean": shuffled_mean,
                    "permuted_mean": permuted_mean,
                    "permuted_donor": perm_fid,
                    "uplift": target_mean - base[fid]["mean"],
                    "gate_d_diff": target_mean - shuffled_mean,
                    "d5_diff": target_mean - permuted_mean,
                    "locality": locality_shifts,
                }
                rows.append(r); done_keys.add((seed, k, fid))
                rows_fp.write(json.dumps(r) + "\n"); rows_fp.flush()

                if (ti + 1) % 100 == 0:
                    recent = [x for x in rows if x["seed"] == seed and x["k"] == k][-100:]
                    print(f"  {ti+1}/{len(test_fids)} "
                          f"uplift={sum(x['uplift'] for x in recent)/max(1,len(recent)):+.2f} "
                          f"gateD={sum(x['gate_d_diff'] for x in recent)/max(1,len(recent)):+.2f} "
                          f"D5={sum(x['d5_diff'] for x in recent)/max(1,len(recent)):+.2f} "
                          f"({time.time()-t0:.0f}s)", flush=True)

    rows_fp.close()

    # aggregate per (k)
    summary = {}
    for k in args.k_values:
        cur = [r for r in rows if r["k"] == k]
        if not cur: continue
        n = len(cur)
        loc_shifts = [abs(s["shift"]) for r in cur for s in r["locality"]]
        summary[f"k{k}"] = {
            "n_obs": n,
            "mean_uplift_nats": sum(r["uplift"] for r in cur) / n,
            "mean_gate_d_diff_nats": sum(r["gate_d_diff"] for r in cur) / n,
            "mean_d5_diff_nats": sum(r["d5_diff"] for r in cur) / n,
            "frac_target_beats_base": sum(1 for r in cur if r["uplift"] > 0) / n,
            "frac_target_beats_shuffled": sum(1 for r in cur if r["gate_d_diff"] > 0) / n,
            "frac_target_beats_permuted": sum(1 for r in cur if r["d5_diff"] > 0) / n,
            "locality_median_abs_shift_nats": sorted(loc_shifts)[len(loc_shifts)//2] if loc_shifts else 0.0,
        }
    json.dump(summary, open(out / "phi1_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
