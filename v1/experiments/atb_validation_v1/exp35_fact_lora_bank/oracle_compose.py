"""Exp35 Φ1 — Oracle composition scan.

For each test fact i (solo_pass only), patch a budget of k facts —
including i and k-1 distractors drawn from train+val — and measure:

  target_margin   — margin on i's read prompts (Gate B)
  shuffled_margin — margin on i's read prompts when i is REPLACED by
                    a random fact (out of pool) — Gate D / anti-cheat C2
  distractor_margins — margin on each j's read prompts (locality C5)

Sweep k ∈ {1, 2, 5, 10, 25, 50} × seeds {0,1,2}.

Plus k=0 bit-equal check (C9): no patch → margins == base. We compute
this once and use it as the base.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402
from build_bank import (  # noqa: E402
    first_target_id, apply_factors, restore, margin_at_last, assert_bit_equal,
)

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "exp31_learned_k_adapter" / "data" / "splits"


def margins_for_fact(model, tok, row, t_new, t_true):
    canon = row["prompt"].format(row["subject"])
    paraphrases = list(row.get("paraphrase_prompts", []))[:2]
    read_prompts = [canon] + paraphrases
    margins = [margin_at_last(model, tok, p, t_new, t_true) for p in read_prompts]
    return margins, sum(margins) / len(margins)


def random_unit_rank1(d_out, d_in, dtype, device, gen):
    """A controlled-norm random rank-1 (b,a). Norm matched to median."""
    b = torch.randn(d_out, generator=gen, dtype=torch.float32) * 0.3
    a = torch.randn(d_in, generator=gen, dtype=torch.float32) * 0.01
    return b.to(dtype).to(device), a.to(dtype).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(HERE / "bank.pt"))
    ap.add_argument("--n-test-eval", type=int, default=125)
    ap.add_argument("--k-values", type=int, nargs="+",
                    default=[1, 2, 5, 10, 25, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-locality-per-k", type=int, default=5,
                    help="number of distractors j to measure margin shift for")
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[load model] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bank_blob = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank_blob["entries"]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Move factors lazily; keep CPU master, send to device when patching.
    def get_factor(fid):
        e = entries[fid]
        return (e["b"].to(device, dtype=dtype),
                e["a"].to(device, dtype=dtype))

    train_pool = [fid for fid, e in entries.items()
                  if e["split"] in ("train", "val") and e["solo_pass"]
                  and not e.get("norm_outlier", False)]
    test_facts = [fid for fid, e in entries.items()
                  if e["split"] == "test" and e["solo_pass"]
                  and not e.get("norm_outlier", False)][: args.n_test_eval]
    print(f"distractor pool = {len(train_pool)}  test = {len(test_facts)}", flush=True)

    W_ref = model.model.layers[args.edit_layer].mlp.down_proj.weight.data.clone()

    # --- bit-equal sanity (C9, k=0) ---
    print("[Φ1 bit-equal k=0 sanity]", flush=True)
    pivot = test_facts[0]
    e0 = entries[pivot]
    t_new = first_target_id(tok, e0["target_new"])
    t_true = first_target_id(tok, e0["target_true"])
    base_margins, base_mean = margins_for_fact(model, tok,
        {"prompt": e0.get("prompt", ""), "subject": e0["subject"],
         "paraphrase_prompts": []}, t_new, t_true) if False else (None, None)
    # use original row from splits for richer prompts
    test_rows = {row["id"]: row for row in json.load(open(SPLITS / "test.json"))}
    train_rows = {row["id"]: row for row in json.load(open(SPLITS / "train.json"))}
    val_rows = {row["id"]: row for row in json.load(open(SPLITS / "val.json"))}
    all_rows = {**train_rows, **val_rows, **test_rows}

    # base margins (k=0 means no patch) — for test AND train_pool (locality)
    base = {}
    base_fids = list(set(test_facts) | set(train_pool))
    print(f"[Φ1] precomputing base margins for {len(base_fids)} facts "
          f"(test+pool, for locality)…", flush=True)
    t0_base = time.time()
    for n_done, fid in enumerate(base_fids):
        e = entries[fid]
        row = all_rows[fid]
        t_new = first_target_id(tok, e["target_new"])
        t_true = first_target_id(tok, e["target_true"])
        margins, mean = margins_for_fact(model, tok, row, t_new, t_true)
        base[fid] = {"margins": margins, "mean": mean,
                     "t_new": t_new, "t_true": t_true}
        if (n_done + 1) % 100 == 0:
            print(f"  base {n_done+1}/{len(base_fids)} ({time.time()-t0_base:.0f}s)",
                  flush=True)
    assert_bit_equal(model, args.edit_layer, W_ref)
    test_mean = sum(base[fid]["mean"] for fid in test_facts) / len(test_facts)
    print(f"  base mean margin over test = {test_mean:+.3f}", flush=True)

    # --- main sweep ---
    rows = []
    for seed in args.seeds:
        for k in args.k_values:
            print(f"\n[Φ1] seed={seed}  k={k}", flush=True)
            gen = torch.Generator().manual_seed(1000 + seed * 100 + k)
            t0 = time.time()
            for ti, fid in enumerate(test_facts):
                e = entries[fid]
                row = all_rows[fid]
                t_new = base[fid]["t_new"]
                t_true = base[fid]["t_true"]

                # sample k-1 distractors
                idx = torch.randperm(len(train_pool), generator=gen).tolist()[:k - 1]
                distractor_fids = [train_pool[j] for j in idx]
                factors = [get_factor(fid)] + [get_factor(d) for d in distractor_fids]

                # variant A: correct (i + distractors)
                W_old = apply_factors(model, args.edit_layer, factors)
                try:
                    target_margins, target_mean = margins_for_fact(
                        model, tok, row, t_new, t_true)
                    # locality: measure margin shift on first n_loc distractors
                    locality_shifts = []
                    n_loc = min(args.n_locality_per_k, len(distractor_fids))
                    for dj in distractor_fids[:n_loc]:
                        drow = all_rows[dj]
                        de = entries[dj]
                        dt_new = base[dj]["t_new"]
                        dt_true = base[dj]["t_true"]
                        _, dj_mean = margins_for_fact(model, tok, drow, dt_new, dt_true)
                        locality_shifts.append({
                            "id": dj,
                            "patched_mean": dj_mean,
                            "base_mean": base[dj]["mean"],
                            "shift": dj_mean - base[dj]["mean"],
                        })
                finally:
                    restore(model, args.edit_layer, W_old)
                assert_bit_equal(model, args.edit_layer, W_ref)

                # variant B: shuffled — replace i's factor with random rank-1
                d_out = W_ref.shape[0]; d_in = W_ref.shape[1]
                b_rand, a_rand = random_unit_rank1(d_out, d_in, dtype, device, gen)
                # scale b_rand to match median fact norm
                med_norm = max(1e-3, e["delta_norm"])
                cur_norm = float((b_rand.float().norm() * a_rand.float().norm()).item())
                b_rand = b_rand * (med_norm / max(cur_norm, 1e-6))
                shuffled_factors = [(b_rand, a_rand)] + [get_factor(d) for d in distractor_fids]
                W_old = apply_factors(model, args.edit_layer, shuffled_factors)
                try:
                    _, shuffled_mean = margins_for_fact(model, tok, row, t_new, t_true)
                finally:
                    restore(model, args.edit_layer, W_old)
                assert_bit_equal(model, args.edit_layer, W_ref)

                rows.append({
                    "seed": seed, "k": k, "fact_id": fid,
                    "base_mean": base[fid]["mean"],
                    "target_mean": target_mean,
                    "shuffled_mean": shuffled_mean,
                    "uplift": target_mean - base[fid]["mean"],
                    "gate_d_diff": target_mean - shuffled_mean,
                    "locality": locality_shifts,
                })

                if (ti + 1) % 25 == 0:
                    recent = rows[-25:]
                    print(f"  {ti+1}/{len(test_facts)}  "
                          f"uplift={sum(r['uplift'] for r in recent)/25:+.2f}  "
                          f"gateD={sum(r['gate_d_diff'] for r in recent)/25:+.2f}  "
                          f"({time.time()-t0:.0f}s)", flush=True)

            # quick mid-k summary
            cur = [r for r in rows if r["seed"] == seed and r["k"] == k]
            uplift = sum(r["uplift"] for r in cur) / len(cur)
            gd = sum(r["gate_d_diff"] for r in cur) / len(cur)
            pos_b = sum(1 for r in cur if r["target_mean"] > r["base_mean"]) / len(cur)
            pos_d = sum(1 for r in cur if r["gate_d_diff"] > 0) / len(cur)
            print(f"  k={k} seed={seed}: uplift={uplift:+.2f}  "
                  f"gateD_mean={gd:+.2f}  posB={pos_b:.0%}  posD={pos_d:.0%}",
                  flush=True)

    # --- aggregate ---
    with open(out / "phi1_cells.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        if n == 0: return 0.0
        return xs[n // 2] if n % 2 else 0.5 * (xs[n//2 - 1] + xs[n//2])

    summary = {}
    for k in args.k_values:
        for seed in args.seeds:
            cur = [r for r in rows if r["seed"] == seed and r["k"] == k]
            n = len(cur)
            if n == 0: continue
            summary[f"k{k}_seed{seed}"] = {
                "n_facts": n,
                "mean_uplift_nats": sum(r["uplift"] for r in cur) / n,
                "mean_gate_d_diff_nats": sum(r["gate_d_diff"] for r in cur) / n,
                "frac_target_beats_base": sum(1 for r in cur if r["uplift"] > 0) / n,
                "frac_target_beats_shuffled": sum(1 for r in cur if r["gate_d_diff"] > 0) / n,
                "frac_target_above_zero": sum(1 for r in cur if r["target_mean"] > 0) / n,
            }
        # cross-seed aggregate
        cur = [r for r in rows if r["k"] == k]
        n = len(cur)
        loc_shifts = [s["shift"] for r in cur for s in r["locality"]]
        summary[f"k{k}_agg"] = {
            "n_obs": n,
            "mean_uplift_nats": sum(r["uplift"] for r in cur) / n,
            "mean_gate_d_diff_nats": sum(r["gate_d_diff"] for r in cur) / n,
            "frac_target_beats_base": sum(1 for r in cur if r["uplift"] > 0) / n,
            "frac_target_beats_shuffled": sum(1 for r in cur if r["gate_d_diff"] > 0) / n,
            "frac_target_above_zero": sum(1 for r in cur if r["target_mean"] > 0) / n,
            "locality_median_abs_shift_nats": median([abs(x) for x in loc_shifts]) if loc_shifts else 0.0,
            "locality_mean_shift_nats": (sum(loc_shifts) / len(loc_shifts)) if loc_shifts else 0.0,
            "locality_n": len(loc_shifts),
        }

    json.dump(summary, open(out / "phi1_summary.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
