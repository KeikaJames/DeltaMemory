"""Exp35b — 08: D4 layer-ablation (downscoped to N=1000 subset).

Build a small bank at edit_layer ∈ {3, 5, 7, 9} on 1000 train+val facts,
then measure single-fact margin uplift (k=1) on a 200-fact eval subset.

This is a downscoped ablation — not the main bank — to test the locus
assumption that L=5 is near-optimal. Full 10k bank stays at L=5.
"""

from __future__ import annotations

import argparse
import importlib.util as iu
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
_spec = iu.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = iu.module_from_spec(_spec); _spec.loader.exec_module(_bb)
first_target_id = _bb.first_target_id
subject_last_pos = _bb.subject_last_pos
capture_k_star = _bb.capture_k_star
compute_v_star = _bb.compute_v_star
apply_factors = _bb.apply_factors
restore = _bb.restore
margin_at_last = _bb.margin_at_last
assert_bit_equal = _bb.assert_bit_equal


def margins_for(model, tok, row, t_new, t_true):
    canon = row["prompt"].format(row["subject"])
    paras = list(row.get("paraphrase_prompts", []))[:2]
    ms = [margin_at_last(model, tok, p, t_new, t_true) for p in [canon] + paras]
    return sum(ms) / len(ms)


def build_solo_factor(model, tok, row, layer):
    subj = row["subject"]; prompt = row["prompt"].format(subj)
    t_new = first_target_id(tok, row["target_new"])
    t_true = first_target_id(tok, row["target_true"])
    sp = subject_last_pos(tok, prompt, subj)
    k_star = capture_k_star(model, tok, prompt, layer, sp)
    v_star = compute_v_star(model, tok, prompt, layer, t_new, t_true, 25, 0.5)
    W = model.model.layers[layer].mlp.down_proj.weight.data.float()
    k = k_star.float(); v = v_star.float()
    denom = (k * k).sum() + 1e-2
    b = (v - W @ k) / denom
    a = k
    norm = float((b.norm() * a.norm()).item())
    return b, a, norm, t_new, t_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35b" / "d4_layer_ablation.json"))
    args = ap.parse_args()

    seed_everything(0)
    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    test = json.load(open(DATA / "splits" / "test.json"))[: args.n_eval]
    print(f"[eval] {len(test)} facts", flush=True)

    summary = {}
    for L in args.layers:
        print(f"\n[L={L}]", flush=True)
        W_ref = model.model.layers[L].mlp.down_proj.weight.data.clone()
        rows = []
        t0 = time.time()
        for i, r in enumerate(test):
            try:
                b, a, norm, t_new, t_true = build_solo_factor(model, tok, r, L)
            except Exception as ex:
                print(f"  skip {r['id']}: {ex}", flush=True)
                continue
            base = margins_for(model, tok, r, t_new, t_true)
            assert_bit_equal(model, L, W_ref)
            b_d = b.to(device, dtype=dtype); a_d = a.to(device, dtype=dtype)
            W_old = apply_factors(model, L, [(b_d, a_d)])
            try:
                pat = margins_for(model, tok, r, t_new, t_true)
            finally:
                restore(model, L, W_old)
            assert_bit_equal(model, L, W_ref)
            rows.append({"id": r["id"], "base": base, "patched": pat,
                         "uplift": pat - base, "delta_norm": norm})
            if (i + 1) % 50 == 0:
                rec = rows[-50:]
                print(f"  {i+1}/{len(test)} uplift={sum(r['uplift'] for r in rec)/len(rec):+.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        n = len(rows)
        summary[f"L{L}"] = {
            "n": n,
            "mean_uplift": sum(r["uplift"] for r in rows) / max(1, n),
            "frac_pos": sum(1 for r in rows if r["uplift"] > 0) / max(1, n),
            "mean_delta_norm": sum(r["delta_norm"] for r in rows) / max(1, n),
        }
        print(f"  L={L}: {summary[f'L{L}']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
