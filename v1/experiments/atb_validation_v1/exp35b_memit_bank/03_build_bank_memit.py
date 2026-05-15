"""Exp35b — 03: Build MEMIT-preconditioned bank on N=10k facts.

Differences from Exp35 build_bank.py:
  * Use 10k corpus (data/splits/{train,val,test}.json).
  * MEMIT preconditioning: a = C^-1 k* (load cov_L5.pt produced by step 02).
  * Solo-pass gate is COMPUTED but NOT filtered out (D1).
  * Saves bank.pt with all 10k entries.
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

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
sys.path.insert(0, str(EXP35))
import importlib.util as _u
_spec = _u.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = _u.module_from_spec(_spec); _spec.loader.exec_module(_bb)

first_target_id = _bb.first_target_id
subject_last_pos = _bb.subject_last_pos
capture_k_star = _bb.capture_k_star
compute_v_star = _bb.compute_v_star
apply_factors = _bb.apply_factors
restore = _bb.restore
margin_at_last = _bb.margin_at_last
assert_bit_equal = _bb.assert_bit_equal


def memit_factors(W, k_star, v_star, C_inv):
    """MEMIT-preconditioned rank-1 update: a = C^-1 k*, b = (v* - W k*) / (k* . a).

    Resulting Δ = b a^T, where the rank-1 form is preserved.
    """
    k = k_star.to(W.dtype)
    v = v_star.to(W.dtype)
    C_inv_d = C_inv.to(device=W.device, dtype=W.dtype)
    a = C_inv_d @ k
    # Normalise by k^T a so that Δ k = v - Wk (the rank-1 ROME identity)
    denom = float((k.float() @ a.float()).item())
    if abs(denom) < 1e-8:
        denom = 1e-8 if denom >= 0 else -1e-8
    a = a / denom
    Wk = W @ k
    b = v - Wk
    return b, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--v-steps", type=int, default=25)
    ap.add_argument("--v-lr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1e-2, help="unused; here for parity")
    ap.add_argument("--cov", default=str(DATA / "cov_L5.pt"))
    ap.add_argument("--out", default=str(DATA / "bank.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-limit", type=int, default=0, help="0 = all 10k")
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[load] covariance {args.cov}", flush=True)
    cov_obj = torch.load(args.cov, map_location="cpu", weights_only=False)
    C_inv = cov_obj["C_inv"]
    print(f"[load] C_inv shape {tuple(C_inv.shape)}, inv_residual={cov_obj.get('inv_residual_norm','n/a')}", flush=True)

    train = json.load(open(DATA / "splits" / "train.json"))
    val = json.load(open(DATA / "splits" / "val.json"))
    test = json.load(open(DATA / "splits" / "test.json"))
    all_facts = [("train", i, r) for i, r in enumerate(train)] \
                + [("val", i, r) for i, r in enumerate(val)] \
                + [("test", i, r) for i, r in enumerate(test)]
    if args.n_limit > 0:
        all_facts = all_facts[:args.n_limit]
    print(f"[facts] bank size = {len(all_facts)} (train={len(train)} val={len(val)} test={len(test)})", flush=True)

    W = model.model.layers[args.edit_layer].mlp.down_proj.weight
    W_ref = W.data.clone()

    entries = {}
    norms = []
    t0 = time.time()

    # checkpoint resume
    if Path(args.out).exists():
        prev = torch.load(args.out, map_location="cpu", weights_only=False)
        entries = prev.get("entries", {})
        print(f"[resume] {len(entries)} already done", flush=True)

    for idx, (split, sidx, row) in enumerate(all_facts):
        fid = row["id"]
        if fid in entries:
            continue
        t_new = first_target_id(tok, row["target_new"])
        t_true = first_target_id(tok, row["target_true"])
        canon = row["prompt"].format(row["subject"])
        pos = subject_last_pos(tok, canon, row["subject"])

        k_star = capture_k_star(model, tok, canon, args.edit_layer, pos)
        v_star = compute_v_star(model, tok, canon, args.edit_layer,
                                t_new, t_true, args.v_steps, args.v_lr)

        b, a = memit_factors(W, k_star, v_star, C_inv)
        delta_norm = float((b.float().norm() * a.float().norm()).item())
        norms.append(delta_norm)

        W_old = apply_factors(model, args.edit_layer, [(b, a)])
        try:
            paraphrases = list(row.get("paraphrase_prompts", []))[:2]
            read_prompts = [canon] + paraphrases
            margins = [margin_at_last(model, tok, p, t_new, t_true) for p in read_prompts]
            canonical_pass = margins[0] > 0
            para_pass = sum(1 for m in margins[1:] if m > 0)
            solo_pass = bool(canonical_pass and para_pass >= 1)
        finally:
            restore(model, args.edit_layer, W_old)
        assert_bit_equal(model, args.edit_layer, W_ref)

        entries[fid] = {
            "split": split,
            "split_idx": sidx,
            "fact_idx_global": idx,
            "subject": row["subject"],
            "relation": row["relation"],
            "target_true": row["target_true"],
            "target_new": row["target_new"],
            "b": b.cpu().to(torch.float32),
            "a": a.cpu().to(torch.float32),
            "delta_norm": delta_norm,
            "solo_margins": [float(m) for m in margins],
            "solo_pass": solo_pass,
        }

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_facts):
            recent = list(entries.values())[-100:]
            recent_pass = sum(1 for e in recent if e["solo_pass"]) / max(1, len(recent))
            elapsed = time.time() - t0
            rate = (len(entries) - len(prev.get("entries", {}) if 'prev' in dir() else {})) / max(1e-3, elapsed) if False else (idx + 1) / max(1e-3, elapsed)
            eta_min = (len(all_facts) - idx - 1) / max(1e-3, rate) / 60
            print(f"  {idx+1}/{len(all_facts)} ({elapsed:.0f}s) "
                  f"recent_solo_pass={recent_pass:.0%} "
                  f"‖Δ‖_F median(last100)={torch.tensor(norms[-100:]).median().item():.2f} "
                  f"eta={eta_min:.1f}m", flush=True)
            torch.save({"entries": entries}, args.out)

    # outlier flag (C6/D7-prep)
    norms_t = torch.tensor([e["delta_norm"] for e in entries.values()])
    med = float(norms_t.median())
    cutoff = 3.0 * med
    for fid, e in entries.items():
        e["norm_outlier"] = e["delta_norm"] > cutoff

    summary = {
        "meta": {"edit_layer": args.edit_layer, "v_steps": args.v_steps,
                 "v_lr": args.v_lr, "seed": args.seed,
                 "n_total": len(all_facts), "preconditioner": "MEMIT_C_inv"},
        "solo_pass_rate": sum(1 for e in entries.values() if e["solo_pass"]) / max(1, len(entries)),
        "solo_pass_by_split": {
            s: {
                "n": sum(1 for e in entries.values() if e["split"] == s),
                "pass": sum(1 for e in entries.values() if e["split"] == s and e["solo_pass"]),
            } for s in ("train", "val", "test")
        },
        "delta_norm_median": med,
        "delta_norm_max": float(norms_t.max()),
        "delta_norm_outlier_cutoff": cutoff,
        "n_norm_outliers": sum(1 for e in entries.values() if e["norm_outlier"]),
    }
    torch.save({"entries": entries, "summary": summary}, args.out)
    json.dump(summary, open(DATA / "bank_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
