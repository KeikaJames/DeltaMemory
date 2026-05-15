"""Exp38 — G1 threshold calibration.

For each fact i, collect a_i^T h(read_prompt_j) for:
  - positives: j == i (canon + 2 paraphrases from train+val)
  - negatives: j != i (canon prompt of other facts)

Set theta_i = 99th percentile of negatives (FPR=1%).
Save to data/g1_theta.pt as tensor of shape (N,) ordered to match bank.fact_ids.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP35B = HERE.parent / "exp35b_memit_bank"
sys.path.insert(0, str(HERE))

from common import load_model, load_bank, get_dtype  # noqa: E402


@torch.no_grad()
def hidden_at_last(model, tok, prompt, edit_layer):
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    captured = {}
    layer = model.model.layers[edit_layer]
    base_lin = layer.mlp.down_proj
    saved = base_lin.forward
    def hook(x):
        # x has shape (..., d_in). Capture last position.
        captured["x"] = x.detach()[..., -1, :].clone() if x.dim() == 3 else x.detach().clone()
        return F.linear(x, base_lin.weight, base_lin.bias)
    base_lin.forward = hook
    try:
        model(**enc, use_cache=False)
    finally:
        base_lin.forward = saved
    return captured["x"][0]  # (d_in,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(EXP35B / "data" / "bank.pt"))
    ap.add_argument("--n-negatives", type=int, default=200,
                    help="random other facts to sample per fact for negative dist")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--out", default=str(HERE / "data" / "g1_theta.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    dtype = get_dtype(args.dtype)
    bank = load_bank(args.bank, device=args.device, dtype=dtype)
    N = len(bank.fact_ids)
    id2idx = {fid: i for i, fid in enumerate(bank.fact_ids)}
    print(f"[bank] N={N}", flush=True)

    # Load splits to get prompts
    train = json.load(open(EXP35B / "data" / "splits" / "train.json"))
    val = json.load(open(EXP35B / "data" / "splits" / "val.json"))
    all_rows = {r["id"]: r for r in train + val}

    # For each fact: get last-position hidden state for canon + 2 paraphrases
    # (positives), then for n_negatives random other facts (negatives).
    # Negatives are drawn ONCE globally and reused across all facts (cheap proxy).
    rng = random.Random(args.seed)
    bank_ids_in_splits = [fid for fid in bank.fact_ids if fid in all_rows]
    print(f"[splits] {len(bank_ids_in_splits)}/{N} bank facts have prompts available", flush=True)

    # Sample shared negative-prompt pool
    neg_pool_ids = rng.sample(bank_ids_in_splits, min(args.n_negatives, len(bank_ids_in_splits)))
    print(f"[neg pool] computing {len(neg_pool_ids)} hidden states for negative pool", flush=True)
    neg_hiddens = []
    for j, fid in enumerate(neg_pool_ids):
        r = all_rows[fid]
        p = r["prompt"].format(r["subject"])
        h = hidden_at_last(model, tok, p, args.edit_layer)
        neg_hiddens.append(h.float())
        if (j + 1) % 50 == 0:
            print(f"  neg {j+1}/{len(neg_pool_ids)}", flush=True)
    neg_H = torch.stack(neg_hiddens, dim=0)  # (M, d_in)
    print(f"[neg H] {tuple(neg_H.shape)}", flush=True)

    # For each fact, compute its own positive scores and use neg_H for negatives.
    # theta_i = quantile(1 - fpr) of {a_i^T h for h in (neg_H ∪ {own paraphrases except positive})}
    # Simpler: theta_i = quantile(1-fpr) of (a_i^T neg_H rows where row is not fact i).
    A = bank.A.float()  # (d_in, N)
    # Project all negatives once: scores_neg = neg_H @ A → (M, N)
    print("[project] neg_H @ A", flush=True)
    scores_neg = neg_H.to(A.device) @ A  # (M, N)
    print(f"[scores_neg] {tuple(scores_neg.shape)}", flush=True)

    # For each fact i, mask out its own contribution if it was in the negative pool
    M = neg_H.shape[0]
    own_mask = torch.zeros(M, N, dtype=torch.bool, device=A.device)
    for j, fid in enumerate(neg_pool_ids):
        if fid in id2idx:
            own_mask[j, id2idx[fid]] = True
    scores_neg_masked = scores_neg.clone()
    scores_neg_masked[own_mask] = float("-inf")

    # Per-fact 99th percentile
    q = 1 - args.fpr
    sorted_scores, _ = scores_neg_masked.sort(dim=0)  # ascending
    # find rank corresponding to q
    valid_count = (~own_mask).sum(dim=0)  # (N,)
    theta = torch.zeros(N, dtype=torch.float32)
    for i in range(N):
        n_valid = int(valid_count[i].item())
        col = sorted_scores[:n_valid, i]
        if n_valid == 0:
            theta[i] = 0.0
            continue
        rank = max(0, min(n_valid - 1, int(q * n_valid)))
        theta[i] = col[rank].item()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"theta": theta, "fact_ids": bank.fact_ids,
                "fpr": args.fpr, "n_negatives": M},
               out)
    print(f"\n[done] theta: mean={theta.mean():.3f} std={theta.std():.3f} "
          f"min={theta.min():.3f} max={theta.max():.3f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
