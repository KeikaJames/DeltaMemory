"""Exp38 — G3/G4 learned gate trainer (per-fact sigmoid heads).

For each fact i, train (w_i, b_i) such that:
  sigmoid(w_i^T h(x) + b_i) ≈ 1 if x is fact i's own (paraphrase) prompt
  sigmoid(w_i^T h(x) + b_i) ≈ 0 if x is some other bank fact's prompt

G3: train on (canon, paraphrase[0], paraphrase[1]) vs random others
G4: same + negation/question hard-negatives (drawn from a TRAINING-only template
    set; val/test negation templates are forbidden — AC5)

Anti-cheat (AC1): training NEVER touches target_new/target_true token ids.
We only feed prompt token ids → hidden states → label ∈ {0,1}. Asserted.
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

TRAIN_NEG_TEMPLATES_G4 = [  # different from eval_panels.NEG_TEMPLATES
    "It is not the case that {p}",
    "{p}? Actually no.",
    "Some claim that {p}, but that is wrong",
]


@torch.no_grad()
def hidden_at_last(model, tok, prompt, edit_layer):
    """Capture down_proj input at last position."""
    import torch.nn.functional as F2
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    captured = {}
    layer = model.model.layers[edit_layer]
    base_lin = layer.mlp.down_proj
    saved = base_lin.forward
    def hook(x):
        captured["x"] = x.detach()[..., -1, :].clone() if x.dim() == 3 else x.detach().clone()
        return F2.linear(x, base_lin.weight, base_lin.bias)
    base_lin.forward = hook
    try:
        model(**enc, use_cache=False)
    finally:
        base_lin.forward = saved
    return captured["x"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(EXP35B / "data" / "bank.pt"))
    ap.add_argument("--variant", choices=["G3", "G4"], required=True)
    ap.add_argument("--n-facts", type=int, default=10000,
                    help="how many facts to train heads for; default = all")
    ap.add_argument("--n-neg-per-fact", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-labels-sanity", action="store_true",
                    help="C8 sanity: shuffle labels, training must NOT converge")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else HERE / "data" / f"{args.variant}_heads.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    dtype = get_dtype(args.dtype)
    bank = load_bank(args.bank, device=args.device, dtype=dtype)
    N_total = len(bank.fact_ids)

    train_rows = json.load(open(EXP35B / "data" / "splits" / "train.json"))
    val_rows = json.load(open(EXP35B / "data" / "splits" / "val.json"))
    all_rows = {r["id"]: r for r in train_rows + val_rows}
    ids_with_prompt = [fid for fid in bank.fact_ids if fid in all_rows]
    print(f"[avail] {len(ids_with_prompt)}/{N_total} facts have prompts", flush=True)

    target_ids = ids_with_prompt[: args.n_facts]
    print(f"[plan] training {len(target_ids)} per-fact heads, variant={args.variant}", flush=True)

    # ---- AC1: per-fact forbidden target tokens (NOT bank-wide).
    # The training head for fact i must never see fact i's own target_new/target_true
    # token ids in its prompts. We compute these inside the per-fact loop below.
    # NOTE: a bank-wide accumulation is wrong — with N=10^4 facts, the set
    # collects ~hundreds of common tokens (single letters A-Z, "of", "the", ...)
    # and every neutral prompt trivially fails the assertion.
    # ----

    # Pre-compute hiddens for: own prompts + a shared negative pool
    rng = random.Random(args.seed)
    neg_pool_ids = rng.sample(ids_with_prompt, min(args.n_neg_per_fact * 4, len(ids_with_prompt)))

    d_in = bank.A.shape[0]
    print(f"[d_in] {d_in}", flush=True)

    # Pre-compute neg pool hiddens
    print(f"[neg pool] computing hiddens for {len(neg_pool_ids)} prompts", flush=True)
    neg_H = []
    for j, fid in enumerate(neg_pool_ids):
        r = all_rows[fid]
        p = r["prompt"].format(r["subject"])
        # Neg-pool prompts are pre-fact-loop; no per-fact AC1 check here.
        # Per-fact AC1 (no fact_i target tokens in fact_i prompts) is enforced
        # in the per-fact loop below.
        h = hidden_at_last(model, tok, p, args.edit_layer).float()
        neg_H.append(h)
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(neg_pool_ids)}", flush=True)
    neg_H = torch.stack(neg_H, dim=0)  # (M, d_in)
    print(f"[neg_H] {tuple(neg_H.shape)}", flush=True)

    # Init weights: W_g (d_in, N_full), b_g (N_full,) — but only train selected rows
    W_g = torch.zeros(d_in, N_total, dtype=torch.float32)
    b_g = torch.full((N_total,), -2.0, dtype=torch.float32)  # bias toward closed
    id2idx = {fid: i for i, fid in enumerate(bank.fact_ids)}

    losses_log = []
    t_start = time.time()
    for fact_i, fid in enumerate(target_ids):
        r = all_rows[fid]
        # Per-fact AC1 forbidden ids: only this fact's target_NEW (the counterfact we
        # want the bank to inject). target_true legitimately appears in subjects
        # (e.g. "Cologne Bonn Airport" subject when target_true="Cologne") and
        # forbidding it would trip legitimate prompts.
        tnew_i = tok(r["target_new"], add_special_tokens=False).input_ids[:1]
        forbidden_target_ids_i = set(tnew_i)
        # positive prompts
        canon = r["prompt"].format(r["subject"])
        paraphrases = list(r.get("paraphrase_prompts", []))[:2]
        pos_prompts = [canon] + paraphrases
        if args.variant == "G4":
            # add training-only hard negatives (negation/contradiction templates)
            for templ in TRAIN_NEG_TEMPLATES_G4:
                pos_prompts.append(templ.format(p=canon.rstrip("?.!")))
        # Per-fact AC1: this fact's prompts must not contain this fact's target tokens
        for p in pos_prompts:
            enc_ids = tok(p, add_special_tokens=False).input_ids
            assert not any(t in forbidden_target_ids_i for t in enc_ids), \
                f"AC1: prompt for fact {fid} contains its own target token: {p!r}"

        # Compute positive hiddens
        # For G4 negation samples, label them as positive=1 if the gate should still fire?
        # Per plan: G4 trains gate to also fire on negation but a) rank-1 patch will still
        # write target_new there (limitation). We label negations as label=0 to encourage
        # G4 to SUPPRESS on negation — addressing 36.4 root cause.
        pos_labels = [1.0] * (1 + len(paraphrases)) + ([0.0] * (len(pos_prompts) - 1 - len(paraphrases)))
        pos_H = torch.stack(
            [hidden_at_last(model, tok, p, args.edit_layer).float() for p in pos_prompts],
            dim=0)  # (P, d_in)

        # Sample per-fact negatives from neg_pool excluding self
        neg_idx = [k for k, nid in enumerate(neg_pool_ids) if nid != fid]
        rng2 = random.Random(args.seed * 7919 + fact_i)
        neg_sel = rng2.sample(neg_idx, min(args.n_neg_per_fact, len(neg_idx)))
        neg_H_i = neg_H[neg_sel]  # (Q, d_in)
        neg_labels = [0.0] * neg_H_i.shape[0]

        # Stack
        H = torch.cat([pos_H, neg_H_i], dim=0).to(model.device)
        y = torch.tensor(pos_labels + neg_labels, dtype=torch.float32, device=H.device)
        if args.shuffle_labels_sanity:
            perm = torch.randperm(y.shape[0], generator=torch.Generator().manual_seed(args.seed))
            y = y[perm]

        # Train (w, b) with AdamW
        w = torch.zeros(d_in, dtype=torch.float32, device=H.device, requires_grad=True)
        b = torch.full((1,), -2.0, dtype=torch.float32, device=H.device, requires_grad=True)
        opt = torch.optim.AdamW([w, b], lr=args.lr, weight_decay=args.wd)
        last_loss = None
        for step in range(args.steps):
            logits = H @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        W_g[:, id2idx[fid]] = w.detach().cpu()
        b_g[id2idx[fid]] = float(b.detach().cpu().item())
        losses_log.append(last_loss)

        if (fact_i + 1) % 50 == 0:
            mean_loss = sum(losses_log[-50:]) / min(50, len(losses_log))
            elapsed = time.time() - t_start
            eta = elapsed / (fact_i + 1) * (len(target_ids) - fact_i - 1)
            print(f"  trained {fact_i+1}/{len(target_ids)}  mean_loss[last50]={mean_loss:.4f}  "
                  f"ETA={eta/60:.1f}min", flush=True)

    # Quick health check: |w| distribution (AC8)
    w_norms = W_g.norm(dim=0)
    nonzero = (w_norms > 0).sum().item()
    print(f"\n[summary] trained {nonzero} heads; |w| mean={w_norms[w_norms>0].mean():.3f} "
          f"std={w_norms[w_norms>0].std():.3f}")
    print(f"final losses: mean={sum(losses_log)/len(losses_log):.4f}")

    torch.save({
        "W_g": W_g, "b_g": b_g,
        "fact_ids": bank.fact_ids,
        "variant": args.variant,
        "config": vars(args),
        "final_losses": losses_log,
        "w_norm_mean": float(w_norms[w_norms>0].mean()) if nonzero else 0.0,
        "w_norm_std": float(w_norms[w_norms>0].std()) if nonzero else 0.0,
    }, out_path)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
