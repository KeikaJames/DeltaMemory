"""Phase B2 — train bank-side K/V projectors (resurrect v1's K-projector inside LPL).

Phase B's null showed that with frozen W_K/W_V and only learnable bank_gate
(a per-position scalar), the static Exp35b residual-space b-vectors cannot
be turned into queryable K/V at layer 9.  This script adds a small learnable
linear projection on the bank h-vectors BEFORE they are fed to W_K/W_V, so
that bank semantics get re-aligned to the target layer's QK frame.

Specifically, replace bank K = W_K^l · h_b with
    bank K = W_K^l · (P_K · h_b + h_b)         (residual projector)
and similarly P_V for V.  P_K, P_V ∈ R^{d×d}, init to zero so step-0 ≡ Phase B.

Compares pre/post training:
    base                         (no LPL)
    LPL+bank+gate-only (Phase B baseline, post-train)
    LPL+bank+gate+projector      (Phase B2)
    LPL+rand-bank+gate+projector (control)

Decision: bridge_unlocked iff Phase-B2 NLL < base by a meaningful margin
AND Phase-B2 < random-bank-with-same-projector.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from atb_validation_v1._lib import load_model
from exp42_lpl import AttentionBank, LPLHeads, install_lpl_patch
from exp42_lpl.qwen3_lpl_patch import LPLState, lpl_state_scope
sys.path.insert(0, str(HERE))
from importlib import import_module
phase_a = import_module("01_phase_a_frozen")
nll_on_answer = phase_a.nll_on_answer


def build_prompt(subj, rel): return f"{subj} {rel}"


def apply_projector(bank, layer, b_raw, P):
    """Re-project b_raw with (I + P) and write to bank.slots[layer]."""
    # P: nn.Linear with bias=False, weight init 0 → identity behavior at step 0
    proj = (b_raw + P(b_raw)).to(dtype=bank.slots[layer].dtype)
    bank.frozen = False
    bank.slots[layer] = proj
    bank.tags[layer] = [(0, 0)] * proj.shape[0]
    bank.frozen = True


def forward_lpl_k2(model, bank, heads, enc, *, grad=False, randomize_bank=False):
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    if randomize_bank:
        for l, t in enumerate(bank.slots):
            if t.shape[0] == 0:
                continue
            n = torch.randn_like(t)
            n = n / (n.norm(dim=-1, keepdim=True) + 1e-9) * t.norm(dim=-1, keepdim=True)
            bank.slots[l] = n.to(dtype=t.dtype, device=t.device)
    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, input_ids, ans_start):
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank_pt", default=str(REPO / "v1/experiments/atb_validation_v1/exp35b_memit_bank/data/bank.pt"))
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=40)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--bank_n_preload", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=0, help="0=full d×d, else low-rank")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "phase_b2_kproj_results.json"))
    args = p.parse_args()

    blob = torch.load(args.bank_pt, map_location="cpu", weights_only=False)
    entries = blob["entries"]
    keys = list(entries.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    train_keys = [k for k in keys if entries[k].get("split") == "train" and entries[k].get("solo_pass")][:args.n_train]
    test_keys = [k for k in keys if entries[k].get("split") == "test" and entries[k].get("solo_pass")][:args.n_eval]
    preload_keys = [k for k in keys if entries[k].get("split") == "train" and entries[k].get("solo_pass")][args.n_train: args.n_train + args.bank_n_preload]

    train_items = [(entries[k]["subject"], entries[k]["relation"], entries[k]["target_true"]) for k in train_keys]
    test_items = [(entries[k]["subject"], entries[k]["relation"], entries[k]["target_true"]) for k in test_keys]
    print(f"[B2] train_n={len(train_items)} test_n={len(test_items)} preload_n={len(preload_keys)}")

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.bank_n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    # Raw b vectors (fp32 on device, kept around for re-projection)
    b_raw = torch.stack([entries[k]["b"].float() for k in preload_keys], dim=0)
    b_raw = b_raw / (b_raw.norm(dim=-1, keepdim=True) + 1e-9) * 15.0
    b_raw = b_raw.to(device=args.device, dtype=torch.float32)
    print(f"[B2] b_raw shape={tuple(b_raw.shape)} norm={b_raw.norm(dim=-1).mean():.2f}")

    # Projector: zero-init residual linear so step 0 == Phase B baseline
    if args.rank > 0:
        # Low-rank: P = U @ V, U[d,r] init small, V[r,d] init zero
        class LowRankProj(nn.Module):
            def __init__(self, d, r):
                super().__init__()
                self.U = nn.Linear(r, d, bias=False)
                self.V = nn.Linear(d, r, bias=False)
                nn.init.zeros_(self.U.weight)
                nn.init.normal_(self.V.weight, std=0.02)
            def forward(self, x): return self.U(self.V(x))
        P = LowRankProj(d, args.rank).to(args.device).float()
    else:
        P = nn.Linear(d, d, bias=False).to(args.device).float()
        nn.init.zeros_(P.weight)

    apply_projector(bank, args.bank_layer, b_raw, P)

    # ---------- baseline (projector still zero ≡ identity-pass) ----------
    def eval_base(items):
        nlls = []
        for sj, rl, tg in items:
            prompt = build_prompt(sj, rl)
            full = prompt + " " + tg
            enc = tok(full, return_tensors="pt").to(args.device)
            ans_start = tok(prompt, return_tensors="pt").input_ids.shape[1]
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans_start))
        return sum(nlls) / len(nlls)

    def eval_lpl(items, randomize=False):
        nlls = []
        for sj, rl, tg in items:
            prompt = build_prompt(sj, rl)
            full = prompt + " " + tg
            enc = tok(full, return_tensors="pt").to(args.device)
            ans_start = tok(prompt, return_tensors="pt").input_ids.shape[1]
            with torch.no_grad():
                apply_projector(bank, args.bank_layer, b_raw, P)
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False, randomize_bank=randomize)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans_start))
        return sum(nlls) / len(nlls)

    base = eval_base(test_items)
    pre_lpl = eval_lpl(test_items)
    pre_rand = eval_lpl(test_items, randomize=True)
    print(f"[B2] === before training ===")
    print(f"  base                NLL = {base:.4f}")
    print(f"  LPL+bank (zero-P)   NLL = {pre_lpl:.4f}   Δ={pre_lpl-base:+.4f}")
    print(f"  LPL+rand (zero-P)   NLL = {pre_rand:.4f}   Δ={pre_rand-base:+.4f}")

    # ---------- training: projector + bank_gate jointly ----------
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    n_train_params = sum(p.numel() for p in trainable)
    print(f"[B2] trainable params: {n_train_params:,}  (P: {sum(p.numel() for p in P.parameters()):,})")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    print(f"[B2] === training {args.steps} steps lr={args.lr} ===")
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        prompt = build_prompt(sj, rl)
        full = prompt + " " + tg
        enc = tok(full, return_tensors="pt").to(args.device)
        ans_start = tok(prompt, return_tensors="pt").input_ids.shape[1]

        # Re-apply projector each step so gradient flows through P -> bank.slots
        # We must do this *with* grad enabled so the bank contents carry grad.
        bank.frozen = False
        proj = (b_raw + P(b_raw)).to(dtype=torch.bfloat16)
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, 0)] * proj.shape[0]
        bank.frozen = True

        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans_start)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 20 == 0 or step == 0:
            recent = sum(losses[-20:]) / min(20, len(losses))
            print(f"  step {step+1:3d}/{args.steps}  loss(avg20)={recent:.4f}  ({time.time()-t0:.1f}s)")
    print(f"[B2] training done in {time.time()-t0:.1f}s")

    post_lpl = eval_lpl(test_items)
    post_rand = eval_lpl(test_items, randomize=True)
    print(f"[B2] === after training ===")
    print(f"  base                NLL = {base:.4f}")
    print(f"  LPL+bank+P (real)   NLL = {post_lpl:.4f}   Δ_vs_base={post_lpl-base:+.4f}  Δ_vs_pre={post_lpl-pre_lpl:+.4f}")
    print(f"  LPL+rand+P (ctl)    NLL = {post_rand:.4f}  Δ_vs_base={post_rand-base:+.4f}")

    bridge = (post_lpl < base - 0.05) and (post_lpl < post_rand - 0.02)
    print(f"\n[B2] bridge_unlocked? {bridge}  (real beats base by ≥0.05 AND beats rand by ≥0.02)")

    out = {
        "n_train": len(train_items), "n_test": len(test_items),
        "preload_n": int(b_raw.shape[0]), "lr": args.lr, "steps": args.steps,
        "rank": args.rank,
        "n_train_params": n_train_params,
        "before": {"base": base, "lpl": pre_lpl, "rand": pre_rand},
        "after": {"base": base, "lpl": post_lpl, "rand": post_rand},
        "bridge_unlocked": bool(bridge),
        "loss_first20": losses[:20],
        "loss_last20": losses[-20:],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[B2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
