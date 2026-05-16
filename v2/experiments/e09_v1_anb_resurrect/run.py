"""e09 — Resurrect v1 ANB (Attention Native Bank) with v2's K-projector.

Context:
  v1's ANB (Attention Native Bank, v1/deltamemory/memory/attn_native_bank.py)
  was the original approach: bank stored raw hidden states, attention
  concatenated them as K/V with NO projector. Result: essentially zero signal
  (Δ ≈ 0). Phase B2 showed that adding a rank-64 K-projector drops NLL from
  12.13 to 6.30 (Δ = -5.83) on the same data.

Hypothesis (e09):
  The K-projector is the active ingredient. Adding it to v1's ANB architecture
  should revive the bank from null to significant signal.

Design:
  - Two conditions per run (mode flag):
      (a) v1_orig  — ANB without projector (frozen to identity)
      (b) v2_kproj — ANB + rank-64 trainable projector
  - SAME bank (512 b-vectors from Exp35b), SAME data, SAME eval set.
  - Both modes use v2/core/qwen3_lpl_patch machinery (bank-attention concat)
    but with projector behavior controlled by the mode flag.

Pass criterion:
  (a) Δ ≈ 0 (within ±0.3) — reproduces v1 null result
  (b) Δ ≤ -2.0          — K-projector revives the bank

Output: e09_{mode}_seed{seed}.json with before/after NLL controls.

NOTE: "v1-equivalent" reduction:
  For v1_orig mode, we FREEZE the K-projector to identity (P.requires_grad=False,
  weights remain at init) and do NOT train it. This is mathematically identical
  to v1's "no projector" path because the projector is initialized as:
      x -> x + 0 * U(V(x))  =  x    (U is zero-init)
  The qwen3_lpl_patch's bank concatenation (lines 125-135 in qwen3_lpl_patch.py)
  handles the K/V projection mechanics, exactly matching v1 ANB's intent.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa,
    data_io,
)


def forward_lpl_k2(model, bank, heads, enc, *, grad=False, randomize_bank=False, zero_bank=False):
    """Two-round LPL forward: round 1 writes to bank, round 2 reads from bank."""
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    
    # Optional bank manipulation (for controls)
    if randomize_bank or zero_bank:
        for l, t in enumerate(bank.slots):
            if t.shape[0] == 0:
                continue
            if zero_bank:
                bank.slots[l] = torch.zeros_like(t)
            else:
                n = torch.randn_like(t)
                n = n / (n.norm(dim=-1, keepdim=True) + 1e-9) * t.norm(dim=-1, keepdim=True)
                bank.slots[l] = n.to(dtype=t.dtype, device=t.device)
    
    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, input_ids, ans_start):
    """Compute causal LM loss over answer span."""
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def main():
    p = argparse.ArgumentParser(description="e09: resurrect v1 ANB with K-projector")
    p.add_argument("--mode", required=True, choices=["v1_orig", "v2_kproj"],
                   help="v1_orig: ANB without projector (frozen); v2_kproj: ANB + rank-64 trainable projector")
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else HERE / f"e09_{args.mode}_seed{args.seed}.json"
    print(f"[e09:{args.mode}] seed={args.seed} steps={args.steps} lr={args.lr} rank={args.rank}")

    # Load data
    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]

    # Build train/test split (standard Phase B2 split)
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng = random.Random(args.seed)
    rng.shuffle(train_keys); rng.shuffle(test_keys)
    train_keys = train_keys[:args.n_train]
    test_keys = test_keys[:args.n_eval]
    
    # Preload keys (disjoint from train)
    preload_pool = data_io.filter_keys(entries, split="train", solo_pass=True)
    preload_pool = [k for k in preload_pool if k not in set(train_keys)]
    preload_keys = preload_pool[:args.n_preload]

    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    print(f"[e09:{args.mode}] train={len(train_items)} test={len(test_items)} preload={len(preload_keys)}")

    # Load model
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    # Build bank + heads
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    # Build b-vectors (raw hidden states from MEMIT)
    b_raw = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                     device=args.device, dtype=torch.float32)
    print(f"[e09:{args.mode}] b_raw shape={b_raw.shape}")

    # Create K-projector
    P = make_projector(d, rank=args.rank).to(args.device).float()

    # MODE CONTROL: v1_orig freezes projector (identity), v2_kproj trains it
    if args.mode == "v1_orig":
        print(f"[e09:v1_orig] FREEZING projector to identity (no training)")
        for param in P.parameters():
            param.requires_grad_(False)
        trainable = list(heads.bank_gate_heads.parameters())
    elif args.mode == "v2_kproj":
        print(f"[e09:v2_kproj] Projector TRAINABLE (rank={args.rank})")
        trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    def apply_proj(zero_bank=False):
        """Apply projector (identity or trained) and load into bank."""
        with torch.no_grad():
            proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
        if zero_bank:
            proj = torch.zeros_like(proj)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True

    apply_proj()

    # Evaluation helpers
    def eval_base(items):
        """Eval with base model only (no bank)."""
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    def eval_lpl(items, *, randomize=False, zero_bank=False, bank_off=False):
        """Eval with LPL bank (real/rand/zero/off controls)."""
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            apply_proj(zero_bank=zero_bank or bank_off)
            if bank_off:
                # Truly disable: empty slots
                bank.frozen = False
                bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device, dtype=torch.bfloat16)
                bank.tags[args.bank_layer] = []
                bank.frozen = True
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False,
                                     randomize_bank=randomize, zero_bank=False)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    # === PRE-TRAIN EVAL ===
    print(f"[e09:{args.mode}] Running pre-train eval...")
    base = eval_base(test_items)
    pre_real = eval_lpl(test_items)
    pre_rand = eval_lpl(test_items, randomize=True)
    pre_zero = eval_lpl(test_items, zero_bank=True)
    pre_off = eval_lpl(test_items, bank_off=True)
    print(f"[e09:{args.mode}] BEFORE: base={base:.4f}  real={pre_real:.4f}  "
          f"rand={pre_rand:.4f}  zero={pre_zero:.4f}  off={pre_off:.4f}")

    # === TRAINING ===
    print(f"[e09:{args.mode}] Training for {args.steps} steps...")
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        
        # Rebuild bank with current projector each step
        bank.frozen = False
        proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True

        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        
        if (step + 1) % 25 == 0 or step == 0:
            recent = sum(losses[-25:]) / min(25, len(losses))
            print(f"  [e09:{args.mode}] step {step+1}/{args.steps} loss(avg25)={recent:.4f} ({time.time()-t0:.1f}s)")
    
    print(f"[e09:{args.mode}] Training done in {time.time()-t0:.1f}s")

    # === POST-TRAIN EVAL ===
    print(f"[e09:{args.mode}] Running post-train eval...")
    post_real = eval_lpl(test_items)
    post_rand = eval_lpl(test_items, randomize=True)
    post_zero = eval_lpl(test_items, zero_bank=True)
    post_off = eval_lpl(test_items, bank_off=True)
    print(f"[e09:{args.mode}] AFTER:  base={base:.4f}  real={post_real:.4f}  "
          f"rand={post_rand:.4f}  zero={post_zero:.4f}  off={post_off:.4f}")

    # === VERDICT ===
    delta = base - post_real
    if args.mode == "v1_orig":
        # v1_orig: should get Δ ≈ 0 (null result like original v1 ANB)
        passed = abs(delta) <= 0.3
        rule = "Δ ≈ 0 (within ±0.3) — reproduces v1 null result"
    elif args.mode == "v2_kproj":
        # v2_kproj: should get Δ ≤ -2.0 (projector revives the bank)
        passed = delta <= -2.0
        rule = "Δ ≤ -2.0 — K-projector revives the bank"
    else:
        passed = None
        rule = "unknown mode"

    verdict = {"pass": passed, "delta": delta, "rule": rule}
    print(f"[e09:{args.mode}] VERDICT: {verdict}")

    # === OUTPUT ===
    out = {
        "mode": args.mode,
        "seed": args.seed,
        "model": args.model,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "before": {
            "base": base,
            "real": pre_real,
            "rand": pre_rand,
            "zero": pre_zero,
            "off": pre_off,
        },
        "after": {
            "base": base,
            "real": post_real,
            "rand": post_rand,
            "zero": post_zero,
            "off": post_off,
        },
        "verdict": verdict,
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
        "n_train_params": sum(p.numel() for p in trainable),
        "projector_trainable": args.mode == "v2_kproj",
    }
    
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e09:{args.mode}] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
