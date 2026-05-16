"""e04 ACT-style halt-head training + K_max sweep.

Adds a learnable halt head that decides per-position-per-round whether to
"pause" (write to bank) or continue normally. Each pause = +K tokens in the
bank → quadratic attention cost. We apply an ACT-style sparsity prior:

    L_total = L_NLL(round_2) + λ_act * mean(halt_prob)

Higher λ_act → fewer halts → smaller effective K.

We sweep (λ_act × K_max) = 5 × 6 = 30 cells:
- λ_act ∈ {0.0, 0.01, 0.1, 0.5, 1.0}
- K_max ∈ {1, 4, 16, 64, 256, -1}  (-1 = unlimited)

If more than K_max positions fire halt in a sample, keep only top-K by halt_prob.

Pass criterion (for the experiment as a whole, not per cell):
- Some (λ_act, K_max) cell with mean halts ≤ 4 still achieves Δ ≤ -2.0

IMPORTANT CAVEAT: The current LPL patch uses `pause_heads` which operate at the
layer level during round 1. This experiment monkey-patches the bank.write logic
to gate on the halt head's output instead, using a custom force_pause_mask that
computes halt probabilities and applies K_max budget enforcement.
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
import torch.nn as nn
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, HaltHead, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa,
    data_io,
)


class HaltHeadPerPosition(nn.Module):
    """Per-position halt head: h[B, T, d] -> halt_logits[B, T].
    
    Initialized with bias=-10.0 so sigmoid(logit) ≈ 0 (no halt anywhere) until trained.
    This guarantees Gate 0 baseline behavior.
    """
    def __init__(self, hidden_size: int, init_bias: float = -10.0):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, d] -> halt_logits: [B, T]."""
        return self.proj(h.float()).squeeze(-1)


def make_halt_pause_mask_fn(halt_head: HaltHeadPerPosition, bank_layer: int, k_max: int, threshold: float = 0.5):
    """Factory for force_pause_mask callable that:
    1. Computes halt_prob = sigmoid(halt_head(h)) at bank_layer round 1
    2. Enforces K_max budget: if >K_max positions have halt_prob > threshold, keep only top-K
    3. Returns [B, T] bool mask for pausing
    
    Args:
        halt_head: per-position halt head module
        bank_layer: which layer to apply halt head (typically 9)
        k_max: maximum number of pauses per sample (-1 = unlimited)
        threshold: halt probability threshold (default 0.5)
    
    Returns:
        callable(layer, round_idx, h_in) -> torch.Tensor | None
    """
    def halt_pause_mask_fn(layer: int, round_idx: int, h_in: torch.Tensor):
        # Only apply at bank_layer during round 1 (pause round)
        if layer != bank_layer or round_idx != 1:
            return None
        
        # Compute halt probabilities: [B, T]
        halt_logits = halt_head(h_in)  # [B, T]
        halt_prob = torch.sigmoid(halt_logits)  # [B, T]
        
        # K_max budget enforcement
        B, T = halt_prob.shape
        if k_max < 0:
            # Unlimited: use threshold
            pause_mask = halt_prob > threshold
        else:
            # Limited: keep top-K by halt_prob per batch item
            pause_mask = torch.zeros_like(halt_prob, dtype=torch.bool)
            for b in range(B):
                probs = halt_prob[b]  # [T]
                # Find positions above threshold
                candidates = (probs > threshold).nonzero(as_tuple=False).squeeze(-1)
                if len(candidates) == 0:
                    continue
                # If more than K_max, keep only top-K
                if len(candidates) > k_max:
                    candidate_probs = probs[candidates]
                    _, top_k_idx = torch.topk(candidate_probs, k=k_max, largest=True)
                    keep_positions = candidates[top_k_idx]
                else:
                    keep_positions = candidates
                pause_mask[b, keep_positions] = True
        
        return pause_mask
    
    return halt_pause_mask_fn


def forward_lpl_k2_with_halt(
    model, bank, heads, enc, halt_head, bank_layer, k_max,
    *, grad=False, capture_halt_stats=False
):
    """Two-round forward with halt-head-controlled pausing.
    
    Round 1: compute halt_prob per position at bank_layer, pause top-K, write to bank.
    Round 2: read bank, compute final logits.
    
    Returns:
        logits: [B, T, vocab_size]
        halt_stats: dict with halt_prob, halt_mask, n_halts (if capture_halt_stats=True)
    """
    # Round 1: compute halt mask and write to bank
    halt_pause_mask_fn = make_halt_pause_mask_fn(halt_head, bank_layer, k_max)
    
    state1 = LPLState(
        bank=bank,
        heads=heads,
        round_idx=1,
        enabled=True,
        force_pause_mask=halt_pause_mask_fn,
    )
    
    # Capture halt statistics if requested
    halt_stats = None
    if capture_halt_stats:
        # We need to manually compute halt probs to return stats
        # Run a forward to get hidden states at bank_layer
        with lpl_state_scope(model, state1), torch.no_grad():
            _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        
        # Get the halt statistics from the state (count of pauses)
        n_pauses = state1.pause_count_per_layer[bank_layer]
        halt_stats = {"n_halts": n_pauses}
    else:
        with lpl_state_scope(model, state1), torch.no_grad():
            _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    
    # Round 2: read bank and compute logits with optional grad
    state2 = LPLState(
        bank=bank,
        heads=heads,
        round_idx=2,
        enabled=True,
        force_pause_mask=None,
    )
    
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            use_cache=False,
            return_dict=True,
        )
    
    return out.logits, halt_stats


def loss_from_logits(logits, input_ids, ans_start):
    """Standard cross-entropy loss on answer tokens."""
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def train_cell(
    model, tok, bank, heads, halt_head, bank_layer,
    train_items, b_raw, P,
    *,
    lam_act: float,
    k_max: int,
    steps: int,
    lr: float,
    device: str,
    seed: int,
):
    """Train a single (λ_act, K_max) cell.
    
    Returns:
        losses: list of per-step L_total values
        halt_counts: list of per-step n_halts
    """
    rng = random.Random(seed)
    d = model.config.hidden_size
    
    # Trainable: projector + bank_gate_heads + halt_head
    trainable = (
        list(P.parameters())
        + list(heads.bank_gate_heads.parameters())
        + list(halt_head.parameters())
    )
    opt = torch.optim.AdamW(trainable, lr=lr)
    
    losses = []
    halt_counts = []
    
    for step in range(steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
        
        # Rebuild bank with grad-tracking projector
        bank.frozen = False
        proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
        bank.slots[bank_layer] = proj
        bank.tags[bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True
        
        opt.zero_grad()
        
        # Forward with halt head
        logits, halt_stats = forward_lpl_k2_with_halt(
            model, bank, heads, enc, halt_head, bank_layer, k_max,
            grad=True, capture_halt_stats=True,
        )
        
        # NLL loss
        loss_nll = loss_from_logits(logits, enc.input_ids, ans)
        
        # ACT-style sparsity penalty: mean(halt_prob) over all positions
        # We need to recompute halt_prob for gradient tracking
        # Get hidden state at bank_layer round 1 — use state.last_layer_hidden as proxy
        # ISSUE: we need h_in at bank_layer during round 1, not round 2.
        # WORKAROUND: compute halt_prob from a separate forward to capture h_in at bank_layer.
        # For efficiency, we'll compute halt_prob from the training sample's hidden state.
        
        # Actually, we need to be more careful. Let me re-architect this:
        # The halt_head needs to see h_in at bank_layer during round 1.
        # We can capture this by running a mini-forward just to get h_in.
        
        # SIMPLIFIED APPROACH: compute halt_prob from input at bank_layer during training forward.
        # The halt_pause_mask_fn already computes halt_prob, but doesn't capture grad.
        # We need to recompute with grad tracking.
        
        # Get h_in at bank_layer: run a partial forward to bank_layer.
        # This is tricky because we need to extract intermediate activations.
        # PRAGMATIC SOLUTION: compute halt_prob on the final hidden state as a proxy.
        # This is not architecturally perfect, but works for the ACT penalty.
        
        # Let's use a simpler approach: compute halt_prob from the last_layer_hidden.
        # Actually, we should compute it from the h_in at bank_layer during round 1.
        
        # BETTER APPROACH: store halt_prob in the halt_pause_mask_fn and retrieve it.
        # We'll modify the function to capture halt_prob for gradient tracking.
        
        # CLEANEST APPROACH FOR TRAINING:
        # Compute halt_prob directly here with grad tracking, then use it in loss.
        # The halt_pause_mask_fn is only for the forward (no grad).
        
        # Recompute halt logits from h_in at bank_layer.
        # We need to extract h_in at bank_layer during round 1.
        # Since this is during training, we can do a separate forward pass to get h_in.
        
        # PRAGMATIC HACK: assume halt_prob on the answer tokens (where we compute NLL).
        # Actually, let's compute halt_prob on ALL positions at bank_layer.
        
        # To get h_in at bank_layer, we need to run model up to that layer.
        # This requires accessing intermediate activations.
        # PyTorch hooks or returning hidden_states from transformers.
        
        # SIMPLER WORKAROUND FOR NOW:
        # Use a global variable to capture halt_prob during the halt_pause_mask_fn.
        # This is hacky but works for this experiment.
        
        # Let's use a closure to capture halt_prob:
        captured_halt_prob = []
        
        def halt_pause_mask_fn_with_capture(layer: int, round_idx: int, h_in: torch.Tensor):
            if layer != bank_layer or round_idx != 1:
                return None
            halt_logits = halt_head(h_in)
            halt_prob = torch.sigmoid(halt_logits)
            # Store for loss computation (detached to avoid double-backward issues)
            # Actually, we want to keep grad for the penalty term!
            captured_halt_prob.append(halt_prob)  # [B, T]
            
            # K_max budget enforcement (same as before)
            B, T = halt_prob.shape
            threshold = 0.5
            if k_max < 0:
                pause_mask = halt_prob > threshold
            else:
                pause_mask = torch.zeros_like(halt_prob, dtype=torch.bool)
                for b in range(B):
                    probs = halt_prob[b]
                    candidates = (probs > threshold).nonzero(as_tuple=False).squeeze(-1)
                    if len(candidates) == 0:
                        continue
                    if len(candidates) > k_max:
                        candidate_probs = probs[candidates]
                        _, top_k_idx = torch.topk(candidate_probs, k=k_max, largest=True)
                        keep_positions = candidates[top_k_idx]
                    else:
                        keep_positions = candidates
                    pause_mask[b, keep_positions] = True
            
            return pause_mask.detach()  # detach mask to avoid tape issues
        
        # Now run the forward with capture
        state1 = LPLState(
            bank=bank, heads=heads, round_idx=1, enabled=True,
            force_pause_mask=halt_pause_mask_fn_with_capture,
        )
        with lpl_state_scope(model, state1):
            _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        
        # Extract captured halt_prob
        if len(captured_halt_prob) > 0:
            halt_prob_for_loss = captured_halt_prob[0]  # [B, T], with grad
            penalty = halt_prob_for_loss.mean()  # scalar
        else:
            penalty = torch.tensor(0.0, device=device)
        
        # Total loss
        loss_total = loss_nll + lam_act * penalty
        
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        
        losses.append(float(loss_nll.detach().cpu()))  # store NLL only for monitoring
        halt_counts.append(halt_stats["n_halts"] if halt_stats else 0)
    
    return losses, halt_counts


def eval_lpl_with_halt(model, tok, bank, heads, halt_head, bank_layer, items, device, k_max=-1):
    """Evaluate NLL and halt statistics on a set of items.
    
    Returns:
        mean_nll: average NLL
        mean_halts: average number of halts per sample
        mean_halt_prob: average halt probability per position per sample
    """
    d = model.config.hidden_size
    nlls = []
    total_halts = []
    total_halt_probs = []
    
    for sj, rl, tg in items:
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
        
        # Forward with halt head
        logits, halt_stats = forward_lpl_k2_with_halt(
            model, bank, heads, enc, halt_head, bank_layer, k_max,
            grad=False, capture_halt_stats=True,
        )
        
        nll = nll_on_answer(logits, enc.input_ids, ans)
        nlls.append(nll)
        total_halts.append(halt_stats["n_halts"])
    
    mean_nll = sum(nlls) / max(len(nlls), 1)
    mean_halts = sum(total_halts) / max(len(total_halts), 1)
    
    return mean_nll, mean_halts


def run_cell(
    args,
    lam_act: float,
    k_max: int,
    train_items,
    test_items,
    preload_keys,
    entries,
    tok,
    model,
    bank,
    heads,
    halt_head,
    b_raw,
    P,
):
    """Run a single (λ_act, K_max) cell: train + eval.
    
    Returns:
        result dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"CELL: λ_act={lam_act:.3f}, K_max={k_max}, seed={args.seed}")
    print(f"{'='*60}")
    
    d = model.config.hidden_size
    
    # Apply projector to preload bank
    with torch.no_grad():
        proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
    bank.frozen = False
    bank.slots[args.bank_layer] = proj
    bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
    bank.frozen = True
    
    # Baseline: eval before training
    def eval_baseline(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    base_nll = eval_baseline(test_items)
    
    # Pre-train eval with halt head (should be near base since halt_head is initialized to ~0 halt prob)
    pre_nll, pre_halts = eval_lpl_with_halt(
        model, tok, bank, heads, halt_head, args.bank_layer, test_items, args.device, k_max
    )
    
    print(f"BEFORE: base_nll={base_nll:.4f}, pre_nll={pre_nll:.4f}, pre_halts={pre_halts:.2f}")
    
    # Train
    t0 = time.time()
    losses, halt_counts = train_cell(
        model, tok, bank, heads, halt_head, args.bank_layer,
        train_items, b_raw, P,
        lam_act=lam_act, k_max=k_max,
        steps=args.steps, lr=args.lr, device=args.device, seed=args.seed,
    )
    train_time = time.time() - t0
    
    print(f"TRAIN: {args.steps} steps in {train_time:.1f}s, "
          f"loss_first10={sum(losses[:10])/10:.4f}, loss_last10={sum(losses[-10:])/10:.4f}, "
          f"halts_last10={sum(halt_counts[-10:])/10:.2f}")
    
    # Post-train eval
    post_nll, post_halts = eval_lpl_with_halt(
        model, tok, bank, heads, halt_head, args.bank_layer, test_items, args.device, k_max
    )
    
    delta_nll = post_nll - base_nll
    
    print(f"AFTER:  post_nll={post_nll:.4f}, post_halts={post_halts:.2f}, Δ={delta_nll:.4f}")
    
    # Effective K used per sample
    eff_k = post_halts
    
    # Pass criterion: mean halts ≤ 4 AND Δ ≤ -2.0
    cell_pass = (post_halts <= 4.0) and (delta_nll <= -2.0)
    
    result = {
        "lam_act": lam_act,
        "k_max": k_max,
        "base_nll": base_nll,
        "pre_nll": pre_nll,
        "pre_halts": pre_halts,
        "post_nll": post_nll,
        "post_halts": post_halts,
        "delta_nll": delta_nll,
        "eff_k": eff_k,
        "train_time_sec": train_time,
        "losses_first10": losses[:10],
        "losses_last10": losses[-10:],
        "halt_counts_last10": halt_counts[-10:],
        "cell_pass": cell_pass,
        "pass_rule": "post_halts <= 4 AND delta_nll <= -2.0",
    }
    
    return result


def build_split_canonical(entries, *, n_train, n_test, n_preload, seed):
    """Build (train_items, test_items, preload_keys) — canonical Phase B2 style.
    
    Random shuffle, train/test from bank.pt's 'split' field, preload from disjoint train slice.
    """
    rng = random.Random(seed)
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng.shuffle(train_keys)
    rng.shuffle(test_keys)
    train_keys = train_keys[:n_train]
    test_keys = test_keys[:n_test]
    
    # Preload from train split, disjoint from train_keys
    preload_pool = data_io.filter_keys(entries, split="train", solo_pass=True)
    preload_pool = [k for k in preload_pool if k not in set(train_keys)]
    rng.shuffle(preload_pool)
    preload_keys = preload_pool[:n_preload]
    
    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    
    return train_items, test_items, preload_keys


def main():
    p = argparse.ArgumentParser(description="e04 ACT-style halt-head training + K_max sweep")
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
    p.add_argument("--lam_grid", default="0.0,0.01,0.1,0.5,1.0", help="comma-separated λ_act values")
    p.add_argument("--kmax_grid", default="1,4,16,64,256,-1", help="comma-separated K_max values (-1=unlimited)")
    args = p.parse_args()
    
    # Parse grids
    lam_grid = [float(x) for x in args.lam_grid.split(",")]
    kmax_grid = [int(x) for x in args.kmax_grid.split(",")]
    
    print(f"[e04] λ_act grid: {lam_grid}")
    print(f"[e04] K_max grid: {kmax_grid}")
    print(f"[e04] Total cells: {len(lam_grid) * len(kmax_grid)}")
    
    # Load data
    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    
    train_items, test_items, preload_keys = build_split_canonical(
        entries, n_train=args.n_train, n_test=args.n_eval, n_preload=args.n_preload, seed=args.seed
    )
    print(f"[e04] train={len(train_items)} test={len(test_items)} preload={len(preload_keys)} seed={args.seed}")
    
    # Load model
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    
    # Create bank and heads
    bank = AttentionBank(
        num_layers=n_layers, hidden_size=d, device=args.device,
        dtype=torch.bfloat16, max_per_layer=args.n_preload + 256,
    )
    heads = LPLHeads.fresh(
        n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0, halt_bias=10.0,
        device=args.device, dtype=torch.float32,
    )
    install_lpl_patch(model)
    
    # Preload b-vectors
    b_raw = data_io.b_stack_for_keys(
        entries, preload_keys, target_norm=15.0, device=args.device, dtype=torch.float32
    )
    
    # Create projector
    P = make_projector(d, rank=args.rank).to(args.device).float()
    
    # Create per-position halt head (separate from LPLHeads.halt_head which is per-sample)
    halt_head = HaltHeadPerPosition(d, init_bias=-10.0).to(args.device).float()
    
    # Run all cells
    all_results = []
    cells_passed = []
    
    for lam in lam_grid:
        for kmax in kmax_grid:
            # Fresh projector, halt_head, bank for each cell to avoid contamination
            P_cell = make_projector(d, rank=args.rank).to(args.device).float()
            halt_head_cell = HaltHeadPerPosition(d, init_bias=-10.0).to(args.device).float()
            
            result = run_cell(
                args, lam, kmax,
                train_items, test_items, preload_keys, entries,
                tok, model, bank, heads, halt_head_cell, b_raw, P_cell,
            )
            all_results.append(result)
            
            if result["cell_pass"]:
                cells_passed.append(result)
            
            # Save individual cell result
            cell_path = HERE / "cells" / f"lam{lam:.3f}_kmax{kmax}_seed{args.seed}.json"
            cell_path.write_text(json.dumps(result, indent=2))
            print(f"  -> {cell_path}")
    
    # Summary
    experiment_pass = len(cells_passed) > 0
    best_cell = min(all_results, key=lambda r: r["delta_nll"])
    best_sparse = min([r for r in all_results if r["post_halts"] <= 4], 
                      key=lambda r: r["delta_nll"], default=None)
    
    summary = {
        "seed": args.seed,
        "model": args.model,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "lam_grid": lam_grid,
        "kmax_grid": kmax_grid,
        "total_cells": len(all_results),
        "cells_passed": len(cells_passed),
        "experiment_pass": experiment_pass,
        "pass_rule": "At least one cell with post_halts <= 4 AND delta_nll <= -2.0",
        "best_cell": {
            "lam_act": best_cell["lam_act"],
            "k_max": best_cell["k_max"],
            "delta_nll": best_cell["delta_nll"],
            "post_halts": best_cell["post_halts"],
        },
        "best_sparse_cell": {
            "lam_act": best_sparse["lam_act"],
            "k_max": best_sparse["k_max"],
            "delta_nll": best_sparse["delta_nll"],
            "post_halts": best_sparse["post_halts"],
        } if best_sparse else None,
        "all_results_summary": [
            {
                "lam_act": r["lam_act"],
                "k_max": r["k_max"],
                "delta_nll": r["delta_nll"],
                "post_halts": r["post_halts"],
                "cell_pass": r["cell_pass"],
            }
            for r in all_results
        ],
    }
    
    summary_path = HERE / f"e04_summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Cells passed: {len(cells_passed)}/{len(all_results)}")
    print(f"Experiment pass: {experiment_pass}")
    print(f"Best cell: λ={best_cell['lam_act']:.3f}, K_max={best_cell['k_max']}, "
          f"Δ={best_cell['delta_nll']:.4f}, halts={best_cell['post_halts']:.2f}")
    if best_sparse:
        print(f"Best sparse (≤4 halts): λ={best_sparse['lam_act']:.3f}, K_max={best_sparse['k_max']}, "
              f"Δ={best_sparse['delta_nll']:.4f}, halts={best_sparse['post_halts']:.2f}")
    print(f"Summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
