"""e05 cross-model LPL driver — QWEN3-ONLY architecture check.

CRITICAL: The v2 LPL patch (qwen3_lpl_patch.py) is Qwen3-specific. This driver:
- Defaults to Qwen/Qwen3-1.7B (smaller variant, hidden_size=2048).
- Supports Qwen/Qwen3-4B-Instruct-2507 and Qwen/Qwen3-8B-Instruct-2507.
- For non-Qwen3 models: EXIT WITH WARNING (model_type check).

Design:
- Preload 512 b-vectors from Exp35b bank.pt at layer args.bank_layer (default 9).
- Exp35b bank.pt b-vectors are 2560-dim (from Qwen3-4B). For Qwen3-1.7B
  (hidden_size=2048), dimensions MISMATCH. We handle this via:
      * Fixed random Gaussian projection (torch.manual_seed=0) from 2560→2048.
      * L2-renorm to 15.0 after projection.
      * Documented in JSON output: "dim_projection_applied": true.
- Train rank-64 (I+P) projector + bank_gate_heads for 200 steps, lr=2e-4.
- Report BEFORE/AFTER (base / real / rand / zero / off) NLL on 120-item test set.
- Pass criterion: Δ_real ≤ -1.0 AND Δ_rand ≥ -0.2 (signal exists, random control flat).

Output: v2/experiments/e05_cross_model/e05_{model_safe}_seed{seed}.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
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
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa,
    data_io,
)


def forward_lpl_k2(model, bank, heads, enc, *, grad=False, randomize_bank=False, zero_bank=False):
    """Two-round LPL forward: round1 (pause + populate bank), round2 (read bank)."""
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    
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
    """Cross-entropy loss on answer tokens."""
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def check_qwen3_compatibility(model):
    """Exit with warning if model is not Qwen3 architecture."""
    model_type = getattr(model.config, "model_type", "unknown")
    if model_type != "qwen3":
        print(f"[e05:FATAL] Model type '{model_type}' is NOT Qwen3.")
        print("[e05:FATAL] The LPL patch (qwen3_lpl_patch.py) ONLY supports 'qwen3' architecture.")
        print("[e05:FATAL] Running on incompatible models will SILENTLY BREAK.")
        print("[e05:FATAL] Supported models: Qwen/Qwen3-1.7B, Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-8B-Instruct-2507")
        sys.exit(1)
    print(f"[e05] Model type: {model_type} ✓ (Qwen3 compatible)")


def maybe_project_b_vectors(b_raw, target_hidden_size, device):
    """Project b-vectors if dimension mismatch. Returns (b_out, projection_applied: bool)."""
    b_dim = b_raw.shape[1]
    if b_dim == target_hidden_size:
        return b_raw, False
    
    print(f"[e05] Dimension mismatch: b_raw={b_dim}, model hidden_size={target_hidden_size}")
    print(f"[e05] Applying fixed random Gaussian projection {b_dim}→{target_hidden_size} (seed=0)")
    
    # Fixed random projection for reproducibility
    torch.manual_seed(0)
    projection_matrix = torch.randn(b_dim, target_hidden_size, dtype=torch.float32)
    # Normalize columns to preserve variance
    projection_matrix = projection_matrix / (projection_matrix.norm(dim=0, keepdim=True) + 1e-9)
    projection_matrix = projection_matrix.to(device=device)
    
    # Apply projection: [N, b_dim] @ [b_dim, target_hidden_size] → [N, target_hidden_size]
    b_projected = b_raw @ projection_matrix
    # Renormalize to L2=15.0
    b_projected = b_projected / (b_projected.norm(dim=-1, keepdim=True) + 1e-9) * 15.0
    
    print(f"[e05] Projection applied. New shape: {b_projected.shape}, L2 norm: {b_projected.norm(dim=-1).mean():.2f}")
    return b_projected, True


def main():
    p = argparse.ArgumentParser(description="e05 cross-model LPL (Qwen3-only)")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B",
                   help="Model name (Qwen3 architecture only)")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--bank_layer", type=int, default=9,
                   help="Layer index for bank preload (adjust for smaller models)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    
    # Safe filename from model name
    model_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model.split("/")[-1])
    out_path = Path(args.out) if args.out else HERE / f"e05_{model_safe}_seed{args.seed}.json"
    
    print(f"[e05] Loading model: {args.model}")
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    
    # CRITICAL: Check Qwen3 compatibility
    check_qwen3_compatibility(model)
    
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"[e05] Model config: n_layers={n_layers}, hidden_size={d}")
    
    # Adjust bank_layer if it exceeds model layers
    if args.bank_layer >= n_layers:
        old_layer = args.bank_layer
        args.bank_layer = min(args.bank_layer, n_layers - 1)
        print(f"[e05] WARNING: bank_layer {old_layer} >= {n_layers}, clamped to {args.bank_layer}")
    
    # Load bank data
    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    
    # Build train/test split
    rng = random.Random(args.seed)
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng.shuffle(train_keys); rng.shuffle(test_keys)
    train_keys = train_keys[:args.n_train]
    test_keys = test_keys[:args.n_eval]
    
    # Preload keys from train, disjoint from train_keys
    preload_pool = [k for k in data_io.filter_keys(entries, split="train", solo_pass=True)
                    if k not in set(train_keys)]
    preload_keys = preload_pool[:args.n_preload]
    
    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    
    print(f"[e05] train={len(train_items)} test={len(test_items)} preload={len(preload_keys)} seed={args.seed}")
    
    # Build LPL components
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)
    
    # Load and project b-vectors
    b_raw = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                      device=args.device, dtype=torch.float32)
    print(f"[e05] Loaded b_raw: shape={b_raw.shape}, dtype={b_raw.dtype}")
    
    b_raw, dim_projection_applied = maybe_project_b_vectors(b_raw, d, args.device)
    
    # Projector
    P = make_projector(d, rank=args.rank).to(args.device).float()
    
    def apply_proj(zero_bank=False):
        """Apply (I+P) to b_raw and load into bank at args.bank_layer."""
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
        """Base model NLL (no LPL)."""
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    def eval_lpl(items, *, randomize=False, zero_bank=False, bank_off=False):
        """LPL NLL with various bank modes."""
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            apply_proj(zero_bank=zero_bank or bank_off)
            if bank_off:
                # Truly disable: empty bank slots
                bank.frozen = False
                bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device, dtype=torch.bfloat16)
                bank.tags[args.bank_layer] = []
                bank.frozen = True
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False,
                                     randomize_bank=randomize, zero_bank=False)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    # === BEFORE training ===
    print("[e05] Evaluating BEFORE training...")
    base = eval_base(test_items)
    pre_real = eval_lpl(test_items)
    pre_rand = eval_lpl(test_items, randomize=True)
    pre_zero = eval_lpl(test_items, zero_bank=True)
    pre_off = eval_lpl(test_items, bank_off=True)
    print(f"[e05] BEFORE: base={base:.4f}  real={pre_real:.4f}  "
          f"rand={pre_rand:.4f}  zero={pre_zero:.4f}  off={pre_off:.4f}")
    
    # === TRAINING ===
    print(f"[e05] Training for {args.steps} steps...")
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []
    t0 = time.time()
    
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        
        # Rebuild bank with grad-tracking projector
        bank.frozen = False
        proj = (b_raw + P(b_raw)).to(dtype=torch.bfloat16)
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
            print(f"  [e05] step {step+1}/{args.steps} loss(avg25)={recent:.4f} ({time.time()-t0:.1f}s)")
    
    print(f"[e05] Training done in {time.time()-t0:.1f}s")
    
    # === AFTER training ===
    print("[e05] Evaluating AFTER training...")
    post_real = eval_lpl(test_items)
    post_rand = eval_lpl(test_items, randomize=True)
    post_zero = eval_lpl(test_items, zero_bank=True)
    post_off = eval_lpl(test_items, bank_off=True)
    print(f"[e05] AFTER:  base={base:.4f}  real={post_real:.4f}  "
          f"rand={post_rand:.4f}  zero={post_zero:.4f}  off={post_off:.4f}")
    
    # === VERDICT ===
    delta_real = base - post_real
    delta_rand = base - post_rand
    pass_real = delta_real >= 1.0
    pass_rand = delta_rand <= 0.2
    verdict = {
        "pass": pass_real and pass_rand,
        "delta_real": delta_real,
        "delta_rand": delta_rand,
        "criterion_real": "Δ_real ≥ 1.0 (signal exists)",
        "criterion_rand": "Δ_rand ≤ 0.2 (random control flat)",
        "pass_real": pass_real,
        "pass_rand": pass_rand,
    }
    
    # Gate stats
    gate_stats = {}
    for li, gh in enumerate(heads.bank_gate_heads):
        w = gh.proj.weight.detach().float().cpu()
        b_ = gh.proj.bias.detach().float().cpu()
        gate_stats[f"layer{li}"] = {"w_norm": float(w.norm()), "b": float(b_)}
    
    # Output JSON
    out = {
        "experiment": "e05_cross_model",
        "model": args.model,
        "model_type": model.config.model_type,
        "seed": args.seed,
        "n_layers": n_layers,
        "hidden_size": d,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "dim_projection_applied": dim_projection_applied,
        "dim_projection_note": (
            f"b-vectors projected from 2560→{d} via fixed Gaussian (seed=0), renorm L2=15.0"
            if dim_projection_applied else "No projection needed (dimensions match)"
        ),
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
        "gate_stats_sample": {k: gate_stats[k] for k in list(gate_stats)[:6]},
        "elapsed_sec": time.time() - t0,
    }
    
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e05] -> {out_path}")
    print(f"[e05] Verdict: {verdict}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
