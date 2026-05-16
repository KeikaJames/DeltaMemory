"""e07 — Per-layer K-projector ablation.

Test whether installing INDEPENDENT rank-64 K-projectors at MULTIPLE layers
(each writing to its own bank slot) beats single-layer (layer 9).

Experimental matrix:
| condition | layers             | total params |
| --------- | ------------------ | ------------ |
| baseline  | {9}                | 1× P         |
| triple    | {3, 9, 21}         | 3× P         |
| 6-layer   | {3, 9, 15, 21, 27, 33} | 6× P     |
| all-even  | {0,2,4,...,34}     | 18× P        |

Each layer gets its OWN rank-64 LowRankProj. Preload bank slots for ALL
chosen layers (use bank_round1 hidden states from those layers).

For each layer_set: install LPL patch on ALL layers (the patch is global),
train all projectors jointly (single optimizer over all P params + all
bank_gate params).

Eval: BEFORE / AFTER NLL with bank-real, bank-zero, bank-off (across all
layers — if any layer is "off", that layer reverts to base attention).

Pass criterion: Δ_real(triple) - Δ_real(single) ≤ -0.5 (multi-layer adds
meaningful additional gain over single).

Output: v2/experiments/e07_perlayer_kproj/e07_seed{seed}.json with one cell
per layer_set.

NOTE on multi-layer patching:
The qwen3_lpl_patch.install_lpl_patch() patches ALL layers globally. Each
patched layer reads from its own bank.slots[layer_idx]. So to use multiple
layers, we:
1. Create separate projectors for each layer in the layer_set
2. Populate bank.slots[layer_idx] for each chosen layer
3. Train all projectors jointly
The patch automatically handles per-layer bank reads and per-layer bank_gate.
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
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa,
    data_io,
)


def forward_lpl_k2(model, bank, heads, enc, *, grad=False, zero_bank_layers=None):
    """Two-round LPL forward.
    
    Args:
        zero_bank_layers: if given, set bank.slots[layer] = zeros for these layers.
    """
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    
    # Apply zero-bank if requested
    if zero_bank_layers is not None:
        for layer_idx in zero_bank_layers:
            if bank.slots[layer_idx].shape[0] > 0:
                bank.slots[layer_idx] = torch.zeros_like(bank.slots[layer_idx])
    
    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, input_ids, ans_start):
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def disjoint_split(entries, n_train, n_test, n_preload, seed):
    """Build train/test/preload with strict (subject ∪ relation) disjoint."""
    keys = list(entries.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    subjects, relations = set(), set()
    train, test = [], []
    for k in keys:
        e = entries[k]
        if not e.get("solo_pass"):
            continue
        s = e["subject"]
        r = e["relation"]
        if len(train) < n_train:
            train.append(k)
            subjects.add(s)
            relations.add(r)
        else:
            if s in subjects or r in relations:
                continue
            test.append(k)
            if len(test) >= n_test:
                break
    # preload from disjoint train pool
    pool = [k for k in keys if entries[k].get("solo_pass") and k not in set(train)]
    preload = pool[:n_preload]
    return (data_io.items_for_keys(entries, train),
            data_io.items_for_keys(entries, test), preload)


def run_layer_set(
    layers: list[int],
    tok, model, entries,
    n_train: int, n_eval: int, n_preload: int,
    steps: int, lr: float, rank: int, seed: int, device: str
):
    """Train and eval a single layer_set configuration.
    
    Returns a dict with before/after metrics.
    """
    train_items, test_items, preload_keys = disjoint_split(
        entries, n_train, n_eval, n_preload, seed
    )
    
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    
    # Create fresh bank and heads for this run
    bank = AttentionBank(
        num_layers=n_layers, hidden_size=d, device=device,
        dtype=torch.bfloat16, max_per_layer=n_preload + 16
    )
    heads = LPLHeads.fresh(
        n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
        halt_bias=10.0, device=device, dtype=torch.float32
    )
    
    # Load preload vectors (same for all layers, from base model hidden states)
    b_raw = data_io.b_stack_for_keys(
        entries, preload_keys, target_norm=15.0,
        device=device, dtype=torch.float32
    )
    
    # Create one projector PER layer
    Ps = nn.ModuleDict({
        str(layer_idx): make_projector(d, rank=rank).to(device).float()
        for layer_idx in layers
    }).to(device)
    
    def apply_proj(zero_layers=None):
        """Populate bank slots for all chosen layers."""
        zero_layers = zero_layers or []
        bank.frozen = False
        for layer_idx in layers:
            P = Ps[str(layer_idx)]
            with torch.no_grad():
                if layer_idx in zero_layers:
                    proj = torch.zeros(b_raw.shape[0], d, device=device, dtype=torch.bfloat16)
                else:
                    proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
            bank.slots[layer_idx] = proj
            bank.tags[layer_idx] = [(0, -1)] * proj.shape[0]
        bank.frozen = True
    
    def apply_proj_trainable():
        """Populate bank slots with grad-tracking projectors (for training)."""
        bank.frozen = False
        for layer_idx in layers:
            P = Ps[str(layer_idx)]
            proj = (b_raw + P(b_raw)).to(dtype=torch.bfloat16)
            bank.slots[layer_idx] = proj
            bank.tags[layer_idx] = [(0, -1)] * proj.shape[0]
        bank.frozen = True
    
    # Initial bank population
    apply_proj()
    
    def eval_base(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    def eval_lpl(items, *, zero_layers=None, bank_off=False):
        """Eval with LPL.
        
        zero_layers: list of layer indices to zero out.
        bank_off: if True, empty ALL bank slots (disable bank entirely).
        """
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
            
            if bank_off:
                # Empty all bank slots to disable
                bank.frozen = False
                for layer_idx in layers:
                    bank.slots[layer_idx] = torch.empty(0, d, device=device, dtype=torch.bfloat16)
                    bank.tags[layer_idx] = []
                bank.frozen = True
            else:
                apply_proj(zero_layers=zero_layers)
            
            logits = forward_lpl_k2(
                model, bank, heads, enc, grad=False,
                zero_bank_layers=None  # already applied in apply_proj
            )
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    # === PRE-TRAIN EVAL ===
    base = eval_base(test_items)
    pre_real = eval_lpl(test_items)
    pre_zero = eval_lpl(test_items, zero_layers=layers)  # zero ALL active layers
    pre_off = eval_lpl(test_items, bank_off=True)
    
    print(f"  [e07:layers={layers}] BEFORE: base={base:.4f}  real={pre_real:.4f}  "
          f"zero={pre_zero:.4f}  off={pre_off:.4f}")
    
    # === TRAINING ===
    rng = random.Random(seed)
    # Collect all trainable params: all projectors + bank_gate heads for active layers
    trainable = list(Ps.parameters())
    for layer_idx in layers:
        trainable.extend(heads.bank_gate_heads[layer_idx].parameters())
    
    opt = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    t0 = time.time()
    
    for step in range(steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
        
        apply_proj_trainable()
        
        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        
        if (step + 1) % 50 == 0 or step == 0:
            recent = sum(losses[-50:]) / min(50, len(losses))
            elapsed = time.time() - t0
            print(f"    step {step+1}/{steps} loss(avg50)={recent:.4f} ({elapsed:.1f}s)")
    
    train_time = time.time() - t0
    print(f"  [e07:layers={layers}] training done in {train_time:.1f}s")
    
    # === POST-TRAIN EVAL ===
    post_real = eval_lpl(test_items)
    post_zero = eval_lpl(test_items, zero_layers=layers)
    post_off = eval_lpl(test_items, bank_off=True)
    
    print(f"  [e07:layers={layers}] AFTER:  base={base:.4f}  real={post_real:.4f}  "
          f"zero={post_zero:.4f}  off={post_off:.4f}")
    
    # Compute deltas
    delta_real = post_real - base
    delta_zero = post_zero - base
    delta_off = post_off - base
    
    n_params_per_proj = sum(p.numel() for p in Ps[str(layers[0])].parameters())
    n_params_total = sum(p.numel() for p in trainable)
    
    return {
        "layers": layers,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "before": {
            "base": float(base),
            "real": float(pre_real),
            "zero": float(pre_zero),
            "off": float(pre_off),
        },
        "after": {
            "base": float(base),
            "real": float(post_real),
            "zero": float(post_zero),
            "off": float(post_off),
        },
        "deltas": {
            "real": float(delta_real),
            "zero": float(delta_zero),
            "off": float(delta_off),
        },
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
        "n_params_per_proj": n_params_per_proj,
        "n_params_total": n_params_total,
        "train_time_s": train_time,
    }


def main():
    p = argparse.ArgumentParser(description="e07 per-layer K-projector ablation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument(
        "--layer_sets",
        default="9;3,9,21;3,9,15,21,27,33",
        help="Semicolon-separated list of layer sets, e.g. '9;3,9,21;0,2,4,6,8'"
    )
    args = p.parse_args()
    
    # Parse layer_sets
    layer_sets = []
    for part in args.layer_sets.split(";"):
        layer_list = [int(x.strip()) for x in part.split(",")]
        layer_sets.append(layer_list)
    
    print(f"[e07] Running {len(layer_sets)} layer configurations:")
    for ls in layer_sets:
        print(f"  {ls}")
    
    # Load data
    blob = data_io.load_bank_blob()
    entries = blob["entries"]
    
    # Load model (shared across all runs)
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    
    # Install LPL patch ONCE (it patches ALL layers globally)
    install_lpl_patch(model, state=None)
    
    print(f"[e07] Model loaded: {model.config.num_hidden_layers} layers, "
          f"d={model.config.hidden_size}")
    print(f"[e07] Training params: steps={args.steps}, lr={args.lr}, rank={args.rank}")
    print(f"[e07] Data: n_train={args.n_train}, n_eval={args.n_eval}, "
          f"n_preload={args.n_preload}")
    
    # Run each layer_set configuration
    results = []
    for i, layers in enumerate(layer_sets, 1):
        print(f"\n[e07] === Configuration {i}/{len(layer_sets)}: layers={layers} ===")
        cell = run_layer_set(
            layers=layers,
            tok=tok,
            model=model,
            entries=entries,
            n_train=args.n_train,
            n_eval=args.n_eval,
            n_preload=args.n_preload,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank,
            seed=args.seed,
            device=args.device,
        )
        results.append(cell)
    
    # Compute pass criterion: triple vs single
    # Find single-layer (typically [9]) and triple (typically [3,9,21])
    single_result = next((r for r in results if len(r["layers"]) == 1), None)
    triple_result = next((r for r in results if len(r["layers"]) == 3), None)
    
    pass_criterion = None
    if single_result and triple_result:
        delta_single = single_result["deltas"]["real"]
        delta_triple = triple_result["deltas"]["real"]
        improvement = delta_triple - delta_single  # More negative = better
        pass_criterion = {
            "delta_single": delta_single,
            "delta_triple": delta_triple,
            "improvement": improvement,
            "pass": improvement <= -0.5,
            "rule": "Δ_real(triple) - Δ_real(single) ≤ -0.5 (multi-layer adds gain)"
        }
        print(f"\n[e07] PASS CRITERION: improvement={improvement:.4f}  "
              f"pass={pass_criterion['pass']}")
    
    # Write output
    out_path = HERE / f"e07_seed{args.seed}.json"
    output = {
        "experiment": "e07_perlayer_kproj",
        "seed": args.seed,
        "model": args.model,
        "config": {
            "n_train": args.n_train,
            "n_eval": args.n_eval,
            "n_preload": args.n_preload,
            "steps": args.steps,
            "lr": args.lr,
            "rank": args.rank,
        },
        "layer_sets": [r["layers"] for r in results],
        "results": results,
        "pass_criterion": pass_criterion,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[e07] -> {out_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
