"""e11 noise robustness (deep falsifier battery following H2c collapsed-bank alarm).

**Context**: v2/experiments/e01_anticheat_b2 H2c_collapsed_bank showed that even
a bank with ALL 512 rows collapsed to the mean vector (i.e. zero row-distinctness,
single unique vector replicated N times) produced Δ = -4.81 NLL drop on the eval
set — nearly identical to canonical's -3.90. This suggests the projector might be
learning to exploit ANY extra attention keys, not the memory content itself.

This experiment runs SEVEN bank-content alterations, each holding everything
else (preload pipeline, training hyperparams, eval) IDENTICAL to the canonical
B2 setup. Each variant is a surgical bank mutation. The goal: determine whether
the NLL drop is driven by memory content, memory structure, attention-slot count,
or pure projector-training artifacts.

**Variants**:
    n1_iid_gaussian            replace bank rows with iid N(0, 1), L2-renorm to 15.0
    n2_uniform_sphere          rows = uniform unit vectors × 15.0
    n3_single_row_replicated   pick 1 real b-vector, replicate to all N rows
    n4_single_random_replicated pick 1 random Gaussian vector, replicate to all rows
    n5_constant_vector         all rows = fixed constant vector (e.g. e_1 × 15.0)
    n6_real_bank_K1            real bank but only N=1 row preloaded (tests: does N>1 matter?)
    n7_real_bank_K0_pure_proj  NO bank rows, but install LPL + train projector anyway

**Interpretation table** (if Δ ≤ -2 for any variant):
| variant | if Δ ≤ -2 | meaning |
|---|---|---|
| n1, n2 | pure noise still helps | claim is hollow (gain from attention machinery, not memory) |
| n3, n4, n5 | yes | confirms H2c: distinctness doesn't matter |
| n6 (K=1) | yes | even 1 slot suffices → about attention slot count |
| n7 (K=0) | yes | projector training alone causes gain → memory entirely ablated, claim dead |

If n7 shows Δ ≈ 0, memory is at least NECESSARY. If n7 shows Δ ≤ -2, that is
**terminal** for the v2 thesis as currently framed.

**Config**: same as e01_anticheat_b2 canonical: Qwen3-4B-Instruct-2507, layer 9,
rank 64, lr 2e-4, 200 steps, n_train=120, n_eval=120, n_preload=512 (except n6/n7
override), seed 0 default.

Output: v2/experiments/e11_noise_robustness/e11_{variant}_seed{seed}.json with:
- before/after NLL (base, real, rand, zero, off)
- bank_mean_norm (mean L2 norm of bank rows at preload)
- bank_distinctness (mean pairwise distance between bank rows — for n3/n4/n5 ~= 0)
- nll_drop
- verdict flag

Usage:
    python3 v2/experiments/e11_noise_robustness/run.py --variant n1_iid_gaussian --seed 0
    python3 v2/experiments/e11_noise_robustness/run.py --variant n7_real_bank_K0_pure_proj --seed 0
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


def forward_lpl_k2(model, bank, heads, enc, *, grad=False, randomize_bank=False, zero_bank=False):
    """2-round LPL forward: round 1 (no bank), round 2 (bank + gate).

    If randomize_bank=True, re-randomize bank slots before round 2 (for
    'rand' eval mode). If zero_bank=True, replace bank with zeros before round 2.
    """
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
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def compute_bank_stats(b_raw: torch.Tensor) -> dict:
    """Compute mean norm and mean pairwise distance (distinctness).

    b_raw: [N, d] tensor of bank rows.
    Returns: {"mean_norm": float, "mean_pairwise_dist": float}
    """
    norms = b_raw.norm(dim=-1)
    mean_norm = float(norms.mean())
    
    # pairwise distance: for large N this is O(N^2) memory, subsample if needed
    N = b_raw.shape[0]
    if N > 256:
        # subsample 256 random rows to compute distinctness estimate
        idx = torch.randperm(N)[:256]
        sub = b_raw[idx]
    else:
        sub = b_raw
    
    # pairwise L2 distance
    # dist[i,j] = ||sub[i] - sub[j]||
    # expand: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    norms_sq = (sub ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
    dots = sub @ sub.T  # [N, N]
    dists_sq = norms_sq + norms_sq.T - 2 * dots
    dists_sq = torch.clamp(dists_sq, min=0.0)
    dists = torch.sqrt(dists_sq)
    
    # mean over upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
    if mask.sum() == 0:
        mean_dist = 0.0
    else:
        mean_dist = float(dists[mask].mean())
    
    return {"mean_norm": mean_norm, "mean_pairwise_dist": mean_dist}


def mutate_bank(b_raw: torch.Tensor, variant: str, seed: int, device: str) -> torch.Tensor:
    """Apply the bank content mutation for the given variant.

    Returns a new [N, d] tensor with the mutation applied.
    """
    N, d = b_raw.shape
    
    if variant == "n1_iid_gaussian":
        # iid N(0, 1), L2-renorm to 15.0
        torch.manual_seed(seed)
        b_new = torch.randn(N, d, dtype=torch.float32, device=device)
        b_new = b_new / (b_new.norm(dim=-1, keepdim=True) + 1e-9) * 15.0
        return b_new
    
    elif variant == "n2_uniform_sphere":
        # uniform unit vectors × 15.0
        torch.manual_seed(seed)
        b_new = torch.randn(N, d, dtype=torch.float32, device=device)
        b_new = b_new / (b_new.norm(dim=-1, keepdim=True) + 1e-9) * 15.0
        return b_new
    
    elif variant == "n3_single_row_replicated":
        # pick 1 real row, replicate to all N
        torch.manual_seed(seed)
        idx = torch.randint(0, N, (1,)).item()
        b_new = b_raw[idx:idx+1].expand(N, d).contiguous()
        return b_new
    
    elif variant == "n4_single_random_replicated":
        # pick 1 random Gaussian vector, replicate to all rows
        torch.manual_seed(seed)
        single = torch.randn(1, d, dtype=torch.float32, device=device)
        single = single / (single.norm(dim=-1, keepdim=True) + 1e-9) * 15.0
        b_new = single.expand(N, d).contiguous()
        return b_new
    
    elif variant == "n5_constant_vector":
        # all rows = fixed constant vector (e.g. e_1 × 15.0)
        b_new = torch.zeros(N, d, dtype=torch.float32, device=device)
        b_new[:, 0] = 15.0
        return b_new
    
    elif variant == "n6_real_bank_K1":
        # real bank but only 1 row (pick random or first)
        # actually for this variant we will override N in the caller, but for
        # safety if called with N>1, just slice
        return b_raw[:1].clone()
    
    elif variant == "n7_real_bank_K0_pure_proj":
        # NO bank rows — return empty tensor
        return torch.empty(0, d, dtype=torch.float32, device=device)
    
    else:
        raise ValueError(f"Unknown variant: {variant}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=["n1_iid_gaussian", "n2_uniform_sphere",
                            "n3_single_row_replicated", "n4_single_random_replicated",
                            "n5_constant_vector", "n6_real_bank_K1",
                            "n7_real_bank_K0_pure_proj"])
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
    
    # Override n_preload for n6 and n7
    if args.variant == "n6_real_bank_K1":
        args.n_preload = 1
    elif args.variant == "n7_real_bank_K0_pure_proj":
        args.n_preload = 0
    
    out_path = Path(args.out) if args.out else HERE / f"e11_{args.variant}_seed{args.seed}.json"
    
    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    
    # Same split strategy as e01 canonical
    keys = list(entries.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng.shuffle(train_keys); rng.shuffle(test_keys)
    train_keys = train_keys[:args.n_train]
    test_keys = test_keys[:args.n_eval]
    
    # Preload pool: from train split, disjoint from train_keys
    preload_pool = data_io.filter_keys(entries, split="train", solo_pass=True)
    preload_pool = [k for k in preload_pool if k not in set(train_keys)]
    
    if args.n_preload > 0:
        preload_keys = preload_pool[:args.n_preload]
    else:
        preload_keys = []
    
    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    
    print(f"[e11:{args.variant}] train={len(train_items)} test={len(test_items)} "
          f"preload={len(preload_keys)} seed={args.seed}")
    
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=max(args.n_preload + 16, 16))
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)
    
    # Load real bank rows
    if len(preload_keys) > 0:
        b_raw = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                          device=args.device, dtype=torch.float32)
        # Apply variant mutation
        b_raw = mutate_bank(b_raw, args.variant, args.seed, args.device)
        bank_stats = compute_bank_stats(b_raw.cpu())
    else:
        # n7: no bank rows
        b_raw = torch.empty(0, d, device=args.device, dtype=torch.float32)
        bank_stats = {"mean_norm": 0.0, "mean_pairwise_dist": 0.0}
    
    print(f"[e11:{args.variant}] bank_stats: norm={bank_stats['mean_norm']:.2f} "
          f"distinctness={bank_stats['mean_pairwise_dist']:.2f}")
    
    P = make_projector(d, rank=args.rank).to(args.device).float()
    
    def apply_proj(zero_bank=False):
        if b_raw.shape[0] == 0:
            # n7: no bank rows, leave empty
            bank.frozen = False
            bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device, dtype=torch.bfloat16)
            bank.tags[args.bank_layer] = []
            bank.frozen = True
            return
        with torch.no_grad():
            proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
        if zero_bank:
            proj = torch.zeros_like(proj)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True
    
    apply_proj()
    
    def eval_base(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    def eval_lpl(items, *, randomize=False, zero_bank=False, bank_off=False):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            apply_proj(zero_bank=zero_bank or bank_off)
            if bank_off:
                # truly disable: empty slots
                bank.frozen = False
                bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device, dtype=torch.bfloat16)
                bank.tags[args.bank_layer] = []
                bank.frozen = True
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False,
                                     randomize_bank=randomize, zero_bank=False)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    base = eval_base(test_items)
    pre_real = eval_lpl(test_items)
    pre_rand = eval_lpl(test_items, randomize=True)
    pre_zero = eval_lpl(test_items, zero_bank=True)
    pre_off = eval_lpl(test_items, bank_off=True)
    print(f"[e11:{args.variant}] BEFORE: base={base:.4f}  real={pre_real:.4f}  "
          f"rand={pre_rand:.4f}  zero={pre_zero:.4f}  off={pre_off:.4f}")
    
    # === training ===
    rng = random.Random(args.seed)
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        
        # rebuild bank with grad-tracking projector each step
        bank.frozen = False
        if b_raw.shape[0] == 0:
            # n7: no bank rows
            bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device, dtype=torch.bfloat16)
            bank.tags[args.bank_layer] = []
        else:
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
            print(f"  [e11:{args.variant}] step {step+1}/{args.steps} loss(avg25)={recent:.4f} ({time.time()-t0:.1f}s)")
    print(f"[e11:{args.variant}] training done in {time.time()-t0:.1f}s")
    
    # === post-train eval ===
    post_real = eval_lpl(test_items)
    post_rand = eval_lpl(test_items, randomize=True)
    post_zero = eval_lpl(test_items, zero_bank=True)
    post_off = eval_lpl(test_items, bank_off=True)
    print(f"[e11:{args.variant}] AFTER:  base={base:.4f}  real={post_real:.4f}  "
          f"rand={post_rand:.4f}  zero={post_zero:.4f}  off={post_off:.4f}")
    
    # Δ = post_real - base (signed; negative = NLL drop = "noise helped").
    # PASS means: noise/degenerate-bank did NOT reduce NLL by >= 2 — i.e. the
    # *information* claim survives. FAIL means: noise/degenerate bank still
    # produced a strong NLL drop, so the bank-content interpretation is hollow.
    delta_signed = post_real - base
    nll_drop = base - post_real  # kept for backward compat (== -delta_signed)
    threshold = 2.0  # "significant" NLL drop magnitude in either direction
    helped = delta_signed <= -threshold  # noise/degenerate bank reduced NLL by >= 2

    # Verdict logic per variant
    verdict = {}

    if args.variant in ["n1_iid_gaussian", "n2_uniform_sphere"]:
        verdict = {
            "pass": not helped,
            "rule": f"pure noise should NOT reduce NLL by >=2 (got Δ_signed={delta_signed:+.2f})",
            "interpretation": "FAIL = noise still helps → claim hollow" if helped else "PASS = noise does not help"
        }
    elif args.variant in ["n3_single_row_replicated", "n4_single_random_replicated", "n5_constant_vector"]:
        verdict = {
            "pass": not helped,
            "rule": f"zero-distinctness bank should NOT reduce NLL by >=2 (got Δ_signed={delta_signed:+.2f})",
            "interpretation": "FAIL = zero distinctness helps → H2c confirmed" if helped else "PASS = distinctness matters"
        }
    elif args.variant == "n6_real_bank_K1":
        verdict = {
            "pass": not helped,
            "rule": f"single bank slot should NOT reduce NLL by >=2 (got Δ_signed={delta_signed:+.2f})",
            "interpretation": "FAIL = 1 slot suffices → about slot count not content" if helped else "PASS = multiple slots needed"
        }
    elif args.variant == "n7_real_bank_K0_pure_proj":
        verdict = {
            "pass": not helped,
            "rule": f"no bank (K=0) should NOT reduce NLL by >=2 (got Δ_signed={delta_signed:+.2f})",
            "interpretation": "FAIL = no bank helps → claim DEAD (terminal)" if helped else "PASS = memory necessary"
        }
    else:
        verdict = {"pass": None, "rule": "unknown variant", "interpretation": "N/A"}
    verdict["delta_signed"] = delta_signed
    verdict["nll_drop"] = nll_drop

    
    out = {
        "variant": args.variant,
        "seed": args.seed,
        "model": args.model,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "bank_stats": bank_stats,
        "before": {
            "base": base,
            "real": pre_real,
            "rand": pre_rand,
            "zero": pre_zero,
            "off": pre_off
        },
        "after": {
            "base": base,
            "real": post_real,
            "rand": post_rand,
            "zero": post_zero,
            "off": post_off
        },
        "nll_drop": nll_drop,
        "verdict": verdict,
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
        "n_train_params": sum(p.numel() for p in trainable),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e11:{args.variant}] -> {out_path}")
    print(f"[e11:{args.variant}] verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
