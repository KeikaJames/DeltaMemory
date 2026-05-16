"""e20b — trainable bank diagnostic.

Phase C / north-star pilot #2.

Following e20a (frozen projector → all deltas ≈ 0, projector confirmed essential),
this pilot inverts: freeze projector + gate heads, and make b_A vectors themselves
trainable nn.Parameters. Question: can information flow into the bank content
itself, producing item-specific memory?

If after training, Δ_A_init ≥ 1.0 AND Δ_A_after_evict << Δ_A_init (because b_B
is random and untrained), then content lives in the bank → north-star asymmetry
achievable by making the bank trainable.

Output: v2/experiments/e20_trainable_bank/seed{S}.json
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
    make_projector, load_model, nll_on_answer, encode_qa, data_io,
)


def forward_lpl_k2(model, bank, heads, enc, *, grad=False):
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
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


def eval_base(model, tok, items, device):
    nlls = []
    for sj, rl, tg in items:
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
        model.lpl_state = None
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
        nlls.append(nll_on_answer(logits, enc.input_ids, ans))
    return sum(nlls) / max(len(nlls), 1)


def eval_lpl(model, tok, bank, heads, items, device):
    nlls = []
    for sj, rl, tg in items:
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
        logits = forward_lpl_k2(model, bank, heads, enc, grad=False)
        nlls.append(nll_on_answer(logits, enc.input_ids, ans))
    return sum(nlls) / max(len(nlls), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--train_on", choices=["setA", "heldout"], default="setA",
                   help="memorization (setA) vs generalization (held-out)")
    args = p.parse_args()

    print(f"[e20b] device={args.device} model={args.model} seed={args.seed} N={args.N} lr={args.lr}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    all_keys = data_io.filter_keys(entries, solo_pass=True)
    rng = random.Random(args.seed)
    rng.shuffle(all_keys)
    setA_keys = all_keys[:args.N]
    setB_keys = all_keys[args.N:2*args.N]
    rest_keys = all_keys[2*args.N:]
    train_keys = rest_keys[:args.n_train]

    setA_items = data_io.items_for_keys(entries, setA_keys)
    setB_items = data_io.items_for_keys(entries, setB_keys)
    train_items = data_io.items_for_keys(entries, train_keys)
    print(f"[e20b] setA={len(setA_items)} setB={len(setB_items)} train={len(train_items)}")

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    install_lpl_patch(model)

    b_A_init = data_io.b_stack_for_keys(entries, setA_keys, target_norm=15.0,
                                         device=args.device, dtype=torch.float32)
    b_A = nn.Parameter(b_A_init.clone())
    b_B = data_io.b_stack_for_keys(entries, setB_keys, target_norm=15.0,
                                    device=args.device, dtype=torch.float32)

    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.N)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    for pp in heads.bank_gate_heads.parameters():
        pp.requires_grad_(False)
    P = make_projector(d, rank=args.rank).to(args.device).float()
    for pp in P.parameters():
        pp.requires_grad_(False)
    print(f"[e20b] trainable: b_A only ({b_A.numel()} elements)")

    def install_A():
        proj = (b_A + P(b_A)).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True

    def install_B():
        with torch.no_grad():
            proj = (b_B + P(b_B)).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(1, -1)] * proj.shape[0]
        bank.frozen = True

    print("[e20b] baselines...")
    base_A = eval_base(model, tok, setA_items, args.device)
    base_B = eval_base(model, tok, setB_items, args.device)
    print(f"[e20b] baseline: A={base_A:.4f} B={base_B:.4f}")

    opt = torch.optim.AdamW([b_A], lr=args.lr)
    trng = random.Random(args.seed)
    train_pool = setA_items if args.train_on == "setA" else train_items
    print(f"[e20b] training b_A only on {args.train_on} ({len(train_pool)} items), "
          f"lr={args.lr}, steps={args.steps}...")
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = trng.choice(train_pool)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        install_A()
        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([b_A], 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0 or step == 0:
            recent = sum(losses[-50:]) / min(50, len(losses))
            print(f"    step {step+1}/{args.steps} loss(avg50)={recent:.4f}")
    print(f"[e20b] training done ({time.time()-t0:.1f}s)")

    install_A()
    nll_A_init = eval_lpl(model, tok, bank, heads, setA_items, args.device)
    delta_A_init = base_A - nll_A_init
    print(f"[e20b] Δ_A_initial = {delta_A_init:.4f}")

    install_B()
    nll_A_after = eval_lpl(model, tok, bank, heads, setA_items, args.device)
    delta_A_after = base_A - nll_A_after
    print(f"[e20b] Δ_A_after_evict = {delta_A_after:.4f}")

    nll_B = eval_lpl(model, tok, bank, heads, setB_items, args.device)
    delta_B = base_B - nll_B
    print(f"[e20b] Δ_B = {delta_B:.4f}")

    bank.frozen = False
    bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device, dtype=torch.bfloat16)
    bank.tags[args.bank_layer] = []
    bank.frozen = True
    nll_A_zero = eval_lpl(model, tok, bank, heads, setA_items, args.device)
    delta_A_zero = base_A - nll_A_zero
    print(f"[e20b] Δ_A_zero = {delta_A_zero:.4f}")

    asymmetry = delta_A_init - delta_A_after
    ab_gap = delta_A_after - delta_B
    north_star_pass = (delta_A_init >= 3.0 and
                       delta_A_after <= 1.0 and
                       delta_B <= 1.0)
    verdict = {
        "north_star_pass": bool(north_star_pass),
        "rule": "Δ_A_init≥3.0 AND Δ_A_after_evict≤1.0 AND Δ_B≤1.0",
        "delta_A_init": delta_A_init, "delta_A_after": delta_A_after,
        "delta_B": delta_B, "delta_A_zero": delta_A_zero,
        "asymmetry_A_init_vs_after": asymmetry,
        "AB_gap_after_evict_vs_B": ab_gap,
    }
    print(f"[e20b] verdict: {verdict}")

    with torch.no_grad():
        drift = (b_A.detach() - b_A_init).norm() / b_A_init.norm()
    print(f"[e20b] b_A relative drift = {float(drift):.4f}")

    result = {
        "experiment": "e20b_trainable_bank",
        "model": args.model, "seed": args.seed, "N": args.N,
        "bank_layer": args.bank_layer, "rank": args.rank,
        "lr": args.lr, "steps": args.steps,
        "n_train": len(train_pool), "train_on": args.train_on,
        "n_setA": len(setA_items), "n_setB": len(setB_items),
        "n_trainable_params": b_A.numel(),
        "base_A": base_A, "base_B": base_B,
        "nll_A_init": nll_A_init, "nll_A_after": nll_A_after,
        "nll_B": nll_B, "nll_A_zero": nll_A_zero,
        "delta_A_init": delta_A_init, "delta_A_after": delta_A_after,
        "delta_B": delta_B, "delta_A_zero": delta_A_zero,
        "asymmetry_A_init_vs_after": asymmetry,
        "AB_gap_after_evict_vs_B": ab_gap,
        "b_A_relative_drift": float(drift),
        "verdict": verdict,
        "loss_first10": losses[:10], "loss_last10": losses[-10:],
    }
    out_path = HERE / f"seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[e20b] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
