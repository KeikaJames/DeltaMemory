"""e12: Long-term + Short-term memory coexistence driver.

Demonstrates that AttentionBank can simultaneously hold:
  1. Long-term knowledge (N_LT b-vectors preloaded from Exp35b, frozen).
  2. Short-term "scratch" content (N_ST items written via direct bank.write() 
     during a simulated round-1 pause, as if from in-context processing).

The key question: does ST write interfere with LT recall, or vice versa?

Design:
  - Preload N_LT Exp35b entries at bank_layer (default layer 9) as LT memory.
  - Train rank-64 projector (canonical e01 setup) for 200 steps on LT items.
  - Post-train: freeze projector, create 60 held-out "ST items" not in LT set.
  - Three eval regimes:
      * LT-only:  bank = LT preload only. Eval 60 LT items → Δ_LT.
      * ST-only:  bank = LT + ST writes. Eval 60 ST items → Δ_ST.
      * LT+ST:    bank = LT + ST writes. Eval 60 LT + 60 ST items → Δ_LT, Δ_ST.

Pass criteria:
  - LT-only Δ ≤ -1.5 (canonical B2 reproduction at 512 entries).
  - ST items Δ ≤ -1.0 in LT+ST regime (working memory works).
  - LT items in LT+ST regime within 0.3 of LT-only (no mutual interference).

Output: e12_seed{seed}.json with 4 Δs: LT-only, ST-only, LT+ST/LT, LT+ST/ST.

CLI: --seed, --device, --model, --n_LT (default 512), --n_ST (default 60),
     --n_train (default 120), --steps, --lr, --rank, --bank_layer.
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
    AttentionBank,
    LPLHeads,
    install_lpl_patch,
    LPLState,
    lpl_state_scope,
    make_projector,
    residual_apply,
    load_model,
    nll_on_answer,
    encode_qa,
    data_io,
)


def forward_lpl_k2(model, bank, heads, enc, *, grad=False):
    """Canonical 2-round LPL forward: round 1 pause-write, round 2 bank-read."""
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)

    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False, return_dict=True
        )
    return out.logits


def loss_from_logits(logits, input_ids, ans_start):
    pred = logits[0, ans_start - 1 : -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def build_splits(entries, n_train, n_LT, n_ST, seed):
    """Split into train, LT-eval, ST-eval with disjoint keys.
    
    Returns:
      train_items: list of (subj, rel, targ) for projector training.
      LT_preload_keys: list of keys for preloading LT bank entries.
      LT_eval_items: subset of LT items for eval (60 items).
      ST_eval_items: completely disjoint set of 60 items (held-out).
    """
    rng = random.Random(seed)
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng.shuffle(train_keys)
    rng.shuffle(test_keys)

    # Allocate pools: train, LT preload, LT eval, ST eval
    train_keys_pool = train_keys[:n_train]
    # LT preload + eval from train side, disjoint from train_keys_pool
    preload_pool = [k for k in train_keys if k not in set(train_keys_pool)]
    LT_preload_keys = preload_pool[:n_LT]
    # LT eval subset: first 60 of LT_preload_keys
    LT_eval_keys = LT_preload_keys[:min(60, len(LT_preload_keys))]
    # ST eval: completely disjoint, drawn from test side
    ST_eval_keys = test_keys[:n_ST]

    train_items = data_io.items_for_keys(entries, train_keys_pool)
    LT_eval_items = data_io.items_for_keys(entries, LT_eval_keys)
    ST_eval_items = data_io.items_for_keys(entries, ST_eval_keys)

    return train_items, LT_preload_keys, LT_eval_items, ST_eval_items


def encode_ST_content(tok, model, entries, ST_eval_keys, device):
    """Encode ST items as hidden vectors at last-token position via a single forward.
    
    Returns: [N_ST, d] tensor of h-vectors.
    """
    # For each ST item, encode its QA pair and do a forward to get last hidden.
    ST_hs = []
    for k in ST_eval_keys:
        e = entries[k]
        prompt = f"{e['subject']} {e['relation']}"
        target = e["target_str"]
        enc, _, _ = encode_qa(tok, prompt, target, device)
        # forward without LPL (vanilla pass-through)
        with torch.no_grad():
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        # extract last-layer last-token hidden
        h_last = out.hidden_states[-1][0, -1, :]  # [d]
        ST_hs.append(h_last.cpu())
    return torch.stack(ST_hs, dim=0)  # [N_ST, d]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_LT", type=int, default=512, help="Number of LT preload entries")
    p.add_argument("--n_ST", type=int, default=60, help="Number of ST (short-term) entries")
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else HERE / f"e12_seed{args.seed}.json"

    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]

    train_items, LT_preload_keys, LT_eval_items, ST_eval_items = build_splits(
        entries, args.n_train, args.n_LT, args.n_ST, args.seed
    )
    print(
        f"[e12] train={len(train_items)} LT_preload={len(LT_preload_keys)} "
        f"LT_eval={len(LT_eval_items)} ST_eval={len(ST_eval_items)} seed={args.seed}"
    )

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    bank = AttentionBank(
        num_layers=n_layers,
        hidden_size=d,
        device=args.device,
        dtype=torch.bfloat16,
        max_per_layer=args.n_LT + args.n_ST + 16,
    )
    heads = LPLHeads.fresh(
        n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0, halt_bias=10.0, device=args.device, dtype=torch.float32
    )
    install_lpl_patch(model)

    # Preload LT b-vectors
    b_LT = data_io.b_stack_for_keys(entries, LT_preload_keys, target_norm=15.0, device=args.device, dtype=torch.float32)
    print(f"[e12] preloaded LT b-vectors: {b_LT.shape}")

    # Build projector
    P = make_projector(d, rank=args.rank).to(args.device).float()

    def apply_LT_proj():
        """Apply projector to LT b-vectors and load into bank at bank_layer."""
        with torch.no_grad():
            proj = residual_apply(P, b_LT).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True

    apply_LT_proj()

    def eval_base(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    def eval_lpl(items):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    base_LT = eval_base(LT_eval_items)
    pre_real_LT = eval_lpl(LT_eval_items)
    print(f"[e12] BEFORE (LT-eval): base={base_LT:.4f}  real={pre_real_LT:.4f}")

    # === Training phase: train projector on train_items (use LT bank) ===
    rng = random.Random(args.seed)
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)

        # Rebuild bank with grad-enabled projector
        bank.frozen = False
        proj = (b_LT + P(b_LT)).to(dtype=torch.bfloat16)
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
            print(f"  [e12] step {step+1}/{args.steps} loss(avg25)={recent:.4f} ({time.time()-t0:.1f}s)")
    print(f"[e12] training done in {time.time()-t0:.1f}s")

    # Freeze projector for eval
    for param in P.parameters():
        param.requires_grad_(False)
    for param in heads.bank_gate_heads.parameters():
        param.requires_grad_(False)

    # === Eval 1: LT-only (bank = LT preload only, no ST writes) ===
    apply_LT_proj()
    LT_only_base = eval_base(LT_eval_items)
    LT_only_lpl = eval_lpl(LT_eval_items)
    LT_only_delta = LT_only_lpl - LT_only_base
    print(f"[e12] LT-only: base={LT_only_base:.4f} lpl={LT_only_lpl:.4f} Δ={LT_only_delta:.4f}")

    # === Prepare ST content: encode ST items as hidden vectors ===
    # We simulate ST writes as if pause-head had fired at these positions.
    # For simplicity, we extract last-layer hidden from a single forward (no LPL).
    ST_eval_keys = [k for k in entries.keys() if entries[k] in ST_eval_items]
    ST_hs = encode_ST_content(tok, model, entries, ST_eval_keys, args.device)
    print(f"[e12] ST content encoded: {ST_hs.shape}")

    # === Eval 2: ST-only (bank = LT + ST, eval on ST items only) ===
    # Direct write ST content into bank.slots[bank_layer] AFTER LT preload.
    apply_LT_proj()
    bank.frozen = False
    # Concatenate LT + ST projections into bank (ST items already projected via model's last layer)
    # We use the encoded ST_hs as-is (they are already hidden vectors from the model).
    # Apply a residual projection to ST content for consistency (train a trivial I+0 on them, or use as-is).
    # For this experiment, we apply same projector (I+P) to ST_hs for parity.
    ST_proj = residual_apply(P, ST_hs.to(args.device).float()).to(dtype=torch.bfloat16)
    combined = torch.cat([bank.slots[args.bank_layer], ST_proj], dim=0)
    bank.slots[args.bank_layer] = combined
    bank.tags[args.bank_layer].extend([(1, -1)] * ST_proj.shape[0])  # mark as round 1 writes
    bank.frozen = True
    print(f"[e12] bank after ST writes: {bank.slots[args.bank_layer].shape}")

    ST_only_base = eval_base(ST_eval_items)
    ST_only_lpl = eval_lpl(ST_eval_items)
    ST_only_delta = ST_only_lpl - ST_only_base
    print(f"[e12] ST-only: base={ST_only_base:.4f} lpl={ST_only_lpl:.4f} Δ={ST_only_delta:.4f}")

    # === Eval 3: LT+ST mix (same bank, eval both LT and ST items) ===
    # Bank already contains LT + ST. Eval on LT items first.
    LT_ST_LT_base = eval_base(LT_eval_items)
    LT_ST_LT_lpl = eval_lpl(LT_eval_items)
    LT_ST_LT_delta = LT_ST_LT_lpl - LT_ST_LT_base
    print(f"[e12] LT+ST (LT items): base={LT_ST_LT_base:.4f} lpl={LT_ST_LT_lpl:.4f} Δ={LT_ST_LT_delta:.4f}")

    # Eval on ST items (bank unchanged).
    LT_ST_ST_base = eval_base(ST_eval_items)
    LT_ST_ST_lpl = eval_lpl(ST_eval_items)
    LT_ST_ST_delta = LT_ST_ST_lpl - LT_ST_ST_base
    print(f"[e12] LT+ST (ST items): base={LT_ST_ST_base:.4f} lpl={LT_ST_ST_lpl:.4f} Δ={LT_ST_ST_delta:.4f}")

    # === Pass criteria ===
    # 1. LT-only Δ ≤ -1.5
    # 2. ST items Δ ≤ -1.0 in LT+ST regime
    # 3. LT items in LT+ST regime within 0.3 of LT-only (no interference)
    pass_LT_only = LT_only_delta <= -1.5
    pass_ST_works = LT_ST_ST_delta <= -1.0
    pass_no_interference = abs(LT_ST_LT_delta - LT_only_delta) <= 0.3

    verdict = {
        "pass_LT_only": pass_LT_only,
        "pass_ST_works": pass_ST_works,
        "pass_no_interference": pass_no_interference,
        "overall": pass_LT_only and pass_ST_works and pass_no_interference,
        "rules": [
            "LT-only Δ ≤ -1.5",
            "ST items Δ ≤ -1.0 in LT+ST regime",
            "LT items in LT+ST regime within 0.3 of LT-only",
        ],
    }

    out = {
        "experiment": "e12_LT_ST_coexist",
        "seed": args.seed,
        "model": args.model,
        "n_train": len(train_items),
        "n_LT": len(LT_preload_keys),
        "n_ST": len(ST_eval_items),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "results": {
            "LT_only": {
                "base": LT_only_base,
                "lpl": LT_only_lpl,
                "delta": LT_only_delta,
            },
            "ST_only": {
                "base": ST_only_base,
                "lpl": ST_only_lpl,
                "delta": ST_only_delta,
            },
            "LT_ST_mix_LT_items": {
                "base": LT_ST_LT_base,
                "lpl": LT_ST_LT_lpl,
                "delta": LT_ST_LT_delta,
            },
            "LT_ST_mix_ST_items": {
                "base": LT_ST_ST_base,
                "lpl": LT_ST_ST_lpl,
                "delta": LT_ST_ST_delta,
            },
        },
        "verdict": verdict,
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
        "n_train_params": sum(p.numel() for p in trainable),
    }

    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e12] -> {out_path}  verdict={verdict['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
