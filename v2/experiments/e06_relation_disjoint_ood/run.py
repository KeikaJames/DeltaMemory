"""e06 — Relation-Disjoint OOD test.

Tests whether the K-projector generalizes across relations or merely
memorizes per-relation routing. Splits Exp35b relations into TRAIN_RELS
and TEST_RELS with EMPTY intersection:
    - preload: 512 b-vectors from TRAIN_RELS only
    - train: 120 items from TRAIN_RELS
    - test: 120 items from TEST_RELS (HARD OOD: never seen during training)

Pass criterion: Δ NLL on OOD test_rel ≤ -1.0 (projector generalizes).

CLI: --seed, --device, --model, --n_preload, --n_train, --n_eval, --steps,
     --lr, --rank, --bank_layer (default 9).

Output: JSON to v2/experiments/e06_relation_disjoint_ood/e06_seed{seed}.json.
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


def forward_lpl_k2(model, bank, heads, enc, *, grad=False, randomize_bank=False):
    """Two-round forward: R1 pauses (inactive), R2 retrieves from bank."""
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
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, input_ids, ans_start):
    """Standard causal LM loss over answer span."""
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def build_split(entries, *, train_frac, n_preload, n_train, n_test, seed):
    """Split relations into TRAIN_RELS and TEST_RELS with EMPTY intersection.
    
    - preload: n_preload solo_pass items from TRAIN_RELS only
    - train: n_train items from TRAIN_RELS (disjoint from preload)
    - test: n_test items from TEST_RELS (HARD OOD)
    """
    train_rels, test_rels = data_io.split_disjoint_relations(
        entries, train_frac=train_frac, seed=seed
    )
    train_rels_set = set(train_rels)
    test_rels_set = set(test_rels)
    
    print(f"[e06] TRAIN_RELS={len(train_rels)} TEST_RELS={len(test_rels)} "
          f"intersection={len(train_rels_set & test_rels_set)}")
    assert len(train_rels_set & test_rels_set) == 0, "Relations must be disjoint!"
    
    # Filter keys by relation
    all_keys = list(entries.keys())
    train_rel_keys = [k for k in all_keys
                      if entries[k].get("solo_pass") and 
                      data_io.relation_of(entries, k) in train_rels_set]
    test_rel_keys = [k for k in all_keys
                     if entries[k].get("solo_pass") and 
                     data_io.relation_of(entries, k) in test_rels_set]
    
    rng = random.Random(seed)
    rng.shuffle(train_rel_keys)
    rng.shuffle(test_rel_keys)
    
    # Preload from TRAIN_RELS
    preload_keys = train_rel_keys[:n_preload]
    # Train from TRAIN_RELS (disjoint from preload)
    train_keys = train_rel_keys[n_preload:n_preload + n_train]
    # Test from TEST_RELS (OOD)
    test_keys = test_rel_keys[:n_test]
    
    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    
    return train_items, test_items, preload_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    
    out_path = Path(args.out) if args.out else HERE / f"e06_seed{args.seed}.json"
    
    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    
    train_items, test_items, preload_keys = build_split(
        entries, train_frac=args.train_frac, n_preload=args.n_preload,
        n_train=args.n_train, n_test=args.n_eval, seed=args.seed,
    )
    print(f"[e06] train={len(train_items)} test_ood={len(test_items)} "
          f"preload={len(preload_keys)} seed={args.seed}")
    
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)
    
    b_raw = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                      device=args.device, dtype=torch.float32)
    
    P = make_projector(d, rank=args.rank).to(args.device).float()
    
    def apply_proj():
        with torch.no_grad():
            proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
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
    
    def eval_lpl(items, *, randomize=False):
        nlls = []
        for sj, rl, tg in items:
            enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
            apply_proj()
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False,
                                     randomize_bank=randomize)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)
    
    # BEFORE: train-rel test and OOD test-rel
    base_train = eval_base(train_items)
    base_test = eval_base(test_items)
    pre_train_real = eval_lpl(train_items)
    pre_train_rand = eval_lpl(train_items, randomize=True)
    pre_test_real = eval_lpl(test_items)
    pre_test_rand = eval_lpl(test_items, randomize=True)
    
    print(f"[e06] BEFORE train-rel: base={base_train:.4f} real={pre_train_real:.4f} "
          f"rand={pre_train_rand:.4f}")
    print(f"[e06] BEFORE test-rel (OOD): base={base_test:.4f} real={pre_test_real:.4f} "
          f"rand={pre_test_rand:.4f}")
    
    # === training ===
    rng = random.Random(args.seed)
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []
    t0 = time.time()
    
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        
        # Rebuild bank with grad-tracking projector each step
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
            print(f"  [e06] step {step+1}/{args.steps} loss(avg25)={recent:.4f} "
                  f"({time.time()-t0:.1f}s)")
    
    print(f"[e06] training done in {time.time()-t0:.1f}s")
    
    # === post-train eval ===
    post_train_real = eval_lpl(train_items)
    post_train_rand = eval_lpl(train_items, randomize=True)
    post_test_real = eval_lpl(test_items)
    post_test_rand = eval_lpl(test_items, randomize=True)
    
    print(f"[e06] AFTER train-rel: base={base_train:.4f} real={post_train_real:.4f} "
          f"rand={post_train_rand:.4f}")
    print(f"[e06] AFTER test-rel (OOD): base={base_test:.4f} real={post_test_real:.4f} "
          f"rand={post_test_rand:.4f}")
    
    # Verdict
    delta_train = post_train_real - base_train
    delta_test_ood = post_test_real - base_test
    verdict = {
        "pass": delta_test_ood <= -1.0,
        "rule": "Δ NLL on OOD test_rel ≤ -1.0 (projector generalizes across relations)",
        "delta_train": delta_train,
        "delta_test_ood": delta_test_ood,
    }
    
    out = {
        "experiment": "e06_relation_disjoint_ood",
        "seed": args.seed,
        "model": args.model,
        "n_train": len(train_items),
        "n_test_ood": len(test_items),
        "n_preload": len(preload_keys),
        "train_frac": args.train_frac,
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "before": {
            "train_rel": {
                "base": base_train,
                "real": pre_train_real,
                "rand": pre_train_rand,
            },
            "test_rel_ood": {
                "base": base_test,
                "real": pre_test_real,
                "rand": pre_test_rand,
            },
        },
        "after": {
            "train_rel": {
                "base": base_train,
                "real": post_train_real,
                "rand": post_train_rand,
            },
            "test_rel_ood": {
                "base": base_test,
                "real": post_test_real,
                "rand": post_test_rand,
            },
        },
        "verdict": verdict,
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
        "n_train_params": sum(p.numel() for p in trainable),
        "elapsed_s": time.time() - t0,
    }
    
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e06] -> {out_path}  verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
