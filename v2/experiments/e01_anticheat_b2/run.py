"""e01 anti-cheat suite — falsifiers for Phase B2's Δ=−5.83 NLL claim.

Re-implements Phase B2 training/eval on top of v2/core, then sweeps a
``--variant`` flag to inject specific cheap-explanation tests:

    h1_bank_off       train as B2, EVAL with bank disabled       (H1)
    h2_shuffle_b      row-level dim shuffle of preload b-vectors (H2)
    h3_disjoint_split entity+relation disjoint train/test split  (H3)
    h4_zero_bank      replace bank slots with zeros at eval      (H4)
    h5_n_sweep        --n_preload sweep                          (H5)
    h6_layer_sweep    --bank_layer sweep                         (H6)
    h7_rand_train     train with RANDOM bank from step 0         (H7)
    h8_kl_neutral     post-train logit KL vs base on neutral     (H8)
    h9_cross_smoke    quick Qwen3-1.7B sanity (use --model)      (H9)
    h10_gate_hist     log gate stats per layer                   (H10)
    canonical         vanilla B2 reproduce                        (sanity)

Output: JSON in this directory, one per --variant per --seed.
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


def shuffle_b_dims(b: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Shuffle each row's dimensions independently (per-row permutation)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = torch.empty_like(b)
    for i in range(b.shape[0]):
        perm = torch.randperm(b.shape[1], generator=g)
        out[i] = b[i, perm]
    return out


def build_split(entries, *, variant, n_train, n_test, n_preload, seed):
    """Build (train_items, test_items, preload_keys) according to variant.

    Default = same as Phase B2 (random shuffle, train/test from bank.pt's
    'split' field, preload from disjoint train slice).

    h3_disjoint_split = enforce strict ENTITY (subject) AND RELATION disjoint
    between train and test. Preload still from train side.
    """
    keys = list(entries.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    if variant == "h3_disjoint_split":
        # build sets of (subj, rel) pairs that don't share subj OR rel
        subjects, relations = set(), set()
        train_keys, test_keys = [], []
        for k in keys:
            e = entries[k]
            if not e.get("solo_pass"):
                continue
            subj = e["subject"]; rel = e["relation"]
            if len(train_keys) < n_train:
                train_keys.append(k)
                subjects.add(subj); relations.add(rel)
            else:
                if subj in subjects or rel in relations:
                    continue
                test_keys.append(k)
                if len(test_keys) >= n_test:
                    break
        # preload from train side, disjoint from train_keys
        preload_pool = [k for k in keys if entries[k].get("split") == "train"
                        and entries[k].get("solo_pass") and k not in set(train_keys)]
        preload_keys = preload_pool[:n_preload]
    else:
        train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
        test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
        rng.shuffle(train_keys); rng.shuffle(test_keys)
        train_keys = train_keys[:n_train]
        test_keys = test_keys[:n_test]
        preload_pool = data_io.filter_keys(entries, split="train", solo_pass=True)
        preload_pool = [k for k in preload_pool if k not in set(train_keys)]
        preload_keys = preload_pool[:n_preload]

    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    return train_items, test_items, preload_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=["canonical", "h1_bank_off", "h2_shuffle_b",
                            "h3_disjoint_split", "h4_zero_bank", "h5_n_sweep",
                            "h6_layer_sweep", "h7_rand_train", "h8_kl_neutral",
                            "h9_cross_smoke", "h10_gate_hist"])
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

    out_path = Path(args.out) if args.out else HERE / f"e01_{args.variant}_seed{args.seed}.json"

    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]

    train_items, test_items, preload_keys = build_split(
        entries, variant=args.variant, n_train=args.n_train,
        n_test=args.n_eval, n_preload=args.n_preload, seed=args.seed,
    )
    print(f"[e01:{args.variant}] train={len(train_items)} test={len(test_items)} "
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
    if args.variant == "h2_shuffle_b":
        b_raw = shuffle_b_dims(b_raw.cpu(), seed=args.seed).to(args.device)
        print("[e01:h2] applied row-level dim shuffle to b_raw")

    P = make_projector(d, rank=args.rank).to(args.device).float()

    def apply_proj(zero_bank=False):
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
    print(f"[e01:{args.variant}] BEFORE: base={base:.4f}  real={pre_real:.4f}  "
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
        if args.variant == "h7_rand_train":
            # random bank during training — reuse rand generation per step
            n = torch.randn_like(b_raw)
            n = n / (n.norm(dim=-1, keepdim=True) + 1e-9) * 15.0
            proj_in = n
        else:
            proj_in = b_raw
        proj = (proj_in + P(proj_in)).to(dtype=torch.bfloat16)
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
            print(f"  [e01:{args.variant}] step {step+1}/{args.steps} loss(avg25)={recent:.4f} ({time.time()-t0:.1f}s)")
    print(f"[e01:{args.variant}] training done in {time.time()-t0:.1f}s")

    # === post-train eval ===
    post_real = eval_lpl(test_items)
    post_rand = eval_lpl(test_items, randomize=True)
    post_zero = eval_lpl(test_items, zero_bank=True)
    post_off = eval_lpl(test_items, bank_off=True)
    print(f"[e01:{args.variant}] AFTER:  base={base:.4f}  real={post_real:.4f}  "
          f"rand={post_rand:.4f}  zero={post_zero:.4f}  off={post_off:.4f}")

    # gate stats (H10)
    gate_stats = {}
    for li, gh in enumerate(heads.bank_gate_heads):
        w = gh.proj.weight.detach().float().cpu()
        b_ = gh.proj.bias.detach().float().cpu()
        gate_stats[f"layer{li}"] = {"w_norm": float(w.norm()), "b": float(b_)}

    # decision per variant
    verdict = {}
    if args.variant == "h1_bank_off":
        verdict = {"pass": post_off >= base - 0.05, "rule": "post_off >= base-0.05"}
    elif args.variant == "h2_shuffle_b":
        verdict = {"pass": (post_real - 6.30) >= 4.0, "rule": "post_real degrades >=4.0 vs B2 baseline 6.30"}
    elif args.variant == "h3_disjoint_split":
        verdict = {"pass": (base - post_real) >= 1.0, "rule": "Δ NLL <= -1.0 even with disjoint split"}
    elif args.variant == "h4_zero_bank":
        verdict = {"pass": abs(post_zero - base) <= 0.05, "rule": "post_zero ~= base ±0.05"}
    elif args.variant == "h7_rand_train":
        verdict = {"pass": (post_real - 6.30) >= 4.0, "rule": "rand-trained cannot reach B2 6.30 (gap >=4.0)"}
    elif args.variant == "canonical":
        verdict = {"pass": (base - post_real) >= 5.0, "rule": "B2 canonical reproduce: Δ NLL <= -5.0"}
    else:
        verdict = {"pass": None, "rule": "non-binary; see numbers"}

    out = {
        "variant": args.variant, "seed": args.seed, "model": args.model,
        "n_train": len(train_items), "n_test": len(test_items),
        "n_preload": len(preload_keys), "bank_layer": args.bank_layer,
        "rank": args.rank, "lr": args.lr, "steps": args.steps,
        "before": {"base": base, "real": pre_real, "rand": pre_rand,
                    "zero": pre_zero, "off": pre_off},
        "after":  {"base": base, "real": post_real, "rand": post_rand,
                    "zero": post_zero, "off": post_off},
        "verdict": verdict,
        "loss_first25": losses[:25], "loss_last25": losses[-25:],
        "n_train_params": sum(p.numel() for p in trainable),
        "gate_stats_sample": {k: gate_stats[k] for k in list(gate_stats)[:6]},
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e01:{args.variant}] -> {out_path}  verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
