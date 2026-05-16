"""e02 — Phase B2 scale matrix.

Sweep the four most informative axes of B2 to characterize where the
NLL-drop signal lives:

    --n_preload  ∈ {512, 2048, 8192}
    --n_train    ∈ {120, 1000, 5000}
    --layers     ∈ {single-9, multi-9-15-21}
    --steps      ∈ {200, 1000}

Use canonical anti-cheat machinery from e01 (real vs random vs zero vs off)
and a strict (subject ∪ relation)-disjoint train/test split so leakage
cannot inflate the curve.

Output: one JSON per cell under v2/experiments/e02_scale_matrix/cells/.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa, data_io,
)


def forward_lpl_k2(model, bank, heads, enc, *, grad=False, randomize_bank=False):
    s1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, s1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    if randomize_bank:
        for l, t in enumerate(bank.slots):
            if t.shape[0] == 0: continue
            n = torch.randn_like(t)
            n = n / (n.norm(dim=-1, keepdim=True) + 1e-9) * t.norm(dim=-1, keepdim=True)
            bank.slots[l] = n.to(dtype=t.dtype, device=t.device)
    s2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, s2), ctx:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, ids, ans):
    pred = logits[0, ans-1:-1, :]; gold = ids[0, ans:]
    return F.cross_entropy(pred.float(), gold)


def disjoint_split(entries, n_train, n_test, n_preload, seed):
    keys = list(entries.keys())
    rng = random.Random(seed); rng.shuffle(keys)
    subjects, relations = set(), set()
    train, test = [], []
    for k in keys:
        e = entries[k]
        if not e.get("solo_pass"): continue
        s = e["subject"]; r = e["relation"]
        if len(train) < n_train:
            train.append(k); subjects.add(s); relations.add(r)
        else:
            if s in subjects or r in relations: continue
            test.append(k)
            if len(test) >= n_test: break
    pool = [k for k in keys if entries[k].get("solo_pass") and k not in set(train)]
    preload = pool[:n_preload]
    return (data_io.items_for_keys(entries, train),
            data_io.items_for_keys(entries, test), preload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n_train", type=int, required=True)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--n_preload", type=int, required=True)
    p.add_argument("--layers", default="single", choices=["single", "multi"])
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--multi_layers", default="9,15,21")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cells_dir = HERE / "cells"; cells_dir.mkdir(exist_ok=True)
    tag = f"n{args.n_preload}_t{args.n_train}_{args.layers}_s{args.steps}_seed{args.seed}"
    out_path = cells_dir / f"{tag}.json"

    blob = data_io.load_bank_blob(); entries = blob["entries"]
    train_items, test_items, preload_keys = disjoint_split(
        entries, args.n_train, args.n_eval, args.n_preload, args.seed)
    print(f"[e02:{tag}] train={len(train_items)} test={len(test_items)} preload={len(preload_keys)}")

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters(): pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    layers_list = [args.bank_layer] if args.layers == "single" \
        else [int(x) for x in args.multi_layers.split(",")]

    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    b_raw = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                      device=args.device, dtype=torch.float32)

    Ps = nn.ModuleList([make_projector(d, rank=args.rank) for _ in layers_list]).to(args.device).float()

    def apply_proj():
        bank.frozen = False
        for li, P in zip(layers_list, Ps):
            with torch.no_grad():
                proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
            bank.slots[li] = proj
            bank.tags[li] = [(0, -1)] * proj.shape[0]
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
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False, randomize_bank=randomize)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    base = eval_base(test_items)
    pre_real = eval_lpl(test_items)
    pre_rand = eval_lpl(test_items, randomize=True)
    print(f"[e02:{tag}] BEFORE base={base:.4f} real={pre_real:.4f} rand={pre_rand:.4f}")

    rng = random.Random(args.seed)
    trainable = list(Ps.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []; t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        bank.frozen = False
        for li, P in zip(layers_list, Ps):
            proj = (b_raw + P(b_raw)).to(dtype=torch.bfloat16)
            bank.slots[li] = proj
            bank.tags[li] = [(0, -1)] * proj.shape[0]
        bank.frozen = True

        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step+1) % 100 == 0 or step == 0:
            recent = sum(losses[-50:]) / min(50, len(losses))
            print(f"  [e02:{tag}] step {step+1}/{args.steps} loss(avg50)={recent:.4f} ({time.time()-t0:.1f}s)")
    print(f"[e02:{tag}] training done in {time.time()-t0:.1f}s")

    post_real = eval_lpl(test_items)
    post_rand = eval_lpl(test_items, randomize=True)
    print(f"[e02:{tag}] AFTER base={base:.4f} real={post_real:.4f} rand={post_rand:.4f}")

    out = {
        "tag": tag, "n_preload": args.n_preload, "n_train": args.n_train,
        "layers": layers_list, "steps": args.steps, "rank": args.rank, "seed": args.seed,
        "before": {"base": base, "real": pre_real, "rand": pre_rand},
        "after":  {"base": base, "real": post_real, "rand": post_rand},
        "delta_real": post_real - base, "delta_rand": post_rand - base,
        "loss_first50": losses[:50], "loss_last50": losses[-50:],
        "n_train_params": sum(p.numel() for p in trainable),
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e02:{tag}] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
