"""Phase D-1 — matched LoRA / plain residual adapter baseline.

Compares three methods at the **same attention layer** (default L=9) on the
**same train/test split** as e01/e11, with **matched trainable parameter
counts** (~0.33M, the rank-64 K-projector budget) and the **same training
schedule** (lr 2e-4, 200 steps, bf16 MPS, frozen base).

Methods (--method)
------------------
    plain_adapter     : residual MLP h -> h + U(V(h)) on layer-L hidden state
    lora_q            : LoRA on layer-L q_proj (rank chosen to match params)
    lora_qk           : LoRA on both q_proj and k_proj (half rank each)

The aim is to falsify or confirm the v2 final claim:

    > The bank+projector mechanism is empirically closer to parameter-efficient
    > adaptation than to content-addressed memory retrieval.

This script is a negative-control PEFT baseline, not the bank-projector
runner. Compare its outputs against the E10/E11/E20 family, which use the
bank/projector path. If a plain residual adapter with a matched parameter
budget absorbs the same train/test split, then the early v2 lift is not
specific evidence for content-addressed bank retrieval.

Output: ``phase_d_lora_<method>_seed<s>_r<rank>.json``
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
    load_model, nll_on_answer, encode_qa, data_io,
)


class ResidualAdapter(nn.Module):
    """h -> h + U(V(h)), zero-init on U so step 0 is identity."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.V = nn.Linear(d, r, bias=False)
        self.U = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.V.weight, std=0.02)
        nn.init.zeros_(self.U.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.U(self.V(x))


class LoRA(nn.Module):
    """Standard LoRA: delta(x) = B(A(x)), zero-init B."""

    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float = 1.0):
        super().__init__()
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scale


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _resolve_layer(model, layer_idx: int):
    """Get the nn.Module for transformer block layer_idx."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    raise RuntimeError("Cannot locate transformer layers on model.")


def install_plain_adapter(model, layer_idx: int, adapter: ResidualAdapter) -> callable:
    """Register a forward hook on layer L's output that applies adapter to hidden state.

    Returns an uninstall function that removes the hook.
    """
    block = _resolve_layer(model, layer_idx)

    def hook(_module, _input, output):
        # output of a decoder block is typically a tuple (hidden_states, ...) or just hidden.
        if isinstance(output, tuple):
            h = output[0]
            h2 = adapter(h.to(adapter.U.weight.dtype)).to(h.dtype)
            return (h2,) + output[1:]
        else:
            h2 = adapter(output.to(adapter.U.weight.dtype)).to(output.dtype)
            return h2

    handle = block.register_forward_hook(hook)
    return handle.remove


def install_lora_on_proj(model, layer_idx: int, proj_names, lora_modules) -> callable:
    """Install LoRA add-ons on listed projection submodules of layer L.

    proj_names: list of attribute names under layer.self_attn, e.g. ['q_proj'] or ['q_proj','k_proj'].
    lora_modules: list of LoRA modules aligned to proj_names.
    Returns uninstall callable.
    """
    block = _resolve_layer(model, layer_idx)
    attn = block.self_attn
    handles = []
    for name, lora in zip(proj_names, lora_modules):
        proj = getattr(attn, name)
        def make_hook(lora_ref):
            def hook(_m, _inp, out):
                x = _inp[0]
                delta = lora_ref(x.to(lora_ref.A.weight.dtype)).to(out.dtype)
                return out + delta
            return hook
        handles.append(proj.register_forward_hook(make_hook(lora)))

    def remove_all():
        for h in handles:
            h.remove()
    return remove_all


def evaluate(model, tok, items, device):
    nlls = []
    for sj, rl, tg in items:
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
        nlls.append(nll_on_answer(logits, enc.input_ids, ans))
    return sum(nlls) / max(len(nlls), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["plain_adapter", "lora_q", "lora_qk"])
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_eval", type=int, default=120)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    out_path = Path(args.out) if args.out else HERE / (
        f"phase_d_lora_{args.method}_seed{args.seed}_r{args.rank}.json"
    )

    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    split_rng = random.Random(args.seed)
    split_rng.shuffle(train_keys); split_rng.shuffle(test_keys)
    train_rng = random.Random(args.seed)
    train_keys = train_keys[:args.n_train]
    test_keys = test_keys[:args.n_eval]
    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)
    print(f"[lora:{args.method}] train={len(train_items)} test={len(test_items)} seed={args.seed}")

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    d = model.config.hidden_size
    # For Qwen3-4B: hidden=2560, num_kv_heads*head_dim = k/v dim
    kv_dim = model.config.num_key_value_heads * model.config.head_dim if hasattr(model.config, "head_dim") \
        else model.config.hidden_size // model.config.num_attention_heads * model.config.num_key_value_heads

    if args.method == "plain_adapter":
        adapter = ResidualAdapter(d, args.rank).to(args.device).float()
        uninstall = install_plain_adapter(model, args.bank_layer, adapter)
        trainables = list(adapter.parameters())
    elif args.method == "lora_q":
        # full hidden -> q (= num_attn_heads * head_dim, typically = d for non-GQA)
        # In Qwen3 q_proj: in=d, out=d (num_attn_heads*head_dim)
        q_out = model.model.layers[args.bank_layer].self_attn.q_proj.out_features
        lq = LoRA(d, q_out, args.rank).to(args.device).float()
        uninstall = install_lora_on_proj(model, args.bank_layer, ["q_proj"], [lq])
        trainables = list(lq.parameters())
    elif args.method == "lora_qk":
        q_out = model.model.layers[args.bank_layer].self_attn.q_proj.out_features
        k_out = model.model.layers[args.bank_layer].self_attn.k_proj.out_features
        r_half = max(args.rank // 2, 1)
        lq = LoRA(d, q_out, r_half).to(args.device).float()
        lk = LoRA(d, k_out, r_half).to(args.device).float()
        uninstall = install_lora_on_proj(model, args.bank_layer, ["q_proj", "k_proj"], [lq, lk])
        trainables = list(lq.parameters()) + list(lk.parameters())
    else:
        raise ValueError(args.method)

    n_params = sum(t.numel() for t in trainables)
    print(f"[lora:{args.method}] trainable params = {n_params:,}")

    nll_before_base = evaluate(model, tok, test_items, args.device)
    print(f"[lora:{args.method}] before: test_nll={nll_before_base:.4f}")

    opt = torch.optim.AdamW(trainables, lr=args.lr)
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        sj, rl, tg = train_rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        out = model(**enc, use_cache=False)
        logits = out.logits
        pred = logits[0, ans - 1: -1, :]
        gold = enc.input_ids[0, ans:]
        loss = F.cross_entropy(pred.float(), gold)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{args.steps} loss={loss.item():.4f}")

    train_time = time.time() - t0

    nll_after = evaluate(model, tok, test_items, args.device)
    print(f"[lora:{args.method}] after:  test_nll={nll_after:.4f}  delta={nll_after - nll_before_base:+.4f}")

    uninstall()

    nll_uninstalled = evaluate(model, tok, test_items, args.device)
    print(f"[lora:{args.method}] uninstalled: test_nll={nll_uninstalled:.4f}  (sanity = before)")

    result = {
        "experiment": "phase_d_lora",
        "method": args.method,
        "seed": args.seed,
        "rank": args.rank,
        "model": args.model,
        "bank_layer": args.bank_layer,
        "lr": args.lr,
        "steps": args.steps,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "n_params_trainable": n_params,
        "test_nll_before": nll_before_base,
        "test_nll_after": nll_after,
        "delta_nll": nll_after - nll_before_base,
        "test_nll_uninstalled": nll_uninstalled,
        "train_time_s": train_time,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[lora:{args.method}] -> {out_path}")


if __name__ == "__main__":
    main()
