"""Exp34 — ROME-style parameter edit (Phase C positive control).

Algorithm: rank-1 edit to `mlp.down_proj.weight` at a chosen layer L.

  k* = activation entering down_proj at subject's last token (write prompt).
  v* = optimized residual addition at layer L (output) that flips
       target_true → target_new on the canonical read prompt.
  W_new = W_old + (v* − W_old k*) ⊗ k* / (‖k*‖² + λ)

Then evaluate margin = logp(target_new) − logp(target_true) on
paraphrase prompts, restore W, and proceed to next fact.

Three variants per fact:
  base        — no edit (baseline margin)
  edited      — correct k*    (Gate B)
  shuffled_k  — someone else's k* (Gate D: identity binding)

Gate B PASS if median (edited − base) > 0.5 nats AND edited > 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402


SPLITS = Path(__file__).resolve().parents[1] / "exp31_learned_k_adapter" / "data" / "splits"


def first_target_id(tokenizer, target: str) -> int:
    ids = tokenizer(" " + target.strip(), add_special_tokens=False).input_ids
    return int(ids[0])


def subject_last_pos(tokenizer, prompt: str, subject: str) -> int:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    for variant in (" " + subject.strip(), subject.strip()):
        sids = tokenizer(variant, add_special_tokens=False).input_ids
        if not sids:
            continue
        for i in range(len(ids) - len(sids), -1, -1):
            if ids[i:i + len(sids)] == sids:
                return i + len(sids) - 1
    return len(ids) - 1


def capture_k_star(model, tokenizer, prompt: str, edit_layer: int, pos: int):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    layer = model.model.layers[edit_layer]
    captured = {}

    def hook(module, inp, out):
        captured["x"] = inp[0][0, pos].detach().clone()

    h = layer.mlp.down_proj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        h.remove()
    return captured["x"]


def compute_v_star(model, tokenizer, read_prompt: str, edit_layer: int,
                   t_new_id: int, t_true_id: int, n_steps: int, lr: float):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    d_model = model.config.hidden_size
    delta = torch.zeros(d_model, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    layer = model.model.layers[edit_layer]

    enc = tokenizer(read_prompt, return_tensors="pt", add_special_tokens=True).to(device)
    last_pos = int(enc["attention_mask"][0].sum().item() - 1)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            h_out = out[0]
            h_out[0, last_pos] = h_out[0, last_pos] + delta.to(h_out.dtype)
            return (h_out,) + out[1:]
        out[0, last_pos] = out[0, last_pos] + delta.to(out.dtype)
        return out

    h = layer.register_forward_hook(hook)
    try:
        for _ in range(n_steps):
            opt.zero_grad()
            out = model(**enc, use_cache=False)
            logp = F.log_softmax(out.logits[0, last_pos].float(), dim=-1)
            loss = -(logp[t_new_id] - 0.5 * logp[t_true_id]) + 1e-3 * (delta ** 2).sum()
            loss.backward()
            opt.step()
    finally:
        h.remove()
    return delta.detach()


def apply_rome_edit(model, edit_layer: int, k_star, v_star, lam: float = 1e-2):
    W = model.model.layers[edit_layer].mlp.down_proj.weight
    W_old = W.data.clone()
    k = k_star.to(W.dtype)
    v = v_star.to(W.dtype)
    Wk = W @ k
    denom = float((k.float() @ k.float()).item()) + lam
    W.data = W_old + torch.outer(v - Wk, k) / denom
    return W_old


def restore_weights(model, edit_layer: int, W_old):
    model.model.layers[edit_layer].mlp.down_proj.weight.data = W_old


@torch.no_grad()
def margin_at_last(model, tokenizer, prompt: str, t_new: int, t_true: int) -> float:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, use_cache=False)
    last = int(enc["attention_mask"][0].sum().item() - 1)
    logp = F.log_softmax(out.logits[0, last].float(), dim=-1)
    return float(logp[t_new].item() - logp[t_true].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n-test", type=int, default=125)
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--v-steps", type=int, default=25)
    ap.add_argument("--v-lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "run_qwen_exp34"))
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    test = json.load(open(SPLITS / "test.json"))[: args.n_test]
    print(f"n_test = {len(test)}  edit_layer = {args.edit_layer}", flush=True)

    print("\n[Φ1] capture k*", flush=True)
    t0 = time.time()
    k_stars = []
    for i, row in enumerate(test):
        wp = row["prompt"].format(row["subject"])
        pos = subject_last_pos(tok, wp, row["subject"])
        k_stars.append(capture_k_star(model, tok, wp, args.edit_layer, pos))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(test)}  ({time.time()-t0:.0f}s)", flush=True)

    g = torch.Generator().manual_seed(1000 + args.seed)
    perm = torch.randperm(len(test), generator=g).tolist()

    rows = []
    print("\n[Φ2] ROME edit + eval per fact", flush=True)
    t0 = time.time()
    for i, row in enumerate(test):
        t_new = first_target_id(tok, row["target_new"])
        t_true = first_target_id(tok, row["target_true"])
        read_prompts = [row["prompt"].format(row["subject"])] + \
                        list(row.get("paraphrase_prompts", []))[:2]

        base_margins = [margin_at_last(model, tok, p, t_new, t_true) for p in read_prompts]
        base_mean = sum(base_margins) / len(base_margins)

        v_star = compute_v_star(model, tok, read_prompts[0], args.edit_layer,
                                t_new, t_true, n_steps=args.v_steps, lr=args.v_lr)

        W_old = apply_rome_edit(model, args.edit_layer, k_stars[i], v_star)
        try:
            edited_margins = [margin_at_last(model, tok, p, t_new, t_true) for p in read_prompts]
        finally:
            restore_weights(model, args.edit_layer, W_old)
        edited_mean = sum(edited_margins) / len(edited_margins)

        W_old = apply_rome_edit(model, args.edit_layer, k_stars[perm[i]], v_star)
        try:
            shuf_margins = [margin_at_last(model, tok, p, t_new, t_true) for p in read_prompts]
        finally:
            restore_weights(model, args.edit_layer, W_old)
        shuf_mean = sum(shuf_margins) / len(shuf_margins)

        rows.append({
            "fact_idx": i, "id": row["id"],
            "base_mean": base_mean,
            "edited_mean": edited_mean,
            "shuffled_k_mean": shuf_mean,
            "edited_minus_base": edited_mean - base_mean,
            "edited_minus_shuffled": edited_mean - shuf_mean,
        })

        if (i + 1) % 10 == 0:
            recent = rows[-10:]
            print(f"  {i+1}/{len(test)}  "
                  f"base={sum(r['base_mean'] for r in recent)/len(recent):+.2f}  "
                  f"edited={sum(r['edited_mean'] for r in recent)/len(recent):+.2f}  "
                  f"shuf={sum(r['shuffled_k_mean'] for r in recent)/len(recent):+.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with open(out_dir / "cells.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def mean(key):
        return sum(r[key] for r in rows) / len(rows)

    def frac_pos(key):
        return sum(1 for r in rows if r[key] > 0) / len(rows)

    summary = {
        "meta": {"n_test": len(test), "edit_layer": args.edit_layer,
                 "v_steps": args.v_steps, "v_lr": args.v_lr, "seed": args.seed},
        "base_mean": mean("base_mean"),
        "edited_mean": mean("edited_mean"),
        "shuffled_k_mean": mean("shuffled_k_mean"),
        "mean_edited_minus_base": mean("edited_minus_base"),
        "mean_edited_minus_shuffled": mean("edited_minus_shuffled"),
        "frac_edited_beats_base": frac_pos("edited_minus_base"),
        "frac_edited_beats_shuffled": frac_pos("edited_minus_shuffled"),
        "frac_edited_above_zero": sum(1 for r in rows if r["edited_mean"] > 0) / len(rows),
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
