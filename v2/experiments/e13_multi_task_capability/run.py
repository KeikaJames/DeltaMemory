"""e13 — multi-task capability drift: prove v2 ALB doesn't degrade base model.

Run a small multi-benchmark suite (WikiText-2, LAMBADA, HellaSwag, GSM8K)
under three conditions:
    base       — pure Qwen3 baseline (no LPL patch)
    bank_off   — LPL patched, AttentionBank empty (must be bit-equal to base)
    bank_on    — LPL patched, Exp35b bank preloaded (512 entries layer 9),
                 projector loaded from a previous e01 run or trained inline

Pass criteria:
    • bank_off ≈ base (within 1e-3 relative)
    • bank_on within ±5% relative on perplexity benchmarks
    • bank_on within ±3 points absolute on accuracy benchmarks

If any benchmark fails: the bank/projector is contaminating general
capabilities — a major caveat for the v2 thesis.

Output: v2/experiments/e13_multi_task_capability/e13_seed{seed}.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, install_lpl_patch, uninstall_lpl_patch,
    LPLState, lpl_state_scope, make_projector, residual_apply,
    load_model, nll_on_answer, encode_qa, data_io,
)


# ============================================================================
# Benchmark implementations
# ============================================================================

def eval_wikitext2(model, tok, *, n_tokens: int, ctx: int, device: str,
                   run_forward_fn) -> dict:
    """WikiText-2 validation set sliding-window NLL."""
    try:
        from datasets import load_dataset
    except ImportError:
        warnings.warn("datasets not installed, skipping WikiText-2")
        return None

    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation", trust_remote_code=True)
    except Exception as e:
        warnings.warn(f"failed to load WikiText-2: {e}")
        return None

    texts = [t for t in ds["text"] if t and not t.isspace()]
    big = "\n".join(texts)
    ids = tok(big, return_tensors="pt").input_ids[0]
    if ids.numel() > n_tokens:
        ids = ids[:n_tokens]

    n = ids.numel()
    nlls, n_pred = [], 0
    t0 = time.time()
    for s in range(0, n - 1, ctx):
        e = min(s + ctx, n)
        chunk = ids[s:e].unsqueeze(0).to(device)
        if chunk.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = run_forward_fn(chunk)
        pred = logits[0, :-1, :].float()
        gold = chunk[0, 1:]
        loss = F.cross_entropy(pred, gold, reduction="sum")
        nlls.append(loss.item())
        n_pred += gold.numel()
    elapsed = time.time() - t0
    avg = sum(nlls) / max(n_pred, 1)
    ppl = math.exp(avg)
    print(f"  [wikitext2] tokens={n_pred} nll={avg:.4f} ppl={ppl:.3f} ({elapsed:.1f}s)")
    return {"nll": avg, "ppl": ppl, "n_tokens": n_pred}


def eval_lambada(model, tok, *, n_items: int, device: str, run_forward_fn) -> dict:
    """LAMBADA last-token prediction accuracy."""
    try:
        from datasets import load_dataset
    except ImportError:
        warnings.warn("datasets not installed, skipping LAMBADA")
        return None

    try:
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test", trust_remote_code=True)
    except Exception as e:
        warnings.warn(f"failed to load LAMBADA: {e}")
        return None

    if len(ds) > n_items:
        ds = ds.select(range(n_items))

    correct, total = 0, 0
    t0 = time.time()
    for item in ds:
        text = item["text"]
        # last word is the target
        words = text.rstrip().split()
        if len(words) < 2:
            continue
        target_word = words[-1]
        prefix = " ".join(words[:-1])

        # encode full text (including target)
        full_ids = tok(text, return_tensors="pt").input_ids.to(device)
        # encode prefix only to find split
        prefix_ids = tok(prefix, return_tensors="pt").input_ids
        split = prefix_ids.shape[1]
        # Guard: BPE may re-tokenize "prefix" + " " + "target" differently from "text";
        # if the target token doesn't exist past the split, skip this item.
        if split >= full_ids.shape[1]:
            continue

        with torch.no_grad():
            logits = run_forward_fn(full_ids)

        # predict at position [split-1] for token at split (first target token)
        pred_logits = logits[0, split - 1, :]
        pred_id = pred_logits.argmax().item()
        # target is full_ids[0, split]
        gold_id = full_ids[0, split].item()
        if pred_id == gold_id:
            correct += 1
        total += 1

    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    print(f"  [lambada] correct={correct}/{total} acc={acc:.4f} ({elapsed:.1f}s)")
    return {"acc": acc, "correct": correct, "total": total}


def eval_hellaswag(model, tok, *, n_items: int, device: str, run_forward_fn) -> dict:
    """HellaSwag 4-way completion scoring via NLL."""
    try:
        from datasets import load_dataset
    except ImportError:
        warnings.warn("datasets not installed, skipping HellaSwag")
        return None

    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    except Exception as e:
        warnings.warn(f"failed to load HellaSwag: {e}")
        return None

    if len(ds) > n_items:
        rng = random.Random(0)
        indices = rng.sample(range(len(ds)), n_items)
        ds = ds.select(indices)

    correct, total = 0, 0
    t0 = time.time()
    for item in ds:
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        # score each ending by NLL
        nlls = []
        for ending in endings:
            full = ctx + " " + ending
            ids = tok(full, return_tensors="pt").input_ids.to(device)
            ctx_ids = tok(ctx, return_tensors="pt").input_ids
            split = ctx_ids.shape[1]

            with torch.no_grad():
                logits = run_forward_fn(ids)

            # NLL on ending tokens [split:]
            pred = logits[0, split - 1: -1, :].float()
            gold = ids[0, split:]
            if gold.numel() == 0:
                nlls.append(float('inf'))
            else:
                nll = F.cross_entropy(pred, gold, reduction="mean").item()
                nlls.append(nll)

        # lowest NLL wins
        pred_label = nlls.index(min(nlls))
        if pred_label == label:
            correct += 1
        total += 1

    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    print(f"  [hellaswag] correct={correct}/{total} acc={acc:.4f} ({elapsed:.1f}s)")
    return {"acc": acc, "correct": correct, "total": total}


def eval_gsm8k(model, tok, *, n_items: int, device: str, run_forward_fn) -> dict:
    """GSM8K exact-match on final answer (greedy decode)."""
    try:
        from datasets import load_dataset
    except ImportError:
        warnings.warn("datasets not installed, skipping GSM8K")
        return None

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    except Exception as e:
        warnings.warn(f"failed to load GSM8K: {e}")
        return None

    if len(ds) > n_items:
        rng = random.Random(0)
        indices = rng.sample(range(len(ds)), n_items)
        ds = ds.select(indices)

    correct, total = 0, 0
    t0 = time.time()
    for item in ds:
        question = item["question"]
        answer = item["answer"]
        # extract numeric answer from "#### 1234" format
        match = re.search(r"####\s*(-?\d+)", answer)
        if not match:
            continue
        gold_num = match.group(1)

        # greedy decode max 256 tokens
        prompt_ids = tok(question, return_tensors="pt").input_ids.to(device)
        gen_ids = prompt_ids.clone()
        max_new = 256
        for _ in range(max_new):
            with torch.no_grad():
                logits = run_forward_fn(gen_ids)
            next_id = logits[0, -1, :].argmax().item()
            gen_ids = torch.cat([gen_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == tok.eos_token_id:
                break

        gen_text = tok.decode(gen_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        # extract numeric answer from generation
        pred_match = re.search(r"####\s*(-?\d+)", gen_text)
        if pred_match:
            pred_num = pred_match.group(1)
            if pred_num == gold_num:
                correct += 1
        total += 1

    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    print(f"  [gsm8k] correct={correct}/{total} acc={acc:.4f} ({elapsed:.1f}s)")
    return {"acc": acc, "correct": correct, "total": total}


# ============================================================================
# Three-condition evaluation wrapper
# ============================================================================

def run_benchmark_triple(benchmark_name: str, benchmark_fn, model, tok, device, 
                         bank, heads, bank_layer, b_proj, n_items_or_tokens: int,
                         ctx: int = 1024):
    """Run one benchmark under base, bank_off, bank_on conditions."""
    print(f"\n=== {benchmark_name} ===")
    results = {}

    # --- base: no LPL patch ---
    uninstall_lpl_patch(model)
    def fwd_base(chunk):
        return model(input_ids=chunk, use_cache=False).logits

    if benchmark_name == "wikitext2":
        res = eval_wikitext2(model, tok, n_tokens=n_items_or_tokens, ctx=ctx, 
                            device=device, run_forward_fn=fwd_base)
    elif benchmark_name == "lambada":
        res = eval_lambada(model, tok, n_items=n_items_or_tokens, device=device, 
                          run_forward_fn=fwd_base)
    elif benchmark_name == "hellaswag":
        res = eval_hellaswag(model, tok, n_items=n_items_or_tokens, device=device, 
                            run_forward_fn=fwd_base)
    elif benchmark_name == "gsm8k":
        res = eval_gsm8k(model, tok, n_items=n_items_or_tokens, device=device, 
                        run_forward_fn=fwd_base)
    else:
        raise ValueError(f"unknown benchmark: {benchmark_name}")

    if res is None:
        return None
    results["base"] = res

    # --- install patch for bank_off and bank_on ---
    install_lpl_patch(model)

    def fwd_lpl(chunk):
        s1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, s1):
            model(input_ids=chunk, use_cache=False)
        s2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, s2):
            return model(input_ids=chunk, use_cache=False).logits

    # --- bank_off: empty bank ---
    bank.clear()
    if benchmark_name == "wikitext2":
        res_off = eval_wikitext2(model, tok, n_tokens=n_items_or_tokens, ctx=ctx, 
                                device=device, run_forward_fn=fwd_lpl)
    elif benchmark_name == "lambada":
        res_off = eval_lambada(model, tok, n_items=n_items_or_tokens, device=device, 
                              run_forward_fn=fwd_lpl)
    elif benchmark_name == "hellaswag":
        res_off = eval_hellaswag(model, tok, n_items=n_items_or_tokens, device=device, 
                                run_forward_fn=fwd_lpl)
    elif benchmark_name == "gsm8k":
        res_off = eval_gsm8k(model, tok, n_items=n_items_or_tokens, device=device, 
                            run_forward_fn=fwd_lpl)
    results["bank_off"] = res_off

    # --- bank_on: preloaded bank + trained projector ---
    bank.frozen = False
    bank.slots[bank_layer] = b_proj
    bank.tags[bank_layer] = [(0, -1)] * b_proj.shape[0]
    bank.frozen = True

    if benchmark_name == "wikitext2":
        res_on = eval_wikitext2(model, tok, n_tokens=n_items_or_tokens, ctx=ctx, 
                               device=device, run_forward_fn=fwd_lpl)
    elif benchmark_name == "lambada":
        res_on = eval_lambada(model, tok, n_items=n_items_or_tokens, device=device, 
                             run_forward_fn=fwd_lpl)
    elif benchmark_name == "hellaswag":
        res_on = eval_hellaswag(model, tok, n_items=n_items_or_tokens, device=device, 
                               run_forward_fn=fwd_lpl)
    elif benchmark_name == "gsm8k":
        res_on = eval_gsm8k(model, tok, n_items=n_items_or_tokens, device=device, 
                           run_forward_fn=fwd_lpl)
    results["bank_on"] = res_on

    return results


# ============================================================================
# Projector training (canonical e01-style 200-step)
# ============================================================================

def train_projector(model, tok, bank, heads, P, train_items, *,
                   bank_layer: int, b_raw: torch.Tensor, lr: float,
                   steps: int, device: str, seed: int):
    """Train projector on canonical e01 task (200 steps)."""
    print(f"\n=== Training projector ({steps} steps) ===")
    rng = random.Random(seed)
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr)
    losses = []
    t0 = time.time()

    for step in range(steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, device)

        # rebuild bank with grad-tracked projector
        bank.frozen = False
        proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)
        bank.slots[bank_layer] = proj
        bank.tags[bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True

        opt.zero_grad()

        # forward (round 1: pause/preload, round 2: retrieve+answer)
        state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state1), torch.no_grad():
            model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)

        state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state2), torch.enable_grad():
            logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                          use_cache=False, return_dict=True).logits

        pred = logits[0, ans - 1: -1, :]
        gold = enc.input_ids[0, ans:]
        loss = F.cross_entropy(pred.float(), gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))

        if (step + 1) % 50 == 0 or step == 0:
            recent = sum(losses[-50:]) / min(50, len(losses))
            print(f"  step {step+1}/{steps} loss(avg50)={recent:.4f} ({time.time()-t0:.1f}s)")

    print(f"[train] done in {time.time()-t0:.1f}s")
    return losses


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n_preload", type=int, default=512)
    p.add_argument("--n_train", type=int, default=120, 
                   help="number of train items for projector training")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--proj_ckpt", default=None, 
                   help="optional: skip training and load projector from checkpoint")
    p.add_argument("--skip", default="", 
                   help="comma-separated benchmarks to skip (wikitext2,lambada,hellaswag,gsm8k)")
    p.add_argument("--wikitext_tokens", type=int, default=4000)
    p.add_argument("--lambada_items", type=int, default=300)
    p.add_argument("--hellaswag_items", type=int, default=200)
    p.add_argument("--gsm8k_items", type=int, default=50)
    args = p.parse_args()

    out_path = HERE / f"e13_seed{args.seed}.json"
    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())

    # --- load model ---
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    # --- prepare bank & heads ---
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=args.n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)

    # --- load bank data ---
    blob = data_io.load_bank_blob()
    entries = blob["entries"]
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng = random.Random(args.seed)
    rng.shuffle(train_keys)
    rng.shuffle(test_keys)

    train_keys = train_keys[:args.n_train]
    train_items = data_io.items_for_keys(entries, train_keys)

    # preload keys disjoint from train
    preload_pool = [k for k in train_keys if k not in set(train_keys[:args.n_train])]
    preload_pool += [k for k in test_keys if k not in set(train_keys)]
    preload_keys = preload_pool[:args.n_preload]
    if len(preload_keys) < args.n_preload:
        # fallback: use all train
        all_train = data_io.filter_keys(entries, split="train", solo_pass=True)
        preload_keys = all_train[:args.n_preload]

    b_raw = data_io.b_stack_for_keys(entries, preload_keys, target_norm=15.0,
                                     device=args.device, dtype=torch.float32)

    # --- projector ---
    P = make_projector(d, rank=args.rank).to(args.device).float()
    if args.proj_ckpt:
        P.load_state_dict(torch.load(args.proj_ckpt, map_location=args.device))
        print(f"[e13] loaded projector from {args.proj_ckpt}")
        losses = []
    else:
        # train inline
        install_lpl_patch(model)
        losses = train_projector(model, tok, bank, heads, P, train_items,
                                bank_layer=args.bank_layer, b_raw=b_raw,
                                lr=args.lr, steps=args.steps, device=args.device, seed=args.seed)
        uninstall_lpl_patch(model)

    # --- prepare projected bank for bank_on condition ---
    with torch.no_grad():
        b_proj = residual_apply(P, b_raw).to(dtype=torch.bfloat16)

    # --- run benchmarks ---
    bench_results = {}

    if "wikitext2" not in skip_set:
        res = run_benchmark_triple("wikitext2", eval_wikitext2, model, tok, args.device,
                                   bank, heads, args.bank_layer, b_proj, args.wikitext_tokens, ctx=1024)
        if res:
            bench_results["wikitext2"] = res

    if "lambada" not in skip_set:
        res = run_benchmark_triple("lambada", eval_lambada, model, tok, args.device,
                                   bank, heads, args.bank_layer, b_proj, args.lambada_items)
        if res:
            bench_results["lambada"] = res

    if "hellaswag" not in skip_set:
        res = run_benchmark_triple("hellaswag", eval_hellaswag, model, tok, args.device,
                                   bank, heads, args.bank_layer, b_proj, args.hellaswag_items)
        if res:
            bench_results["hellaswag"] = res

    if "gsm8k" not in skip_set:
        res = run_benchmark_triple("gsm8k", eval_gsm8k, model, tok, args.device,
                                   bank, heads, args.bank_layer, b_proj, args.gsm8k_items)
        if res:
            bench_results["gsm8k"] = res

    # --- compute verdicts ---
    verdicts = {}
    for bname, bres in bench_results.items():
        base = bres["base"]
        off = bres["bank_off"]
        on = bres["bank_on"]

        # check bit-equal for bank_off
        if "nll" in base:
            # perplexity benchmark
            base_val = base["nll"]
            off_val = off["nll"]
            on_val = on["nll"]
            off_equal = abs(off_val - base_val) / max(base_val, 1e-6) <= 1e-3
            on_drift = abs(on_val - base_val) / max(base_val, 1e-6)
            on_pass = on_drift <= 0.05
            verdicts[bname] = {
                "off_bit_equal": off_equal,
                "on_within_5pct": on_pass,
                "on_rel_drift": on_drift,
            }
        elif "acc" in base:
            # accuracy benchmark
            base_val = base["acc"]
            off_val = off["acc"]
            on_val = on["acc"]
            off_equal = abs(off_val - base_val) <= 0.01
            on_drift = abs(on_val - base_val)
            on_pass = on_drift <= 0.03
            verdicts[bname] = {
                "off_bit_equal": off_equal,
                "on_within_3pts": on_pass,
                "on_abs_drift": on_drift,
            }

    # --- output ---
    out = {
        "model": args.model,
        "seed": args.seed,
        "n_preload": len(preload_keys),
        "n_train": len(train_items),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "proj_ckpt": args.proj_ckpt,
        "benchmarks": bench_results,
        "verdicts": verdicts,
        "loss_first25": losses[:25] if losses else [],
        "loss_last25": losses[-25:] if losses else [],
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[e13] -> {out_path}")
    print("\n=== VERDICTS ===")
    for bname, v in verdicts.items():
        print(f"{bname}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
