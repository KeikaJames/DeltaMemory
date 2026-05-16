"""e21 — counterfactual injection demo (the "make AI lie" test).

Phase C pivot.  e20c proved the soft-attention-uniform-bank lift is a global
style attractor, NOT item-specific memory.  v1-style usability requires the
result the user wants: install bank → model greedy-decodes a *chosen wrong
answer* on a *specific factual prompt* it normally gets right.

Protocol per fact:
  1. Verify base model greedy-decodes the truth_prefix correctly (no bank).
  2. Spin up a single-slot bank at layer L.  Initialize one trainable b vector.
  3. Train b (and only b) to minimize NLL of the *counterfactual* answer
     conditioned on the prompt.  Projector frozen at random init.
  4. Re-decode the prompt with the trained single-slot bank installed.
     Pass iff: greedy decode now starts with the counterfactual_prefix.
  5. Sanity cross-check: with the trained-for-fact-i bank installed, query
     fact-j (different prompt). Decode must NOT be affected (the bank should
     be a content-targeted injector, not a global style shifter).

Pass = (a) base says truth, (b) bank flips it to counterfactual, (c)
cross-prompt unaffected for ≥ N/2 of the cross pairs.

Output: v2/experiments/e21_counterfactual_injection/results.json
"""
from __future__ import annotations

import argparse
import json
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
    make_projector, load_model,
)

FACTS = [
    ("The capital of France is",            " Paris",     " Berlin"),
    ("The capital of Japan is",             " Tokyo",     " Beijing"),
    ("The largest planet in our solar system is", " Jupiter", " Saturn"),
    ("The author of Romeo and Juliet is",   " William",   " Charles"),
    ("Water boils at",                      " 100",       " 50"),
    ("The Great Wall is located in",        " China",     " India"),
]


def greedy_decode(model, tok, prompt, device, *, bank=None, heads=None, max_new=12):
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    attn = enc.attention_mask
    if bank is None:
        model.lpl_state = None
        with torch.no_grad():
            out = model.generate(input_ids, attention_mask=attn,
                                  max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
    else:
        state = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state), torch.no_grad():
            out = model.generate(input_ids, attention_mask=attn,
                                  max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
    new_ids = out[0, input_ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True)


def train_one_b(model, tok, prompt, counterfactual, bank, heads, P, device,
                bank_layer, *, steps=200, lr=5e-3, b_init_norm=15.0, verbose=False):
    d = model.config.hidden_size
    g = torch.Generator(device="cpu").manual_seed(0)
    b = torch.randn(1, d, generator=g).to(device=device, dtype=torch.float32)
    b = b * (b_init_norm / b.norm())
    b = nn.Parameter(b.clone())

    full = prompt + counterfactual
    enc_full = tok(full, return_tensors="pt").to(device)
    enc_prompt = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc_prompt.input_ids.shape[1]
    ans_start = prompt_len
    input_ids = enc_full.input_ids
    attn = enc_full.attention_mask

    def install():
        proj = (b + P(b)).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[bank_layer] = proj
        bank.tags[bank_layer] = [(0, -1)]
        bank.frozen = True

    opt = torch.optim.AdamW([b], lr=lr)
    losses = []
    for step in range(steps):
        install()
        state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state1), torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state2), torch.enable_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
        logits = out.logits
        pred = logits[0, ans_start - 1: -1, :].float()
        gold = input_ids[0, ans_start:]
        loss = F.cross_entropy(pred, gold)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([b], 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (step + 1) % 50 == 0:
            print(f"      step {step+1}/{steps} loss={loss.item():.4f}")
    install()
    return b, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    args = p.parse_args()

    print(f"[e21] device={args.device} model={args.model} layer={args.bank_layer}")
    torch.manual_seed(0)

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    install_lpl_patch(model)

    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=1)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                            halt_bias=10.0, device=args.device, dtype=torch.float32)
    for pp in heads.bank_gate_heads.parameters():
        pp.requires_grad_(False)
    P = make_projector(d, rank=args.rank).to(args.device).float()
    for pp in P.parameters():
        pp.requires_grad_(False)

    print("\n=== Phase 0: verify base model says the truth ===")
    base_decodes = []
    for prompt, truth, counter in FACTS:
        gen = greedy_decode(model, tok, prompt, args.device, max_new=8)
        ok = gen.lower().lstrip().startswith(truth.lower().lstrip())
        base_decodes.append((prompt, truth, counter, gen, ok))
        print(f"  [{'OK' if ok else 'XX'}] {prompt!r} -> {gen!r}  (truth={truth!r})")
    usable_facts = [t for t in base_decodes if t[4]]
    print(f"\nusable facts: {len(usable_facts)}/{len(FACTS)}")
    if not usable_facts:
        print("[e21] no fact base-decodes to truth — abort")
        return 1

    print("\n=== Phase 1: train one b per fact, decode with bank ===")
    trained_banks = []  # (prompt, truth, counter, b_tensor)
    per_fact = []
    for prompt, truth, counter, base_gen, _ in usable_facts:
        print(f"\n[FACT] prompt={prompt!r}  truth={truth!r}  counter={counter!r}")
        t0 = time.time()
        b, losses = train_one_b(model, tok, prompt, counter, bank, heads, P,
                                  args.device, args.bank_layer,
                                  steps=args.steps, lr=args.lr, verbose=True)
        with_gen = greedy_decode(model, tok, prompt, args.device,
                                  bank=bank, heads=heads, max_new=12)
        flipped = with_gen.lower().lstrip().startswith(counter.lower().lstrip())
        # save b for cross-test
        trained_banks.append((prompt, truth, counter, b.detach().clone()))
        per_fact.append({
            "prompt": prompt, "truth": truth, "counterfactual": counter,
            "base_decode": base_gen, "with_bank_decode": with_gen,
            "flipped_to_counterfactual": bool(flipped),
            "loss_first": losses[0], "loss_last": losses[-1],
            "train_time_s": time.time() - t0,
        })
        print(f"  no bank:   {base_gen!r}")
        print(f"  with bank: {with_gen!r}")
        print(f"  flipped to counterfactual: {flipped}  (loss {losses[0]:.3f} -> {losses[-1]:.3f})")

    print("\n=== Phase 2: cross-prompt independence ===")
    cross = []
    def install_b(b_tensor):
        proj = (b_tensor + P(b_tensor)).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, -1)]
        bank.frozen = True

    for i, (pi, ti, ci, bi) in enumerate(trained_banks):
        for j, (pj, tj, cj, _) in enumerate(trained_banks):
            if i == j:
                continue
            install_b(bi)
            gen = greedy_decode(model, tok, pj, args.device,
                                bank=bank, heads=heads, max_new=8)
            unaffected = gen.lower().lstrip().startswith(tj.lower().lstrip())
            wrong_flip = gen.lower().lstrip().startswith(ci.lower().lstrip())
            cross.append({"bank_for": pi, "prompt": pj, "decode": gen,
                          "still_truth": bool(unaffected),
                          "leaked_counterfactual": bool(wrong_flip)})
            tag = "OK" if unaffected else ("LEAK" if wrong_flip else "DRIFT")
            print(f"  [{tag}] bank=<{pi[:25]}…>  prompt=<{pj[:25]}…>  -> {gen!r}")

    n_flips = sum(1 for f in per_fact if f["flipped_to_counterfactual"])
    n_cross_ok = sum(1 for c in cross if c["still_truth"])
    n_cross_total = len(cross)
    print("\n=== summary ===")
    print(f"counterfactual injection success: {n_flips}/{len(per_fact)} facts")
    print(f"cross-prompt independence:        {n_cross_ok}/{n_cross_total} pairs preserved truth")
    success = (n_flips >= max(1, len(per_fact) // 2)) and (n_cross_ok >= n_cross_total // 2)
    print(f"overall pass: {success}")

    result = {
        "experiment": "e21_counterfactual_injection",
        "model": args.model, "bank_layer": args.bank_layer,
        "steps": args.steps, "lr": args.lr,
        "base_phase": [{"prompt": p, "truth": t, "counter": c,
                         "decode": g, "base_correct": ok}
                        for p, t, c, g, ok in base_decodes],
        "per_fact": per_fact, "cross": cross,
        "n_flips": n_flips, "n_facts_eval": len(per_fact),
        "n_cross_truth_preserved": n_cross_ok, "n_cross_total": n_cross_total,
        "overall_pass": bool(success),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\n[e21] -> {HERE / 'results.json'}")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
