"""e20c — adversarial audit on the e20b north-star result.

We've shown (e20b): when b_A is trainable and projector is frozen, Δ_A_init ≈ 5 nat
and evaporates on evict. Before accepting that as proof of item-specific memory,
this audit runs the falsifiers a skeptic would demand:

  (1) **Shuffle-within-setA**: permute b_A rows in the bank. If lift survives,
      content is "bag-of-setA" not item-specific.
  (2) **Held-out items**: split setA = (trainA 256, heldA 256). Train b_A using
      only trainA items as gradient drivers; b_A_held rows are in the bank
      (so they receive gradient via soft-attention coupling) but heldA items
      never appear in the loss. Compare Δ on trainA vs heldA.
  (3) **Capability drift**: with bank installed, measure NLL on an unrelated
      text chunk (WikiText-style sample drawn from baseline bank entries).
      If bank breaks general LM ability, north-star is hollow.
  (4) **Generative demo**: greedy-decode the answer for one trainA item
      with/without bank. If "with bank" produces the gold token where
      "without bank" doesn't, that's the usability proof.

Output: v2/experiments/e20c_adversarial_audit/seed{S}.json
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
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=512)
    args = p.parse_args()

    print(f"[e20c] device={args.device} model={args.model} seed={args.seed} N={args.N}")
    torch.manual_seed(args.seed); random.seed(args.seed)

    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    all_keys = data_io.filter_keys(entries, solo_pass=True)
    rng = random.Random(args.seed); rng.shuffle(all_keys)
    setA_keys = all_keys[:args.N]
    setB_keys = all_keys[args.N:2*args.N]
    drift_keys = all_keys[2*args.N:2*args.N + 64]

    half = args.N // 2
    trainA_keys = setA_keys[:half]
    heldA_keys = setA_keys[half:]

    setA_items = data_io.items_for_keys(entries, setA_keys)
    trainA_items = data_io.items_for_keys(entries, trainA_keys)
    heldA_items = data_io.items_for_keys(entries, heldA_keys)
    setB_items = data_io.items_for_keys(entries, setB_keys)
    drift_items = data_io.items_for_keys(entries, drift_keys)
    print(f"[e20c] setA={len(setA_items)} (train={len(trainA_items)}, held={len(heldA_items)}) "
          f"setB={len(setB_items)} drift={len(drift_items)}")

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

    def install_A_perm(perm):
        proj = (b_A[perm] + P(b_A[perm])).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(0, -1)] * proj.shape[0]
        bank.frozen = True

    identity = torch.arange(args.N, device=args.device)

    def install_A():
        install_A_perm(identity)

    def install_B():
        with torch.no_grad():
            proj = (b_B + P(b_B)).to(dtype=torch.bfloat16)
        bank.frozen = False
        bank.slots[args.bank_layer] = proj
        bank.tags[args.bank_layer] = [(1, -1)] * proj.shape[0]
        bank.frozen = True

    def clear_bank():
        bank.frozen = False
        bank.slots[args.bank_layer] = torch.empty(0, d, device=args.device, dtype=torch.bfloat16)
        bank.tags[args.bank_layer] = []
        bank.frozen = True

    print("[e20c] baselines...")
    base_trainA = eval_base(model, tok, trainA_items, args.device)
    base_heldA = eval_base(model, tok, heldA_items, args.device)
    base_B = eval_base(model, tok, setB_items, args.device)
    base_drift = eval_base(model, tok, drift_items, args.device)
    print(f"[e20c] baseline trainA={base_trainA:.4f} heldA={base_heldA:.4f} "
          f"setB={base_B:.4f} drift={base_drift:.4f}")

    opt = torch.optim.AdamW([b_A], lr=args.lr)
    trng = random.Random(args.seed)
    losses = []
    print(f"[e20c] training b_A on trainA only ({len(trainA_items)} items), "
          f"steps={args.steps}, lr={args.lr}...")
    t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = trng.choice(trainA_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        install_A()
        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([b_A], 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 100 == 0 or step == 0:
            recent = sum(losses[-50:]) / min(50, len(losses))
            print(f"    step {step+1}/{args.steps} loss(avg50)={recent:.4f}")
    print(f"[e20c] training done ({time.time()-t0:.1f}s)")

    install_A()
    nll_trainA_id = eval_lpl(model, tok, bank, heads, trainA_items, args.device)
    nll_heldA_id = eval_lpl(model, tok, bank, heads, heldA_items, args.device)
    print(f"[e20c] identity bank: trainA={nll_trainA_id:.4f} heldA={nll_heldA_id:.4f}")

    perm = torch.randperm(args.N, generator=torch.Generator(device="cpu").manual_seed(args.seed + 100)).to(args.device)
    install_A_perm(perm)
    nll_trainA_shuf = eval_lpl(model, tok, bank, heads, trainA_items, args.device)
    nll_heldA_shuf = eval_lpl(model, tok, bank, heads, heldA_items, args.device)
    print(f"[e20c] shuffled b_A: trainA={nll_trainA_shuf:.4f} heldA={nll_heldA_shuf:.4f}")

    install_A()
    nll_drift_withbank = eval_lpl(model, tok, bank, heads, drift_items, args.device)
    print(f"[e20c] bank-installed drift NLL = {nll_drift_withbank:.4f} (baseline {base_drift:.4f})")

    install_B()
    nll_trainA_B = eval_lpl(model, tok, bank, heads, trainA_items, args.device)
    nll_setB_B = eval_lpl(model, tok, bank, heads, setB_items, args.device)

    clear_bank()
    nll_trainA_zero = eval_lpl(model, tok, bank, heads, trainA_items, args.device)

    delta_trainA = base_trainA - nll_trainA_id
    delta_heldA = base_heldA - nll_heldA_id
    delta_trainA_shuf = base_trainA - nll_trainA_shuf
    delta_heldA_shuf = base_heldA - nll_heldA_shuf
    delta_trainA_after_evict = base_trainA - nll_trainA_B
    delta_B = base_B - nll_setB_B
    delta_trainA_zero = base_trainA - nll_trainA_zero
    drift_delta = base_drift - nll_drift_withbank

    print()
    print("=== adversarial audit ===")
    print(f"Δ trainA (identity)          = {delta_trainA:+.4f}")
    print(f"Δ heldA  (identity, not trained)= {delta_heldA:+.4f}")
    print(f"Δ trainA (shuffled b_A)      = {delta_trainA_shuf:+.4f}    ← falsifier")
    print(f"Δ heldA  (shuffled b_A)      = {delta_heldA_shuf:+.4f}")
    print(f"Δ trainA (after evict→b_B)   = {delta_trainA_after_evict:+.4f}")
    print(f"Δ setB   (b_B)               = {delta_B:+.4f}")
    print(f"Δ trainA (empty bank)        = {delta_trainA_zero:+.4f}")
    print(f"Δ drift  (b_A installed)     = {drift_delta:+.4f}   ← capability drift")

    print("\n[e20c] generative demo (greedy, 8 tokens) on first trainA item:")
    sj, rl, tg = trainA_items[0]
    prompt = f"{sj} {rl}"
    enc, _, ans = encode_qa(tok, prompt, tg, args.device)
    prompt_ids = enc.input_ids[:, :ans]
    with torch.no_grad():
        model.lpl_state = None
        out_no = model.generate(prompt_ids, max_new_tokens=8, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        gen_no = tok.decode(out_no[0, ans:], skip_special_tokens=True)
        install_A()
        state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state2):
            out_yes = model.generate(prompt_ids, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        gen_yes = tok.decode(out_yes[0, ans:], skip_special_tokens=True)
    print(f"  prompt    : {prompt!r}")
    print(f"  gold ans  : {tg!r}")
    print(f"  no bank   : {gen_no!r}")
    print(f"  with bank : {gen_yes!r}")

    item_specific_pass = (delta_trainA - delta_trainA_shuf) >= 1.0
    held_generalization = delta_heldA
    drift_ok = abs(drift_delta) <= 0.3
    print()
    print(f"item-specific (Δ_id − Δ_shuf ≥ 1.0)  : {item_specific_pass}  "
          f"(gap = {delta_trainA - delta_trainA_shuf:+.4f})")
    print(f"held-out generalization (Δ_heldA)    : {delta_heldA:+.4f}")
    print(f"capability drift (|Δ_drift|≤0.3)     : {drift_ok}  ({drift_delta:+.4f})")

    result = {
        "experiment": "e20c_adversarial_audit",
        "model": args.model, "seed": args.seed, "N": args.N, "half": half,
        "bank_layer": args.bank_layer, "rank": args.rank,
        "lr": args.lr, "steps": args.steps,
        "base_trainA": base_trainA, "base_heldA": base_heldA,
        "base_setB": base_B, "base_drift": base_drift,
        "nll_trainA_identity": nll_trainA_id,
        "nll_heldA_identity": nll_heldA_id,
        "nll_trainA_shuffled": nll_trainA_shuf,
        "nll_heldA_shuffled": nll_heldA_shuf,
        "nll_trainA_after_evict_to_B": nll_trainA_B,
        "nll_setB_B": nll_setB_B,
        "nll_trainA_zero": nll_trainA_zero,
        "nll_drift_with_bank": nll_drift_withbank,
        "delta_trainA": delta_trainA,
        "delta_heldA": delta_heldA,
        "delta_trainA_shuffled": delta_trainA_shuf,
        "delta_heldA_shuffled": delta_heldA_shuf,
        "delta_trainA_after_evict": delta_trainA_after_evict,
        "delta_B": delta_B,
        "delta_trainA_zero": delta_trainA_zero,
        "drift_delta": drift_delta,
        "item_specific_gap": float(delta_trainA - delta_trainA_shuf),
        "item_specific_pass": bool(item_specific_pass),
        "drift_ok": bool(drift_ok),
        "demo_prompt": prompt, "demo_gold": tg,
        "demo_no_bank": gen_no, "demo_with_bank": gen_yes,
        "loss_first10": losses[:10], "loss_last10": losses[-10:],
    }
    (HERE / f"seed{args.seed}.json").write_text(json.dumps(result, indent=2))
    print(f"\n[e20c] -> {HERE / f'seed{args.seed}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
