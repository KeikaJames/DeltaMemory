"""e17 negation robustness — content-specificity falsifier.

MOTIVATION (from v2/verdicts/E01_VERDICT.md): H2c showed that even a
*collapsed* bank (all rows identical = mean of all b-vectors) still gives
Δ=−4.62 NLL drop after projector training. This raises a terminal threat:
what if the bank doesn't transmit *factual content* at all, but just acts as
a generic smoothing mechanism — extra K/V slots that the projector spreads
out to "catch" the decoder's attention regardless of what's being asked?

If the v2 mechanism *does* transmit content-specific information via the
bank, then NEGATED queries (where the "correct" answer is flipped) should
suppress or invert the gain. If the gain persists *even when the target is
wrong*, the projector is just a content-free loss-landscape smoother.

EXPERIMENT DESIGN:
Take 120 standard Exp35b QA prompts. Format: "{subject} {relation}".
Build NEGATED variants using two templates:

  T1: "Contrary to popular belief, {subject} {relation} not {target_true} but"
      → expected completion: random_target (a different fact's answer)

  T2: "{subject} {relation} INCORRECT ANSWER: {target_true}. CORRECT ANSWER:"
      → expected completion: random_target

For each item, sample a random_target from a *different* entry in the test set.

Train a canonical projector for 200 steps on STANDARD (non-negated) prompts.
Eval on FOUR sets:

  (a) standard prompts, standard targets (control — should see big Δ)
  (b) standard prompts, random targets (negative control — signal should NOT
      transfer to wrong answers if bank is content-specific)
  (c) negated prompts (T1+T2), random-target labels (the "correct" answer per
      the negated prompt is now the random_target — if ALB transmits content,
      it should NOT help because the bank's b-vector encodes target_true, not
      random_target)
  (d) negated prompts (T1+T2), target_true labels (the original truth, now
      "wrong" per the prompt's negation — if ALB is content-blind, Δ≈0; if ALB
      transmits content, it might *still help* despite the prompt's negation,
      because the bank encodes target_true)

INTERPRETATION TABLE:
| set | if Δ ≤ -1           | if Δ ≈ 0                | meaning                                     |
|-----|---------------------|-------------------------|---------------------------------------------|
| (a) | expected            | bug                     | sanity passes                               |
| (b) | content-blind: fail | content-specific: ✓     | does signal transfer to wrong target?       |
| (c) | content-blind: fail | content-specific: ✓     | does signal transfer via negated prompt?    |
| (d) | content-specific: ✓ | content-blind: expected | bank's content "overrides" prompt negation? |

The most informative comparison: (b). If Δ_b ≤ −1, the bank doesn't encode
"the answer is target_true" — it just encodes "any answer is more likely here".
That would be terminal for v2's claim of content-specific transmission.

OUTPUT: v2/experiments/e17_negation_robustness/e17_seed{seed}.json with:
    - delta_a, delta_b, delta_c, delta_d
    - NLL_base, NLL_lpl for each of a/b/c/d
    - templates used
    - verdict: {"pass": bool, "interpretation": str}
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


def forward_lpl_k2(model, bank, heads, enc, *, grad=False):
    """Standard 2-pass LPL forward: round 1 (preload), round 2 (exec)."""
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


def build_negated_items(items, entries_dict, seed, template_id):
    """Build negated (prompt, target) pairs.

    For each item (subj, rel, tg_true), construct a negated prompt and a
    random_target from another item.

    template_id:
      1 => "Contrary to popular belief, {subj} {rel} not {tg_true} but"
      2 => "{subj} {rel} INCORRECT ANSWER: {tg_true}. CORRECT ANSWER:"

    Returns: list of (negated_prompt, random_target) tuples.
    """
    rng = random.Random(seed + template_id)
    all_targets = list({e["target_true"] for e in entries_dict.values()})
    negated = []
    for subj, rel, tg_true in items:
        # sample random target != tg_true
        pool = [t for t in all_targets if t != tg_true]
        if not pool:
            pool = all_targets  # fallback
        rand_tg = rng.choice(pool)
        if template_id == 1:
            prompt = f"Contrary to popular belief, {subj} {rel} not {tg_true} but"
        else:
            prompt = f"{subj} {rel} INCORRECT ANSWER: {tg_true}. CORRECT ANSWER:"
        negated.append((prompt, rand_tg))
    return negated


def main():
    p = argparse.ArgumentParser(description="e17 negation robustness")
    p.add_argument("--seed", type=int, default=0)
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
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else HERE / f"e17_seed{args.seed}.json"

    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]

    # Build train/test splits (standard, non-negated)
    rng = random.Random(args.seed)
    train_keys = data_io.filter_keys(entries, split="train", solo_pass=True)
    test_keys = data_io.filter_keys(entries, split="test", solo_pass=True)
    rng.shuffle(train_keys); rng.shuffle(test_keys)
    train_keys = train_keys[:args.n_train]
    test_keys = test_keys[:args.n_eval]
    # preload from train side, disjoint from train_keys
    preload_pool = data_io.filter_keys(entries, split="train", solo_pass=True)
    preload_pool = [k for k in preload_pool if k not in set(train_keys)]
    preload_keys = preload_pool[:args.n_preload]

    train_items = data_io.items_for_keys(entries, train_keys)
    test_items = data_io.items_for_keys(entries, test_keys)

    # Build negated variants for test_items
    negated_t1 = build_negated_items(test_items, entries, args.seed, template_id=1)
    negated_t2 = build_negated_items(test_items, entries, args.seed, template_id=2)
    print(f"[e17] train={len(train_items)} test={len(test_items)} preload={len(preload_keys)} seed={args.seed}")
    print(f"[e17] negated_t1={len(negated_t1)} negated_t2={len(negated_t2)}")

    # Load model
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

    # ===== TRAINING (on standard prompts) =====
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        enc, _, ans = encode_qa(tok, f"{sj} {rl}", tg, args.device)
        # rebuild bank with grad-tracking projector
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
            print(f"  [e17] step {step+1}/{args.steps} loss(avg25)={recent:.4f} ({time.time()-t0:.1f}s)")
    print(f"[e17] training done in {time.time()-t0:.1f}s")

    # ===== EVALUATION =====
    # Helper: eval with base model (no bank)
    def eval_base(prompts_and_targets):
        nlls = []
        for prompt, target in prompts_and_targets:
            enc, _, ans = encode_qa(tok, prompt, target, args.device)
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    # Helper: eval with LPL bank
    def eval_lpl(prompts_and_targets):
        nlls = []
        for prompt, target in prompts_and_targets:
            enc, _, ans = encode_qa(tok, prompt, target, args.device)
            apply_proj()
            logits = forward_lpl_k2(model, bank, heads, enc, grad=False)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans))
        return sum(nlls) / max(len(nlls), 1)

    # (a) standard prompts, standard targets — control
    print("[e17] eval (a) standard prompts, standard targets...")
    items_a = [(f"{sj} {rl}", tg) for sj, rl, tg in test_items]
    base_a = eval_base(items_a)
    lpl_a = eval_lpl(items_a)
    delta_a = lpl_a - base_a

    # (b) standard prompts, random targets — negative control
    print("[e17] eval (b) standard prompts, random targets...")
    # reuse negated_t1's random_targets (they're already sampled differently)
    items_b = [(f"{sj} {rl}", negated_t1[i][1]) for i, (sj, rl, tg) in enumerate(test_items)]
    base_b = eval_base(items_b)
    lpl_b = eval_lpl(items_b)
    delta_b = lpl_b - base_b

    # (c) negated prompts (T1+T2), random-target labels (the "correct" answer per negation)
    print("[e17] eval (c) negated prompts, random targets...")
    items_c_t1 = [(prompt, rand_tg) for prompt, rand_tg in negated_t1]
    items_c_t2 = [(prompt, rand_tg) for prompt, rand_tg in negated_t2]
    base_c_t1 = eval_base(items_c_t1)
    lpl_c_t1 = eval_lpl(items_c_t1)
    base_c_t2 = eval_base(items_c_t2)
    lpl_c_t2 = eval_lpl(items_c_t2)
    base_c = (base_c_t1 + base_c_t2) / 2
    lpl_c = (lpl_c_t1 + lpl_c_t2) / 2
    delta_c = lpl_c - base_c

    # (d) negated prompts (T1+T2), target_true labels (the original truth, now "wrong")
    print("[e17] eval (d) negated prompts, target_true labels...")
    items_d_t1 = [(negated_t1[i][0], tg) for i, (sj, rl, tg) in enumerate(test_items)]
    items_d_t2 = [(negated_t2[i][0], tg) for i, (sj, rl, tg) in enumerate(test_items)]
    base_d_t1 = eval_base(items_d_t1)
    lpl_d_t1 = eval_lpl(items_d_t1)
    base_d_t2 = eval_base(items_d_t2)
    lpl_d_t2 = eval_lpl(items_d_t2)
    base_d = (base_d_t1 + base_d_t2) / 2
    lpl_d = (lpl_d_t1 + lpl_d_t2) / 2
    delta_d = lpl_d - base_d

    print(f"[e17] RESULTS:")
    print(f"  (a) standard/standard:   base={base_a:.4f} lpl={lpl_a:.4f} Δ={delta_a:.4f}")
    print(f"  (b) standard/random:     base={base_b:.4f} lpl={lpl_b:.4f} Δ={delta_b:.4f}")
    print(f"  (c) negated/random:      base={base_c:.4f} lpl={lpl_c:.4f} Δ={delta_c:.4f}")
    print(f"  (d) negated/target_true: base={base_d:.4f} lpl={lpl_d:.4f} Δ={delta_d:.4f}")

    # VERDICT
    content_specific = True
    reasons = []
    if delta_a <= -1.0:
        reasons.append("(a) passes: Δ_a ≤ -1 (sanity)")
    else:
        reasons.append("(a) FAIL: Δ_a > -1 (sanity broken)")
        content_specific = False

    if delta_b > -0.5:
        reasons.append("(b) passes: bank does NOT help random targets on standard prompts")
    else:
        reasons.append("(b) FAIL: Δ_b ≤ -0.5 → bank is content-blind (terminal)")
        content_specific = False

    if delta_c > -0.5:
        reasons.append("(c) passes: bank does NOT help random targets on negated prompts")
    else:
        reasons.append("(c) FAIL: Δ_c ≤ -0.5 → bank helps wrong answers (content-blind)")
        content_specific = False

    if delta_d <= -1.0:
        reasons.append("(d) content-specific evidence: bank still helps target_true despite negation")
    else:
        reasons.append("(d) neutral: bank does not override prompt's negation")

    verdict = {
        "pass": content_specific,
        "interpretation": " | ".join(reasons),
        "thresholds": {
            "a_sanity": "Δ_a ≤ -1.0",
            "b_content_blind_fail": "Δ_b ≤ -0.5",
            "c_content_blind_fail": "Δ_c ≤ -0.5",
            "d_content_override": "Δ_d ≤ -1.0",
        },
    }

    out = {
        "seed": args.seed,
        "model": args.model,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "n_preload": len(preload_keys),
        "bank_layer": args.bank_layer,
        "rank": args.rank,
        "lr": args.lr,
        "steps": args.steps,
        "templates": {
            "T1": "Contrary to popular belief, {subject} {relation} not {target_true} but",
            "T2": "{subject} {relation} INCORRECT ANSWER: {target_true}. CORRECT ANSWER:",
        },
        "results": {
            "a_standard_standard": {"base": base_a, "lpl": lpl_a, "delta": delta_a},
            "b_standard_random": {"base": base_b, "lpl": lpl_b, "delta": delta_b},
            "c_negated_random": {"base": base_c, "lpl": lpl_c, "delta": delta_c,
                                 "t1": {"base": base_c_t1, "lpl": lpl_c_t1},
                                 "t2": {"base": base_c_t2, "lpl": lpl_c_t2}},
            "d_negated_true": {"base": base_d, "lpl": lpl_d, "delta": delta_d,
                               "t1": {"base": base_d_t1, "lpl": lpl_d_t1},
                               "t2": {"base": base_d_t2, "lpl": lpl_d_t2}},
        },
        "verdict": verdict,
        "loss_first25": losses[:25],
        "loss_last25": losses[-25:],
    }

    out_path.write_text(json.dumps(out, indent=2))
    print(f"[e17] -> {out_path}")
    print(f"[e17] verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
