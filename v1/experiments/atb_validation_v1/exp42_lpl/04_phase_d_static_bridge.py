"""Phase D — bridge Exp35b/38 static bank to LPL AttentionBank.

We load 10K (subject, relation, target_true, b ∈ R^{2560}) entries produced by
exp35b_memit_bank (MEMIT-preconditioned edits at Qwen3-4B layer 5) and inject
the `b` vectors as STATIC, FROZEN entries into LPL's AttentionBank at a chosen
layer.  We then build a small eval set from a random subsample of those same
(subject, relation, target_true) triples, run base vs LPL K=2 (no pause, no
training), and measure NLL on the target_true token.

If the gain is positive AND collapses under AC2 (random-replace bank), the
LPL bank is acting as a real long-term memory channel — bridging the static
Fact-LoRA Bank work (Exp35b/38) into the dynamic LPL.

Bank entries are loaded at LAYER 9 by default (closest to edit_layer=5 among
our canonical active set {9,18,27}).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))

import torch
from atb_validation_v1._lib import load_model
from exp42_lpl import AttentionBank, LPLHeads, LPLConfig, install_lpl_patch
from exp42_lpl.runtime import LPLRuntime as RT
from exp42_lpl.qwen3_lpl_patch import LPLState, lpl_state_scope
sys.path.insert(0, str(HERE))
from importlib import import_module
phase_a = import_module("01_phase_a_frozen")
nll_on_answer = phase_a.nll_on_answer


def build_prompt(subject: str, relation: str) -> str:
    """Mirror Exp35b's prompt template."""
    return f"{subject} {relation}"


def evaluate(items, tok, model, bank, heads, device, *,
             K: int, bank_active: bool, randomize_bank: bool = False):
    """Run a list of (subject, relation, target_true) through the model.
    K=1: round-1 only (Gate 0 + no second forward; bank unused).
    K=2 with bank_active: round-2 sees preloaded bank.
    """
    nlls = []
    for subj, rel, tgt in items:
        prompt = build_prompt(subj, rel)
        full = prompt + " " + tgt
        enc = tok(full, return_tensors="pt").to(device)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        ans_start = prompt_ids.shape[1]

        if K == 1 and not bank_active:
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans_start))
            continue

        # K=2 LPL with preloaded (frozen) bank, no pauses.
        # Round 1 sees no bank (round_idx=1). Round 2 sees the preloaded bank.
        # We bypass LPLRuntime.clear_bank with a manual two-shot.
        state1 = LPLState(bank=bank, heads=heads, round_idx=1,
                          enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state1):
            with torch.no_grad():
                _ = model(input_ids=enc.input_ids,
                          attention_mask=enc.attention_mask,
                          use_cache=False, return_dict=True)
        # Optionally noise-replace the bank between rounds (AC2)
        if randomize_bank:
            for l, t in enumerate(bank.slots):
                if t.shape[0] == 0: continue
                noise = torch.randn_like(t)
                target_norm = t.norm(dim=-1, keepdim=True)
                noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-9) * target_norm
                bank.slots[l] = noise.to(dtype=t.dtype, device=t.device)
        state2 = LPLState(bank=bank, heads=heads, round_idx=2,
                          enabled=True, force_pause_mask=None)
        with lpl_state_scope(model, state2):
            with torch.no_grad():
                out2 = model(input_ids=enc.input_ids,
                             attention_mask=enc.attention_mask,
                             use_cache=False, return_dict=True)
        nlls.append(nll_on_answer(out2.logits, enc.input_ids, ans_start))
    return nlls


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bank_pt",
                   default=str(REPO / "v1/experiments/atb_validation_v1"
                                      "/exp35b_memit_bank/data/bank.pt"))
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n_eval", type=int, default=40)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--bank_n_preload", type=int, default=512,
                   help="number of b-vectors to preload (FIFO cap will trim)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "phase_d_static_bridge.json"))
    p.add_argument("--rescale_norm", type=float, default=0.0,
                   help="if >0, L2-normalize each b vector and rescale to this norm")
    args = p.parse_args()

    print(f"[phase_d] loading bank from {args.bank_pt}")
    blob = torch.load(args.bank_pt, map_location="cpu", weights_only=False)
    entries = blob["entries"]
    keys = list(entries.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    # Use TEST-split entries that passed solo (model "knows" the fact via edit).
    # For our LPL test we want target_TRUE — the natural fact — which the base
    # model should know moderately well; bank should boost it.
    test_keys = [k for k in keys if entries[k].get("split") == "test"
                 and entries[k].get("solo_pass", False)]
    print(f"[phase_d] {len(test_keys)} eligible test-split solo-pass entries")
    eval_keys = test_keys[: args.n_eval]
    preload_keys = test_keys[args.n_eval : args.n_eval + args.bank_n_preload]

    # Build eval list (subject, relation, target_true)
    eval_items = [(entries[k]["subject"], entries[k]["relation"],
                   entries[k]["target_true"]) for k in eval_keys]
    # Sanity print 2 prompts
    for sj, rl, tg in eval_items[:2]:
        print(f"  prompt: '{build_prompt(sj, rl)}'  → ' {tg}'")

    print(f"[phase_d] loading model")
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"[phase_d] model: {n_layers} layers, d={d}")

    bank = AttentionBank(num_layers=n_layers, hidden_size=d,
                         device=args.device, dtype=torch.bfloat16,
                         max_per_layer=args.bank_n_preload + 16)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=10.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    # Preload b vectors at args.bank_layer
    b_stack = torch.stack([entries[k]["b"].float() for k in preload_keys],
                          dim=0)  # [N, 2560]
    if args.rescale_norm > 0:
        b_stack = b_stack / (b_stack.norm(dim=-1, keepdim=True) + 1e-9) * args.rescale_norm
        print(f"[phase_d] rescaled b vectors to norm={args.rescale_norm}")
    b_stack = b_stack.to(device=args.device, dtype=torch.bfloat16)
    bank.slots[args.bank_layer] = b_stack
    bank.tags[args.bank_layer] = [(0, 0)] * b_stack.shape[0]
    bank.frozen = True
    print(f"[phase_d] preloaded {b_stack.shape[0]} b-vectors at layer "
          f"{args.bank_layer}  (||·|| mean={b_stack.float().norm(dim=-1).mean():.2f})")

    # === Conditions =====================================================
    # base (K=1, no bank)
    print("[phase_d] running base ...")
    nll_base = evaluate(eval_items, tok, model, bank, heads, args.device,
                        K=1, bank_active=False)
    # LPL K=2 with real preloaded bank
    print("[phase_d] running LPL K=2 + preloaded bank ...")
    nll_lpl = evaluate(eval_items, tok, model, bank, heads, args.device,
                       K=2, bank_active=True)
    # AC: noise-replace bank between rounds
    print("[phase_d] running AC: random-bank K=2 ...")
    # we need a fresh preload before each random-bank pass, since AC modifies in-place
    # restore real bank
    bank.frozen = False
    bank.slots[args.bank_layer] = b_stack
    bank.tags[args.bank_layer] = [(0, 0)] * b_stack.shape[0]
    bank.frozen = True
    nll_random = evaluate(eval_items, tok, model, bank, heads, args.device,
                          K=2, bank_active=True, randomize_bank=True)
    # restore for analysis
    bank.frozen = False
    bank.slots[args.bank_layer] = b_stack
    bank.tags[args.bank_layer] = [(0, 0)] * b_stack.shape[0]
    bank.frozen = True

    m_base = sum(nll_base) / len(nll_base)
    m_lpl = sum(nll_lpl) / len(nll_lpl)
    m_rand = sum(nll_random) / len(nll_random)
    print(f"\n[phase_d] N={len(eval_items)}  bank_layer={args.bank_layer}  "
          f"preload_N={b_stack.shape[0]}")
    print(f"  base                   mean_nll={m_base:.4f}")
    print(f"  LPL K=2 +static bank   mean_nll={m_lpl:.4f}   Δ={m_lpl-m_base:+.4f}")
    print(f"  AC random-bank K=2     mean_nll={m_rand:.4f}   Δ={m_rand-m_base:+.4f}")

    out = {
        "n_eval": len(eval_items),
        "bank_layer": args.bank_layer,
        "preload_n": int(b_stack.shape[0]),
        "mean_nll_base": m_base,
        "mean_nll_lpl_k2_bank": m_lpl,
        "mean_nll_lpl_k2_random_bank": m_rand,
        "delta_lpl_vs_base": m_lpl - m_base,
        "delta_random_vs_base": m_rand - m_base,
        "per_prompt": {
            "base": nll_base, "lpl": nll_lpl, "random_bank": nll_random
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[phase_d] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
