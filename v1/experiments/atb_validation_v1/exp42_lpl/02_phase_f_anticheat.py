"""Phase F (early) — Anti-cheat suite on frozen-heads Phase A configuration.

Runs the same 24-prompt eval set with FOUR perturbations of the LPL K=2 pause
path and reports the NLL deltas. If the Phase A gains survive AC1-AC3, they're
likely real; if they collapse, it was bookkeeping cheating.

Conditions
  base               — unpatched
  lpl_pause_K2       — Phase A canonical (pause at {9,18,27}, K=2)
  AC1_shuffle_layers — pause at 3 *random* layers (different per run)
  AC2_random_bank    — pause at canonical layers, but bank h-entries are
                       replaced by random gaussian noise (matched norm)
  AC3_no_bank_read   — bank_gate forced to 0 → equivalent to skipping the
                       bank concat (pure pause+K=2 with no memory read)
  AC4_K1_pause       — pause-skip happens but only K=1 round (no second
                       forward) → tests "is even the pause itself helpful"

For each set we report mean NLL and Δ vs base. A condition that gives ≈ same
gain as the canonical pause is a CHEAT (the mechanism we attributed it to
was not necessary). A condition that wipes the gain is a SUPPORT (that
mechanism was load-bearing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))

import random
import torch
import torch.nn.functional as F

from atb_validation_v1._lib import load_model
from exp42_lpl import (
    AttentionBank,
    LPLHeads,
    LPLConfig,
    LPLRuntime,
    install_lpl_patch,
)
from exp42_lpl.runtime import LPLRuntime as RT

# Reuse same datasets
sys.path.insert(0, str(HERE))
from importlib import import_module
phase_a = import_module("01_phase_a_frozen")
SETS = phase_a.SETS
nll_on_answer = phase_a.nll_on_answer
make_force_mask = phase_a.make_force_mask


def replace_bank_with_noise(bank: AttentionBank, scale_per_layer: list[float]) -> None:
    """In-place: replace each layer's bank tensor with gaussian noise of the
    same shape, scaled to match per-row L2 norm."""
    for l, t in enumerate(bank.slots):
        if t.shape[0] == 0:
            continue
        noise = torch.randn_like(t)
        # match per-row L2 norm
        target = t.norm(dim=-1, keepdim=True)
        noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-9) * target
        bank.slots[l] = noise.to(dtype=t.dtype, device=t.device)


def eval_condition(name, items, tok, model, device,
                   *, mode: str, active_layers: set[int],
                   heads: LPLHeads, bank: AttentionBank,
                   seed: int = 0):
    rng = random.Random(seed)
    n_layers = model.config.num_hidden_layers
    results = []
    for prompt, gold in items:
        full = prompt + gold
        enc = tok(full, return_tensors="pt").to(device)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        ans_start = prompt_ids.shape[1]
        seq_len = enc.input_ids.shape[1]

        if mode == "base":
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nll = nll_on_answer(logits, enc.input_ids, ans_start)
            results.append(nll)
            continue

        if mode == "canonical":
            layers_use = active_layers
        elif mode == "shuffle_layers":
            layers_use = set(rng.sample(range(n_layers), k=len(active_layers)))
        else:
            layers_use = active_layers

        K = 1 if mode == "K1_pause" else 2
        # Optionally zero bank_gate
        if mode == "no_bank_read":
            # temporarily set bank_gate bias to -20 (sigmoid ≈ 0)
            old_biases = []
            for h in heads.bank_gate_heads:
                old_biases.append(h.proj.bias.detach().clone())
                with torch.no_grad():
                    h.proj.bias.fill_(-20.0)

        fpm = make_force_mask(layers_use, prompt_len=ans_start,
                              seq_len=seq_len, device=device,
                              pause_pos=ans_start - 1)
        bank.clear()
        cfg = LPLConfig(K_max=K, enabled=True, force_pause_mask=fpm)
        rt = RT(model, heads, bank, cfg)

        if mode == "random_bank":
            # Run round 1 to populate the bank, then noise-replace, then round 2
            # by calling forward with K=1 first, then K=2 second forward manually.
            # Simpler: call forward as K=2 but inject noise *between* rounds via
            # a small wrapper. We do it by running forward(K=1), corrupting bank,
            # then forward(K=1) again — but the runtime's bank-clear-on-start
            # default is fine here because we pass clear_bank=False for round 2.
            with torch.no_grad():
                # round 1 (populates bank)
                rt1 = RT(model, heads, bank, LPLConfig(K_max=1, enabled=True,
                                                       force_pause_mask=fpm))
                _ = rt1.forward(enc.input_ids, attention_mask=enc.attention_mask,
                                clear_bank=True)
                # corrupt bank
                replace_bank_with_noise(bank, [1.0] * n_layers)
                # round 2 (treats current round as 2 via round_idx)
                # we hack by calling runtime with K_max=1 but round_idx=2
                from exp42_lpl.qwen3_lpl_patch import LPLState, lpl_state_scope
                state = LPLState(bank=bank, heads=heads, round_idx=2,
                                 enabled=True, force_pause_mask=fpm)
                with lpl_state_scope(model, state):
                    out = model(input_ids=enc.input_ids,
                                attention_mask=enc.attention_mask,
                                use_cache=False, return_dict=True)
                logits = out.logits
        else:
            with torch.no_grad():
                res = rt.forward(enc.input_ids, attention_mask=enc.attention_mask)
                logits = res.logits

        if mode == "no_bank_read":
            for h, b in zip(heads.bank_gate_heads, old_biases):
                with torch.no_grad():
                    h.proj.bias.copy_(b)

        nll = nll_on_answer(logits, enc.input_ids, ans_start)
        results.append(nll)
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default=str(HERE / "phase_f_anticheat.json"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    bank = AttentionBank(num_layers=n_layers, hidden_size=d,
                         device=args.device, dtype=torch.bfloat16,
                         max_per_layer=512)
    heads = LPLHeads.fresh(n_layers, d,
                           pause_bias=-20.0, bank_gate_bias=10.0, halt_bias=10.0,
                           device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    active = {n_layers // 4, n_layers // 2, (3 * n_layers) // 4}
    modes = ["base", "canonical", "shuffle_layers", "random_bank",
             "no_bank_read", "K1_pause"]

    summary = {}
    for set_name, items in SETS.items():
        print(f"\n[ac] === {set_name} ({len(items)}) ===")
        per_mode = {}
        for mode in modes:
            nlls = eval_condition(set_name, items, tok, model, args.device,
                                  mode=mode, active_layers=active,
                                  heads=heads, bank=bank, seed=args.seed)
            mean = sum(nlls) / len(nlls)
            per_mode[mode] = {"mean_nll": mean, "per_prompt": nlls}
            tag = "←base" if mode == "base" else (
                f"Δ={mean - per_mode['base']['mean_nll']:+.4f}")
            print(f"  {mode:18s}  mean_nll={mean:.4f}   {tag}")
        summary[set_name] = per_mode

    # Verdict
    print("\n[ac] === verdict matrix (Δ NLL vs base; negative = gain) ===")
    for set_name, per_mode in summary.items():
        b = per_mode["base"]["mean_nll"]
        deltas = {m: per_mode[m]["mean_nll"] - b for m in modes if m != "base"}
        print(f"  {set_name:10s}  " + "  ".join(f"{m}={d:+.3f}" for m, d in deltas.items()))

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ac] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
