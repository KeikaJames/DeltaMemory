"""Gate 0 sanity test for LPL.

Runs Qwen3-4B-Instruct-2507 in two configurations on the same input batch:

  (1) BASE — unpatched model.
  (2) LPL  — patched model with empty bank, pause heads disabled (bias=-10 ⇒
            σ→0 ⇒ pause_mask is all-False), K_max=1.

Asserts that the resulting logits are bit-identical up to a tiny tolerance.

This is the LPL analog of v1 ``tests/test_attn_native_bank.py`` Gate 0.

Usage:
    python v1/experiments/atb_validation_v1/exp42_lpl/00_gate0_sanity.py \
        --model Qwen/Qwen3-4B-Instruct-2507 --device mps
"""
from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))  # so `exp42_lpl` is importable

import torch

from atb_validation_v1._lib import load_model
from exp42_lpl import (
    AttentionBank,
    LPLHeads,
    LPLConfig,
    LPLRuntime,
    install_lpl_patch,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max abs diff between base and LPL logits")
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    print(f"[gate0] loading {args.model} on {args.device} dtype={args.dtype} ...")
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()

    cfg_h = model.config
    n_layers = cfg_h.num_hidden_layers
    d_model = cfg_h.hidden_size
    print(f"[gate0] model: {n_layers} layers, d={d_model}")

    enc = tok([args.prompt, args.prompt + " known as"],
              return_tensors="pt", padding=True).to(args.device)

    # --- (1) BASE forward (unpatched) -----------------------------------
    with torch.no_grad():
        base_out = model(**enc, use_cache=False, return_dict=True)
    base_logits = base_out.logits.detach().float().cpu()
    print(f"[gate0] base logits shape={tuple(base_logits.shape)} "
          f"mean={base_logits.mean().item():.5f}")

    # --- (2) Install LPL patch with disabled state ----------------------
    bank = AttentionBank(
        num_layers=n_layers, hidden_size=d_model,
        device=args.device, dtype=getattr(torch, {"bf16": "bfloat16",
                                                  "fp16": "float16",
                                                  "fp32": "float32"}[args.dtype]),
    )
    # Pause bias deeply negative ⇒ σ(linear(h)) ≈ 0 ⇒ pause_mask all False.
    heads = LPLHeads.fresh(
        num_layers=n_layers, hidden_size=d_model,
        pause_bias=-20.0,        # ⇒ p_pause ≈ 2e-9 ⇒ p_pause > 0.5 is False
        bank_gate_bias=10.0,
        halt_bias=10.0,
        device=args.device, dtype=torch.float32,
    )
    install_lpl_patch(model)

    cfg = LPLConfig(K_max=1, enabled=True)
    runtime = LPLRuntime(model, heads, bank, cfg)
    with torch.no_grad():
        res = runtime.forward(enc["input_ids"], attention_mask=enc.get("attention_mask"))
    lpl_logits = res.logits.detach().float().cpu()
    print(f"[gate0] lpl  logits shape={tuple(lpl_logits.shape)} "
          f"rounds={res.rounds_used} bank_size={res.bank_total_size_after} "
          f"pause_per_layer_sum={sum(res.pause_count_per_layer)}")

    diff = (base_logits - lpl_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[gate0] |Δlogits|: max={max_diff:.3e}  mean={mean_diff:.3e}  tol={args.tol:.0e}")

    # Sanity: no pause should have happened.
    if sum(res.pause_count_per_layer) != 0:
        print("[gate0] FAIL — pause head fired despite bias=-20")
        return 2
    # Sanity: bank must be empty
    if res.bank_total_size_after != 0:
        print("[gate0] FAIL — bank non-empty after K=1 with no pauses")
        return 3
    if max_diff > args.tol:
        print(f"[gate0] FAIL — logits diverged beyond tol")
        return 1
    print("[gate0] PASS — LPL with empty bank + no pause + K=1 is bit-equal to base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
