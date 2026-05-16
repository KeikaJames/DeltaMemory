"""Gate 0b — bank-augmented path smoke test.

Confirms that when we *force* a pause at last_token in a few layers during
round 1, then run round 2 (where the bank is non-empty), the patched
attention path executes without shape/dtype errors and produces logits that
differ from base (proving bank K/V actually entered the softmax).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))

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
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    enc = tok([args.prompt], return_tensors="pt").to(args.device)
    B, T = enc["input_ids"].shape

    # ----- base logits -------------------------------------------------
    with torch.no_grad():
        base_logits = model(**enc, use_cache=False).logits.float().cpu()

    # ----- patch + force pause at last position for round 1, then run K=2
    bank = AttentionBank(num_layers=n_layers, hidden_size=d,
                         device=args.device, dtype=torch.bfloat16)
    heads = LPLHeads.fresh(n_layers, d,
                           pause_bias=-20.0, bank_gate_bias=10.0, halt_bias=-10.0,
                           device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    force = torch.zeros(B, T, dtype=torch.bool, device=args.device)
    force[:, -1] = True  # always pause on last token

    cfg = LPLConfig(K_max=2, enabled=True, force_pause_mask=force, use_halt_head=False)
    runtime = LPLRuntime(model, heads, bank, cfg)
    with torch.no_grad():
        res = runtime.forward(enc["input_ids"], attention_mask=enc.get("attention_mask"))
    lpl_logits = res.logits.float().cpu()
    print(f"[gate0b] rounds_used={res.rounds_used}  bank_total={res.bank_total_size_after}  "
          f"pauses_per_layer_sum={sum(res.pause_count_per_layer)}")
    # Expect: bank_total = n_layers * 1 (last position) after round 1; round 2 also pauses last
    # → expect 2 * n_layers total writes (rounds 1 and 2).

    # Round-1 logits (per_round_logits[0]): bank empty during round 1 so the
    # only difference from base is the pause skip at last position.
    r1 = res.per_round_logits[0].float().cpu()
    # Round-2 logits use bank (non-empty) ⇒ should diverge from base for the
    # active (non-paused) positions.
    diff_r2 = (lpl_logits[:, :-1, :] - base_logits[:, :-1, :]).abs()
    print(f"[gate0b] r2 active-pos |Δ| max={diff_r2.max().item():.3e} "
          f"mean={diff_r2.mean().item():.3e}")
    if diff_r2.max().item() == 0.0:
        print("[gate0b] FAIL — bank K/V had no effect on round 2")
        return 1
    print("[gate0b] PASS — bank-augmented attention path executes and modifies logits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
