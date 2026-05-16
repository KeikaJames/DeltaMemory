"""e09 summarize — Compare v1_orig (no projector) vs v2_kproj (with projector).

Usage:
    python summarize.py [--seed SEED]

Reads e09_v1_orig_seedN.json and e09_v2_kproj_seedN.json, prints a verdict
comparing the two conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    v1_path = HERE / f"e09_v1_orig_seed{args.seed}.json"
    v2_path = HERE / f"e09_v2_kproj_seed{args.seed}.json"

    if not v1_path.exists():
        print(f"ERROR: {v1_path} not found. Run with --mode v1_orig first.")
        return 1
    if not v2_path.exists():
        print(f"ERROR: {v2_path} not found. Run with --mode v2_kproj first.")
        return 1

    v1 = json.loads(v1_path.read_text())
    v2 = json.loads(v2_path.read_text())

    print("=" * 80)
    print(f"e09 SUMMARY — seed={args.seed}")
    print("=" * 80)
    print()
    print("HYPOTHESIS:")
    print("  v1 ANB (no projector) got zero signal. Adding a rank-64 K-projector")
    print("  should revive the bank. If true, the projector is the active ingredient.")
    print()
    print("EXPERIMENTAL CONDITIONS:")
    print(f"  Bank: {v1['n_preload']} b-vectors at layer {v1['bank_layer']}")
    print(f"  Train: {v1['n_train']} items, {v1['steps']} steps, lr={v1['lr']}")
    print(f"  Test:  {v1['n_test']} items")
    print()
    print("-" * 80)
    print("(a) v1_orig — ANB without projector (frozen to identity)")
    print("-" * 80)
    print(f"  BEFORE: base={v1['before']['base']:.4f}  real={v1['before']['real']:.4f}  "
          f"rand={v1['before']['rand']:.4f}  zero={v1['before']['zero']:.4f}  off={v1['before']['off']:.4f}")
    print(f"  AFTER:  base={v1['after']['base']:.4f}  real={v1['after']['real']:.4f}  "
          f"rand={v1['after']['rand']:.4f}  zero={v1['after']['zero']:.4f}  off={v1['after']['off']:.4f}")
    delta_v1 = v1['after']['base'] - v1['after']['real']
    print(f"  Δ = {delta_v1:+.4f}  (pass: {v1['verdict']['pass']}) — {v1['verdict']['rule']}")
    print(f"  Trainable params: {v1['n_train_params']:,} (gate heads only)")
    print()
    print("-" * 80)
    print("(b) v2_kproj — ANB + rank-64 trainable K-projector")
    print("-" * 80)
    print(f"  BEFORE: base={v2['before']['base']:.4f}  real={v2['before']['real']:.4f}  "
          f"rand={v2['before']['rand']:.4f}  zero={v2['before']['zero']:.4f}  off={v2['before']['off']:.4f}")
    print(f"  AFTER:  base={v2['after']['base']:.4f}  real={v2['after']['real']:.4f}  "
          f"rand={v2['after']['rand']:.4f}  zero={v2['after']['zero']:.4f}  off={v2['after']['off']:.4f}")
    delta_v2 = v2['after']['base'] - v2['after']['real']
    print(f"  Δ = {delta_v2:+.4f}  (pass: {v2['verdict']['pass']}) — {v2['verdict']['rule']}")
    print(f"  Trainable params: {v2['n_train_params']:,} (projector + gate heads)")
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    # Compute verdict
    both_pass = v1['verdict']['pass'] and v2['verdict']['pass']
    projector_effect = delta_v2 - delta_v1
    
    print(f"  v1_orig (no projector): Δ = {delta_v1:+.4f}  {'✓' if v1['verdict']['pass'] else '✗'}")
    print(f"  v2_kproj (with projector): Δ = {delta_v2:+.4f}  {'✓' if v2['verdict']['pass'] else '✗'}")
    print(f"  Projector effect: Δ_v2 - Δ_v1 = {projector_effect:+.4f}")
    print()
    
    if both_pass:
        print("  ✓ HYPOTHESIS CONFIRMED:")
        print("    v1 ANB alone got ~zero signal (as in original v1 experiments).")
        print("    Adding the K-projector revived the bank to significant signal.")
        print("    CONCLUSION: The K-projector is the active ingredient.")
    elif v1['verdict']['pass'] and not v2['verdict']['pass']:
        print("  ✗ PARTIAL RESULT:")
        print("    v1_orig reproduced the null result (Δ ≈ 0) as expected.")
        print("    BUT v2_kproj did NOT achieve Δ ≤ -2.0.")
        print("    Projector helped but not enough. Possible explanations:")
        print("      - Need more training steps")
        print("      - Need different hyperparameters")
        print("      - v2 infrastructure differs subtly from v1")
    elif not v1['verdict']['pass'] and v2['verdict']['pass']:
        print("  ⚠ UNEXPECTED:")
        print("    v1_orig did NOT reproduce the null result.")
        print("    BUT v2_kproj achieved significant signal.")
        print("    Something about the v2 setup differs from v1. Investigate:")
        print("      - Bank construction (b-vector normalization?)")
        print("      - Attention mechanics (qwen3_lpl_patch vs v1 ANB)")
        print("      - Training dynamics")
    else:
        print("  ✗ BOTH CONDITIONS FAILED:")
        print("    Neither v1_orig nor v2_kproj met pass criteria.")
        print("    Possible issues:")
        print("      - Bad data split or seed")
        print("      - Infrastructure bugs")
        print("      - Hyperparameters need tuning")
    
    print()
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
