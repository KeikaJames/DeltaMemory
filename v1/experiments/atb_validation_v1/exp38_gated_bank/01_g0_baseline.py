"""Exp38 — G0 baseline reproduction of Exp35b numbers via gated runtime.

Sanity check: G0 with mask should give identical results to Exp35b's apply_factors.
Run a small subset to verify the gated_patches runtime is correct, then optionally
scale to full panels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import load_model, load_bank, get_dtype, seed_everything  # noqa: E402
from gates import gate_G0_baseline, GATE_FNS  # noqa: E402
from eval_panels import panel_phi1, panel_37_C, panel_neg_36_4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(HERE.parent / "exp35b_memit_bank" / "data" / "bank.pt"))
    ap.add_argument("--n-test", type=int, default=200, help="Φ1 test facts")
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--skip-37c", action="store_true")
    ap.add_argument("--skip-neg", action="store_true")
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp38" / "G0"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    dtype = get_dtype(args.dtype)
    print("[load bank]", flush=True)
    bank = load_bank(args.bank, device=args.device, dtype=dtype)
    print(f"[bank] N={len(bank.fact_ids)} A={tuple(bank.A.shape)} B={tuple(bank.B.shape)}", flush=True)

    gate_fn = GATE_FNS["G0"]
    gate_ctx = {"edit_layer": args.edit_layer}

    results = {"variant": "G0", "config": vars(args)}

    t0 = time.time()
    print("\n=== Φ1 oracle composition ===", flush=True)
    results["phi1"] = panel_phi1(model, tok, bank, gate_fn, gate_ctx,
                                 n_test=args.n_test, k_values=tuple(args.k_values),
                                 seeds=tuple(args.seeds))
    print(f"[time] Φ1 done in {time.time()-t0:.1f}s", flush=True)

    if not args.skip_37c:
        t0 = time.time()
        print("\n=== 37.C cross-talk ===", flush=True)
        results["cross_talk_37c"] = panel_37_C(model, tok, bank, gate_fn, gate_ctx,
                                               n_probes=50, n_patch_sets=20)
        print(f"[time] 37.C done in {time.time()-t0:.1f}s", flush=True)

    if not args.skip_neg:
        t0 = time.time()
        print("\n=== 36.4 negation ===", flush=True)
        results["negation_36_4"] = panel_neg_36_4(model, tok, bank, gate_fn, gate_ctx,
                                                  n_facts=100)
        print(f"[time] 36.4 done in {time.time()-t0:.1f}s", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out/'results.json'}")


if __name__ == "__main__":
    main()
