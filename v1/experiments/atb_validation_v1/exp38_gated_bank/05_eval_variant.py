"""Exp38 — full evaluation matrix runner.

Given a variant in {G0, G1, G2, G3, G4, G5}, load the appropriate gate state
and run the 6 panels: Φ1, Φ3*, 36.4 negation, 37.C cross-talk, HellaSwag, D6 ppl.

(*Φ3 routed eval reuses Exp35b's router; for G2 we replace router with retrieval.)
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

from common import load_model, load_bank, get_dtype  # noqa: E402
from gates import GATE_FNS  # noqa: E402
from eval_panels import panel_phi1, panel_37_C, panel_neg_36_4, panel_hellaswag  # noqa: E402


def build_gate_ctx(variant: str, args, bank):
    ctx = {"edit_layer": args.edit_layer}
    if variant == "G1":
        d = torch.load(HERE / "data" / "g1_theta.pt", map_location="cpu", weights_only=False)
        theta = d["theta"]
        # Ensure ordering matches bank.fact_ids
        assert d["fact_ids"] == bank.fact_ids, "theta fact_ids mismatch"
        ctx["theta"] = theta.to(bank.A.device)
    elif variant == "G2":
        ctx["k_r"] = args.k_r
    elif variant in ("G2cos", "G2l2"):
        ctx["k_r"] = args.k_r
        ctx["A_full"] = bank.A  # (d_in, N) — already on bank.A.device
    elif variant in ("G3", "G4"):
        d = torch.load(HERE / "data" / f"{variant}_heads.pt", map_location="cpu", weights_only=False)
        assert d["fact_ids"] == bank.fact_ids, "head fact_ids mismatch"
        ctx["W_g"] = d["W_g"].to(bank.A.device)
        ctx["b_g"] = d["b_g"].to(bank.A.device)
    elif variant == "G5":
        d_g3 = torch.load(HERE / "data" / "G3_heads.pt", map_location="cpu", weights_only=False)
        ctx["W_g"] = d_g3["W_g"].to(bank.A.device)
        ctx["b_g"] = d_g3["b_g"].to(bank.A.device)
        ctx["k_r"] = args.k_r
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(GATE_FNS.keys()), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(HERE.parent / "exp35b_memit_bank" / "data" / "bank.pt"))
    ap.add_argument("--k-r", type=int, default=5, help="for G2/G5 retrieval")
    ap.add_argument("--n-phi1", type=int, default=1500)
    ap.add_argument("--phi1-k", type=int, nargs="+", default=[1, 10, 100, 1000])
    ap.add_argument("--phi1-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-neg-facts", type=int, default=600)
    ap.add_argument("--n-cross-probes", type=int, default=100)
    ap.add_argument("--n-cross-sets", type=int, default=50)
    ap.add_argument("--n-hellaswag", type=int, default=1000)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["phi1", "37c", "neg", "hellaswag"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.variant if args.variant not in ("G2", "G2cos", "G2l2") else f"{args.variant}_kr{args.k_r}"
    out = Path(args.out) if args.out else HERE / "run_qwen_exp38" / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    dtype = get_dtype(args.dtype)
    bank = load_bank(args.bank, device=args.device, dtype=dtype)

    gate_fn = GATE_FNS[args.variant]
    gate_ctx = build_gate_ctx(args.variant, args, bank)

    results = {"variant": args.variant, "tag": tag, "config": vars(args)}

    if "phi1" not in args.skip:
        t0 = time.time()
        print(f"\n=== Φ1 ({args.n_phi1} facts × {args.phi1_k} × {args.phi1_seeds}) ===", flush=True)
        results["phi1"] = panel_phi1(model, tok, bank, gate_fn, gate_ctx,
                                     n_test=args.n_phi1,
                                     k_values=tuple(args.phi1_k),
                                     seeds=tuple(args.phi1_seeds))
        print(f"[time] Φ1 {time.time()-t0:.1f}s", flush=True)

    if "37c" not in args.skip:
        t0 = time.time()
        print(f"\n=== 37.C cross-talk ===", flush=True)
        results["cross_talk_37c"] = panel_37_C(model, tok, bank, gate_fn, gate_ctx,
                                               n_probes=args.n_cross_probes,
                                               n_patch_sets=args.n_cross_sets)
        print(f"[time] 37.C {time.time()-t0:.1f}s", flush=True)

    if "neg" not in args.skip:
        t0 = time.time()
        print(f"\n=== 36.4 negation ===", flush=True)
        results["negation_36_4"] = panel_neg_36_4(model, tok, bank, gate_fn, gate_ctx,
                                                  n_facts=args.n_neg_facts)
        print(f"[time] 36.4 {time.time()-t0:.1f}s", flush=True)

    if "hellaswag" not in args.skip:
        t0 = time.time()
        print(f"\n=== HellaSwag ===", flush=True)
        results["hellaswag"] = panel_hellaswag(model, tok, bank, gate_fn, gate_ctx,
                                               n_examples=args.n_hellaswag)
        print(f"[time] HellaSwag {time.time()-t0:.1f}s", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] {out/'results.json'}")


if __name__ == "__main__":
    main()
