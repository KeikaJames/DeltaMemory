"""Exp35b — 04b: D6 capability audit (WikiText-103 ppl drift)."""
from __future__ import annotations

import argparse
import importlib.util as iu
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
_spec = iu.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = iu.module_from_spec(_spec); _spec.loader.exec_module(_bb)
apply_factors = _bb.apply_factors
restore = _bb.restore
assert_bit_equal = _bb.assert_bit_equal


@torch.no_grad()
def ppl_on(model, tok, prompts, max_len=128):
    nll_sum = 0.0
    n_tok = 0
    device = next(model.parameters()).device
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        if enc["input_ids"].size(1) < 2:
            continue
        out = model(**enc, use_cache=False)
        logits = out.logits[0, :-1].float()
        tgt = enc["input_ids"][0, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
        nll_sum += nll
        n_tok += tgt.numel()
    return nll_sum / max(1, n_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--edit-layer", type=int, default=5)
    ap.add_argument("--bank", default=str(DATA / "bank.pt"))
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35b" / "d6_capability.json"))
    args = ap.parse_args()

    print("[load wikitext]", flush=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    prompts = [x["text"] for x in ds if x["text"].strip() and len(x["text"]) > 64][: args.n_prompts]
    print(f"[wikitext] {len(prompts)} prompts", flush=True)

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    W_ref = model.model.layers[args.edit_layer].mlp.down_proj.weight.data.clone()

    print("[base ppl]", flush=True)
    t0 = time.time()
    base_nll = ppl_on(model, tok, prompts)
    print(f"base nll/token = {base_nll:.4f} ({time.time()-t0:.0f}s)", flush=True)
    assert_bit_equal(model, args.edit_layer, W_ref)

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    fids = list(entries.keys())
    torch.manual_seed(0)
    pick = torch.randperm(len(fids))[: args.k].tolist()
    factors = [(entries[fids[i]]["b"].to(device, dtype=dtype),
                entries[fids[i]]["a"].to(device, dtype=dtype)) for i in pick]

    print(f"[patched ppl k={args.k}]", flush=True)
    W_old = apply_factors(model, args.edit_layer, factors)
    try:
        t0 = time.time()
        pat_nll = ppl_on(model, tok, prompts)
        print(f"patched nll/token = {pat_nll:.4f} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        restore(model, args.edit_layer, W_old)
    assert_bit_equal(model, args.edit_layer, W_ref)

    import math
    base_ppl = math.exp(base_nll)
    pat_ppl = math.exp(pat_nll)
    drift = (pat_ppl - base_ppl) / base_ppl
    summary = {
        "k": args.k,
        "n_prompts": len(prompts),
        "base_nll_per_tok": base_nll,
        "patched_nll_per_tok": pat_nll,
        "base_ppl": base_ppl,
        "patched_ppl": pat_ppl,
        "ppl_drift_frac": drift,
        "pre_registered_max_drift": 0.05,
        "pass": drift < 0.05,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
