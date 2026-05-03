"""Conservation-law regression for non-Gemma-4 architectures via ArchAdapter.

Runs on real HF models. For each (model, family) pair:
  1. Load the frozen LLM in bf16.
  2. Build AttnNativePatcher (auto-picks ArchAdapter via class-name match).
  3. Run a short prompt through the unpatched model -> logits_orig.
  4. Run the same prompt with the patcher installed but bank empty -> logits_patched.
  5. Assert torch.equal(logits_orig, logits_patched) -- bit-equal.

This is the K7 gate: any new ArchAdapter must produce *identical* output to
the upstream forward when the bank is empty / alpha=0. Without this we cannot
trust the family on the benchmark.

Usage:
    python tests/conservation_real_models.py --models gemma-4-E2B qwen3-4b
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running from repo root without installing package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank


MODEL_REGISTRY = {
    "gemma-4-E2B":   "google/gemma-4-E2B",
    "gemma-4-31b":   "google/gemma-4-31B-it",
    "qwen3-4b":      "Qwen/Qwen3-4B-Instruct-2507",
    "deepseek-32b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "glm-4-9b":      "THUDM/glm-4-9b-chat",
}


def conservation_check(model_id: str, prompt: str, dtype: torch.dtype, device: str) -> dict:
    print(f"\n=== {model_id} ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits_orig = model(**enc).logits.detach().clone()

    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)  # empty
    print(f"  adapter = {patcher.adapter.name}, num_layers = {patcher.num_layers}", flush=True)

    with patcher.patched(), patcher.injecting(bank=bank, alpha=0.0), torch.no_grad():
        logits_patched = model(**enc).logits.detach().clone()

    diff = (logits_orig - logits_patched).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    bit_equal = torch.equal(logits_orig, logits_patched)
    print(f"  max-abs-diff = {max_abs:.3e}  mean-abs-diff = {mean_abs:.3e}  bit_equal = {bit_equal}", flush=True)

    del model, patcher, bank, logits_orig, logits_patched
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"model": model_id, "max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "bit_equal": bit_equal}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["gemma-4-E2B"], choices=list(MODEL_REGISTRY))
    p.add_argument("--prompt", default="Q: Who is the mayor of Paris?\nA:")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=1e-3,
                   help="max-abs-diff tolerance; bit_equal is the hard gate.")
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"device = {args.device}, dtype = {args.dtype}", flush=True)
    results = []
    for short in args.models:
        try:
            r = conservation_check(MODEL_REGISTRY[short], args.prompt, dtype, args.device)
            results.append(r)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
            results.append({"model": MODEL_REGISTRY[short], "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY ===", flush=True)
    fail = 0
    for r in results:
        if "error" in r:
            print(f"  {r['model']:50s}  ERROR  {r['error']}", flush=True)
            fail += 1
        else:
            tag = "PASS" if r["max_abs_diff"] < args.threshold else "FAIL"
            print(f"  {r['model']:50s}  {tag}  max-abs={r['max_abs_diff']:.3e}  bit_equal={r['bit_equal']}", flush=True)
            if tag == "FAIL":
                fail += 1
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
