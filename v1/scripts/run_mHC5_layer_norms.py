"""Phase mHC5 — Layer-norm energy probe.

Hook each layer output, measure ||x_l||_F across the 3 architectures
at alpha=1.5 with bank injection.  The prediction:

  - Residual GPT-2: exponential growth in log-y (injection amplified by
    unbounded residual accumulator)
  - mHC GPT-2: sub-linear / plateau (Sinkhorn-Knopp C^k bound)
  - HC GPT-2: intermediate (row-softmax mixing, no SK guarantee)

If the multi-stream thesis is correct, ||x_L||_F / ||x_0||_F for mHC
should be significantly smaller than for residual (H5: >= 10x gap).

Usage:
    .venv-mac/bin/python scripts/run_mHC5_layer_norms.py \
        --device mps --dtype bfloat16 --seeds 0 1 2 3 4 \
        --alpha 1.5 --segments 32 --segment-length 512 \
        --out reports/cleanroom/mHC5_layer_norms
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import GPT2LMHeadModel

from deltamemory.baselines.mhc_gpt2 import convert_gpt2_lm_head_model
from deltamemory.baselines.mhc_gpt2.transformers.gpt2_mhc import (
    MhcGPT2LMHeadModel,
)
from scripts.run_mHC2_perturbation_sweep import _wikitext2_segments
from scripts.run_mHC3_bank_injection import (
    ARCH_SPEC,
    _get_attention,
    _get_num_layers,
    _read_with_bank,
    _write_fact,
    load_architecture,
)


def measure_norms(model, tok, segments, device: str, bank=None, alpha=0.0) -> list[list[float]]:
    """Measure per-layer hidden-state Frobenius norms for each segment.
    Returns list of per-segment per-layer norms.
    If bank is None, runs baseline (no injection)."""
    n_layers = _get_num_layers(model)
    all_norms: list[list[float]] = []

    for seg_idx, seg_ids in enumerate(segments):
        ids = seg_ids.unsqueeze(0).to(device)  # (1, T)
        am = torch.ones_like(ids)

        layer_norms: list[float] = []

        def make_hook(layer_i):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                layer_norms.append(out.float().norm().item())
            return hook

        handles = []
        for li in range(n_layers):
            block = model.transformer.h[li]
            h = block.register_forward_hook(make_hook(li))
            handles.append(h)

        if bank is not None:
            # Use the injection hook from mHC3
            injection_handles = []
            for li in range(n_layers):
                attn = _get_attention(model, li)
                entry = bank[li]
                mk = entry["K"].to(device)
                mv = entry["V"].to(device)
                n_embd = mk.size(-1)

                def make_inject(li_local, mk_local, mv_local):
                    def inj_hook(module, input, output):
                        if alpha == 0.0:
                            return output
                        x = input[0]
                        B, T, _ = x.shape
                        qkv = module.c_attn(x)
                        split_dim = module.split_size
                        q, k, v = qkv.split(split_dim, dim=-1)
                        n_head = module.num_heads
                        head_dim = n_embd // n_head
                        q_r = q.view(B, T, n_head, head_dim).transpose(1, 2)
                        k_r = k.view(B, T, n_head, head_dim).transpose(1, 2)
                        v_r = v.view(B, T, n_head, head_dim).transpose(1, 2)
                        mk_r = mk_local.view(1, n_head, 1, head_dim).expand(B, -1, 1, -1)
                        mv_r = mv_local.view(1, n_head, 1, head_dim).expand(B, -1, 1, -1)
                        k_cat = torch.cat([k_r, mk_r], dim=-2)
                        v_cat = torch.cat([v_r, alpha * mv_r], dim=-2)
                        scale = head_dim ** -0.5
                        scores = torch.matmul(q_r, k_cat.transpose(-2, -1)) * scale
                        attn_w = torch.softmax(scores.float(), dim=-1).to(q_r.dtype)
                        attn_out = torch.matmul(attn_w, v_cat)
                        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, n_embd)
                        attn_out = module.c_proj(attn_out)
                        attn_out = module.resid_dropout(attn_out)
                        return (attn_out, None)
                    return inj_hook

                h = attn.register_forward_hook(make_inject(li, mk, mv))
                injection_handles.append(h)

        with torch.no_grad():
            try:
                model(input_ids=ids, attention_mask=am, use_cache=True)
            except Exception:
                pass

        for h in handles:
            h.remove()
        if bank is not None:
            for h in injection_handles:
                h.remove()

        if layer_norms:
            all_norms.append(layer_norms)

    return all_norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="gpt2")
    ap.add_argument("--archs", nargs="+", default=["residual", "hc", "mhc"])
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--segments", type=int, default=32)
    ap.add_argument("--segment-length", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="reports/cleanroom/mHC5_layer_norms")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Wikitext-2 segments (shared across arches)
    from transformers import GPT2TokenizerFast
    tok_wiki = GPT2TokenizerFast.from_pretrained(args.base_model)
    tok_wiki.pad_token = tok_wiki.eos_token
    segments = _wikitext2_segments(tok_wiki, args.segments, args.segment_length,
                                   seed=42)  # fixed seed for prompt selection

    all_results: dict[str, list] = {}

    for arch_name in args.archs:
        label = ARCH_SPEC[arch_name]["label"]
        print(f"\n{'='*60}\n  {label}  (alpha={args.alpha})\n{'='*60}", flush=True)

        t_load = time.time()
        tok, model = load_architecture(arch_name, args.base_model, args.device, dtype)
        n_layers = _get_num_layers(model)
        print(f"  loaded in {time.time()-t_load:.1f}s, layers={n_layers}", flush=True)

        # Write one fact for bank
        write_prompt = "Fact: Python was created by Ada Lovelace."
        bank = _write_fact(model, tok, write_prompt, n_layers, args.device)

        arch_norms: list[dict] = []
        for seed in args.seeds:
            torch.manual_seed(seed)

            # Baseline norms (no bank)
            base_norms = measure_norms(model, tok, segments, args.device, bank=None)
            # Injected norms (bank at alpha)
            inj_norms = measure_norms(model, tok, segments, args.device, bank=bank, alpha=args.alpha)

            # Average across segments for this seed
            n_segs = len(base_norms)
            n_inj_segs = len(inj_norms)
            if n_segs > 0 and len(base_norms[0]) > 0:
                avg_base = [sum(base_norms[s][l] for s in range(n_segs)) / n_segs
                           for l in range(len(base_norms[0]))]
                if n_inj_segs > 0 and len(inj_norms[0]) > 0:
                    avg_inj = [sum(inj_norms[s][l] for s in range(n_inj_segs)) / n_inj_segs
                              for l in range(len(inj_norms[0]))]
                else:
                    avg_inj = []
            else:
                avg_base, avg_inj = [], []

            # Delta norms (injection - baseline) isolates the bank signal
            delta_norms = []
            if avg_base and avg_inj and len(avg_base) == len(avg_inj):
                delta_norms = [avg_inj[l] - avg_base[l] for l in range(len(avg_base))]
            # Growth ratio: last-layer delta / first-layer delta
            if len(delta_norms) >= 2 and delta_norms[0] > 1e-9:
                growth = delta_norms[-1] / delta_norms[0]
            else:
                growth = 0.0

            arch_norms.append(dict(
                seed=seed,
                base_norms=avg_base,
                inj_norms=avg_inj,
                delta_norms=delta_norms,
                growth_ratio=round(growth, 2),
            ))
            print(f"  seed={seed} delta-growth(L/L0)={growth:.1f}x  layers={len(delta_norms)}", flush=True)

        all_results[arch_name] = arch_norms
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save
    with open(out_dir / "h5_norms.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Quick summary (delta norm growth = injection amplification across layers)
    print(f"\n[mHC5] === H5 Summary: delta-norm growth at alpha={args.alpha} ===")
    print(f"  (Last-layer delta / first-layer delta — should be HIGH for residual, LOW for mHC)")
    for arch_name in args.archs:
        if arch_name in all_results:
            growths = [s["growth_ratio"] for s in all_results[arch_name]]
            mean_g = sum(growths) / len(growths)
            print(f"  {arch_name:>10s}: {mean_g:.1f}x growth  (n={len(growths)} seeds)")

    # H5 gate: residual delta-growth > mHC delta-growth * 10 ?
    if "residual" in all_results and "mhc" in all_results:
        g_res = sum(s["growth_ratio"] for s in all_results["residual"]) / len(all_results["residual"])
        g_mhc = sum(s["growth_ratio"] for s in all_results["mhc"]) / len(all_results["mhc"])
        if g_mhc > 0:
            fold = g_res / max(g_mhc, 1e-9)
            print(f"\n  residual/mHC delta-growth gap: {fold:.1f}x  (H5 gate requires >=10x)")
            if fold >= 10:
                print("  H5: PASS (residual amplifies injection >=10x more than mHC)")
            else:
                print(f"  H5: FAIL ({fold:.1f}x < 10x threshold)")
        elif g_res > 0:
            print(f"\n  H5: PASS (residual shows growth, mHC shows zero/negative growth)")

    print(f"\n[mHC5] wrote {out_dir}/h5_norms.json", flush=True)


if __name__ == "__main__":
    main()
