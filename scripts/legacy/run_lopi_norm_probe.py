"""R-3.5 LOPI Gaussian layer-norm probe.

Validates the H1' / Section §5 mechanism claim from
`reports/cleanroom/lopi_v33/FINDINGS.md`: the Gaussian layer weight
w(ell, t) actually concentrates injection energy at mid-stack and
keeps shallow / deep layers near the unpatched residual norms.

For a fixed (model, neutral_prompt, alpha) we record per-layer L2 norm
of the residual stream (post block.attn output) under three conditions:

* ``base``   — model with no bank injection at all (legacy forward).
* ``A0``     — bank injected, LOPI disabled (raw alpha * M_V add).
* ``A4``     — bank injected, LOPI Gaussian + gamma (no orthogonal).

The expected signature is:

* ``base``  — smooth log-linear norm growth across layers (typical GPT-2 trace).
* ``A0``    — uniform offset across all layers; readout (last 1-2 layers)
              also significantly displaced.
* ``A4``    — bell-shaped offset peaking at mu ~= 0.5L, with shallow and
              deep layers staying close to ``base``.

Output: `reports/cleanroom/lopi_v33/r35_norm_probe.json` + .md table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.lopi import LOPIConfig, LOPIState
from scripts.run_lopi_ablation import _load_model, _read_with_bank_lopi
from scripts.run_mHC3_bank_injection import (
    NEUTRAL_PROMPTS,
    _get_attention,
    _write_fact,
)


@torch.no_grad()
def _record_layer_norms_base(model, tok, prompt, device):
    """Return per-layer mean residual L2 norm under no bank injection."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    norms = []

    def make_hook(idx):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            norms.append((idx, x.float().norm(dim=-1).mean().item()))
        return hook

    handles = []
    for i, blk in enumerate(model.transformer.h):
        handles.append(blk.attn.register_forward_hook(make_hook(i)))
    model(**enc)
    for h in handles:
        h.remove()
    norms.sort()
    return [n for _, n in norms]


@torch.no_grad()
def _record_layer_norms_with_bank(model, tok, prompt, bank, alpha, n_layers, device,
                                  lopi_cfg, lopi_state):
    """Per-layer mean residual L2 norm under bank+LOPI injection."""
    norms = []

    def hook_pre_idx(idx):
        def hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            norms.append((idx, x.float().norm(dim=-1).mean().item()))
        return hook

    pre_handles = []
    for i, blk in enumerate(model.transformer.h):
        pre_handles.append(blk.attn.register_forward_hook(hook_pre_idx(i)))

    _read_with_bank_lopi(
        model, tok, prompt, bank, alpha, n_layers, device,
        lopi_cfg=lopi_cfg, lopi_state=lopi_state,
    )

    for h in pre_handles:
        h.remove()
    norms.sort()
    return [n for _, n in norms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="alpha for high-stress regime where A4 was shown to shield")
    ap.add_argument("--out", default="reports/cleanroom/lopi_v33/r35_norm_probe.json")
    args = ap.parse_args()

    print(f"[r3.5] loading {args.model} on {args.device} ({args.dtype})")
    model, tok = _load_model(args.model, args.device, args.dtype)
    n_layers = model.config.n_layer
    print(f"[r3.5] n_layers={n_layers}")

    prompts = NEUTRAL_PROMPTS  # 5
    fact_text = "Fact: The Sun is a star at the centre of the Solar System."

    bank = _write_fact(model, tok, fact_text, n_layers, args.device)

    # --- per-prompt traces ---
    cfg_a0 = LOPIConfig(enabled=False)
    cfg_a4 = LOPIConfig(enabled=True, orthogonal=False, gaussian=True, derivative=True)
    state_a0 = LOPIState(num_layers=n_layers)
    state_a4 = LOPIState(num_layers=n_layers)

    base_norms_per_prompt = []
    a0_norms_per_prompt = []
    a4_norms_per_prompt = []

    for p in prompts:
        base_norms_per_prompt.append(
            _record_layer_norms_base(model, tok, p, args.device))

        state_a0.reset()
        a0_norms_per_prompt.append(
            _record_layer_norms_with_bank(model, tok, p, bank, args.alpha, n_layers,
                                          args.device, cfg_a0, state_a0))

        state_a4.reset()
        a4_norms_per_prompt.append(
            _record_layer_norms_with_bank(model, tok, p, bank, args.alpha, n_layers,
                                          args.device, cfg_a4, state_a4))

    # ---- aggregate to per-layer mean across prompts ----
    def col_mean(rows):
        return [sum(r[i] for r in rows) / len(rows) for i in range(len(rows[0]))]

    base = col_mean(base_norms_per_prompt)
    a0 = col_mean(a0_norms_per_prompt)
    a4 = col_mean(a4_norms_per_prompt)

    # Per-layer offset relative to base.
    a0_offset = [a0[i] - base[i] for i in range(n_layers)]
    a4_offset = [a4[i] - base[i] for i in range(n_layers)]
    # Relative perturbation: |offset| / base. This normalizes against the
    # natural per-layer scale (final-layer base norms are 5-15x larger than
    # mid-stack, so absolute offsets there look "big" but are small in
    # relative terms).
    a0_rel = [abs(a0_offset[i]) / max(base[i], 1e-6) for i in range(n_layers)]
    a4_rel = [abs(a4_offset[i]) / max(base[i], 1e-6) for i in range(n_layers)]

    # Concentration ratio: max(|offset|) / mean(|offset|).
    # Higher => offset is peaked (Gaussian-shaped).
    def conc(xs):
        absxs = [abs(x) for x in xs]
        m = max(absxs)
        avg = sum(absxs) / len(absxs)
        return m / avg if avg > 0 else float("inf")

    out = {
        "model": args.model,
        "alpha": args.alpha,
        "n_layers": n_layers,
        "n_prompts": len(prompts),
        "per_layer": {
            "base_norm": base,
            "A0_norm": a0,
            "A4_norm": a4,
            "A0_offset": a0_offset,
            "A4_offset": a4_offset,
            "A0_relative": a0_rel,
            "A4_relative": a4_rel,
        },
        "concentration_abs": {
            "A0": conc(a0_offset),
            "A4": conc(a4_offset),
        },
        "concentration_rel": {
            "A0": conc(a0_rel),
            "A4": conc(a4_rel),
        },
        "mean_relative_offset": {
            "A0": sum(a0_rel) / n_layers,
            "A4": sum(a4_rel) / n_layers,
        },
        "argmax_relative_offset_layer": {
            "A0": int(max(range(n_layers), key=lambda i: a0_rel[i])),
            "A4": int(max(range(n_layers), key=lambda i: a4_rel[i])),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # ---- pretty table ----
    print(f"\n{'layer':>5} {'base':>10} {'A0':>10} {'A4':>10} "
          f"{'A0-base':>10} {'A4-base':>10} {'A0_rel':>8} {'A4_rel':>8}")
    for i in range(n_layers):
        print(f"{i:>5d} {base[i]:>10.3f} {a0[i]:>10.3f} {a4[i]:>10.3f} "
              f"{a0_offset[i]:>+10.3f} {a4_offset[i]:>+10.3f} "
              f"{a0_rel[i]:>8.3f} {a4_rel[i]:>8.3f}")
    print(f"\nMean relative perturbation |x_inj-x_base|/|x_base|:")
    print(f"  A0 = {out['mean_relative_offset']['A0']:.3f}")
    print(f"  A4 = {out['mean_relative_offset']['A4']:.3f}  "
          f"(reduction {(1 - out['mean_relative_offset']['A4']/out['mean_relative_offset']['A0'])*100:.1f}%)")
    print(f"\nArgmax relative-offset layer (μ_target = L/2 = {n_layers/2:.1f}):")
    print(f"  A0 = layer {out['argmax_relative_offset_layer']['A0']}")
    print(f"  A4 = layer {out['argmax_relative_offset_layer']['A4']}")
    print(f"\nConcentration (relative): A0={out['concentration_rel']['A0']:.2f}  "
          f"A4={out['concentration_rel']['A4']:.2f}")
    print(f"\n[r3.5] wrote {out_path}")


if __name__ == "__main__":
    main()
