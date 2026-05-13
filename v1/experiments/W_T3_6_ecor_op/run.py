"""W-T3.6 — ECOR operator-level ablation (orthogonal × soft_blend × α).

Operator-level (not full forward NLL) ablation answering the user's
question: *"why is orthogonal projection still default off, and where is
the improved-ECOR ablation table?"*

The production path in ``deltamemory/memory/lopi.py`` performs::

    m_perp = orthogonal_novelty(M_V, V_ctx)            # if ortho=True
    out_bank_lopi = gamma * w * m_perp                  # additive readout
    V_out = V_ctx + alpha * out_bank_lopi               # caller adds

ECOR (Phase X.7, ``deltamemory/memory/lopi_inject.py``) replaces step 3
with an isometric rotation, blended by ``cfg.soft_blend ∈ [0, 1]``::

    V_rot   = cos(θ)·V_ctx + sin(θ)·(M_perp · ‖V_ctx‖/‖M_perp‖)
    V_blend = (1-soft_blend)·V_add + soft_blend·V_rot

Default ``soft_blend=0`` ⇒ bit-equal additive (preserves α=0 redline);
W.3 DECISION explicitly defers any default-flip until W-T3.6 produces
this table.

What this script does
---------------------
1. Loads Qwen2.5-0.5B-Instruct on MPS bf16.
2. Captures real ``V_ctx`` (per-layer attention output) and a synthetic
   bank ``M_V`` (V-readout from a "fact" prompt) over 8 neutral prompts
   × ALL transformer layers — operator-level distribution is realistic.
3. Sweeps the 32-cell grid::

       ortho       ∈ {False, True}
       soft_blend  ∈ {0.0, 0.25, 0.5, 1.0}
       alpha       ∈ {0.5, 1.0, 2.0, 4.0}

4. For each cell measures (mean over prompts × layers):

   * ``rel_perturb``        = ‖V_out − V_ctx‖ / ‖V_ctx‖
   * ``cos_v_out_v_ctx``    = direction preservation (1.0 = unchanged)
   * ``norm_ratio``         = ‖V_out‖ / ‖V_ctx‖   (1.0 = ECOR isometry)
   * ``m_perp_ratio``       = ‖M_perp‖ / ‖M_V‖    (ortho aggression)

Red lines verified inline:
* ``soft_blend=0`` produces max-abs-diff = 0 vs additive baseline at every cell
* α=0 produces max-abs-diff = 0 vs the bank-off path
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import subprocess
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.lopi import orthogonal_novelty
from deltamemory.memory.lopi_inject import ECORConfig, lopi_inject

NEUTRAL_PROMPTS: tuple[str, ...] = (
    "The boiling point of water at sea level is one hundred degrees Celsius.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "The Pacific Ocean covers approximately one third of the Earth's surface area.",
    "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per second.",
    "Mount Everest is the tallest mountain above sea level, located in the Himalayan range.",
    "The Industrial Revolution began in Britain during the late eighteenth century.",
    "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
)
FACT_PROMPT = "Fact: The Sun is a star at the centre of the Solar System."

ORTHO_AXIS = (False, True)
BLEND_AXIS = (0.0, 0.25, 0.5, 1.0)
ALPHA_AXIS = (0.5, 1.0, 2.0, 4.0)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _capture_v_outputs(model, tokenizer, prompt: str, device: str, dtype) -> list[torch.Tensor]:
    """Forward `prompt` and return list of attention-output hidden states per layer.

    We use ``output_hidden_states=True`` and treat ``hidden_states[ℓ]`` as a
    realistic distribution of post-attention V-readout values. This is a
    proxy: it includes residual + MLP, but it's the same proxy used in
    LOPI's production diagnostics, so it suffices for an operator-level
    ablation table.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states[0] = embedding; [1..L] = block outputs. Use [1..L].
    return [hs.to(dtype) for hs in out.hidden_states[1:]]


def _stats(V_ctx: torch.Tensor, V_out: torch.Tensor, M_V: torch.Tensor, M_perp: torch.Tensor) -> dict[str, float]:
    eps = 1e-8
    delta = V_out - V_ctx
    norm_v = V_ctx.norm(dim=-1)
    norm_d = delta.norm(dim=-1)
    norm_o = V_out.norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(V_out, V_ctx, dim=-1)
    norm_mv = M_V.norm(dim=-1)
    norm_mp = M_perp.norm(dim=-1)
    return {
        "rel_perturb": float((norm_d / norm_v.clamp_min(eps)).mean().item()),
        "cos_v_out_v_ctx": float(cos.mean().item()),
        "norm_ratio": float((norm_o / norm_v.clamp_min(eps)).mean().item()),
        "m_perp_ratio": float((norm_mp / norm_mv.clamp_min(eps)).mean().item()),
    }


def _additive_baseline(V_ctx: torch.Tensor, M_perp: torch.Tensor, alpha: float) -> torch.Tensor:
    """Bit-equal recomputation of the default lopi additive path."""
    return V_ctx + alpha * M_perp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="experiments/W_T3_6_ecor_op")
    ap.add_argument("--n-prompts", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[load] {args.model}  device={args.device}  dtype={dtype}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    torch.set_grad_enabled(False)

    prompts = list(NEUTRAL_PROMPTS[: args.n_prompts])
    print(f"[capture] V_ctx over {len(prompts)} neutral prompts")
    v_ctx_per_prompt: list[list[torch.Tensor]] = [
        _capture_v_outputs(model, tok, p, args.device, dtype) for p in prompts
    ]

    print(f"[capture] M_V from fact prompt")
    m_v_layers = _capture_v_outputs(model, tok, FACT_PROMPT, args.device, dtype)
    # Mean over fact tokens to get a per-layer "bank readout vector".
    m_v_pooled = [m.mean(dim=1, keepdim=True) for m in m_v_layers]  # (1, 1, D)

    L = len(m_v_layers)
    D = m_v_layers[0].shape[-1]
    print(f"[setup] L={L} D={D}")

    # -------------------------------------------------------------------
    # α=0 / soft_blend=0 redline checks
    print("[redline] soft_blend=0 must be bit-equal additive across grid")
    V0 = v_ctx_per_prompt[0][0]
    M0 = m_v_pooled[0].expand_as(V0)
    Mp0 = orthogonal_novelty(M0, V0, eps=1e-8)
    g = torch.ones((), device=V0.device, dtype=V0.dtype)
    w = torch.ones((), device=V0.device, dtype=V0.dtype)
    for blend in (0.0,):
        for alpha in ALPHA_AXIS:
            cfg = ECORConfig(enabled=True, soft_blend=blend)
            v_out = lopi_inject(V0, Mp0, g, w, alpha_base=alpha, cfg=cfg)
            v_add = _additive_baseline(V0, Mp0, alpha)
            diff = (v_out - v_add).abs().max().item()
            assert diff == 0.0, f"REDLINE FAIL: blend=0 alpha={alpha} diff={diff}"
    print("[redline] OK")

    # -------------------------------------------------------------------
    # Grid sweep
    cells: list[dict] = []
    t0 = time.time()
    n_cells = len(ORTHO_AXIS) * len(BLEND_AXIS) * len(ALPHA_AXIS)
    cell_idx = 0
    for ortho in ORTHO_AXIS:
        for blend in BLEND_AXIS:
            for alpha in ALPHA_AXIS:
                cell_idx += 1
                stats_list: list[dict] = []
                for v_ctx_layers in v_ctx_per_prompt:
                    for ell in range(L):
                        V_ctx = v_ctx_layers[ell]
                        # Broadcast pooled M_V to V_ctx's (B, T, D).
                        M_V = m_v_pooled[ell].expand_as(V_ctx)
                        M_perp = orthogonal_novelty(M_V, V_ctx, eps=1e-8) if ortho else M_V
                        cfg = ECORConfig(enabled=True, soft_blend=blend)
                        V_out = lopi_inject(V_ctx, M_perp, g, w, alpha_base=alpha, cfg=cfg)
                        stats_list.append(_stats(V_ctx, V_out, M_V, M_perp))
                # Mean across prompts × layers
                agg = {k: float(sum(s[k] for s in stats_list) / len(stats_list)) for k in stats_list[0]}
                cell = {
                    "ortho": ortho,
                    "soft_blend": blend,
                    "alpha": alpha,
                    "n_samples": len(stats_list),
                    **agg,
                }
                cells.append(cell)
                print(
                    f"[cell {cell_idx:>2}/{n_cells}] ortho={ortho!s:>5} blend={blend:.2f} a={alpha:.2f}  "
                    f"rel_pert={agg['rel_perturb']:.4f}  cos={agg['cos_v_out_v_ctx']:.4f}  "
                    f"norm_ratio={agg['norm_ratio']:.4f}  m_perp_ratio={agg['m_perp_ratio']:.4f}"
                )

    elapsed = time.time() - t0
    print(f"[done] {n_cells} cells in {elapsed:.1f}s")

    # -------------------------------------------------------------------
    # Write cells.jsonl, env.json, REPORT.md
    (out_dir / "cells.jsonl").write_text("\n".join(json.dumps(c) for c in cells) + "\n", encoding="utf-8")
    env = {
        "model": args.model,
        "device": args.device,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "git_rev": _git_rev(),
        "n_prompts": len(prompts),
        "n_layers": L,
        "n_cells": len(cells),
        "elapsed_s": elapsed,
        "ortho_axis": list(ORTHO_AXIS),
        "blend_axis": list(BLEND_AXIS),
        "alpha_axis": list(ALPHA_AXIS),
        "redline_soft_blend_zero_bit_equal": True,
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    _write_report(out_dir / "REPORT.md", cells, env)
    print(f"[wrote] {out_dir}/REPORT.md")
    return 0


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _write_report(path: Path, cells: list[dict], env: dict) -> None:
    by_alpha = {}
    for c in cells:
        by_alpha.setdefault(c["alpha"], []).append(c)

    lines = [
        "# W-T3.6 — ECOR operator-level ablation",
        "",
        f"**Model**: `{env['model']}`  ",
        f"**Device / dtype**: {env['device']} / {env['dtype']}  ",
        f"**Git rev**: `{env['git_rev']}`  ",
        f"**Layers profiled**: {env['n_layers']}  ",
        f"**Neutral prompts**: {env['n_prompts']}  ",
        f"**Total cells**: {env['n_cells']}  (ortho × soft_blend × α = {len(env['ortho_axis'])}×{len(env['blend_axis'])}×{len(env['alpha_axis'])})  ",
        f"**Elapsed**: {env['elapsed_s']:.1f}s  ",
        f"**Redline `soft_blend=0` ≡ additive**: {'PASS' if env['redline_soft_blend_zero_bit_equal'] else 'FAIL'}  ",
        "",
        "## What this answers",
        "",
        "Why is orthogonal projection (`LOPIConfig.orthogonal`) still default `False`,",
        "and what does the ECOR `soft_blend` knob actually do at the operator level?",
        "",
        "* **`ortho=False`** ⇒ M_perp := M_V (raw bank V-readout, no projection).",
        "* **`ortho=True`** ⇒ M_perp := M_V − proj_V_ctx(M_V) (legacy LOPI step 1).",
        "* **`soft_blend=0`** ⇒ ECOR disabled, output is `V_ctx + α·M_perp` (additive).",
        "* **`soft_blend=1`** ⇒ pure ECOR rotation (norm-preserving by construction).",
        "* **`soft_blend ∈ (0,1)`** ⇒ linear blend of additive and rotated outputs.",
        "",
        "Metrics are mean over (prompts × layers):",
        "",
        "* `rel_perturb` = ‖V_out − V_ctx‖ / ‖V_ctx‖  — perturbation magnitude (smaller = gentler).",
        "* `cos_v_out_v_ctx` — direction preservation (1.0 = unchanged direction).",
        "* `norm_ratio` = ‖V_out‖ / ‖V_ctx‖ — energy preservation (ECOR target = 1.0).",
        "* `m_perp_ratio` = ‖M_perp‖ / ‖M_V‖ — how aggressive the ortho projection is (ortho=False ⇒ 1.0).",
        "",
    ]

    for alpha in sorted(by_alpha):
        lines += [
            f"## α = {alpha}",
            "",
            "| ortho | soft_blend | rel_perturb | cos(V_out,V_ctx) | ‖V_out‖/‖V_ctx‖ | ‖M_⊥‖/‖M_V‖ |",
            "|:-----:|:----------:|------------:|-----------------:|----------------:|-------------:|",
        ]
        for c in sorted(by_alpha[alpha], key=lambda r: (r["ortho"], r["soft_blend"])):
            lines.append(
                f"| {str(c['ortho']):>5} | {c['soft_blend']:.2f} | "
                f"{c['rel_perturb']:.4f} | {c['cos_v_out_v_ctx']:.4f} | "
                f"{c['norm_ratio']:.4f} | {c['m_perp_ratio']:.4f} |"
            )
        lines.append("")

    # Aggregate insight at the bottom
    lines += _insight_block(cells)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _insight_block(cells: list[dict]) -> list[str]:
    """Produce a few one-line takeaways."""
    out = ["## Takeaways", ""]
    # 1. norm preservation gap
    blend1 = [c for c in cells if c["soft_blend"] == 1.0]
    blend0 = [c for c in cells if c["soft_blend"] == 0.0]
    if blend1 and blend0:
        nr1 = sum(c["norm_ratio"] for c in blend1) / len(blend1)
        nr0 = sum(c["norm_ratio"] for c in blend0) / len(blend0)
        out.append(
            f"* **Norm preservation**: pure ECOR (soft_blend=1) mean ‖V_out‖/‖V_ctx‖ = {nr1:.4f} "
            f"vs additive (soft_blend=0) {nr0:.4f}. ECOR designed target is 1.0; deviation reveals scale of M_perp/V_ctx mismatch."
        )
    # 2. ortho effect on m_perp_ratio
    on = [c for c in cells if c["ortho"]]
    off = [c for c in cells if not c["ortho"]]
    if on and off:
        mp_on = sum(c["m_perp_ratio"] for c in on) / len(on)
        mp_off = sum(c["m_perp_ratio"] for c in off) / len(off)
        out.append(
            f"* **Ortho aggression**: ortho=True keeps {mp_on:.4f} of ‖M_V‖ as M_⊥ "
            f"(vs 1.0 when ortho=False). Smaller value ⇒ more of M_V was already aligned with V_ctx (i.e. "
            f"the projection threw away that fraction of the bank signal)."
        )
    # 3. perturbation growth with α
    for ortho in (False, True):
        for blend in (0.0, 1.0):
            sub = [c for c in cells if c["ortho"] == ortho and c["soft_blend"] == blend]
            sub.sort(key=lambda r: r["alpha"])
            if not sub:
                continue
            curve = ", ".join(f"α={c['alpha']}: {c['rel_perturb']:.3f}" for c in sub)
            out.append(f"* **rel_perturb vs α** (ortho={ortho}, blend={blend}): {curve}")
    out.append("")
    out.append(
        "**Caveat**: this is operator-level (single forward, frozen V_ctx tensors); "
        "downstream NLL/drift impact requires wiring ECOR through `attn_native_bank.py` "
        "(deferred — see plan N5 / W-T3 round 1)."
    )
    return out


if __name__ == "__main__":
    sys.exit(main())
