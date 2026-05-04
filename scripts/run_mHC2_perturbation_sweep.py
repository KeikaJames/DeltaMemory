"""Phase mHC2 — pure-architecture α-perturbation NLL sweep.

This script implements Phase mHC2 of the preregistration
(``docs/preregistration/mHC_alpha_safe_v1.md``):

    For each of the three architectures (residual GPT-2 / unconstrained-HC
    GPT-2 / mHC GPT-2), inject random Gaussian noise ε into the V tensor of
    every attention layer with ``‖ε‖_F = α · ‖V‖_F``, then evaluate
    Wikitext-2 validation NLL across the preregistered α grid and seed grid.

This isolates the *architecture-only* spectral-shield question from the bank
side: no Mneme bank, no learned K-projector, no factual content.  If
the doubly-stochastic constraint really keeps mHC's hidden-state norm bounded
under bounded V perturbations, mHC's NLL curve must stay flat where Residual
GPT-2's blows up.

Output
------
- ``reports/cleanroom/mHC2_perturbation/<arch>/seed<s>/alpha<a>.json``
- ``reports/cleanroom/mHC2_perturbation/<arch>/per_layer_norms.json``
  (only for the ``layernorm_probe`` mode used by H5)
- ``reports/cleanroom/mHC2_perturbation/AGGREGATE.json``

Decision gate (preregistered, mHC2.4)
-------------------------------------
At least 2 of {H1, H2, H5} must show the predicted direction
(uncorrected p < 0.05) on this script's output before Phase mHC3 launches.
This gate exists to conserve compute if the architecture-only spectral
shield does not manifest; null result is reported honestly per §6 of the
preregistration.

Usage (Mac MPS)
---------------
.. code-block:: bash

    .venv-mac/bin/python scripts/run_mHC2_perturbation_sweep.py \\
        --base-model gpt2 \\
        --device mps \\
        --num-segments 32 \\
        --segment-length 512 \\
        --seeds 0 1 2 3 4 \\
        --alphas 0 0.05 0.1 0.5 1 1.5 2 5 10 \\
        --out-dir reports/cleanroom/mHC2_perturbation/

The ``layernorm_probe`` flag additionally records ``‖x_ℓ‖_F`` per layer for
H5's paired visualisation at α=1.5.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel

# Allow `python scripts/...` to import the deltamemory package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deltamemory.baselines.mhc_gpt2 import convert_gpt2_lm_head_model  # noqa: E402

# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


def build_residual(base_model_name: str, device: str, dtype: torch.dtype) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(base_model_name).to(device=device, dtype=dtype).eval()


def build_mhc(base_model_name: str, device: str, dtype: torch.dtype, *, use_sinkhorn: bool):
    base = GPT2LMHeadModel.from_pretrained(base_model_name)
    mhc = convert_gpt2_lm_head_model(
        base,
        mhc_n=4,
        mhc_tmax=20,
        equivalence_init=True,
        offdiag_bias=-50.0,
        use_sinkhorn=use_sinkhorn,
    )
    return mhc.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------------------
# V-perturbation hook
# ---------------------------------------------------------------------------


@dataclass
class _PerturbState:
    alpha: float
    generator: torch.Generator
    v_norms: list[float]


@dataclass
class _NormProbeState:
    """Per-layer hidden-state Frobenius norm probe (preregistration H5).

    Records ``‖x_ℓ‖_F`` at the *output* of every transformer block, plus
    ``‖x_0‖_F`` at the embedding output. We attach forward hooks to the
    model's block list (``transformer.h``) and to the embedding layer
    (``transformer.wte``).
    """

    norms: list[float]  # one entry per layer, in order of forward pass
    enabled: bool = True


def _gpt2_attention_modules(model: nn.Module) -> List[nn.Module]:
    """Return the GPT2Attention modules whose V output we perturb.

    Works for both residual GPT-2 and the vendored mHC GPT-2 (the attention
    module class is the same; only its surrounding routing changes).
    """
    out: list[nn.Module] = []
    for m in model.modules():
        cls = type(m).__name__
        if cls == "GPT2Attention":
            out.append(m)
    return out


def _attach_perturb_hooks(
    attn_modules: List[nn.Module], state: _PerturbState
) -> List[torch.utils.hooks.RemovableHandle]:
    """Wrap GPT2Attention's ``c_attn`` linear so V is perturbed in place.

    GPT-2 fuses Q/K/V into a single ``c_attn`` linear; the output has shape
    ``(B, T, 3*C)`` and V is the last C-slice. We add Gaussian noise of
    Frobenius norm ``α·‖V‖_F`` only to that slice. This matches the
    preregistration § Phase mHC2.1 ("random V noise of norm α·‖V‖").
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for attn in attn_modules:
        c_attn = getattr(attn, "c_attn", None)
        if c_attn is None:
            continue

        def hook(module, inputs, output, _state=state):
            # output: (B, T, 3C). Split into Q, K, V along last dim.
            split = output.shape[-1] // 3
            q, k, v = output[..., :split], output[..., split : 2 * split], output[..., 2 * split :]
            v_norm = torch.linalg.vector_norm(v).item()
            _state.v_norms.append(v_norm)
            if _state.alpha == 0.0:
                return output
            noise = torch.empty(v.shape, device="cpu", dtype=torch.float32)
            noise.normal_(generator=_state.generator)
            noise = noise.to(device=v.device)
            noise_norm = torch.linalg.vector_norm(noise)
            if noise_norm.item() == 0:
                return output
            scale = (_state.alpha * v_norm) / noise_norm.item()
            v = v + (noise * scale).to(dtype=v.dtype)
            return torch.cat([q, k, v], dim=-1)

        handles.append(c_attn.register_forward_hook(hook))
    return handles


def _detach(handles):
    for h in handles:
        h.remove()


def _attach_norm_probes(model: nn.Module, probe: _NormProbeState) -> List[torch.utils.hooks.RemovableHandle]:
    """Attach hidden-state Frobenius norm probes for H5 visualisation.

    For residual GPT-2 we hook ``transformer.h[i]`` outputs (which are
    ``(B, T, C)``).  For mHC GPT-2 the block output is a *stream tensor*
    ``(B, T, n, C)``; we collapse the stream axis with the (already-trained
    or equivalence-init) ``mhc_readout_logits`` softmax to recover the
    effective single-stream hidden state, which is what gets fed to ``ln_f``.

    A single ``probe.norms`` list is reset before each forward pass by the
    caller; hooks append in forward order so the i-th entry corresponds
    to layer index i (with index 0 = embedding output).
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []

    transformer = getattr(model, "transformer", model)

    # Embedding output (||x_0||_F).
    wte = getattr(transformer, "wte", None)
    if wte is not None:
        def _emb_hook(_m, _i, out, _probe=probe):
            if _probe.enabled:
                _probe.norms.append(float(torch.linalg.vector_norm(out).item()))
        handles.append(wte.register_forward_hook(_emb_hook))

    # Per-block outputs.
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        return handles

    readout_logits = getattr(transformer, "mhc_readout_logits", None)

    for block in blocks:
        def _block_hook(_m, _i, out, _probe=probe, _readout=readout_logits):
            if not _probe.enabled:
                return
            # HF GPT2Block returns a tuple (hidden_states, ...) where
            # hidden_states is (B,T,C). MhcGPT2Block returns the stream tensor
            # (B,T,n,C) directly (no tuple).
            hs = out[0] if isinstance(out, tuple) else out
            if hs.dim() == 4 and _readout is not None:
                w = torch.softmax(_readout, dim=0).to(dtype=hs.dtype)
                hs = torch.einsum("btnc,n->btc", hs, w)
            _probe.norms.append(float(torch.linalg.vector_norm(hs).item()))
        handles.append(block.register_forward_hook(_block_hook))

    return handles


# ---------------------------------------------------------------------------
# NLL evaluation
# ---------------------------------------------------------------------------


def _wikitext2_segments(tokenizer, num_segments: int, segment_length: int, seed: int) -> torch.Tensor:
    """Sha-locked Wikitext-2 validation segmentation.

    The seed only controls the *segment selection*; the underlying validation
    text is constant (HuggingFace ``Salesforce/wikitext`` ``wikitext-2-raw-v1``).
    We deterministically pick ``num_segments`` non-overlapping segments of
    length ``segment_length`` from the contiguous tokenization.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total = ids.shape[0]
    needed = num_segments * segment_length
    if total < needed:
        raise RuntimeError(f"Wikitext-2 val too short ({total}) for {num_segments}x{segment_length}")
    # Deterministic, seed-locked, strictly non-overlapping selection.
    # Sample from equally-spaced candidate start positions to guarantee
    # every pair of segments is separated by at least ``segment_length``.
    g = torch.Generator(device="cpu").manual_seed(seed)
    candidate_starts = list(range(0, total - segment_length + 1, segment_length))
    if len(candidate_starts) < num_segments:
        raise RuntimeError(
            f"Wikitext-2 val too short ({total}) for {num_segments}x{segment_length}"
        )
    perm = torch.randperm(len(candidate_starts), generator=g)[:num_segments]
    starts = sorted(candidate_starts[i] for i in perm.tolist())
    segs = torch.stack([ids[s : s + segment_length] for s in starts])
    return segs  # (num_segments, segment_length)


@torch.no_grad()
def evaluate_nll(
    model: nn.Module,
    segments: torch.Tensor,
    device: str,
    *,
    norm_probe: _NormProbeState | None = None,
) -> tuple[float, list[list[float]]]:
    """Return mean per-token NLL (nats) and (optionally) per-segment per-layer norms."""
    model.eval()
    losses: list[float] = []
    per_seg_norms: list[list[float]] = []
    for seg in segments:
        ids = seg.to(device).unsqueeze(0)
        if norm_probe is not None:
            norm_probe.norms = []
        out = model(input_ids=ids, labels=ids)
        losses.append(float(out.loss.item()))
        if norm_probe is not None:
            per_seg_norms.append(list(norm_probe.norms))
    return float(sum(losses) / len(losses)), per_seg_norms


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_one_condition(
    *,
    arch_name: str,
    arch_factory,
    segments: torch.Tensor,
    alphas: List[float],
    probe_alphas: set[float],
    seed: int,
    device: str,
    dtype: torch.dtype,
    out_dir: Path,
) -> dict:
    model = arch_factory()
    attn_modules = _gpt2_attention_modules(model)
    rows: list[dict] = []
    for alpha in alphas:
        gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + int(alpha * 1000))
        state = _PerturbState(alpha=float(alpha), generator=gen, v_norms=[])
        handles = _attach_perturb_hooks(attn_modules, state)
        probe = None
        probe_handles: list = []
        if float(alpha) in probe_alphas:
            probe = _NormProbeState(norms=[])
            probe_handles = _attach_norm_probes(model, probe)
        try:
            t0 = time.time()
            nll, per_seg_norms = evaluate_nll(model, segments, device, norm_probe=probe)
            dt = time.time() - t0
        finally:
            _detach(handles)
            _detach(probe_handles)
        row = dict(
            arch=arch_name,
            alpha=float(alpha),
            seed=int(seed),
            nll=float(nll),
            n_attn_layers=len(attn_modules),
            mean_v_norm=float(sum(state.v_norms) / max(1, len(state.v_norms))),
            wallclock_s=float(dt),
        )
        if per_seg_norms:
            row["per_layer_norms_per_seg"] = per_seg_norms
        rows.append(row)
        print(f"[{arch_name} seed={seed} α={alpha:>5}] NLL={nll:.4f}  "
              f"meanV={row['mean_v_norm']:.3f}  ({dt:.1f}s)")
        # Per-condition file (sha-traceable).
        cond_dir = out_dir / arch_name / f"seed{seed}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        with open(cond_dir / f"alpha{alpha}.json", "w") as f:
            json.dump(row, f, indent=2)
    return dict(arch=arch_name, seed=seed, rows=rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="gpt2", help="HF id of the residual base (default gpt2 = small)")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--num-segments", type=int, default=32)
    p.add_argument("--segment-length", type=int, default=512)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        help="α grid (preregistered)",
    )
    p.add_argument(
        "--archs",
        nargs="+",
        default=["residual", "hc", "mhc"],
        choices=["residual", "hc", "mhc"],
    )
    p.add_argument(
        "--probe-alphas",
        type=float,
        nargs="*",
        default=[1.5],
        help="α values at which to record per-layer hidden-state norms (H5 paired vis).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("reports/cleanroom/mHC2_perturbation"))
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Use the FIRST seed's segmentation as the canonical evaluation set
    # (seed-locked so seeds 1..4 reuse the SAME prompts; seeds vary only the
    # noise stream). This mirrors the preregistration pairing principle.
    segments = _wikitext2_segments(
        tokenizer, args.num_segments, args.segment_length, seed=0
    )

    factories = {
        "residual": lambda: build_residual(args.base_model, device, dtype),
        "hc": lambda: build_mhc(args.base_model, device, dtype, use_sinkhorn=False),
        "mhc": lambda: build_mhc(args.base_model, device, dtype, use_sinkhorn=True),
    }

    aggregate: list[dict] = []
    probe_alphas = set(float(a) for a in args.probe_alphas)
    for arch in args.archs:
        for seed in args.seeds:
            res = run_one_condition(
                arch_name=arch,
                arch_factory=factories[arch],
                segments=segments,
                alphas=args.alphas,
                probe_alphas=probe_alphas,
                seed=seed,
                device=device,
                dtype=dtype,
                out_dir=args.out_dir,
            )
            aggregate.append(res)

    with open(args.out_dir / "AGGREGATE.json", "w") as f:
        json.dump(
            dict(
                base_model=args.base_model,
                device=device,
                dtype=args.dtype,
                num_segments=args.num_segments,
                segment_length=args.segment_length,
                seeds=args.seeds,
                alphas=args.alphas,
                archs=args.archs,
                runs=aggregate,
            ),
            f,
            indent=2,
        )
    print(f"\n[DONE] aggregate written to {args.out_dir / 'AGGREGATE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
