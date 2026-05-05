"""Phase S — Auto-calibration profiler for Universal LOPI (U-LOPI).

Each architecture (Gemma / Qwen / Llama / GLM / GPT-2) has its own
residual-stream scale: ``mu_arch`` and the per-layer (mu_base, sigma_base)
distributions differ by 10-100x.  v3.4 LOPI hard-coded ``norm_base = 10.0``
calibrated to Gemma-4-E2B and silently degraded everywhere else.

This module replaces the global constant with a one-shot cold-start
profile over a small set of neutral prompts.  The forward is
``output_hidden_states=True`` and we collect ``||hidden_states[layer]||_2``
statistics along (B, T).  Both the per-layer mean / std arrays and the
auto-anchored ``mu_arch = argmax_l sigma_base(l)`` (with ties broken
toward the lower index, see ``D-S-3`` in the plan) are persisted alongside
the bank so reloads inherit the calibration.

The profile is **forward-only**.  No nn.Parameter is introduced and the
LLM weights are bit-equal pre/post (verified by
``test_lopi_profiler.py::test_profile_does_not_mutate_weights``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


DEFAULT_PROFILE_CORPUS: tuple[str, ...] = (
    "hello",
    "the cat sat on the mat",
    "1 2 3 4 5",
    "good morning",
    "I went to the store yesterday",
    "你好",
    "今天 天气 不错",
    "一二三四五六七八",
    "请 你 帮 我 一个 忙",
    "the quick brown fox jumps over the lazy dog",
)


@dataclass
class LOPIProfile:
    """Per-architecture residual-stream calibration."""

    model_name: str
    num_layers: int
    mu_base: list[float]      # length L
    sigma_base: list[float]   # length L
    mu_arch: int
    profile_corpus_sha: str
    n_prompts: int
    dtype: str
    eta_sigma: float = 1.0    # auto: 0.7 if cv > 0.5 else 1.0  (D-S-6)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LOPIProfile":
        return LOPIProfile(
            model_name=str(d["model_name"]),
            num_layers=int(d["num_layers"]),
            mu_base=[float(x) for x in d["mu_base"]],
            sigma_base=[float(x) for x in d["sigma_base"]],
            mu_arch=int(d["mu_arch"]),
            profile_corpus_sha=str(d["profile_corpus_sha"]),
            n_prompts=int(d["n_prompts"]),
            dtype=str(d["dtype"]),
            eta_sigma=float(d.get("eta_sigma", 1.0)),
        )


def default_profile_corpus() -> list[str]:
    return list(DEFAULT_PROFILE_CORPUS)


def _corpus_sha(prompts: Sequence[str]) -> str:
    blob = json.dumps(list(prompts), ensure_ascii=False, sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _layer_norm_stats(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[float], list[float]]:
    """Return (mu_base, sigma_base) computed over (B, T) for every layer.

    ``hidden_states`` is the HF ``output_hidden_states`` tuple: ``len = L+1``
    where index 0 is the embedding output and 1..L are after each transformer
    block.  We profile **block outputs** (1..L) because that is the residual
    state consumed by the next layer's attention -- this matches v3.4
    LOPI's ``avg_prev_residual_norm`` semantics.  When ``attention_mask`` is
    provided, padding positions are excluded from the (B, T) population.
    """
    mus: list[float] = []
    sigmas: list[float] = []
    valid_mask = None
    if attention_mask is not None:
        valid_mask = attention_mask.to(dtype=torch.bool, device=hidden_states[0].device)
    for h in hidden_states[1:]:
        # h: (B, T, D)
        norms = torch.linalg.vector_norm(h.float(), ord=2, dim=-1)  # (B, T)
        if valid_mask is not None:
            if tuple(valid_mask.shape) != tuple(norms.shape):
                raise ValueError(
                    "attention_mask shape must match hidden state token shape: "
                    f"mask={tuple(valid_mask.shape)} norms={tuple(norms.shape)}"
                )
            vals = norms[valid_mask]
            if vals.numel() == 0:
                raise ValueError("profile_residuals: attention_mask has no valid tokens")
        else:
            vals = norms.reshape(-1)
        mus.append(float(vals.mean().item()))
        sigmas.append(float(vals.std(unbiased=False).item()))
    return mus, sigmas


def _argmax_low_tiebreak(values: Sequence[float]) -> int:
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def _coefficient_of_variation(sigmas: Sequence[float]) -> float:
    """CV across layers of sigma_base (D-S-6 σ-shrink trigger)."""
    if not sigmas:
        return 0.0
    mean = sum(sigmas) / len(sigmas)
    if mean <= 0:
        return 0.0
    var = sum((s - mean) ** 2 for s in sigmas) / len(sigmas)
    return (var ** 0.5) / mean


def profile_residuals(
    model: Any,
    tokenizer: Any,
    *,
    prompts: Iterable[str] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    max_length: int = 32,
) -> LOPIProfile:
    """One-shot profile of the residual stream.

    Parameters
    ----------
    model
        A HF causal-LM.  Must support ``output_hidden_states=True``.
    tokenizer
        Matching tokenizer.  Padding is set to right-pad with ``pad_token``.
    prompts
        Override for the default neutral corpus (defaults to N=10 mixed
        zh/en short strings).
    device
        Device for the profiling forward.
    dtype
        Optional cast before forward (default: leave model dtype).
    max_length
        Tokenization cap; profile prompts are short by design.

    Returns
    -------
    LOPIProfile
        Per-layer mu/sigma + auto-anchored mu_arch.

    Notes
    -----
    The function does not mutate ``model.training`` permanently and never
    invokes ``backward`` -- weights are bit-equal pre/post.
    """
    prompts_list = list(prompts) if prompts is not None else default_profile_corpus()
    if not prompts_list:
        raise ValueError("profile_residuals: at least one prompt required")

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
        # Mirror HF idiom; ``tokenizer.pad_token = tokenizer.eos_token`` if
        # not already set so batched encode does not error.
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token  # type: ignore[attr-defined]

    enc = tokenizer(
        prompts_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        if was_training:
            model.train()

    hidden = out.hidden_states  # tuple length L+1
    if hidden is None:
        raise RuntimeError(
            "profile_residuals: model did not return hidden_states; "
            "check that output_hidden_states=True is supported."
        )

    mu_base, sigma_base = _layer_norm_stats(hidden, attention_mask=attention_mask)
    mu_arch = _argmax_low_tiebreak(sigma_base)
    cv = _coefficient_of_variation(sigma_base)
    eta_sigma = 0.7 if cv > 0.5 else 1.0

    # dtype best-effort: use first parameter
    try:
        param_dtype = next(model.parameters()).dtype
        dtype_str = str(param_dtype).replace("torch.", "")
    except StopIteration:
        dtype_str = "unknown"

    return LOPIProfile(
        model_name=str(getattr(model, "name_or_path", "<unknown>")),
        num_layers=len(mu_base),
        mu_base=mu_base,
        sigma_base=sigma_base,
        mu_arch=mu_arch,
        profile_corpus_sha=_corpus_sha(prompts_list),
        n_prompts=len(prompts_list),
        dtype=dtype_str,
        eta_sigma=eta_sigma,
    )


def save_profile(profile: LOPIProfile, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(profile.asdict(), fh, indent=2, sort_keys=True, ensure_ascii=False)


def load_profile(path: str | Path) -> LOPIProfile:
    with Path(path).open("r", encoding="utf-8") as fh:
        return LOPIProfile.from_dict(json.load(fh))


__all__ = [
    "LOPIProfile",
    "profile_residuals",
    "default_profile_corpus",
    "save_profile",
    "load_profile",
]
