"""Library for the ATB validation v1 suite (paper-grade harness).

Six experiments share these primitives:

* :class:`Variant` — config describing one ablation cell (method, alpha,
  bank_key_mode, value-scale mode, optional K/V perturbation, SCAR flag).
* :func:`load_model` — bf16/eager HF loader with deterministic seeding.
* :func:`continuation_logp` — sum-of-token logprob for a target continuation.
* :func:`first_token_rank` — rank of the first target token in the next-token
  distribution conditioned on the prompt.
* :func:`evaluate_prompt` — per-prompt metric pack:
  ``recall@1``, ``margin``, ``target_rank``, ``target_new_logprob``,
  ``target_true_logprob``.
* :func:`unrelated_drift` — symmetric JS divergence over 100 fixed neutral
  windows (matches W.6's drift convention).
* :class:`VariantContext` — context manager that installs an
  ``AttnNativePatcher`` and writes the (optionally perturbed) bank.

This module is intentionally a thin facade over the production
``deltamemory.memory.attn_native_bank`` API; no W.6-specific machinery
(method_winner, gpt2 carve-out, smoke mode) leaks in.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Variant config

@dataclass
class Variant:
    """One ablation cell.

    ``method``: ``"none"`` (no ATB) or ``"anb"`` (AttnNativeBank installed).
    ``alpha``: bank read scaling.
    ``bank_key_mode``: ``"pre_rope"`` (default) or ``"post_rope"``.
    ``value_scale_mode``: passed to AttnNativeBank; ``"none"`` disables V-scale.
    ``bank_perturbation``: applied AFTER bank write; one of:
        ``None``                — no perturbation (default).
        ``"shuffled"``          — fact ↔ target permutation (V tensors permuted).
        ``"random_kv"``         — both K and V replaced by per-layer
                                   RMS-matched Gaussian noise.
        ``"random_K_only"``     — K replaced (V kept correct).
        ``"random_V_only"``     — V replaced (K kept correct).
    ``enable_scar``: SCAR is OFF by default in this suite. Setting True is
        explicit and recorded in the manifest.
    ``description``: free-text label for the manifest.
    """
    name: str
    method: str = "anb"
    alpha: float = 1.0
    bank_key_mode: str = "pre_rope"
    value_scale_mode: str = "auto_rms_cap"
    bank_perturbation: Optional[str] = None
    enable_scar: bool = False
    mhc_shield: bool = False
    mhc_kappa: float = 1.0
    bank_separate_softmax: bool = False
    bank_merge_beta: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Determinism / seeding

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loading

def load_model(name: str, device: str = "cuda", dtype: str = "bf16"):
    """Load HF causal LM in eager attention with deterministic flags.

    For large models on a single GPU we use ``device_map={"": device}`` so the
    weights are placed on-device during the from_pretrained call, avoiding the
    duplicate CPU→GPU copy that ``.to(device)`` otherwise triggers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs).to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, device_map={"": device}, **kwargs
        )
    model.eval()
    return tok, model


# ---------------------------------------------------------------------------
# Token-level metrics

@torch.no_grad()
def continuation_logp(
    model: Any,
    tok: Any,
    prompt: str,
    target: str,
    device: str,
) -> tuple[float, list[int]]:
    """Return (sum of per-token logp, target token ids).

    Sums logp over every target token conditioned on prompt + previous
    target tokens. Joining rule mirrors W.6: insert a single space between
    prompt and target unless prompt ends with whitespace.
    """
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    sep = "" if (prompt.endswith(" ") or not prompt) else " "
    full_ids = tok.encode(prompt + sep + target, add_special_tokens=True)
    if len(full_ids) <= len(prompt_ids):
        return float("nan"), []
    target_ids = full_ids[len(prompt_ids):]
    ids = torch.tensor([full_ids], device=device)
    out = model(input_ids=ids, use_cache=False)
    logp = F.log_softmax(out.logits[0].float(), dim=-1)
    total = 0.0
    for i, tid in enumerate(target_ids):
        total += float(logp[len(prompt_ids) - 1 + i, tid].item())
    return total, target_ids


@torch.no_grad()
def first_token_rank(
    model: Any,
    tok: Any,
    prompt: str,
    target_first_id: int,
    device: str,
) -> tuple[int, float]:
    """Return (rank, logp) of ``target_first_id`` as the next token after prompt.

    Rank is 0-indexed (0 = argmax = recall@1 hit).
    """
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    ids = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0, -1].float()
    logp = F.log_softmax(logits, dim=-1)
    sorted_ids = torch.argsort(logits, descending=True)
    rank = int((sorted_ids == target_first_id).nonzero(as_tuple=False).item())
    return rank, float(logp[target_first_id].item())


# ---------------------------------------------------------------------------
# Per-prompt evaluation

def first_token_id(tok: Any, prompt: str, target: str) -> int:
    """Return the id of the *first* token of ``target`` in the joined sequence."""
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    sep = "" if (prompt.endswith(" ") or not prompt) else " "
    full_ids = tok.encode(prompt + sep + target, add_special_tokens=True)
    return full_ids[len(prompt_ids)]


@torch.no_grad()
def evaluate_prompt(
    model: Any,
    tok: Any,
    prompt: str,
    target_new: str,
    target_true: str,
    device: str,
) -> dict[str, Any]:
    """Compute the per-prompt metric pack."""
    logp_new, ids_new = continuation_logp(model, tok, prompt, target_new, device)
    logp_true, _ = continuation_logp(model, tok, prompt, target_true, device)
    target_new_first = ids_new[0] if ids_new else -1
    rank, _ = first_token_rank(model, tok, prompt, target_new_first, device)
    return {
        "target_new_logprob": logp_new,
        "target_true_logprob": logp_true,
        "margin": logp_new - logp_true,
        "target_rank": rank,
        "recall_at_1": (rank == 0),
    }


# ---------------------------------------------------------------------------
# Unrelated drift (JS) — fixed 100 neutral windows

_NEUTRAL_PROMPTS_CACHE: list[str] | None = None
_KL_LAST = 8


def neutral_prompts(n: int = 100, seed: int = 42) -> list[str]:
    """Return ``n`` deterministic neutral Wikitext-2 sentences.

    Resolution order:
      1. on-disk cache at ``experiments/atb_validation_v1/_lib/neutral_100.json``
      2. HuggingFace ``wikitext-2-raw-v1`` train split sampled with ``seed``
         (saved to the cache for reproducibility)
      3. hardcoded fallback (12 sentences × ceil(n/12))
    """
    global _NEUTRAL_PROMPTS_CACHE
    if _NEUTRAL_PROMPTS_CACHE is not None and len(_NEUTRAL_PROMPTS_CACHE) >= n:
        return _NEUTRAL_PROMPTS_CACHE[:n]
    cache_path = Path(__file__).parent / "neutral_100.json"
    if cache_path.exists():
        prompts = json.loads(cache_path.read_text())[:n]
        _NEUTRAL_PROMPTS_CACHE = prompts
        return prompts
    prompts: list[str] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        rng = random.Random(seed)
        cand = [t.strip() for t in ds["text"] if isinstance(t, str)
                and 40 <= len(t.strip()) <= 200 and t.strip().endswith(".")]
        rng.shuffle(cand)
        prompts = cand[:n]
    except Exception:
        prompts = []
    if len(prompts) < n:
        fallback = [
            "The annual report cited supply chain disruptions as a contributing factor.",
            "After the storm passed the harbor reopened to commercial traffic.",
            "Compilers traditionally separate parsing from semantic analysis for clarity.",
            "Glaciers in the region have retreated by several hundred meters this century.",
            "Under the new statute warrants must specify the items to be searched.",
            "The festival drew musicians from a dozen neighboring countries.",
            "Mineral deposits along the ridge attracted prospectors throughout the 1880s.",
            "Modern aircraft engines rely on extensive on-board diagnostic instrumentation.",
            "Economists disagreed about the effect of the tariff on regional employment.",
            "She filed her dissertation in the spring after a four-year residency.",
            "Tropical depressions often dissipate before reaching the temperate latitudes.",
            "Following the reform act the assembly was reconstituted with new electorates.",
        ]
        while len(prompts) < n:
            prompts.extend(fallback)
        prompts = prompts[:n]
    cache_path.write_text(json.dumps(prompts))
    _NEUTRAL_PROMPTS_CACHE = prompts
    return prompts


@torch.no_grad()
def _last_k_logsoftmax(model, tok, prompt: str, device: str, k: int = _KL_LAST):
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=True)], device=device)
    out = model(input_ids=ids, use_cache=False)
    last = out.logits[0, -k:].float()
    return F.log_softmax(last, dim=-1).detach().cpu()


def _js_nats(logp_a: torch.Tensor, logp_b: torch.Tensor) -> float:
    p = torch.exp(logp_a)
    q = torch.exp(logp_b)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-30))
    return float(0.5 * ((p * (logp_a - log_m)).sum(-1) +
                        (q * (logp_b - log_m)).sum(-1)).mean().item())


def _kl_nats(logp_a: torch.Tensor, logp_b: torch.Tensor) -> float:
    p = torch.exp(logp_a)
    return float((p * (logp_a - logp_b)).sum(-1).mean().item())


def unrelated_drift(
    model,
    tok,
    device: str,
    install_variant_ctx,
    n_prompts: int = 100,
) -> tuple[float, float]:
    """Return ``(js_drift_nats, kl_drift_nats)``.

    ``install_variant_ctx`` is a callable returning a context manager that
    installs the variant for the duration of the patched evaluation. The
    baseline pass runs with no patcher installed.
    """
    prompts = neutral_prompts(n=n_prompts)
    base_lps: list[torch.Tensor] = []
    for p in prompts:
        base_lps.append(_last_k_logsoftmax(model, tok, p, device))
    js_vals: list[float] = []
    kl_vals: list[float] = []
    with install_variant_ctx() as _ctx:
        for p, base_lp in zip(prompts, base_lps):
            inj_lp = _last_k_logsoftmax(model, tok, p, device)
            js_vals.append(_js_nats(base_lp, inj_lp))
            kl_vals.append(_kl_nats(base_lp, inj_lp))
    return (float(sum(js_vals) / max(len(js_vals), 1)),
            float(sum(kl_vals) / max(len(kl_vals), 1)))


# ---------------------------------------------------------------------------
# Variant context — install patcher + bank with optional perturbation

class VariantContext:
    """Context manager: install AttnNativePatcher, write bank, optionally
    perturb K/V, then enter ``patcher.injecting(bank, alpha)`` for read.

    For ``method == "none"`` this is a no-op context.
    """

    def __init__(
        self,
        model,
        tok,
        device: str,
        variant: Variant,
        facts: list[dict],
    ) -> None:
        """``facts`` items must contain: id, subject, write_prompt."""
        from deltamemory.memory.attn_native_bank import (
            AttnNativePatcher,
            fresh_bank,
            write_fact,
        )
        self.model = model
        self.tok = tok
        self.device = device
        self.variant = variant
        self.facts = facts
        self._patcher = None
        self._bank = None
        if variant.method == "none":
            return
        self._patcher = AttnNativePatcher(model)
        self._bank = fresh_bank(model)
        # ``fresh_bank`` doesn't accept knobs; set them directly. The ANB
        # patcher reads ``bank_key_mode`` / ``value_scale_mode`` per-call.
        self._bank.value_scale_mode = variant.value_scale_mode
        self._bank.bank_key_mode = variant.bank_key_mode
        for fact in facts:
            write_fact(
                self._patcher,
                self._bank,
                tok,
                write_prompt=fact["write_prompt"],
                fact_id=str(fact["id"]),
                address=fact.get("subject"),
            )
        self._apply_perturbation()

    def _apply_perturbation(self) -> None:
        bank = self._bank
        if bank is None or self.variant.bank_perturbation is None:
            return
        kind = self.variant.bank_perturbation
        if kind == "shuffled":
            # Permute V tensors so each fact's K maps to a wrong V.
            n = len(bank.fact_ids)
            if n < 2:
                return
            perm = list(range(n))
            rng = random.Random(0xA1B0)
            rng.shuffle(perm)
            for li in range(len(bank.M_V)):
                bank.M_V[li] = bank.M_V[li][perm].contiguous()
        elif kind in ("random_kv", "random_K_only", "random_V_only"):
            generator = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
            for li in range(len(bank.M_K)):
                Kt = bank.M_K[li]
                Vt = bank.M_V[li]
                if kind in ("random_kv", "random_K_only"):
                    rms_k = float(Kt.float().pow(2).mean().sqrt().item()) or 1e-3
                    Knew = torch.randn(Kt.shape, generator=generator,
                                       dtype=torch.float32)
                    Knew = Knew / Knew.float().pow(2).mean().sqrt().clamp_min(1e-8)
                    Knew = Knew * rms_k
                    bank.M_K[li] = Knew.to(Kt.device, dtype=Kt.dtype).contiguous()
                if kind in ("random_kv", "random_V_only"):
                    rms_v = float(Vt.float().pow(2).mean().sqrt().item()) or 1e-3
                    Vnew = torch.randn(Vt.shape, generator=generator,
                                       dtype=torch.float32)
                    Vnew = Vnew / Vnew.float().pow(2).mean().sqrt().clamp_min(1e-8)
                    Vnew = Vnew * rms_v
                    bank.M_V[li] = Vnew.to(Vt.device, dtype=Vt.dtype).contiguous()
        else:
            raise ValueError(f"unknown bank_perturbation: {kind}")

    def __enter__(self):
        if self.variant.method == "none":
            return self
        self._patcher.install()
        self._patcher.bank = self._bank
        self._patcher.alpha = float(self.variant.alpha)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.variant.method == "none":
            return
        try:
            self._patcher.bank = None
            self._patcher.alpha = 0.0
        finally:
            self._patcher.remove()


# ---------------------------------------------------------------------------
# Bootstrap CI (paired)

def bootstrap_paired_diff(
    a: list[float],
    b: list[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (mean_diff, lo, hi) for paired ``b - a``."""
    assert len(a) == len(b) and len(a) > 0
    diffs = [bi - ai for ai, bi in zip(a, b)]
    rng = random.Random(seed)
    mean = sum(diffs) / len(diffs)
    boots = []
    n = len(diffs)
    for _ in range(n_resamples):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(alpha / 2 * n_resamples)]
    hi = boots[int((1 - alpha / 2) * n_resamples) - 1]
    return mean, lo, hi


# ---------------------------------------------------------------------------
# CounterFact loading + filter

def load_counterfact(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_alpha_tokens(s: str, k: int = 3) -> tuple[str, ...]:
    return tuple(w.lower() for w in re.findall(r"[A-Za-z]+", s)[:k])


def filter_cf_for_tokenizer(rows: list[dict], tok: Any) -> tuple[list[dict], int]:
    """W.6's filter: paraphrase_prompts present, target_new/target_true
    distinct in their first 3 alpha tokens, both tokenizable.

    Returns (kept_rows, dropped_count).
    """
    kept = []
    dropped = 0
    for r in rows:
        if not r.get("paraphrase_prompts"):
            dropped += 1
            continue
        tn = (r.get("target_new") or "").strip()
        tt = (r.get("target_true") or "").strip()
        if not tn or not tt or _first_alpha_tokens(tn) == _first_alpha_tokens(tt):
            dropped += 1
            continue
        try:
            tok.encode(tn, add_special_tokens=False)
            tok.encode(tt, add_special_tokens=False)
        except Exception:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


# ---------------------------------------------------------------------------
# SHA1

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
