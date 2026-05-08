"""Multi-fact bank runner for Exp7 and beyond.

Key difference from cf_runner: bank_size > 1.  For each (seed, prompt) the
runner:

1. Samples ``bank_size - 1`` distractor facts from the eligible set
   (deterministic: ``random.Random(seed ^ hash(prompt_id))``)
2. Writes all ``bank_size`` facts into a single ``base_bank`` **once**
3. Clones ``base_bank`` and applies each variant's K/V perturbation
4. Evaluates the target query under the perturbed bank

This makes ``shuffled_bank`` a genuine ``bank_size``-row permutation (not a
no-op at bank_size=1) and reduces bank-write overhead by a factor of
``len(variants)`` compared to calling ``cf_runner`` with bank_size=200.

Loop order: seed → prompt → variant (write once, clone N-variants times).
"""

from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from . import (
    Variant,
    evaluate_prompt,
    filter_cf_for_tokenizer,
    first_token_id,
    load_counterfact,
    load_model,
    neutral_prompts,
    seed_everything,
    _js_nats,
    _kl_nats,
    _last_k_logsoftmax,
)


# ---------------------------------------------------------------------------
# Bank cloning

def clone_bank(bank):
    """Return a new AttnNativeBank with cloned M_K / M_V tensors.

    Only copies tensor data and scalar config; LOPI state is freshly
    initialised (disabled by default, so this is functionally identical).
    """
    from deltamemory.memory.attn_native_bank import AttnNativeBank

    new = AttnNativeBank(
        num_layers=bank.num_layers,
        num_kv_heads=bank.num_kv_heads,
        head_dim=bank.head_dim,
        head_dims=list(bank.head_dims),
        num_kv_heads_per_layer=list(bank.num_kv_heads_per_layer),
        device=bank.device,
        dtype=bank.dtype,
        M_K=[k.clone() for k in bank.M_K],
        M_V=[v.clone() for v in bank.M_V],
    )
    new.fact_ids = list(bank.fact_ids)
    new.address_strs = list(getattr(bank, "address_strs", []))
    new.bank_key_mode = bank.bank_key_mode
    new.value_scale_mode = bank.value_scale_mode
    new.bank_temperature = getattr(bank, "bank_temperature", 1.0)
    return new


# ---------------------------------------------------------------------------
# Perturbation (mirrors _lib/__init__.py VariantContext._apply_perturbation)

def apply_perturbation(bank, kind: Optional[str], seed: int) -> None:
    """Apply in-place K/V perturbation to a cloned bank."""
    if kind is None:
        return
    if kind == "shuffled":
        n = len(bank.fact_ids)
        if n < 2:
            return
        perm = list(range(n))
        rng = random.Random(0xA1B0)
        rng.shuffle(perm)
        for li in range(len(bank.M_V)):
            bank.M_V[li] = bank.M_V[li][perm].contiguous()
        return
    if kind in ("random_kv", "random_K_only", "random_V_only"):
        # Seed incorporates the experiment seed for inter-seed reproducibility.
        generator = torch.Generator(device="cpu").manual_seed(0xC0FFEE ^ seed)
        for li in range(len(bank.M_K)):
            Kt = bank.M_K[li]
            Vt = bank.M_V[li]
            if kind in ("random_kv", "random_K_only"):
                rms_k = float(Kt.float().pow(2).mean().sqrt().item()) or 1e-3
                Knew = torch.randn(Kt.shape, generator=generator, dtype=torch.float32)
                Knew = Knew / Knew.float().pow(2).mean().sqrt().clamp_min(1e-8)
                Knew = Knew * rms_k
                bank.M_K[li] = Knew.to(Kt.device, dtype=Kt.dtype).contiguous()
            if kind in ("random_kv", "random_V_only"):
                rms_v = float(Vt.float().pow(2).mean().sqrt().item()) or 1e-3
                Vnew = torch.randn(Vt.shape, generator=generator, dtype=torch.float32)
                Vnew = Vnew / Vnew.float().pow(2).mean().sqrt().clamp_min(1e-8)
                Vnew = Vnew * rms_v
                bank.M_V[li] = Vnew.to(Vt.device, dtype=Vt.dtype).contiguous()
        return
    raise ValueError(f"unknown bank_perturbation: {kind!r}")


# ---------------------------------------------------------------------------
# Build base bank with multiple facts

def _build_base_bank(patcher, model, tok, facts: list[dict],
                     bank_key_mode: str, value_scale_mode: str):
    """Write ``facts`` into a fresh bank and return it.

    Each entry in ``facts`` must contain: id, subject, write_prompt.
    """
    from deltamemory.memory.attn_native_bank import fresh_bank, write_fact

    bank = fresh_bank(model)
    bank.bank_key_mode = bank_key_mode
    bank.value_scale_mode = value_scale_mode
    for fact in facts:
        write_fact(
            patcher, bank, tok,
            write_prompt=fact["write_prompt"],
            fact_id=str(fact["id"]),
            address=fact.get("subject", ""),
        )
    return bank


# ---------------------------------------------------------------------------
# Main run function

def run(
    *,
    model_name: str,
    dtype: str,
    device: str,
    counterfact_path: Path,
    variants: list[Variant],
    seeds: list[int],
    out_dir: Path,
    bank_size: int = 200,
    n_prompts: Optional[int] = None,
    n_neutral: int = 100,
) -> Path:
    """Run all (seed, prompt, variant) cells with bank_size > 1.

    Returns the results.jsonl path.

    Loop order: seed → prompt → variant.
    Bank is written once per (seed, prompt); variants share the base bank
    (each gets a clone with its own perturbation applied).
    """
    from deltamemory.memory.attn_native_bank import AttnNativePatcher

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    log_path = out_dir / "run.log"
    log_fh = log_path.open("a")

    def _log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    _log(f"loading {model_name} dtype={dtype} device={device}")
    tok, model = load_model(model_name, device=device, dtype=dtype)

    rows = load_counterfact(counterfact_path)
    _log(f"counterfact raw rows: {len(rows)}")
    kept, dropped = filter_cf_for_tokenizer(rows, tok)
    _log(f"after filter: kept={len(kept)} dropped={dropped}")

    # Build eligible set and facts lookup.
    facts_by_pid: dict[str, dict] = {}
    for r in kept:
        subject = r["subject"]
        pt = r.get("prompt", "")
        phrase = pt[2:].strip().rstrip(",") if pt.startswith("{}") else None
        if phrase is None:
            continue
        wp = f"Fact: {subject} {phrase} {r['target_new']}."
        facts_by_pid[r["id"]] = {
            "id": r["id"],
            "subject": subject,
            "write_prompt": wp,
        }
    eligible = [r for r in kept if r["id"] in facts_by_pid]
    _log(f"eligible prompts (have relation phrase): {len(eligible)}")

    if n_prompts is not None:
        eligible = eligible[:n_prompts]
        _log(f"truncated to n={len(eligible)} (smoke / dry-run)")

    eligible_ids = [r["id"] for r in eligible]
    patcher = AttnNativePatcher(model)

    # Pre-compute neutral baseline logprobs once (shared across all variants/seeds).
    _log("computing neutral baseline logprobs (100 prompts)...")
    neutral = neutral_prompts(n=n_neutral)
    base_lps: list[torch.Tensor] = []
    for p in neutral:
        base_lps.append(_last_k_logsoftmax(model, tok, p, device))
    _log("baseline done.")

    # Per-variant, per-seed drift: use first 8 eligible facts.
    drift_js: dict[tuple[str, int], float] = {}
    drift_kl: dict[tuple[str, int], float] = {}
    for variant in variants:
        if variant.method == "none":
            for seed in seeds:
                drift_js[(variant.name, seed)] = 0.0
                drift_kl[(variant.name, seed)] = 0.0
            continue
        for seed in seeds:
            seed_everything(seed)
            drift_facts = [facts_by_pid[r["id"]] for r in eligible[:8]]
            drift_bank = _build_base_bank(
                patcher, model, tok, drift_facts,
                bank_key_mode=variant.bank_key_mode,
                value_scale_mode=variant.value_scale_mode,
            )
            apply_perturbation(drift_bank, variant.bank_perturbation, seed)
            js_vals: list[float] = []
            kl_vals: list[float] = []
            patcher.install()
            patcher.bank = drift_bank
            patcher.alpha = float(variant.alpha)
            try:
                for p, blp in zip(neutral, base_lps):
                    ilp = _last_k_logsoftmax(model, tok, p, device)
                    js_vals.append(_js_nats(blp, ilp))
                    kl_vals.append(_kl_nats(blp, ilp))
            finally:
                patcher.bank = None
                patcher.alpha = 0.0
                patcher.remove()
            js = float(sum(js_vals) / max(len(js_vals), 1))
            kl = float(sum(kl_vals) / max(len(kl_vals), 1))
            drift_js[(variant.name, seed)] = js
            drift_kl[(variant.name, seed)] = kl
            _log(f"drift variant={variant.name} seed={seed}: js={js:.4f} kl={kl:.4f}")

    # Check if all anb variants share the same bank_key_mode + value_scale_mode.
    # If so we can write the base_bank ONCE per (seed, prompt) and clone for
    # each variant — the efficiency win of multi_bank_runner.
    anb_variants = [v for v in variants if v.method == "anb"]
    _shared_bkm = (len({v.bank_key_mode for v in anb_variants}) <= 1 and
                   len({v.value_scale_mode for v in anb_variants}) <= 1)
    if anb_variants and _shared_bkm:
        _ref_bank_key_mode = anb_variants[0].bank_key_mode
        _ref_value_scale_mode = anb_variants[0].value_scale_mode
        _log(f"Shared bank config: bank_key_mode={_ref_bank_key_mode} "
             f"value_scale_mode={_ref_value_scale_mode} → write base_bank ONCE per prompt.")
    else:
        _ref_bank_key_mode = _ref_value_scale_mode = None
        _log("Mixed bank configs — writing base_bank per variant.")

    # Main loop: seed → prompt → variant.
    n_cells = 0
    with open(results_path, "a") as out_f:
        for seed in seeds:
            seed_everything(seed)
            _log(f"=== seed={seed} ===")
            for prow in eligible:
                pid = prow["id"]
                target_new = prow["target_new"]
                target_true = prow["target_true"]
                query = (prow["prompt"].format(prow["subject"])
                         if "{}" in prow.get("prompt", "") else prow.get("prompt", ""))

                # Sample distractors (deterministic per seed+pid).
                other_ids = [eid for eid in eligible_ids if eid != pid]
                rng = random.Random(seed ^ (hash(pid) & 0xFFFF_FFFF))
                n_dist = min(bank_size - 1, len(other_ids))
                distractor_ids = rng.sample(other_ids, n_dist)
                all_fact_ids = [pid] + distractor_ids
                all_facts = [facts_by_pid[fid] for fid in all_fact_ids]
                actual_bank_size = len(all_facts)

                # Write shared base_bank once when all anb variants agree.
                shared_base: Any = None
                if _shared_bkm and _ref_bank_key_mode is not None:
                    try:
                        shared_base = _build_base_bank(
                            patcher, model, tok, all_facts,
                            bank_key_mode=_ref_bank_key_mode,
                            value_scale_mode=_ref_value_scale_mode,
                        )
                    except Exception as exc:
                        _log(f"  ERROR building base_bank pid={pid} seed={seed}: {exc}")
                        shared_base = None

                for variant in variants:
                    if variant.method == "none":
                        # No bank: evaluate baseline.
                        try:
                            mp = evaluate_prompt(model, tok, query,
                                                 target_new, target_true, device)
                        except Exception as exc:
                            _log(f"  ERROR pid={pid} variant={variant.name} "
                                 f"seed={seed}: {exc}")
                            continue
                    else:
                        try:
                            # Use shared base_bank (clone) or build per-variant.
                            if shared_base is not None:
                                bank = clone_bank(shared_base)
                            else:
                                bank = _build_base_bank(
                                    patcher, model, tok, all_facts,
                                    bank_key_mode=variant.bank_key_mode,
                                    value_scale_mode=variant.value_scale_mode,
                                )
                            apply_perturbation(bank, variant.bank_perturbation, seed)
                            # Inject and evaluate.
                            patcher.install()
                            patcher.bank = bank
                            patcher.alpha = float(variant.alpha)
                            try:
                                mp = evaluate_prompt(model, tok, query,
                                                     target_new, target_true, device)
                            finally:
                                patcher.bank = None
                                patcher.alpha = 0.0
                                patcher.remove()
                            del bank
                        except Exception as exc:
                            _log(f"  ERROR pid={pid} variant={variant.name} "
                                 f"seed={seed}: {exc}")
                            continue

                    row: dict[str, Any] = {
                        "experiment": out_dir.name,
                        "variant": variant.name,
                        "method": variant.method,
                        "alpha": variant.alpha,
                        "bank_key_mode": variant.bank_key_mode,
                        "bank_perturbation": variant.bank_perturbation,
                        "value_scale_mode": variant.value_scale_mode,
                        "enable_scar": variant.enable_scar,
                        "seed": seed,
                        "prompt_id": pid,
                        "subject": prow["subject"],
                        "relation": prow.get("relation"),
                        "target_new": target_new,
                        "target_true": target_true,
                        "bank_size": actual_bank_size,
                        "js_drift": drift_js.get((variant.name, seed), float("nan")),
                        "kl_drift": drift_kl.get((variant.name, seed), float("nan")),
                        **mp,
                    }
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                    n_cells += 1
                    if n_cells % 50 == 0:
                        _log(f"  ...{n_cells} cells")

                if shared_base is not None:
                    del shared_base

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _log(f"done: {n_cells} cells -> {results_path}")
    log_fh.close()
    return results_path
