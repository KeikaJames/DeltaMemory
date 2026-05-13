"""Bit-equality probe for AttnNativeBank α=0.

For each (model, prompt) pair compare the next-token logits of:
  * baseline forward (no patcher installed)
  * patcher.injecting(bank, alpha=0.0) on a NON-EMPTY bank

Records ``torch.equal`` (strict) and ``(L0 - L1).abs().max()``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from . import (
    Variant,
    VariantContext,
    load_counterfact,
    load_model,
    neutral_prompts,
    seed_everything,
)
from .cf_runner import build_write_prompt


@torch.no_grad()
def _last_logits(model, tok, prompt: str, device: str) -> torch.Tensor:
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=True)],
                       device=device)
    out = model(input_ids=ids, use_cache=False)
    return out.logits[0, -1].detach().to(torch.float32).cpu()


def run(
    *,
    model_name: str,
    dtype: str,
    device: str,
    counterfact_path: Path,
    out_dir: Path,
    n_prompts: int = 100,
    n_facts: int = 8,
    seed: int = 0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    log_path = out_dir / "run.log"
    log = log_path.open("a")

    def _log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    seed_everything(seed)
    _log(f"loading {model_name}")
    tok, model = load_model(model_name, device=device, dtype=dtype)

    cf = load_counterfact(counterfact_path)
    facts: list[dict] = []
    for r in cf:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        facts.append({"id": r["id"], "subject": r["subject"],
                      "write_prompt": wp})
        if len(facts) >= n_facts:
            break
    _log(f"facts loaded: {len(facts)}")

    variant = Variant(name="anb_alpha0", method="anb", alpha=0.0,
                      bank_key_mode="pre_rope",
                      value_scale_mode="auto_rms_cap")

    prompts = neutral_prompts(n=n_prompts)
    _log(f"neutral prompts: {len(prompts)}")
    n_pass = 0
    max_abs = 0.0
    with open(results_path, "a") as out_f:
        # Build VariantContext once so the bank is shared across all prompts.
        ctx = VariantContext(model, tok, device, variant, facts)
        for i, p in enumerate(prompts):
            base = _last_logits(model, tok, p, device)
            with ctx:
                patched = _last_logits(model, tok, p, device)
            eq = bool(torch.equal(base, patched))
            mad = float((base - patched).abs().max().item())
            n_pass += int(eq)
            max_abs = max(max_abs, mad)
            row = {
                "experiment": out_dir.name,
                "variant": variant.name,
                "method": variant.method,
                "alpha": variant.alpha,
                "model": model_name,
                "seed": seed,
                "prompt_id": f"neutral_{i:03d}",
                "torch_equal": eq,
                "max_abs_diff": mad,
                "logits_shape": list(base.shape),
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            if (i + 1) % 20 == 0:
                _log(f"  {i+1}/{len(prompts)} eq_so_far={n_pass} "
                     f"max_abs={max_abs:.3e}")
    _log(f"done: {n_pass}/{len(prompts)} bit-equal, max_abs_diff={max_abs:.3e}")
    log.close()
    return results_path
