"""CounterFact-style variant runner.

For each (variant, seed, prompt) cell:
  * write the single counterfact fact into a fresh bank under variant config
  * apply the variant's K/V perturbation (if any)
  * read the prompt under variant alpha
  * record per-prompt metrics
  * record per-variant unrelated drift (broadcast to rows)

Used by Exp 1 (5 variants), Exp 4 (3 rows), Exp 6 (5 negative-control variants).
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from . import (
    Variant,
    VariantContext,
    evaluate_prompt,
    filter_cf_for_tokenizer,
    first_token_id,
    load_counterfact,
    load_model,
    seed_everything,
    sha1_of_file,
    unrelated_drift,
)


def render_query(prompt_row: dict) -> str:
    pt = prompt_row["prompt"]
    return pt.format(prompt_row["subject"]) if "{}" in pt else pt


def relation_phrase(prompt_row: dict) -> str | None:
    """Drop subject from the front of the natural-language prompt."""
    pt = (prompt_row.get("prompt") or "").strip()
    if pt.startswith("{}"):
        return pt[2:].strip().rstrip(",") or None
    return None


def build_write_prompt(prompt_row: dict, target: str) -> str | None:
    phr = relation_phrase(prompt_row)
    if phr is None:
        return None
    return f"Fact: {prompt_row['subject']} {phr} {target}."


def run(
    *,
    model_name: str,
    dtype: str,
    device: str,
    counterfact_path: Path,
    variants: list[Variant],
    seeds: list[int],
    out_dir: Path,
    n_prompts: int | None = None,
    n_neutral: int = 100,
    drift_every_seed: bool = True,
) -> Path:
    """Run all (variant, seed, prompt) cells and write results.jsonl.

    Returns the results path.
    """
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

    _log(f"loading {model_name} dtype={dtype} device={device}")
    tok, model = load_model(model_name, device=device, dtype=dtype)
    rows = load_counterfact(counterfact_path)
    _log(f"counterfact raw rows: {len(rows)}")
    kept, dropped = filter_cf_for_tokenizer(rows, tok)
    _log(f"after filter kept={len(kept)} dropped={dropped}")
    if n_prompts is not None:
        kept = kept[:n_prompts]
        _log(f"truncated to n={len(kept)} (smoke / dry-run)")

    # Pre-compute per-prompt facts (variants reuse same fact structure).
    facts_by_pid: dict[str, dict] = {}
    for r in kept:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        facts_by_pid[r["id"]] = {
            "id": r["id"],
            "subject": r["subject"],
            "write_prompt": wp,
        }
    eligible = [r for r in kept if r["id"] in facts_by_pid]
    _log(f"eligible prompts (have relation phrase): {len(eligible)}")

    n_cells = 0
    with open(results_path, "a") as out_f:
        for variant in variants:
            _log(f"variant={variant.name} method={variant.method} "
                 f"alpha={variant.alpha} bank_key_mode={variant.bank_key_mode} "
                 f"perturbation={variant.bank_perturbation}")
            for seed in seeds:
                seed_everything(seed)
                # --- per-(variant, seed) drift: install bank with all eligible
                # facts (or 8 sample for big runs) and probe 100 neutral
                # prompts.
                drift_facts = (
                    [facts_by_pid[r["id"]] for r in eligible[:8]]
                    if variant.method != "none" else []
                )
                if drift_every_seed or seed == seeds[0]:
                    def _ctx(_v=variant, _f=drift_facts):
                        return VariantContext(model, tok, device, _v, _f)
                    js, kl = unrelated_drift(model, tok, device, _ctx,
                                              n_prompts=n_neutral)
                    _log(f"  drift seed={seed}: js={js:.4f} kl={kl:.4f}")
                # --- per-prompt cells
                for prow in eligible:
                    fact = facts_by_pid[prow["id"]]
                    query = render_query(prow)
                    target_new = prow["target_new"]
                    target_true = prow["target_true"]
                    try:
                        with VariantContext(model, tok, device, variant,
                                            [fact]):
                            mp = evaluate_prompt(model, tok, query,
                                                 target_new, target_true,
                                                 device)
                    except Exception as exc:
                        _log(f"  ERROR pid={prow['id']} variant={variant.name}"
                             f" seed={seed}: {exc}")
                        continue
                    row = {
                        "experiment": out_dir.name,
                        "variant": variant.name,
                        "method": variant.method,
                        "alpha": variant.alpha,
                        "bank_key_mode": variant.bank_key_mode,
                        "bank_perturbation": variant.bank_perturbation,
                        "value_scale_mode": variant.value_scale_mode,
                        "enable_scar": variant.enable_scar,
                        "seed": seed,
                        "prompt_id": prow["id"],
                        "subject": prow["subject"],
                        "relation": prow.get("relation"),
                        "target_new": target_new,
                        "target_true": target_true,
                        "bank_size": 1,
                        "js_drift": js,
                        "kl_drift": kl,
                        **mp,
                    }
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                    n_cells += 1
                    if n_cells % 50 == 0:
                        _log(f"  ...{n_cells} cells")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _log(f"done: {n_cells} cells -> {results_path}")
    log.close()
    return results_path
