"""α dense-sweep runner for Exp 5.

Bank: ``bank_size`` total facts (1 target + (bank_size-1) distractors)
drawn deterministically from CounterFact-1k.

For each (alpha, seed):
  * write all bank_size facts under default ATB config
  * read the *target* prompt under each alpha
  * record per-alpha metrics including readout norms (o_bank, o_seq)

This is a single-target probe (alpha cliff hunting), not a CF-1k sweep.
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
    evaluate_prompt,
    filter_cf_for_tokenizer,
    load_counterfact,
    load_model,
    seed_everything,
    variant_uses_dynamic_lopi,
)
from .cf_runner import build_write_prompt, render_query


def run(
    *,
    model_name: str,
    dtype: str,
    device: str,
    counterfact_path: Path,
    alphas: list[float],
    seeds: list[int],
    bank_size: int,
    out_dir: Path,
    target_index: int = 0,
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

    _log(f"loading {model_name}")
    tok, model = load_model(model_name, device=device, dtype=dtype)
    cf = load_counterfact(counterfact_path)
    kept, _ = filter_cf_for_tokenizer(cf, tok)
    if len(kept) < bank_size:
        raise RuntimeError(f"need {bank_size} CF rows, only {len(kept)} kept")
    bank_rows = kept[:bank_size]
    target_row = bank_rows[target_index]
    _log(f"bank_size={bank_size} target_pid={target_row['id']} "
         f"subject={target_row['subject']}")

    facts: list[dict] = []
    for r in bank_rows:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        facts.append({"id": r["id"], "subject": r["subject"],
                      "write_prompt": wp})
    _log(f"facts written: {len(facts)}")

    target_query = render_query(target_row)
    target_new = target_row["target_new"]
    target_true = target_row["target_true"]

    n_cells = 0
    with open(results_path, "a") as out_f:
        for seed in seeds:
            seed_everything(seed)
            for alpha in alphas:
                variant = Variant(name=f"alpha_{alpha:.2f}", method="anb",
                                  alpha=alpha, bank_key_mode="pre_rope",
                                  value_scale_mode="auto_rms_cap")
                try:
                    with VariantContext(model, tok, device, variant, facts):
                        mp = evaluate_prompt(model, tok, target_query,
                                             target_new, target_true, device,
                                             preserve_forward_sequence=variant_uses_dynamic_lopi(variant))
                except Exception as exc:
                    _log(f"  ERROR alpha={alpha} seed={seed}: {exc}")
                    continue
                row = {
                    "experiment": out_dir.name,
                    "variant": variant.name,
                    "method": "anb",
                    "alpha": alpha,
                    "seed": seed,
                    "prompt_id": target_row["id"],
                    "bank_size": bank_size,
                    **mp,
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                n_cells += 1
            _log(f"  seed={seed} done ({len(alphas)} alphas)")
    _log(f"done: {n_cells} cells -> {results_path}")
    log.close()
    return results_path
