"""Position-invariance runner for Exp 2.

For each (variant ∈ {pre_rope, post_rope}, fact, position_delta, seed):
  * write the fact at canonical (no filler) position
  * read with N=position_delta tokens of neutral filler prepended

Subject / target tokens never appear in filler.
"""

from __future__ import annotations

import json
import random
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
    neutral_prompts,
    seed_everything,
)
from .cf_runner import build_write_prompt, render_query


def _build_filler(
    tok: Any,
    n_tokens: int,
    forbidden: list[str],
    rng: random.Random,
) -> str:
    """Return text that tokenizes to roughly ``n_tokens`` and contains none of
    the ``forbidden`` substrings (case-insensitive)."""
    if n_tokens == 0:
        return ""
    pool = neutral_prompts(n=200)
    forb_l = [f.lower() for f in forbidden if f]
    rng.shuffle(pool)
    chunks: list[str] = []
    cur = 0
    for p in pool:
        if any(f in p.lower() for f in forb_l):
            continue
        ids = tok.encode(p, add_special_tokens=False)
        chunks.append(p)
        cur += len(ids)
        if cur >= n_tokens:
            break
    text = " ".join(chunks)
    # Truncate to exactly n_tokens (approx).
    ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
    return tok.decode(ids)


def run(
    *,
    model_name: str,
    dtype: str,
    device: str,
    counterfact_path: Path,
    variants: list[Variant],
    seeds: list[int],
    out_dir: Path,
    n_facts: int = 50,
    position_deltas: list[int] | None = None,
) -> Path:
    if position_deltas is None:
        position_deltas = [0, 128, 512, 1024]
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
    facts = []
    for r in kept:
        wp = build_write_prompt(r, r["target_new"])
        if wp is None:
            continue
        facts.append({
            "row": r,
            "fact": {"id": r["id"], "subject": r["subject"],
                     "write_prompt": wp},
        })
        if len(facts) >= n_facts:
            break
    _log(f"selected facts: {len(facts)}")

    n_cells = 0
    with open(results_path, "a") as out_f:
        for variant in variants:
            _log(f"variant={variant.name} mode={variant.bank_key_mode}")
            for seed in seeds:
                seed_everything(seed)
                rng = random.Random(seed)
                for entry in facts:
                    row = entry["row"]
                    fact = entry["fact"]
                    canonical_query = render_query(row)
                    forbidden = [row["subject"], row["target_new"],
                                 row["target_true"]]
                    for pd in position_deltas:
                        filler = _build_filler(tok, pd, forbidden, rng)
                        read_prompt = (filler + " " + canonical_query
                                       if filler else canonical_query)
                        try:
                            with VariantContext(model, tok, device, variant,
                                                [fact]):
                                mp = evaluate_prompt(model, tok, read_prompt,
                                                     row["target_new"],
                                                     row["target_true"],
                                                     device)
                        except Exception as exc:
                            _log(f"  ERROR pid={row['id']} pd={pd}: {exc}")
                            continue
                        rec = {
                            "experiment": out_dir.name,
                            "variant": variant.name,
                            "method": variant.method,
                            "alpha": variant.alpha,
                            "bank_key_mode": variant.bank_key_mode,
                            "seed": seed,
                            "prompt_id": row["id"],
                            "subject": row["subject"],
                            "position_delta": pd,
                            "filler_token_count": len(
                                tok.encode(filler, add_special_tokens=False)
                            ) if filler else 0,
                            "bank_size": 1,
                            **mp,
                        }
                        out_f.write(json.dumps(rec) + "\n")
                        out_f.flush()
                        n_cells += 1
                        if n_cells % 25 == 0:
                            _log(f"  ...{n_cells} cells")
    _log(f"done: {n_cells} cells -> {results_path}")
    log.close()
    return results_path
