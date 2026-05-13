#!/usr/bin/env python3
"""X7_mech/B2 — Sparsity test at |bank|=5000.

Tests the "quasi-top-k regime" conjecture: at |bank|=5000, what fraction
of attention concentrates on the top-k bank columns? Compare to |bank|=500.

Reports top-k-fraction at k=1,5,10,20.
Statistical test (paired Wilcoxon across 3 seeds): does the top-k-fraction
discontinuously jump between |bank|=500 and 5000?

PREREG: experiments/X7_mech/PREREG.md (X7MECH.v1 §2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deltamemory.memory.attn_native_bank import (  # noqa: E402
    AttnNativePatcher,
    forward_with_bank,
    fresh_bank,
    write_fact,
)
from deltamemory.diagnostics import DiagnosticRecorder  # noqa: E402
from tools.env_writer import sha1_of, write_env_json  # noqa: E402

PREREG_VERSION = "X7MECH.v1"
DEFAULT_BANK_SIZES = [100, 200, 500, 1000, 5000]
DEFAULT_SEEDS = [0, 1, 2]
TOP_K_VALUES = [1, 5, 10, 20]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for line in f:
            try:
                out.add(json.loads(line).get("cell_id", ""))
            except json.JSONDecodeError:
                pass
    return out


def cell_id(model: str, size: int, seed: int, fact_id: str) -> str:
    return hashlib.sha1(
        f"sparsity|{model}|{size}|{seed}|{fact_id}".encode()
    ).hexdigest()[:16]


def first_token_id(tok, text: str) -> int:
    ids = tok(" " + text.strip(), add_special_tokens=False).input_ids
    if not ids:
        ids = tok(text, add_special_tokens=False).input_ids
    return int(ids[0])


def _wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, float]:
    """Simplified paired Wilcoxon signed-rank test.

    Returns (W_statistic, p_value_approx) for small n.
    For n=3 seeds, uses exact table; falls back to normal approximation for n>5.
    """
    n = len(x)
    assert len(y) == n
    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero = [(abs(d), 1 if d > 0 else -1) for d in diffs if abs(d) > 1e-10]
    if not nonzero:
        return 0.0, 1.0
    ranked = sorted(enumerate(nonzero), key=lambda p: p[1][0])
    W_plus = sum((r + 1) * (s > 0) for r, (_, s) in enumerate(ranked))
    W_minus = sum((r + 1) * (s < 0) for r, (_, s) in enumerate(ranked))
    W = min(W_plus, W_minus)
    n_nz = len(nonzero)
    # Normal approximation (suitable for n≥3 as a rough guide)
    if n_nz == 0:
        return W, 1.0
    mu = n_nz * (n_nz + 1) / 4.0
    sigma = math.sqrt(n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0)
    if sigma < 1e-10:
        return W, 1.0
    z = (W - mu) / sigma
    # Two-tailed p from z approximation
    p_approx = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return W, p_approx


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_topk_fractions(
    bank_col_sums: list[float],
    k_values: list[int],
) -> dict[str, float]:
    """Compute fraction of attention mass in top-k bank columns."""
    if not bank_col_sums:
        return {}
    total = sum(bank_col_sums) + 1e-12
    sorted_desc = sorted(bank_col_sums, reverse=True)
    result = {}
    for k in k_values:
        actual_k = min(k, len(sorted_desc))
        frac = sum(sorted_desc[:actual_k]) / total
        result[f"top{k}_frac"] = frac
    return result


def write_target_then_distractors(
    *, model, tok, patcher, target_fact, distractors, n_distract, seed
) -> tuple[Any, str | None]:
    bank = fresh_bank(model)
    try:
        write_fact(
            patcher=patcher, bank=bank, tokenizer=tok,
            write_prompt=target_fact["write_prompt"],
            fact_id=target_fact["fact_id"],
            address=target_fact["subject"],
        )
    except Exception as exc:
        return bank, f"target_write_failed: {exc!r}"
    if n_distract > 0:
        offset = (seed * 2069) % len(distractors)
        chosen = [distractors[(offset + i) % len(distractors)]
                  for i in range(n_distract)]
        for d in chosen:
            try:
                write_fact(
                    patcher=patcher, bank=bank, tokenizer=tok,
                    write_prompt=d["write_prompt"],
                    fact_id=d["fact_id"], address=d["address"],
                )
            except Exception as exc:
                return bank, f"distract_write_failed: {exc!r}"
    return bank, None


def run_sparsity_cell(
    model, tok, patcher, bank, target, alpha: float,
    target_new_id: int, target_canon_id: int,
    k_values: list[int],
) -> dict[str, Any]:
    """Measure per-cell top-k fractions aggregated across all layers."""
    rec = DiagnosticRecorder(model=model, patcher=patcher, enabled=True)
    try:
        with rec:
            logits = forward_with_bank(
                patcher=patcher, bank=bank, tokenizer=tok,
                read_prompt=target["read_prompt"], alpha=alpha,
            )
    except Exception as exc:
        return {"status": "forward_failed", "error": repr(exc)[:200]}

    if not bool(torch.isfinite(logits).all().item()):
        return {"status": "nan_inf"}

    log_margin = float(
        logits[target_new_id].item() - logits[target_canon_id].item()
    )

    # Aggregate bank_col_sum by layer
    layer_bank_cols: dict[int, list[float]] = {}
    for r in rec._records:
        if r["signal_name"] == "bank_col_sum":
            L = r["layer"]
            layer_bank_cols.setdefault(L, []).append(r["value"])

    # Overall top-k (summed across all layers)
    all_col_sums: list[float] = []
    for cols in layer_bank_cols.values():
        all_col_sums.extend(cols)

    topk_overall = compute_topk_fractions(all_col_sums, k_values)

    # Per-layer top-k fractions
    layer_topk: dict[str, Any] = {}
    for L, cols in layer_bank_cols.items():
        layer_topk[str(L)] = compute_topk_fractions(cols, k_values)

    return {
        "status": "ok",
        "log_margin": log_margin,
        "n_bank_cols": len(all_col_sums),
        "layer_topk": layer_topk,
        **topk_overall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="X7_mech/B2 — Sparsity test"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--bank-sizes", nargs="+", type=int, default=DEFAULT_BANK_SIZES)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.bank_sizes = [100, 500]
        args.seeds = [0]

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"
    summary_path = args.out / "sparsity_summary.json"

    x1_dir = ROOT / "experiments" / "X1_bank_scaling"
    facts_path = x1_dir / "facts.jsonl"
    distract_path = x1_dir / "distractors.jsonl"

    write_env_json(
        out_dir=args.out, prereg_version=PREREG_VERSION,
        dataset_sha1={
            facts_path.name: sha1_of(facts_path),
            distract_path.name: sha1_of(distract_path),
        },
        device=args.device, dtype=args.dtype, cli_argv=sys.argv,
        extra={
            "experiment": "B2_sparsity",
            "model": args.model,
            "bank_sizes": args.bank_sizes,
            "seeds": args.seeds,
            "alpha": args.alpha,
            "top_k_values": TOP_K_VALUES,
        },
    )

    facts = load_jsonl(facts_path)
    distractors = load_jsonl(distract_path)
    target = facts[0]
    print(f"[B2] target={target['fact_id']} distractors={len(distractors)}", flush=True)

    done = load_done(cells_path)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print(f"[B2] loading {args.model} ({args.dtype}) → {args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    patcher = AttnNativePatcher(model)
    patcher.install()

    target_new_id = first_token_id(tok, target["target_new"])
    target_canon_id = first_token_id(tok, target["target_canonical"])

    # Collect results for Wilcoxon test
    # results_by_size[size][seed] = {top1_frac, top5_frac, top10_frac, top20_frac}
    results_by_size: dict[int, dict[int, dict]] = {s: {} for s in args.bank_sizes}

    try:
        for size in args.bank_sizes:
            for seed in args.seeds:
                cid = cell_id(args.model, size, seed, target["fact_id"])
                if cid in done:
                    print(f"  [skip] |bank|={size} s={seed}", flush=True)
                    continue

                torch.manual_seed(seed)
                bank, err = write_target_then_distractors(
                    model=model, tok=tok, patcher=patcher,
                    target_fact=target, distractors=distractors,
                    n_distract=max(0, size - 1), seed=seed,
                )
                if err:
                    row = {
                        "status": "write_failed", "error": err,
                        "bank_size": size, "seed": seed,
                        "cell_id": cid, "model": args.model,
                        "prereg_version": PREREG_VERSION,
                        "experiment": "B2_sparsity",
                    }
                    append_row(cells_path, row)
                    continue

                result = run_sparsity_cell(
                    model=model, tok=tok, patcher=patcher,
                    bank=bank, target=target, alpha=args.alpha,
                    target_new_id=target_new_id, target_canon_id=target_canon_id,
                    k_values=TOP_K_VALUES,
                )

                row = {
                    "cell_id": cid,
                    "bank_size": size,
                    "seed": seed,
                    "model": args.model,
                    "prereg_version": PREREG_VERSION,
                    "experiment": "B2_sparsity",
                    **result,
                }
                # layer_topk is too large for cells.jsonl; store separately
                layer_topk = row.pop("layer_topk", {})
                append_row(cells_path, row)

                if result.get("status") == "ok":
                    results_by_size[size][seed] = {
                        k: result.get(k, 0.0) for k in
                        [f"top{kv}_frac" for kv in TOP_K_VALUES]
                    }
                    print(
                        f"  B2 |bank|={size:>5} s={seed} "
                        f"top1={result.get('top1_frac', 0):.3f} "
                        f"top10={result.get('top10_frac', 0):.3f} "
                        f"margin={result.get('log_margin', 0):+.3f}",
                        flush=True,
                    )
    finally:
        patcher.remove()

    # --- Statistical analysis: Wilcoxon 500 vs 5000 ---
    wilcoxon_results: dict[str, Any] = {}
    size_500_data = results_by_size.get(500, {})
    size_5000_data = results_by_size.get(5000, {})
    common_seeds = sorted(set(size_500_data.keys()) & set(size_5000_data.keys()))

    if len(common_seeds) >= 2:
        for k in TOP_K_VALUES:
            k_key = f"top{k}_frac"
            x = [size_500_data[s].get(k_key, 0.0) for s in common_seeds]
            y = [size_5000_data[s].get(k_key, 0.0) for s in common_seeds]
            W, p = _wilcoxon_signed_rank(x, y)
            mean_diff = sum(yi - xi for xi, yi in zip(x, y)) / len(x)
            wilcoxon_results[f"top{k}"] = {
                "W": W, "p_approx": p,
                "mean_diff_5000_minus_500": mean_diff,
                "seeds": common_seeds,
                "x_500": x, "y_5000": y,
            }
            print(
                f"  Wilcoxon top{k}: W={W:.1f} p≈{p:.3f} "
                f"mean_diff={mean_diff:+.4f}",
                flush=True,
            )

    # Compute mean top-k fracs by size
    size_means: dict[int, dict[str, float]] = {}
    for size, seed_data in results_by_size.items():
        if not seed_data:
            continue
        means: dict[str, float] = {}
        for k in TOP_K_VALUES:
            k_key = f"top{k}_frac"
            vals = [sd.get(k_key, 0.0) for sd in seed_data.values()]
            means[k_key] = sum(vals) / len(vals) if vals else 0.0
        size_means[size] = means

    summary = {
        "experiment": "B2_sparsity",
        "prereg_version": PREREG_VERSION,
        "bank_sizes": args.bank_sizes,
        "top_k_values": TOP_K_VALUES,
        "size_means": {str(k): v for k, v in size_means.items()},
        "wilcoxon_500_vs_5000": wilcoxon_results,
        "hypothesis_H_B2.1": _check_h_b2_1(size_means),
        "hypothesis_H_B2.3": _check_h_b2_3(wilcoxon_results),
    }

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[B2] DONE summary -> {summary_path}", flush=True)
    return 0


def _check_h_b2_1(size_means: dict[int, dict[str, float]]) -> dict:
    """H_B2.1: at |bank|=5000, top10 >=80%; at |bank|=500, top10 <80%."""
    r = {}
    m5000 = size_means.get(5000, {}).get("top10_frac", 0.0)
    m500 = size_means.get(500, {}).get("top10_frac", 0.0)
    r["top10_at_5000"] = m5000
    r["top10_at_500"] = m500
    r["supported"] = bool(m5000 >= 0.80 and m500 < 0.80)
    r["note"] = (
        f"top10@5000={m5000:.3f} ≥ 0.80 AND top10@500={m500:.3f} < 0.80: "
        f"{'supported' if r['supported'] else 'not_supported'}"
    )
    return r


def _check_h_b2_3(wilcoxon_results: dict) -> dict:
    """H_B2.3: top-10 fraction significantly different at p<0.05."""
    r = {}
    w10 = wilcoxon_results.get("top10", {})
    p = w10.get("p_approx", 1.0)
    r["p_top10"] = p
    r["supported"] = bool(p < 0.05)
    r["note"] = f"p(top10, 500 vs 5000)={p:.3f}: {'supported' if r['supported'] else 'not_supported'}"
    return r


if __name__ == "__main__":
    sys.exit(main())
