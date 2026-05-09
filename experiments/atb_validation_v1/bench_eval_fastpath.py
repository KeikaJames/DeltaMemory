"""Benchmark ATB evaluate_prompt fast-path parity and speed.

This script is intentionally read-only: it loads a model and CounterFact rows,
prints one JSON summary to stdout, and writes no experiment artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import (
    continuation_logp,
    evaluate_prompt,
    filter_cf_for_tokenizer,
    first_token_rank,
    load_counterfact,
    load_model,
)


def _legacy_evaluate_prompt(
    model: Any,
    tok: Any,
    prompt: str,
    target_new: str,
    target_true: str,
    device: str,
) -> dict[str, Any]:
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


def _query(row: dict[str, Any]) -> str:
    prompt = row.get("prompt", "")
    return prompt.format(row["subject"]) if "{}" in prompt else prompt


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: str) -> float | None:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


def _run_cells(
    *,
    mode: str,
    model: Any,
    tok: Any,
    rows: list[dict[str, Any]],
    device: str,
    preserve_forward_sequence: bool = False,
) -> tuple[list[dict[str, Any]], float, float | None]:
    _reset_peak(device)
    _sync(device)
    t0 = time.perf_counter()
    out: list[dict[str, Any]] = []
    for row in rows:
        query = _query(row)
        if mode == "legacy":
            out.append(
                _legacy_evaluate_prompt(
                    model, tok, query, row["target_new"], row["target_true"], device
                )
            )
        elif mode == "fast":
            out.append(
                evaluate_prompt(
                    model,
                    tok,
                    query,
                    row["target_new"],
                    row["target_true"],
                    device,
                    preserve_forward_sequence=preserve_forward_sequence,
                )
            )
        else:
            raise ValueError(f"unknown mode {mode!r}")
    _sync(device)
    return out, time.perf_counter() - t0, _peak_mb(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--counterfact",
        default="experiments/datasets/counterfact_1k.jsonl",
        type=Path,
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--preserve-forward-sequence",
        action="store_true",
        help="Run evaluate_prompt in the legacy three-forward sequence.",
    )
    args = parser.parse_args()

    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    rows = load_counterfact(args.counterfact)
    kept, dropped = filter_cf_for_tokenizer(rows, tok)
    rows_n = kept[: args.n]

    if args.warmup and rows_n:
        row = rows_n[0]
        evaluate_prompt(
            model, tok, _query(row), row["target_new"], row["target_true"], args.device
        )
        _sync(args.device)

    legacy, legacy_s, legacy_peak = _run_cells(
        mode="legacy", model=model, tok=tok, rows=rows_n, device=args.device
    )
    fast, fast_s, fast_peak = _run_cells(
        mode="fast",
        model=model,
        tok=tok,
        rows=rows_n,
        device=args.device,
        preserve_forward_sequence=args.preserve_forward_sequence,
    )

    mismatches = [
        {"index": i, "legacy": a, "fast": b}
        for i, (a, b) in enumerate(zip(legacy, fast))
        if a != b
    ]
    result = {
        "model": args.model,
        "counterfact": str(args.counterfact),
        "device": args.device,
        "dtype": args.dtype,
        "preserve_forward_sequence": args.preserve_forward_sequence,
        "n": len(rows_n),
        "dropped": dropped,
        "parity": not mismatches,
        "mismatches": len(mismatches),
        "legacy_s": legacy_s,
        "fast_s": fast_s,
        "legacy_cell_s": legacy_s / max(len(rows_n), 1),
        "fast_cell_s": fast_s / max(len(rows_n), 1),
        "speedup": legacy_s / fast_s if fast_s > 0 else None,
        "legacy_peak_cuda_mb": legacy_peak,
        "fast_peak_cuda_mb": fast_peak,
        "peak_delta_cuda_mb": (
            None if legacy_peak is None or fast_peak is None else fast_peak - legacy_peak
        ),
    }
    print(json.dumps(result, sort_keys=True))
    if mismatches:
        print(json.dumps({"sample_mismatches": mismatches[:3]}, indent=2, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
