#!/usr/bin/env python3
"""X.1 — bank-size scaling and softmax dilution witness.

Tests whether the target-fact recall log-margin decays as bank size N
grows (default arm = `none`), and whether the existing defense knobs
(`bank_topk`, `bank_separate_softmax`) restore N-invariance.

PREREG: experiments/X1_bank_scaling/PREREG.md (X1.v1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
from tools.env_writer import sha1_of, write_env_json  # noqa: E402

PREREG_VERSION = "X1.v1"
DEFAULT_NS = [1, 10, 100, 1000, 10000]
DEFAULT_ARMS = ["none", "topk_4", "separate_softmax"]
RED_LINE_TOL_FP32 = 1e-4
RED_LINE_TOL_BF16 = 5e-3


def cell_id(model: str, N: int, arm: str, alpha: float, seed: int,
            tfid: str) -> str:
    return hashlib.sha1(
        f"{model}|{N}|{arm}|{alpha}|{seed}|{tfid}".encode()
    ).hexdigest()[:16]


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


def configure_bank(bank, arm: str) -> None:
    """Apply defense-arm settings to the bank object in-place."""
    if arm == "none":
        bank.bank_topk = 0
        bank.bank_separate_softmax = False
        bank.bank_temperature = 1.0
    elif arm == "topk_4":
        bank.bank_topk = 4
        bank.bank_separate_softmax = False
        bank.bank_temperature = 1.0
    elif arm == "separate_softmax":
        bank.bank_topk = 0
        bank.bank_separate_softmax = True
        bank.bank_merge_beta = 1.0
        bank.bank_temperature = 1.0
    else:
        raise ValueError(f"unknown arm: {arm!r}")


def first_token_id(tok, text: str) -> int:
    ids = tok(" " + text.strip(), add_special_tokens=False).input_ids
    return int(ids[0]) if ids else int(tok(text, add_special_tokens=False).input_ids[0])


def run_cell(*, model, tok, patcher, target_fact: dict, distractors: list[dict],
             N: int, arm: str, alpha: float, seed: int, model_name: str,
             dtype_str: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    bank = fresh_bank(model)
    configure_bank(bank, arm)

    # Write target fact first (slot 0).
    try:
        write_fact(
            patcher=patcher, bank=bank, tokenizer=tok,
            write_prompt=target_fact["write_prompt"],
            fact_id=target_fact["fact_id"],
            address=target_fact["subject"],
        )
    except Exception as exc:
        return {"status": "target_write_failed", "error": repr(exc)[:200]}

    # Write distractors deterministically based on seed (rotation of the pack).
    n_distract = max(0, N - 1)
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
                return {"status": "distract_write_failed", "n_written": -1,
                        "error": repr(exc)[:200]}

    read_prompt = target_fact["read_prompt"]
    target_new_id = first_token_id(tok, target_fact["target_new"])
    target_canon_id = first_token_id(tok, target_fact["target_canonical"])

    # Latency probe (single read, wall-clock).
    t0 = time.perf_counter()
    try:
        logits = forward_with_bank(
            patcher=patcher, bank=bank, tokenizer=tok,
            read_prompt=read_prompt, alpha=alpha,
        )
    except Exception as exc:
        return {"status": "forward_failed", "error": repr(exc)[:200]}
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not bool(torch.isfinite(logits).all().item()):
        return {"status": "nan_inf", "N": N, "arm": arm, "alpha": alpha}

    log_margin = float(
        logits[target_new_id].item() - logits[target_canon_id].item()
    )

    res = {
        "status": "ok",
        "model": model_name, "N": N, "arm": arm, "alpha": alpha, "seed": seed,
        "target_fact_id": target_fact["fact_id"],
        "log_margin": log_margin,
        "score_new": float(logits[target_new_id].item()),
        "score_canonical": float(logits[target_canon_id].item()),
        "read_latency_ms": elapsed_ms,
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--Ns", nargs="+", type=int, default=DEFAULT_NS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.Ns = [1, 10, 100]
        args.arms = ["none", "topk_4", "separate_softmax"]
        args.alphas = [0.0, 1.0]
        args.seeds = [0]
        print("[X1] SMOKE: 1 model x 3 N x 3 arm x 2 alpha x 1 seed = "
              "18 cells", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"

    here = Path(__file__).parent
    facts_path = here / "facts.jsonl"
    distract_path = here / "distractors.jsonl"
    write_env_json(
        out_dir=args.out, prereg_version=PREREG_VERSION,
        dataset_sha1={
            facts_path.name: sha1_of(facts_path),
            distract_path.name: sha1_of(distract_path),
        },
        device=args.device, dtype=args.dtype, cli_argv=sys.argv,
        extra={"Ns": args.Ns, "arms": args.arms, "model": args.model},
    )

    facts = load_jsonl(facts_path)
    distractors = load_jsonl(distract_path)
    target = facts[0]
    print(f"[X1] target={target['fact_id']} distractors={len(distractors)}",
          flush=True)
    done = load_done(cells_path)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print(f"[X1] loading {args.model}", flush=True)
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

    try:
        for N in args.Ns:
            for arm in args.arms:
                for seed in args.seeds:
                    for alpha in args.alphas:
                        cid = cell_id(args.model, N, arm, alpha, seed,
                                      target["fact_id"])
                        if cid in done:
                            continue
                        row = run_cell(
                            model=model, tok=tok, patcher=patcher,
                            target_fact=target, distractors=distractors,
                            N=N, arm=arm, alpha=alpha, seed=seed,
                            model_name=args.model, dtype_str=args.dtype,
                        )
                        row["cell_id"] = cid
                        row["prereg_version"] = PREREG_VERSION
                        append_row(cells_path, row)
                        lm = row.get("log_margin")
                        lm_s = (f"{lm:+.3f}"
                                if isinstance(lm, (int, float)) else "n/a")
                        lat = row.get("read_latency_ms", 0.0)
                        print(f"  N={N:>5} arm={arm:>17} a={alpha} "
                              f"s={seed} margin={lm_s} "
                              f"lat={lat:6.1f}ms status={row.get('status')}",
                              flush=True)
    finally:
        patcher.remove()

    print(f"[X1] DONE -> {cells_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
