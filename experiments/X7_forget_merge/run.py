#!/usr/bin/env python3
"""X.7 — LRU forget/merge experimental verdict (H_X7.3).

Protocol B (interleaved): write target fact at t=0, then write N
distractors, periodically reading the target. Under LRU eviction, the
read should update the target's last-access timestamp and keep it
resident; under FIFO, the target gets evicted once N > capacity.

Cells: capacity ∈ {0=unbounded, 16, 64, 256, 1024} × policy ∈ {lru, fifo}
× N ∈ {100, 1000, 4000} × 3 seeds × α = 1.0 (+ α=0 redline).

PREREG: experiments/X7_forget_merge/PREREG.md (X.7.v1).
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

PREREG_VERSION = "X7.v1"
DEFAULT_CAPS = [0, 16, 64, 256, 1024]
DEFAULT_POLS = ["lru", "fifo"]
DEFAULT_NS = [100, 1000, 4000]


def cell_id(model: str, N: int, cap: int, pol: str, alpha: float,
            seed: int, tfid: str) -> str:
    return hashlib.sha1(
        f"{model}|{N}|{cap}|{pol}|{alpha}|{seed}|{tfid}".encode()
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


def first_token_id(tok, text: str) -> int:
    ids = tok(" " + text.strip(), add_special_tokens=False).input_ids
    return int(ids[0]) if ids else int(tok(text, add_special_tokens=False).input_ids[0])


def run_cell(*, model, tok, patcher, target_fact: dict,
             distractors: list[dict], N: int, capacity: int, policy: str,
             alpha: float, seed: int, model_name: str,
             read_period: int) -> dict[str, Any]:
    """Protocol B: write target → interleave (write distractor, periodic read).

    Periodic reads update LRU access metadata. At end, do a final read and
    record log_margin.
    """
    torch.manual_seed(seed)
    bank = fresh_bank(model)
    bank.bank_capacity = int(capacity)
    bank.bank_evict_policy = policy
    bank.bank_topk = 0
    bank.bank_separate_softmax = False
    bank.bank_temperature = 1.0

    try:
        write_fact(
            patcher=patcher, bank=bank, tokenizer=tok,
            write_prompt=target_fact["write_prompt"],
            fact_id=target_fact["fact_id"],
            address=target_fact["subject"],
        )
    except Exception as exc:
        return {"status": "target_write_failed", "error": repr(exc)[:200]}

    read_prompt = target_fact["read_prompt"]
    target_new_id = first_token_id(tok, target_fact["target_new"])
    target_canon_id = first_token_id(tok, target_fact["target_canonical"])

    n_distract = max(0, N - 1)
    interim_reads = 0
    if n_distract > 0:
        offset = (seed * 2069) % len(distractors)
        for i in range(n_distract):
            d = distractors[(offset + i) % len(distractors)]
            try:
                write_fact(
                    patcher=patcher, bank=bank, tokenizer=tok,
                    write_prompt=d["write_prompt"],
                    fact_id=d["fact_id"], address=d["address"],
                )
            except Exception as exc:
                return {"status": "distract_write_failed",
                        "error": repr(exc)[:200], "i": i}
            # Periodic read to update LRU access metadata on the target slot.
            if read_period > 0 and (i + 1) % read_period == 0:
                try:
                    _ = forward_with_bank(
                        patcher=patcher, bank=bank, tokenizer=tok,
                        read_prompt=read_prompt, alpha=alpha,
                    )
                    interim_reads += 1
                except Exception:
                    pass

    # Final read: latency + log_margin.
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
        return {"status": "nan_inf", "N": N, "capacity": capacity,
                "policy": policy, "alpha": alpha}

    log_margin = float(
        logits[target_new_id].item() - logits[target_canon_id].item()
    )

    # Bank residency: was the target slot (first append, write_step==1) kept?
    write_steps = list(getattr(bank, "_x7_write_step", []) or [])
    target_resident = bool(write_steps) and (min(write_steps) == 1)
    bank_size = len(write_steps)

    return {
        "status": "ok",
        "model": model_name, "N": N, "capacity": capacity, "policy": policy,
        "alpha": alpha, "seed": seed,
        "target_fact_id": target_fact["fact_id"],
        "log_margin": log_margin,
        "score_new": float(logits[target_new_id].item()),
        "score_canonical": float(logits[target_canon_id].item()),
        "read_latency_ms": elapsed_ms,
        "interim_reads": interim_reads,
        "target_resident": target_resident,
        "bank_size": bank_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--Ns", nargs="+", type=int, default=DEFAULT_NS)
    ap.add_argument("--caps", nargs="+", type=int, default=DEFAULT_CAPS)
    ap.add_argument("--policies", nargs="+", default=DEFAULT_POLS)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--read-period", type=int, default=32,
                    help="periodic interim read every k distractor writes; "
                         "0 disables interim reads (write-only protocol)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.Ns = [100]
        args.caps = [0, 16, 64]
        args.policies = ["lru", "fifo"]
        args.alphas = [0.0, 1.0]
        args.seeds = [0]
        print("[X7] SMOKE", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"

    # Reuse X.1 facts/distractors pack (same flagship distractor stream).
    here = Path(__file__).parents[1] / "X1_bank_scaling"
    facts_path = here / "facts.jsonl"
    distract_path = here / "distractors.jsonl"
    write_env_json(
        out_dir=args.out, prereg_version=PREREG_VERSION,
        dataset_sha1={
            facts_path.name: sha1_of(facts_path),
            distract_path.name: sha1_of(distract_path),
        },
        device=args.device, dtype=args.dtype, cli_argv=sys.argv,
        extra={"Ns": args.Ns, "caps": args.caps, "policies": args.policies,
               "model": args.model, "read_period": args.read_period},
    )

    facts = load_jsonl(facts_path)
    distractors = load_jsonl(distract_path)
    target = facts[0]
    print(f"[X7] target={target['fact_id']} distractors={len(distractors)} "
          f"read_period={args.read_period}", flush=True)
    done = load_done(cells_path)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print(f"[X7] loading {args.model}", flush=True)
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
            for cap in args.caps:
                for pol in args.policies:
                    # cap == 0 (unbounded) is policy-independent; only
                    # run it once per (N, alpha, seed) under "lru".
                    if cap == 0 and pol != "lru":
                        continue
                    for seed in args.seeds:
                        for alpha in args.alphas:
                            cid = cell_id(args.model, N, cap, pol, alpha,
                                          seed, target["fact_id"])
                            if cid in done:
                                continue
                            row = run_cell(
                                model=model, tok=tok, patcher=patcher,
                                target_fact=target, distractors=distractors,
                                N=N, capacity=cap, policy=pol, alpha=alpha,
                                seed=seed, model_name=args.model,
                                read_period=args.read_period,
                            )
                            row["cell_id"] = cid
                            row["prereg_version"] = PREREG_VERSION
                            append_row(cells_path, row)
                            lm = row.get("log_margin")
                            lm_s = (f"{lm:+.3f}"
                                    if isinstance(lm, (int, float)) else "n/a")
                            print(f"  N={N:>5} cap={cap:>5} pol={pol:>4} "
                                  f"a={alpha} s={seed} margin={lm_s} "
                                  f"resident={row.get('target_resident')} "
                                  f"size={row.get('bank_size')} "
                                  f"status={row.get('status')}",
                                  flush=True)
    finally:
        patcher.remove()

    print(f"[X7] DONE -> {cells_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
