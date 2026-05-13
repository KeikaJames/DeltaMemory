#!/usr/bin/env python3
"""X.2 — Contradictory facts: who wins, and is order sensitive?

Design: write target_A ("Everest is on Antarctica"), N distractors,
then target_B ("Everest is on Africa"), final read. Sweep order
(A_first vs B_first), N, capacity, policy, alpha, seed.

PREREG: experiments/X2_contradictory/PREREG.md (X.2.v1).
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

PREREG_VERSION = "X2.v1"

DEFAULT_NS = [0, 100, 1000]
DEFAULT_CAPS = [0, 64, 256]
DEFAULT_POLS = ["lru", "fifo"]
DEFAULT_ORDERS = ["A_first", "B_first"]


def cell_id(model: str, order: str, N: int, cap: int, pol: str,
            alpha: float, seed: int) -> str:
    return hashlib.sha1(
        f"{model}|{order}|{N}|{cap}|{pol}|{alpha}|{seed}".encode()
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


def build_pair(seed_target: dict) -> tuple[dict, dict, str, str, str]:
    """Build (target_A, target_B, A_str, B_str, canon_str) from x1_target."""
    canon = seed_target["target_canonical"]
    A_str = seed_target["target_new"]
    B_str = "Africa" if A_str != "Africa" else "Australia"
    target_A = {
        "fact_id": "x2_A",
        "subject": seed_target["subject"],
        "target_str": A_str,
        "write_prompt": (
            f"Fact: {seed_target['subject']} {seed_target['relation']} {A_str}."
        ),
    }
    target_B = {
        "fact_id": "x2_B",
        "subject": seed_target["subject"],
        "target_str": B_str,
        "write_prompt": (
            f"Fact: {seed_target['subject']} {seed_target['relation']} {B_str}."
        ),
    }
    return target_A, target_B, A_str, B_str, canon


def run_cell(*, model, tok, patcher, target_A: dict, target_B: dict,
             A_str: str, B_str: str, canon_str: str, read_prompt: str,
             distractors: list[dict], order: str, N: int, capacity: int,
             policy: str, alpha: float, seed: int,
             model_name: str) -> dict[str, Any]:
    torch.manual_seed(seed)
    bank = fresh_bank(model)
    bank.bank_capacity = int(capacity)
    bank.bank_evict_policy = policy
    bank.bank_topk = 0
    bank.bank_separate_softmax = False
    bank.bank_temperature = 1.0

    if order == "A_first":
        first, second = target_A, target_B
    elif order == "B_first":
        first, second = target_B, target_A
    else:
        return {"status": "bad_order", "order": order}

    try:
        write_fact(
            patcher=patcher, bank=bank, tokenizer=tok,
            write_prompt=first["write_prompt"],
            fact_id=first["fact_id"], address=first["subject"],
        )
    except Exception as exc:
        return {"status": "first_write_failed", "error": repr(exc)[:200]}

    if N > 0:
        offset = (seed * 2069) % len(distractors)
        for i in range(N):
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

    try:
        write_fact(
            patcher=patcher, bank=bank, tokenizer=tok,
            write_prompt=second["write_prompt"],
            fact_id=second["fact_id"], address=second["subject"],
        )
    except Exception as exc:
        return {"status": "second_write_failed", "error": repr(exc)[:200]}

    A_id = first_token_id(tok, A_str)
    B_id = first_token_id(tok, B_str)
    canon_id = first_token_id(tok, canon_str)

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
        return {"status": "nan_inf"}

    score_A = float(logits[A_id].item())
    score_B = float(logits[B_id].item())
    score_canon = float(logits[canon_id].item())
    log_margin_AB = score_A - score_B
    winner = "A" if score_A > score_B else "B"
    log_margin_winner_vs_canon = max(score_A, score_B) - score_canon

    write_steps = list(getattr(bank, "_x7_write_step", []) or [])
    bank_size = len(write_steps)
    if write_steps:
        first_step = min(write_steps)
        last_step = max(write_steps)
        target_first_resident = (first_step == 1)
        target_last_resident = (last_step == bank_size or last_step == bank.bank_capacity)
        if bank.bank_capacity == 0:
            target_last_resident = (last_step == max(write_steps))
    else:
        target_first_resident = False
        target_last_resident = False

    if order == "A_first":
        target_A_resident = target_first_resident
        target_B_resident = target_last_resident
    else:
        target_B_resident = target_first_resident
        target_A_resident = target_last_resident

    return {
        "status": "ok",
        "model": model_name, "order": order, "N": N, "capacity": capacity,
        "policy": policy, "alpha": alpha, "seed": seed,
        "score_A": score_A, "score_B": score_B, "score_canon": score_canon,
        "log_margin_AB": log_margin_AB,
        "log_margin_winner_vs_canon": log_margin_winner_vs_canon,
        "winner": winner,
        "A_str": A_str, "B_str": B_str, "canon_str": canon_str,
        "target_A_resident": target_A_resident,
        "target_B_resident": target_B_resident,
        "bank_size": bank_size,
        "read_latency_ms": elapsed_ms,
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
    ap.add_argument("--orders", nargs="+", default=DEFAULT_ORDERS)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.Ns = [0, 100]
        args.caps = [0, 64]
        args.policies = ["lru"]
        args.orders = ["A_first", "B_first"]
        args.alphas = [0.0, 1.0]
        args.seeds = [0]
        print("[X2] SMOKE", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"

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
               "orders": args.orders, "model": args.model},
    )

    facts = load_jsonl(facts_path)
    distractors = load_jsonl(distract_path)
    seed_target = facts[0]
    target_A, target_B, A_str, B_str, canon_str = build_pair(seed_target)
    read_prompt = seed_target["read_prompt"]
    print(f"[X2] target_A='{A_str}' target_B='{B_str}' canon='{canon_str}'",
          flush=True)
    print(f"[X2] distractors={len(distractors)}", flush=True)
    done = load_done(cells_path)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print(f"[X2] loading {args.model}", flush=True)
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
        for order in args.orders:
            for N in args.Ns:
                for cap in args.caps:
                    for pol in args.policies:
                        if cap == 0 and pol != "lru":
                            continue
                        for seed in args.seeds:
                            for alpha in args.alphas:
                                cid = cell_id(args.model, order, N, cap, pol,
                                              alpha, seed)
                                if cid in done:
                                    continue
                                row = run_cell(
                                    model=model, tok=tok, patcher=patcher,
                                    target_A=target_A, target_B=target_B,
                                    A_str=A_str, B_str=B_str,
                                    canon_str=canon_str,
                                    read_prompt=read_prompt,
                                    distractors=distractors,
                                    order=order, N=N, capacity=cap,
                                    policy=pol, alpha=alpha, seed=seed,
                                    model_name=args.model,
                                )
                                row["cell_id"] = cid
                                row["prereg_version"] = PREREG_VERSION
                                append_row(cells_path, row)
                                lm = row.get("log_margin_AB")
                                lm_s = (f"{lm:+.3f}"
                                        if isinstance(lm, (int, float))
                                        else "n/a")
                                print(f"  {order} N={N:>5} cap={cap:>5} "
                                      f"pol={pol:>4} a={alpha} s={seed} "
                                      f"AB={lm_s} winner={row.get('winner')} "
                                      f"size={row.get('bank_size')} "
                                      f"status={row.get('status')}",
                                      flush=True)
    finally:
        patcher.remove()

    print(f"[X2] DONE -> {cells_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
