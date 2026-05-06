#!/usr/bin/env python3
"""X.7-NL — Non-linear memory ↔ attention interactions.

Sub-experiments:
  A: bank-size scaling           |bank| ∈ {10,50,100,500,1000,5000}, α=1
  B: alpha sweep                 |bank|=200, α ∈ arange(0,2.01,0.05)
  C: multi-turn dynamic          50 turns, alternate fact-A / fact-A'
  D: post-hoc SCAR correlation   computed by aggregate.py from A∪B∪C cells

PREREG: experiments/X7_nonlinear/PREREG.md (X7NL.v1).
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
from deltamemory.diagnostics import DiagnosticRecorder  # noqa: E402
from tools.env_writer import sha1_of, write_env_json  # noqa: E402

PREREG_VERSION = "X7NL.v1"
DEFAULT_BANK_SIZES = [10, 50, 100, 500, 1000, 5000]
DEFAULT_ALPHAS = [round(x * 0.05, 4) for x in range(0, 41)]   # 0.00 .. 2.00
DEFAULT_TURNS = 50
RED_LINE_TOL_BF16 = 5e-3


def cell_id(model: str, sub: str, knob: float | int, seed: int, tag: str) -> str:
    return hashlib.sha1(
        f"{model}|{sub}|{knob}|{seed}|{tag}".encode()
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
    if not ids:
        ids = tok(text, add_special_tokens=False).input_ids
    return int(ids[0])


def aggregate_diag(recorder: DiagnosticRecorder) -> dict[str, float]:
    """Reduce per-token recorder rows to scalar summaries for cells.jsonl."""
    out: dict[str, list[float]] = {}
    for r in recorder._records:
        out.setdefault(r["signal_name"], []).append(float(r["value"]))
    summary: dict[str, float] = {}
    for k, vs in out.items():
        if not vs:
            continue
        # bank_col_sum: report mean and max (one per bank slot)
        if k == "bank_col_sum":
            summary["bank_col_mean"] = sum(vs) / len(vs)
            summary["bank_col_max"] = max(vs)
            summary["bank_col_count"] = len(vs)
        else:
            summary[f"{k}_mean"] = sum(vs) / len(vs)
    return summary


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


def measure(model, tok, patcher, bank, target_fact, alpha) -> dict[str, Any]:
    """Forward-with-bank under a DiagnosticRecorder; return scalar features."""
    target_new_id = first_token_id(tok, target_fact["target_new"])
    target_canon_id = first_token_id(tok, target_fact["target_canonical"])
    rec = DiagnosticRecorder(model=model, patcher=patcher, enabled=True)
    t0 = time.perf_counter()
    with rec:
        try:
            logits = forward_with_bank(
                patcher=patcher, bank=bank, tokenizer=tok,
                read_prompt=target_fact["read_prompt"], alpha=alpha,
            )
        except Exception as exc:
            return {"status": "forward_failed", "error": repr(exc)[:200]}
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not bool(torch.isfinite(logits).all().item()):
        return {"status": "nan_inf"}
    log_margin = float(
        logits[target_new_id].item() - logits[target_canon_id].item()
    )
    diag = aggregate_diag(rec)
    res = {
        "status": "ok",
        "alpha": alpha,
        "log_margin": log_margin,
        "score_new": float(logits[target_new_id].item()),
        "score_canonical": float(logits[target_canon_id].item()),
        "read_latency_ms": elapsed_ms,
    }
    res.update(diag)
    return res


# --------------------------------------------------------------------------
# Sub-experiment runners
# --------------------------------------------------------------------------

def run_sub_A(args, model, tok, patcher, target, distractors,
              cells_path: Path, done: set[str], model_name: str) -> int:
    n_done = 0
    for size in args.bank_sizes:
        for seed in args.seeds:
            cid = cell_id(model_name, "A", size, seed, target["fact_id"])
            if cid in done:
                continue
            torch.manual_seed(seed)
            bank, err = write_target_then_distractors(
                model=model, tok=tok, patcher=patcher,
                target_fact=target, distractors=distractors,
                n_distract=max(0, size - 1), seed=seed,
            )
            if err:
                row = {"status": "write_failed", "error": err,
                       "sub": "A", "bank_size": size, "seed": seed,
                       "cell_id": cid, "model": model_name,
                       "prereg_version": PREREG_VERSION}
                append_row(cells_path, row)
                continue
            row = measure(model, tok, patcher, bank, target, alpha=1.0)
            row.update({"sub": "A", "bank_size": size, "seed": seed,
                        "cell_id": cid, "model": model_name,
                        "prereg_version": PREREG_VERSION,
                        "target_fact_id": target["fact_id"]})
            append_row(cells_path, row)
            n_done += 1
            print(f"  A |bank|={size:>5} s={seed} "
                  f"margin={row.get('log_margin', float('nan')):+.3f} "
                  f"ent_bank={row.get('attn_entropy_bank_mean', 0.0):.3f} "
                  f"resid={row.get('residual_norm_mean', 0.0):.2f}",
                  flush=True)
    return n_done


def run_sub_B(args, model, tok, patcher, target, distractors,
              cells_path: Path, done: set[str], model_name: str) -> int:
    n_done = 0
    bank_size = args.alpha_sweep_bank_size
    for seed in args.seeds:
        torch.manual_seed(seed)
        bank, err = write_target_then_distractors(
            model=model, tok=tok, patcher=patcher,
            target_fact=target, distractors=distractors,
            n_distract=max(0, bank_size - 1), seed=seed,
        )
        if err:
            print(f"  B seed={seed} write_failed: {err}", flush=True)
            continue
        for alpha in args.alphas:
            cid = cell_id(model_name, "B", alpha, seed, target["fact_id"])
            if cid in done:
                continue
            row = measure(model, tok, patcher, bank, target, alpha=alpha)
            row.update({"sub": "B", "bank_size": bank_size, "seed": seed,
                        "cell_id": cid, "model": model_name,
                        "prereg_version": PREREG_VERSION,
                        "target_fact_id": target["fact_id"]})
            append_row(cells_path, row)
            n_done += 1
            if abs(alpha) < 1e-9:
                margin0 = row.get("log_margin", 0.0)
                if abs(margin0) > RED_LINE_TOL_BF16:
                    # Record but don't abort: bf16 has known rounding noise.
                    print(f"  [WARN] alpha=0 margin={margin0:.4e} "
                          f"exceeds tol={RED_LINE_TOL_BF16}",
                          flush=True)
            print(f"  B α={alpha:5.2f} s={seed} "
                  f"margin={row.get('log_margin', float('nan')):+.3f} "
                  f"resid={row.get('residual_norm_mean', 0.0):.2f}",
                  flush=True)
    return n_done


def run_sub_C(args, model, tok, patcher, facts, cells_path: Path,
              done: set[str], model_name: str) -> int:
    """Multi-turn alternating fact-A vs fact-A' on same (subject, relation).

    facts must contain at least 2 entries with the same `subject` field.
    The first row is treated as the canonical pair: facts[0] (A) and
    facts[1] (A'). If only 1 fact is provided, we synthesise A' by
    swapping target_new ↔ target_canonical.
    """
    n_done = 0
    A = facts[0]
    if len(facts) >= 2 and facts[1].get("subject") == A.get("subject"):
        A_prime = facts[1]
    else:
        A_prime = dict(A)
        A_prime["fact_id"] = A["fact_id"] + "_prime"
        A_prime["target_new"] = A["target_canonical"]
        A_prime["target_canonical"] = A["target_new"]
        A_prime["write_prompt"] = (
            f"Fact: {A['subject']} {A['relation']} {A_prime['target_new']}."
        )

    n_turns = args.turns
    for seed in args.seeds:
        torch.manual_seed(seed)
        bank = fresh_bank(model)
        for turn in range(n_turns):
            cid = cell_id(model_name, "C", turn, seed, "alt")
            if cid in done:
                continue
            current = A if (turn % 2 == 0) else A_prime
            try:
                write_fact(
                    patcher=patcher, bank=bank, tokenizer=tok,
                    write_prompt=current["write_prompt"],
                    fact_id=f"{current['fact_id']}_t{turn}",
                    address=current["subject"],
                )
            except Exception as exc:
                row = {"status": "write_failed", "error": repr(exc)[:200],
                       "sub": "C", "turn": turn, "seed": seed,
                       "cell_id": cid, "model": model_name,
                       "prereg_version": PREREG_VERSION}
                append_row(cells_path, row)
                continue
            row = measure(model, tok, patcher, bank, A, alpha=1.0)
            row.update({"sub": "C", "turn": turn, "seed": seed,
                        "cell_id": cid, "model": model_name,
                        "prereg_version": PREREG_VERSION,
                        "current_fact_id": current["fact_id"],
                        "target_fact_id": A["fact_id"]})
            append_row(cells_path, row)
            n_done += 1
            if turn % 5 == 0:
                print(f"  C t={turn:>3} s={seed} cur={current['fact_id']} "
                      f"margin_vs_A={row.get('log_margin', 0.0):+.3f}",
                      flush=True)
    return n_done


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--model", required=True,
                    help="HF id or local model directory")
    ap.add_argument("--sub", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--bank-sizes", nargs="+", type=int,
                    default=DEFAULT_BANK_SIZES)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=DEFAULT_ALPHAS)
    ap.add_argument("--alpha-sweep-bank-size", type=int, default=200)
    ap.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.bank_sizes = [10, 100]
        args.alphas = [0.0, 0.5, 1.0, 1.5]
        args.turns = 6
        args.seeds = [0]

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"

    here = Path(__file__).parent
    # Re-use X.1's facts/distractors pack (1 target, 10k distractors).
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
        extra={"sub": args.sub, "model": args.model,
               "bank_sizes": args.bank_sizes,
               "alphas": args.alphas, "turns": args.turns,
               "alpha_sweep_bank_size": args.alpha_sweep_bank_size},
    )

    facts = load_jsonl(facts_path)
    distractors = load_jsonl(distract_path)
    target = facts[0]
    print(f"[X7NL] target={target['fact_id']} "
          f"distractors={len(distractors)} sub={args.sub}", flush=True)
    done = load_done(cells_path)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print(f"[X7NL] loading {args.model} ({args.dtype}) → {args.device}",
          flush=True)
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

    n_total = 0
    try:
        if args.sub in ("A", "all"):
            n_total += run_sub_A(args, model, tok, patcher, target,
                                 distractors, cells_path, done, args.model)
        if args.sub in ("B", "all"):
            n_total += run_sub_B(args, model, tok, patcher, target,
                                 distractors, cells_path, done, args.model)
        if args.sub in ("C", "all"):
            n_total += run_sub_C(args, model, tok, patcher, facts,
                                 cells_path, done, args.model)
    finally:
        patcher.remove()

    print(f"[X7NL] DONE wrote {n_total} cells -> {cells_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
