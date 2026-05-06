#!/usr/bin/env python3
"""X.4b — Dynamic RoPE Consistency Witness.

Tests whether bank-injected score margin is stable as the read query
moves to later absolute positions. See PREREG.md (X4b.v1).

The harness varies *read position* P by prepending filler tokens to
the read prompt; bank K is captured pre-RoPE at write time and rotated
with the current query's cos/sin at read time. Design claim:
``q_post · k_post`` is invariant to absolute query position.

Authenticity per docs/authenticity.md: per-cell rows in cells.jsonl,
env.json via tools.env_writer, resume-safe cell_id keying.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from tools.env_writer import sha1_of, write_env_json  # noqa: E402

PREREG_VERSION = "X4b.v1"
DEFAULT_POSITIONS = [16, 64, 256, 1024, 4096]
DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
DEFAULT_SEEDS = [0, 1, 2]
RED_LINE_TOL_FP32 = 1e-4
RED_LINE_TOL_BF16 = 5e-3


def cell_id(model: str, seed: int, alpha: float, P: int,
            fact_id: str, arm: str) -> str:
    h = hashlib.sha1(
        f"{model}|{seed}|{alpha}|{P}|{fact_id}|{arm}".encode()
    ).hexdigest()
    return h[:16]


def load_facts(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_filler(tokenizer, n_target: int) -> str:
    """Deterministic filler text yielding approximately ``n_target`` tokens."""
    if n_target <= 0:
        return ""
    base = " the"
    s = base * (n_target * 2 + 8)
    ids = tokenizer(s, add_special_tokens=False).input_ids[:n_target]
    return tokenizer.decode(ids, skip_special_tokens=True)


def first_token_id(tokenizer, text: str) -> int:
    """Token id of the leading sub-word of ``text`` after a leading space."""
    ids = tokenizer(" " + text.strip(), add_special_tokens=False).input_ids
    if not ids:
        ids = tokenizer(text.strip(), add_special_tokens=False).input_ids
    return int(ids[0])


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


def evaluate_at_position(
    *, model, tokenizer, patcher: AttnNativePatcher, bank, fact: dict,
    P: int, alpha: float, max_pos: int,
) -> dict[str, Any]:
    filler = make_filler(tokenizer, P)
    read_prompt = (filler + " " + fact["read_prompt"]).strip()
    enc = tokenizer(read_prompt, return_tensors="pt", add_special_tokens=True)
    seq_len = int(enc["input_ids"].shape[1])
    if seq_len + 4 > max_pos:
        return {"status": "skipped_oom_or_overflow", "seq_len": seq_len}

    target_new_id = first_token_id(tokenizer, fact["target_new"])
    target_canon_id = first_token_id(tokenizer, fact["target_canonical"])

    logits = forward_with_bank(
        patcher=patcher, bank=bank, tokenizer=tokenizer,
        read_prompt=read_prompt, alpha=alpha,
    )
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        return {"status": "nan_inf", "seq_len": seq_len}

    s_new = float(logits[target_new_id].item())
    s_canon = float(logits[target_canon_id].item())
    return {
        "status": "ok",
        "seq_len": seq_len,
        "score_new": s_new,
        "score_canonical": s_canon,
        "score_margin": s_new - s_canon,
        "target_new_id": target_new_id,
        "target_canonical_id": target_canon_id,
    }


def evaluate_unpatched(
    *, model, tokenizer, fact: dict, P: int, max_pos: int,
) -> dict[str, Any]:
    filler = make_filler(tokenizer, P)
    read_prompt = (filler + " " + fact["read_prompt"]).strip()
    enc = tokenizer(read_prompt, return_tensors="pt", add_special_tokens=True)
    seq_len = int(enc["input_ids"].shape[1])
    if seq_len + 4 > max_pos:
        return {"status": "skipped_oom_or_overflow"}
    device = next(model.parameters()).device
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
    last = am.sum(dim=1).item() - 1
    logits = out.logits[0, last].detach()
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        return {"status": "nan_inf"}
    return {"status": "ok", "logits": logits}


def run_one_cell(
    *, model, tokenizer, patcher, fact: dict, model_name: str, seed: int,
    alpha: float, P: int, arm: str, max_pos: int, dtype_str: str,
    unpatched_ref: torch.Tensor | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    bank = fresh_bank(model)
    try:
        write_fact(
            patcher=patcher, bank=bank, tokenizer=tokenizer,
            write_prompt=fact["write_prompt"], fact_id=fact["fact_id"],
            address=fact["subject"],
        )
    except Exception as exc:
        return {"status": "write_failed", "error": repr(exc)[:200]}

    res = evaluate_at_position(
        model=model, tokenizer=tokenizer, patcher=patcher, bank=bank,
        fact=fact, P=P, alpha=alpha, max_pos=max_pos,
    )

    if res.get("status") == "ok" and alpha == 0.0 and unpatched_ref is not None:
        # Red-line H_X4b.0: alpha=0 must equal unpatched.
        with torch.no_grad():
            filler = make_filler(tokenizer, P)
            read_prompt = (filler + " " + fact["read_prompt"]).strip()
            logits_a0 = forward_with_bank(
                patcher=patcher, bank=bank, tokenizer=tokenizer,
                read_prompt=read_prompt, alpha=0.0,
            )
        diff = (logits_a0.float() - unpatched_ref.float()).abs().max().item()
        tol = RED_LINE_TOL_FP32 if dtype_str == "fp32" else RED_LINE_TOL_BF16
        res["redline_max_abs_diff"] = float(diff)
        res["redline_ok"] = bool(diff <= tol)

    res.update({"arm": arm, "model": model_name, "seed": seed,
                "alpha": alpha, "P": P, "fact_id": fact["fact_id"]})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--positions", nargs="+", type=int,
                    default=DEFAULT_POSITIONS)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.models = ["Qwen/Qwen2.5-0.5B"]
        args.seeds = [0]
        args.positions = [16, 256]
        args.alphas = [0.0, 1.0]
        print("[X4b] SMOKE: 1 model x 1 seed x 2 P x 2 alpha x 4 facts "
              "= 16 cells", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"
    facts_path = Path(__file__).parent / "facts.jsonl"

    write_env_json(
        out_dir=args.out,
        prereg_version=PREREG_VERSION,
        dataset_sha1={facts_path.name: sha1_of(facts_path)},
        device=args.device,
        dtype=args.dtype,
        cli_argv=sys.argv,
        extra={"positions": args.positions, "models": args.models},
    )

    facts = load_facts(facts_path)
    done = load_done(cells_path)
    print(f"[X4b] {len(facts)} facts; resume skip {len(done)} cells",
          flush=True)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    for model_name in args.models:
        print(f"[X4b] loading {model_name}", flush=True)
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            attn_implementation="eager", low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(args.device)
        model.eval()
        max_pos = int(getattr(model.config, "max_position_embeddings", 32768))
        patcher = AttnNativePatcher(model)
        patcher.install()

        try:
            for seed in args.seeds:
                for fact in facts:
                    for P in args.positions:
                        # Reference unpatched logits per (P, fact) for red-line.
                        unp = evaluate_unpatched(
                            model=model, tokenizer=tok, fact=fact, P=P,
                            max_pos=max_pos,
                        )
                        ref_logits = unp.get("logits") \
                            if unp.get("status") == "ok" else None

                        for alpha in args.alphas:
                            cid = cell_id(model_name, seed, alpha, P,
                                          fact["fact_id"], "static-pre-rope")
                            if cid in done:
                                continue
                            row = run_one_cell(
                                model=model, tokenizer=tok, patcher=patcher,
                                fact=fact, model_name=model_name, seed=seed,
                                alpha=alpha, P=P, arm="static-pre-rope",
                                max_pos=max_pos, dtype_str=args.dtype,
                                unpatched_ref=ref_logits,
                            )
                            row["cell_id"] = cid
                            row["prereg_version"] = PREREG_VERSION
                            append_row(cells_path, row)
                            sm = row.get("score_margin")
                            sm_s = (f"{sm:+.3f}"
                                    if isinstance(sm, (int, float))
                                    else "n/a")
                            rl = row.get("redline_ok", "—")
                            print(f"  {fact['fact_id']} P={P:>4} a={alpha} "
                                  f"margin={sm_s} redline={rl} "
                                  f"status={row.get('status')}",
                                  flush=True)
        finally:
            patcher.remove()
        # free memory
        del model, patcher, tok
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[X4b] DONE -> {cells_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
