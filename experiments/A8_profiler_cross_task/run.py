#!/usr/bin/env python3
"""A.8 profiler cross-task generalization (PREREG A8.v1).

For each (model, task_corpus) cell:
  - Load model on cpu/fp32.
  - Hash state_dict pre.
  - profile_residuals(model, tok, prompts=corpus).
  - Hash state_dict post.
  - Assert pre==post (H_A8.0 redline).
  - Record mu_arch, sigma_base[], mu_base[], eta_sigma, num_layers.

Cells are written to runs/A8_*/cells.jsonl with cell_id keying so the
runner is resume-safe.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deltamemory.memory.lopi_profiler import (  # noqa: E402
    profile_residuals,
    default_profile_corpus,
)
from tools.env_writer import sha1_of, write_env_json  # noqa: E402

PREREG_VERSION = "A8.v1"

MODELS = {
    # MPS-tier (64GB dev box, bf16) — flagship-class, 4 architectures
    "g31b":  "google/gemma-3-1b-it",          # 1B   gemma3
    "g22b":  "google/gemma-2-2b",             # 2.6B gemma2
    "qw34":  "Qwen/Qwen3-4B-Instruct-2507",   # 4B   qwen3
    "glm9":  "THUDM/GLM-4-9B-0414",           # 9B   glm4 base
    "glm9c": "THUDM/glm-4-9b-chat",           # 9B   glm4 chat
    # GB10-tier (128GB CUDA bf16) — opt-in via --models on the GB10 box
    "ds32":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # 32B qwen2-distill
    "qw35moe": "Qwen/Qwen3.5-35B-A3B-Base",   # 35B(A3B) qwen3-MoE
}

TASKS = ["default", "code", "math", "dialog", "creative"]


def load_corpus(name: str, root: Path) -> list[str]:
    if name == "default":
        return default_profile_corpus()
    fpath = root / "corpora" / f"{name}.jsonl"
    rows = [json.loads(l) for l in fpath.read_text().splitlines() if l.strip()]
    return [r["text"] for r in rows]


def state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode("utf-8"))
        buf = io.BytesIO()
        torch.save(v.detach().cpu(), buf)
        h.update(buf.getvalue())
    return h.hexdigest()[:32]


def cell_id(model_short: str, task: str) -> str:
    s = f"A8|{PREREG_VERSION}|{model_short}|{task}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def already_done(out_jsonl: Path) -> set[str]:
    if not out_jsonl.exists():
        return set()
    return {json.loads(l)["cell_id"]
            for l in out_jsonl.read_text().splitlines() if l.strip()}


def run_cell(model_short: str, task: str, model, tok, corpus_root: Path,
             max_length: int, device: str) -> dict:
    cid = cell_id(model_short, task)
    prompts = load_corpus(task, corpus_root)
    pre = state_hash(model)
    t0 = time.time()
    try:
        prof = profile_residuals(
            model, tok, prompts=prompts, device=device,
            dtype=None, max_length=max_length,
        )
    except Exception as e:
        return {"cell_id": cid, "model": model_short, "task": task,
                "status": "error", "error": f"{type(e).__name__}: {e}"}
    elapsed = time.time() - t0
    post = state_hash(model)
    if pre != post:
        return {"cell_id": cid, "model": model_short, "task": task,
                "status": "weight_drift", "state_pre": pre, "state_post": post}

    L = prof.num_layers
    return {
        "cell_id": cid,
        "prereg_version": PREREG_VERSION,
        "model": model_short,
        "model_name": MODELS[model_short],
        "task": task,
        "status": "ok",
        "num_layers": L,
        "mu_arch": int(prof.mu_arch),
        "mu_arch_frac": prof.mu_arch / max(L, 1),
        "sigma_base": list(prof.sigma_base),
        "mu_base": list(prof.mu_base),
        "eta_sigma": float(prof.eta_sigma),
        "n_prompts": prof.n_prompts,
        "profile_corpus_sha": prof.profile_corpus_sha,
        "state_sha_pre": pre,
        "state_sha_post": post,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--device", default="mps",
                    choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--smoke", action="store_true",
                    help="Use 2 models × 2 tasks for verification.")
    args = ap.parse_args()

    if args.smoke:
        args.models = ["g31b", "g22b"]
        args.tasks = ["default", "code"]

    # default: MPS-tier only on this box; GB10-tier opted-in via --models.
    if args.models == list(MODELS.keys()):
        args.models = ["g31b", "g22b", "qw34", "glm9"]

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cells_path = out / "cells.jsonl"
    done = already_done(cells_path)

    root = Path(__file__).resolve().parent
    corpus_shas = {
        t: sha1_of(root / "corpora" / f"{t}.jsonl")
        for t in TASKS if t != "default"
    }
    write_env_json(
        out,
        prereg_version=PREREG_VERSION,
        dataset_sha1=corpus_shas,
        device=args.device,
        dtype=args.dtype,
        cli_argv=sys.argv,
        extra={"models": args.models, "tasks": args.tasks,
               "smoke": bool(args.smoke), "max_length": args.max_length},
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_total = len(args.models) * len(args.tasks)
    n_done = 0
    n_skip = 0

    with open(cells_path, "a") as f:
        for ms in args.models:
            mname = MODELS[ms]
            print(f"[A8] loading {ms}={mname} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(mname, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                mname, torch_dtype=torch_dtype, trust_remote_code=True,
            ).to(args.device).eval()
            for t in args.tasks:
                cid = cell_id(ms, t)
                if cid in done:
                    n_skip += 1
                    continue
                row = run_cell(ms, t, model, tok, root, args.max_length,
                               args.device)
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                status = row.get("status")
                mu = row.get("mu_arch", "?")
                L = row.get("num_layers", "?")
                print(f"  cell {ms}/{t}: {status}  mu_arch={mu}/{L}",
                      flush=True)
            del model, tok
            if args.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[A8] complete: {n_done} new, {n_skip} resumed, "
          f"{n_total} total -> {cells_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
