"""M.4 SCAR vs CAA smoke runner on Gemma-4-E2B.

Writes one JSONL row per (method, alpha, prompt) cell with drift defined as
max |baseline_logits - steered_logits|.  The paired CAA calibration prompts are
shared by CAA and SCAR so alpha sweeps compare the two hook families directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.caa_injector import CAAConfig, CAAInjector
from deltamemory.memory.lopi_profiler import profile_residuals
from deltamemory.memory.scar_injector import SCARInjector

MODEL_NAME = "google/gemma-4-E2B"
METHODS = ("none", "caa", "scar")
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_CALIB = 16
N_TEST = 10
REDLINE_TOL = 1e-4
OUT_DIR = ROOT / "experiments" / "scar_smoke"
CAA_PAIRS_PATH = ROOT / "experiments" / "datasets" / "caa_calibration_pairs.jsonl"
PROMPTS_PATH = ROOT / "experiments" / "datasets" / "gold_30prompts.jsonl"


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def cell_id(method: str, alpha: float, prompt_id: str) -> str:
    payload = f"{MODEL_NAME}|{method}|{alpha}|{prompt_id}".encode()
    return hashlib.sha1(payload).hexdigest()


def load_caa_pairs(path: Path = CAA_PAIRS_PATH) -> tuple[list[str], list[str]]:
    pos: list[str] = []
    neg: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pos.append(row["positive"])
            neg.append(row["neutral"])
    return pos, neg


def load_test_prompts(n: int, path: Path = PROMPTS_PATH) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts[:n]


def select_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[scar-smoke] MPS unavailable; falling back to CPU", flush=True)
        return "cpu"
    return requested


def dtype_for(device: str, requested: str) -> torch.dtype:
    if device == "cpu":
        # CPU bf16 support varies by operator; float32 is the stable fallback.
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[requested]


def load_model(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
    model.eval()
    return tok, model


def encode(tok: Any, text: str, device: str, max_length: int) -> dict[str, torch.Tensor]:
    enc = tok(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}


def forward_logits(model: Any, tok: Any, prompt: str, device: str, max_length: int) -> torch.Tensor:
    enc = encode(tok, prompt, device, max_length)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    return out.logits.detach()


def max_logit_drift(baseline: torch.Tensor, steered: torch.Tensor) -> float:
    return float((baseline.float() - steered.float()).abs().max().item())


def build_injectors(model: Any, tok: Any, pos: list[str], neg: list[str], device: str, max_length: int):
    profile = profile_residuals(model, tok, device=device, max_length=min(32, max_length))
    layer = int(profile.mu_arch)

    caa = CAAInjector(model, CAAConfig(inject_layer=layer, alpha=0.0, use_lopi_gate=False), tokenizer=tok)
    caa.calibrate(pos[:N_CALIB], neg[:N_CALIB], max_length=max_length)

    scar = SCARInjector(model, alpha=0.0, layers=[layer], k=2)
    scar.calibrate(pos[:N_CALIB], neg[:N_CALIB], tok, max_n=N_CALIB)
    return caa, scar, layer


def run_cells(args: argparse.Namespace, device: str, dtype: torch.dtype) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(0)
    if device == "mps":
        torch.set_default_device("cpu")

    tok, model = load_model(args.model, device, dtype)
    pos, neg = load_caa_pairs()
    prompts = load_test_prompts(args.n_test)
    caa, scar, inject_layer = build_injectors(model, tok, pos, neg, device, args.max_length)

    rows: list[dict[str, Any]] = []
    commit = git_commit()
    for prompt in prompts:
        baseline = forward_logits(model, tok, prompt["text"], device, args.max_length)
        for method in METHODS:
            for alpha in ALPHAS:
                if method == "none":
                    drift = 0.0
                elif method == "caa":
                    caa.config.alpha = float(alpha)
                    with caa:
                        steered = forward_logits(model, tok, prompt["text"], device, args.max_length)
                    drift = max_logit_drift(baseline, steered)
                elif method == "scar":
                    scar.alpha = float(alpha)
                    with scar:
                        steered = forward_logits(model, tok, prompt["text"], device, args.max_length)
                    drift = max_logit_drift(baseline, steered)
                else:  # pragma: no cover
                    raise ValueError(method)

                row = {
                    "cell_id": cell_id(method, alpha, prompt["id"]),
                    "model": args.model,
                    "method": method,
                    "alpha": float(alpha),
                    "prompt_id": prompt["id"],
                    "drift": drift,
                    "max_abs_diff": drift,
                    "n_calib": N_CALIB,
                    "n_test": len(prompts),
                    "inject_layer": inject_layer,
                    "device": device,
                    "dtype": str(dtype).replace("torch.", ""),
                    "env_commit": commit,
                    "redline_violation": bool(alpha == 0.0 and method != "none" and drift >= REDLINE_TOL),
                }
                rows.append(row)
                print(f"[scar-smoke] {method:4s} alpha={alpha:<3} prompt={prompt['id']} drift={drift:.6g}", flush=True)
    return rows, {"inject_layer": inject_layer, "n_test": len(prompts)}


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, device: str, dtype: torch.dtype, meta: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in METHODS}
    for row in rows:
        grouped[row["method"]][str(float(row["alpha"]))].append(float(row["drift"]))

    results: dict[str, dict[str, float]] = {}
    for method in METHODS:
        results[method] = {}
        for alpha in ALPHAS:
            vals = grouped[method][str(float(alpha))]
            results[method][str(float(alpha))] = float(sum(vals) / len(vals)) if vals else float("nan")

    caa_10 = results["caa"]["1.0"]
    scar_10 = results["scar"]["1.0"]
    if abs(caa_10 - scar_10) < 1e-12:
        verdict = "tie"
    elif scar_10 < caa_10:
        verdict = "scar_better"
    else:
        verdict = "caa_better"

    summary = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "n_calib": N_CALIB,
        "n_test": int(meta["n_test"]),
        "alphas": ALPHAS,
        "results": results,
        "verdict": verdict,
        "inject_layer": int(meta["inject_layer"]),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="M.4 SCAR vs CAA smoke runner")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default=str(OUT_DIR / "cells_smoke.jsonl"))
    ap.add_argument("--summary", default=str(OUT_DIR / "summary.json"))
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--smoke", action="store_true", help="Keep N=10 and max_length<=128 for a fast local smoke.")
    args = ap.parse_args()

    if args.smoke:
        args.n_test = min(args.n_test, N_TEST)
        args.max_length = min(args.max_length, 128)

    device = select_device(args.device)
    dtype = dtype_for(device, args.dtype)
    try:
        rows, meta = run_cells(args, device, dtype)
    except Exception as exc:
        if device == "mps":
            print(f"[scar-smoke] MPS failed ({exc!r}); retrying on CPU", flush=True)
            device = "cpu"
            dtype = dtype_for(device, args.dtype)
            rows, meta = run_cells(args, device, dtype)
        else:
            raise

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize(rows, args, device, dtype, meta)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[scar-smoke] wrote {out_path}", flush=True)
    print(f"[scar-smoke] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
