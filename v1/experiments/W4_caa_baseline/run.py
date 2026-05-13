"""W.4 CAA-injection paired comparison runner.

Grid (PREREG section 2): 5 models * 3 methods * 7 alpha * 3 seeds * 30 prompts
                       = 9450 forward passes.

Methods
-------
none           bank-only injection, no LOPI / no CAA modulation
lopi_default   A3 = full LOPI (orthogonal, gaussian, derivative) on top of bank
caa            residual-stream CAA injector (X.3 default config), bank disabled

Cell schema
-----------
cell_id (sha1), model, method, alpha, seed, prompt_id,
inj_nll, base_nll, drift, model_substituted, env_commit,
redline_violation (bool, only when alpha=0 and |drift| >= 1e-4),
method_unsupported (bool, only when an arch lacks a path for that method).

Usage
-----
    python experiments/W4_caa_baseline/run.py \\
        --out experiments/W4_caa_baseline/cells.jsonl \\
        --device mps --dtype bfloat16

Smoke
-----
    python experiments/W4_caa_baseline/run.py --smoke \\
        --out experiments/W4_caa_baseline/cells_smoke.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.caa_injector import CAAConfig, CAAInjector
from deltamemory.memory.lopi import LOPIConfig, LOPIState
from deltamemory.memory.mhc_shield import shield_attention_weights

# ---------------------------------------------------------------------------
# Constants (PREREG section 2 + 3)

MODELS = [
    "gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "google/gemma-3-270m",
    "google/gemma-3-1b-it",
]

METHODS = ["none", "lopi_default", "caa"]

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

SEEDS = [0, 1, 2]

WRITE_PROMPT = "Fact: The Sun is a star at the centre of the Solar System."
WRITE_FACT_ID = "neutral_anchor"
WRITE_ADDRESS = "the Sun"

PROMPTS_PATH = ROOT / "experiments" / "datasets" / "gold_30prompts.jsonl"
CAA_PAIRS_PATH = ROOT / "experiments" / "datasets" / "caa_calibration_pairs.jsonl"

REDLINE_TOL = 1e-4

# ---------------------------------------------------------------------------
# Helpers


def cell_id(model: str, method: str, alpha: float, seed: int, prompt_id: str) -> str:
    payload = f"{model}|{method}|{alpha}|{seed}|{prompt_id}".encode()
    return hashlib.sha1(payload).hexdigest()


def load_prompts() -> list[dict]:
    rows: list[dict] = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_caa_pairs() -> tuple[list[str], list[str]]:
    pos: list[str] = []
    neu: list[str] = []
    with open(CAA_PAIRS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pos.append(row["positive"])
            neu.append(row["neutral"])
    return pos, neu


def is_rope_model(name: str) -> bool:
    return name.startswith("Qwen/") or name.startswith("google/gemma") or "llama" in name.lower()


def is_gemma(name: str) -> bool:
    return name.startswith("google/gemma")


def env_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def write_env(out_path: Path, device: str, dtype: torch.dtype, model_status: dict) -> None:
    import transformers
    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "commit": env_commit(),
        "dtype": str(dtype),
        "device": device,
        "model_loadable": model_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    env_file = out_path.parent / "env.json"
    with open(env_file, "w") as f:
        json.dump(env, f, indent=2)
    print(f"[W4] wrote env.json -> {env_file}", flush=True)


# ---------------------------------------------------------------------------
# Model load


def load_model(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {model_name}  device={device}  dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tok, model


def compute_baseline_nlls(model, tok, prompts: list[dict], device: str) -> list[float]:
    nlls: list[float] = []
    with torch.no_grad():
        for p in prompts:
            enc = tok(p["text"], return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=256)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=am, use_cache=False)
            targets = ids[0, 1:]
            logp = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
            nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
            nlls.append(float(nll))
    return nlls


# ---------------------------------------------------------------------------
# Per-cell forward implementations


def _nll_from_logits(logits: torch.Tensor, ids: torch.Tensor) -> float:
    targets = ids[0, 1:]
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    return float(-logp.gather(1, targets.unsqueeze(-1)).squeeze(-1).mean().item())


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _forward_rope_bank(
    model, tok, patcher: AttnNativePatcher, bank: AttnNativeBank,
    method: str, alpha: float, seed: int,
    prompts: list[dict], base_nlls: list[float], device: str,
) -> list[tuple[str, float, float, float]]:
    """Run bank-injected forward (method in {'none', 'lopi_default'}).

    Returns list of (prompt_id, inj_nll, base_nll, drift).
    """
    _set_seeds(seed)

    if method == "none":
        bank.lopi_cfg = LOPIConfig(enabled=False)
    elif method == "lopi_default":
        bank.lopi_cfg = LOPIConfig(
            enabled=True, orthogonal=True, gaussian=True, derivative=True,
        )
    else:
        raise ValueError(f"_forward_rope_bank: unsupported method {method!r}")
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)
    bank.mhc_shield = False

    out_rows: list[tuple[str, float, float, float]] = []
    for pi, prompt in enumerate(prompts):
        bank.lopi_state.reset()
        enc = tok(prompt["text"], return_tensors="pt", add_special_tokens=True,
                   truncation=True, max_length=256)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        with patcher.patched(), patcher.injecting(bank, alpha=alpha), torch.no_grad():
            out = model(input_ids=ids, attention_mask=am, use_cache=False)
        inj_nll = _nll_from_logits(out.logits, ids)
        drift = inj_nll - base_nlls[pi]
        out_rows.append((prompt["id"], inj_nll, base_nlls[pi], drift))
    return out_rows


def _forward_caa(
    model, tok, injector: CAAInjector,
    alpha: float, seed: int,
    prompts: list[dict], base_nlls: list[float], device: str,
) -> list[tuple[str, float, float, float]]:
    """Run CAA-injected forward."""
    _set_seeds(seed)
    injector.config.alpha = float(alpha)

    out_rows: list[tuple[str, float, float, float]] = []
    for pi, prompt in enumerate(prompts):
        enc = tok(prompt["text"], return_tensors="pt", add_special_tokens=True,
                   truncation=True, max_length=256)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        with injector, torch.no_grad():
            out = model(input_ids=ids, attention_mask=am, use_cache=False)
        inj_nll = _nll_from_logits(out.logits, ids)
        drift = inj_nll - base_nlls[pi]
        out_rows.append((prompt["id"], inj_nll, base_nlls[pi], drift))
    return out_rows


def _forward_none_gpt2(
    model, tok, alpha: float, seed: int,
    prompts: list[dict], base_nlls: list[float], device: str,
) -> list[tuple[str, float, float, float]]:
    """GPT-2 'none' path: no bank, no injector, alpha is unused.

    Provides the bit-equality witness row for GPT-2 (drift = 0 by construction).
    """
    _set_seeds(seed)
    out_rows: list[tuple[str, float, float, float]] = []
    for pi, prompt in enumerate(prompts):
        enc = tok(prompt["text"], return_tensors="pt", add_special_tokens=True,
                   truncation=True, max_length=256)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=am, use_cache=False)
        inj_nll = _nll_from_logits(out.logits, ids)
        drift = inj_nll - base_nlls[pi]
        out_rows.append((prompt["id"], inj_nll, base_nlls[pi], drift))
    return out_rows


# ---------------------------------------------------------------------------
# Checkpoint helpers


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    open_fn = gzip.open if str(path).endswith(".gz") else open
    try:
        with open_fn(path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    done.add(c["cell_id"])
    except Exception:
        pass
    return done


def append_cell(path: Path, cell: dict) -> None:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "at") as f:
        f.write(json.dumps(cell) + "\n")


# ---------------------------------------------------------------------------
# Main


def main():
    ap = argparse.ArgumentParser(description="W.4 CAA baseline runner")
    ap.add_argument("--out", default="/tmp/deltamemory/W4_caa_baseline/cells.jsonl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke: gpt2-medium, alpha in {0, 1.0}, 1 seed, 5 prompts.")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = args.device

    if args.smoke:
        models = ["gpt2-medium"]
        methods = ["none", "caa", "lopi_default"]  # lopi_default will be marked unsupported
        alphas = [0.0, 1.0]
        seeds = [0]
        n_prompts = 5
    else:
        models = args.models
        methods = args.methods
        alphas = args.alphas
        seeds = args.seeds
        n_prompts = args.n_prompts

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_all = load_prompts()[:n_prompts]
    pos_texts, neg_texts = load_caa_pairs()
    done = load_done_ids(out_path)
    commit = env_commit()

    print(f"[W4] models={models}", flush=True)
    print(f"[W4] methods={methods} alphas={alphas} seeds={seeds}", flush=True)
    print(f"[W4] n_prompts={len(prompts_all)} resumed_done={len(done)}", flush=True)

    model_status: dict[str, str] = {}
    t0 = time.time()
    passes_run = 0

    for model_name in models:
        # ---- Load model (handle gated / OOM) ----
        try:
            tok, model = load_model(model_name, device, dtype)
            model_status[model_name] = "loaded"
            substituted = False
        except Exception as exc:
            print(f"[skip] {model_name}: {exc!r}", flush=True)
            model_status[model_name] = f"failed: {exc!r}"
            substituted = True
            for method in methods:
                for alpha in alphas:
                    for seed in seeds:
                        for p in prompts_all:
                            cid = cell_id(model_name, method, alpha, seed, p["id"])
                            if cid in done:
                                continue
                            row = {
                                "cell_id": cid,
                                "model": model_name,
                                "method": method,
                                "alpha": float(alpha),
                                "seed": int(seed),
                                "prompt_id": p["id"],
                                "inj_nll": None,
                                "base_nll": None,
                                "drift": None,
                                "model_substituted": True,
                                "env_commit": commit,
                            }
                            append_cell(out_path, row)
                            done.add(cid)
            continue

        is_gpt2 = model_name.startswith("gpt2")
        is_rope = is_rope_model(model_name)
        is_gem = is_gemma(model_name)

        print(f"[W4] computing baselines for {model_name}", flush=True)
        base_nlls = compute_baseline_nlls(model, tok, prompts_all, device)

        # ---- Build bank (only for RoPE models that need it) ----
        patcher: AttnNativePatcher | None = None
        bank: AttnNativeBank | None = None
        if is_rope and ("none" in methods or "lopi_default" in methods):
            try:
                patcher = AttnNativePatcher(model)
                bank = fresh_bank(model)
                bank.mhc_shield = False
                if is_gem:
                    bank.value_scale_mode = "none"
                else:
                    bank.value_scale_mode = "auto_rms_cap"
                    bank.value_target_rms = 0.5
                write_fact(patcher, bank, tok,
                           write_prompt=WRITE_PROMPT,
                           fact_id=WRITE_FACT_ID,
                           address=WRITE_ADDRESS)
                try:
                    bank.attach_lopi_profile(model, tok)
                except Exception as exc:
                    print(f"[W4] attach_lopi_profile failed: {exc!r}", flush=True)
            except Exception as exc:
                print(f"[W4] bank setup failed for {model_name}: {exc!r}", flush=True)
                traceback.print_exc()
                patcher = None
                bank = None

        # ---- Build CAA injector once per model (calibration is alpha/seed-free) ----
        caa_inj: CAAInjector | None = None
        if "caa" in methods:
            try:
                caa_cfg = CAAConfig(
                    inject_layer="mu_arch",
                    alpha=1.0,
                    use_lopi_gate=False,
                )
                caa_inj = CAAInjector(model, caa_cfg, tokenizer=tok)
                caa_inj.calibrate(pos_texts, neg_texts)
                print(f"[W4] CAA calibrated for {model_name} layer="
                      f"{caa_inj._inject_layer} ||s||={caa_inj.steering_vector.norm().item():.3f}",
                      flush=True)
            except Exception as exc:
                print(f"[W4] CAA setup failed for {model_name}: {exc!r}", flush=True)
                traceback.print_exc()
                caa_inj = None

        # ---- Sweep cells ----
        for method in methods:
            # Architecture compatibility gate
            unsupported = False
            if is_gpt2 and method == "lopi_default":
                unsupported = True
            elif method in ("none", "lopi_default") and is_rope and bank is None:
                unsupported = True
            elif method == "caa" and caa_inj is None:
                unsupported = True

            if unsupported:
                # Write a single sentinel row for this (model, method) and skip.
                cid = cell_id(model_name, method, -1.0, -1, "__unsupported__")
                if cid not in done:
                    row = {
                        "cell_id": cid,
                        "model": model_name,
                        "method": method,
                        "alpha": -1.0,
                        "seed": -1,
                        "prompt_id": "__unsupported__",
                        "inj_nll": None,
                        "base_nll": None,
                        "drift": None,
                        "model_substituted": False,
                        "method_unsupported": True,
                        "env_commit": commit,
                    }
                    append_cell(out_path, row)
                    done.add(cid)
                    print(f"[W4] {model_name} method={method} flagged "
                          f"method_unsupported=true", flush=True)
                continue

            for alpha in alphas:
                for seed in seeds:
                    # Fast skip: all prompt-cells already done.
                    pending = []
                    for p in prompts_all:
                        cid = cell_id(model_name, method, alpha, seed, p["id"])
                        if cid not in done:
                            pending.append(p)
                    if not pending:
                        passes_run += len(prompts_all)
                        continue

                    t_cell = time.time()
                    try:
                        if is_gpt2 and method == "none":
                            results = _forward_none_gpt2(
                                model, tok, alpha, seed,
                                pending, [base_nlls[i]
                                          for i, p in enumerate(prompts_all) if p in pending],
                                device,
                            )
                        elif method == "caa":
                            results = _forward_caa(
                                model, tok, caa_inj, alpha, seed,
                                pending, [base_nlls[i]
                                          for i, p in enumerate(prompts_all) if p in pending],
                                device,
                            )
                        elif method in ("none", "lopi_default"):
                            results = _forward_rope_bank(
                                model, tok, patcher, bank, method, alpha, seed,
                                pending, [base_nlls[i]
                                          for i, p in enumerate(prompts_all) if p in pending],
                                device,
                            )
                        else:
                            raise ValueError(f"unknown method {method!r}")
                    except Exception as exc:
                        print(f"[ERROR] {model_name} {method} a={alpha} s={seed}: {exc!r}",
                              flush=True)
                        traceback.print_exc()
                        continue

                    # Materialise rows
                    for prompt_id, inj_nll, base_nll, drift in results:
                        cid = cell_id(model_name, method, alpha, seed, prompt_id)
                        redline = bool(alpha == 0.0 and abs(drift) >= REDLINE_TOL)
                        row = {
                            "cell_id": cid,
                            "model": model_name,
                            "method": method,
                            "alpha": float(alpha),
                            "seed": int(seed),
                            "prompt_id": prompt_id,
                            "inj_nll": float(inj_nll),
                            "base_nll": float(base_nll),
                            "drift": float(drift),
                            "model_substituted": False,
                            "env_commit": commit,
                        }
                        if redline:
                            row["redline_violation"] = True
                            print(f"[REDLINE] {model_name} {method} a=0 s={seed} "
                                  f"prompt={prompt_id} drift={drift:.6e}", flush=True)
                        append_cell(out_path, row)
                        done.add(cid)
                        passes_run += 1

                    elapsed = time.time() - t_cell
                    total_elapsed = time.time() - t0
                    print(
                        f"[{passes_run:>5}] {model_name[:18]:18s} {method:13s} "
                        f"a={alpha:<5.2f} s={seed} cell={elapsed:.1f}s "
                        f"total={total_elapsed/60:.1f}m",
                        flush=True,
                    )

        # ---- Free model ----
        del model, tok
        if caa_inj is not None:
            caa_inj.steering_vector = None
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    write_env(out_path, device, dtype, model_status)
    total_elapsed = time.time() - t0
    print(f"\n[W4] DONE: {passes_run} forward passes in {total_elapsed/60:.1f} min",
          flush=True)
    print(f"[W4] Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
