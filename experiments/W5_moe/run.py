"""W.5 MoE per-expert column-cap runner.

Implements the grid pre-registered in ``experiments/W5_moe/PREREG.md``:

* 1 MoE model × 7 α × 3 seeds × 3 cap_modes × 30 prompts = **1890 cells**
  (PREREG also sweeps κ ∈ {0.5, 1.0, 2.0}; for the smoke we fix κ=1.0).

For each cell we record:

* ``inj_nll`` — mean per-token NLL on the gold prompt with the bank attached.
* ``base_nll`` — same prompt, no patcher (baseline).
* ``drift = inj_nll - base_nll`` (positive = injection raised loss).

Cell key
--------
``cell_id = sha1(f"{model}|{cap_mode}|{alpha}|{seed}|{prompt_id}")``

Smoke vs full
-------------
``--smoke`` runs 1 alpha × 1 seed × 1 cap × 5 prompts = 5 cells.  If the
selected MoE model fails to load, ``BLOCKED.md`` is written documenting
the failure and the runner exits 0 (no faked rows).

Cell flushing: every 100 cells.

Hardware notes (per W.5 PREREG): the full grid targets 128 GB unified
memory or CUDA bf16.  This runner handles loading on whatever device is
available; on a 64 GB MPS box only the smaller MoE checkpoints fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PREREG_VERSION = "W.5.v1"

# MoE model preference order (first one that loads wins).
MOE_MODELS = [
    "Qwen/Qwen3.5-35B-A3B-Base",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V2-Lite-MoE",
]

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = [0, 1, 2]
CAP_MODES = ["none", "global", "per_expert"]
KAPPA = 1.0

DATA_ROOT = ROOT / "experiments" / "datasets"
GOLD_PATH = DATA_ROOT / "gold_30prompts.jsonl"

OUT_DIR = ROOT / "experiments" / "W5_moe"
CELLS_PATH = OUT_DIR / "cells.jsonl"
SMOKE_CELLS_PATH = OUT_DIR / "cells_smoke.jsonl"
ENV_PATH = OUT_DIR / "env.json"
BLOCKED_PATH = OUT_DIR / "BLOCKED.md"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def cell_id(model: str, cap_mode: str, alpha: float, seed: int, prompt_id: str) -> str:
    key = f"{model}|{cap_mode}|{alpha}|{seed}|{prompt_id}"
    return hashlib.sha1(key.encode()).hexdigest()


def env_signature(model_id: Optional[str]) -> dict:
    import platform

    sig = {
        "prereg_version": PREREG_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "model_id": model_id,
        "kappa": KAPPA,
        "alphas": ALPHAS,
        "seeds": SEEDS,
        "cap_modes": CAP_MODES,
    }
    try:
        import subprocess

        sig["env_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        sig["env_commit"] = "unknown"
    return sig


def load_prompts(limit: Optional[int] = None) -> list[dict]:
    out = []
    with open(GOLD_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# Model loading (preference order; first that fits wins)
# ---------------------------------------------------------------------------


def try_load_moe_model() -> tuple[Optional[object], Optional[object], Optional[str], Optional[str]]:
    """Iterate MOE_MODELS, return (model, tokenizer, name, error_trace).

    On success: error_trace is None.  On failure of all models: model and
    tokenizer are None and error_trace contains the joined tracebacks.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    errs = []
    for name in MOE_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation="eager",
                device_map=device,
            )
            model.eval()
            return model, tok, name, None
        except Exception as exc:
            tb = traceback.format_exc()
            errs.append(f"### {name}\n```\n{tb}\n```\n")
    return None, None, None, "\n".join(errs)


def write_blocked(error_trace: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BLOCKED_PATH.write_text(
        f"""# W.5 MoE smoke — BLOCKED

No MoE model from the preference list loaded successfully on this host.

## Preference list (all failed)

{chr(10).join('- ' + m for m in MOE_MODELS)}

## Hardware

* device: {"cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")}
* torch: {torch.__version__}

## Error traces

{error_trace}

## Resolution

This is the expected outcome on the 64 GB development machine: none of
the listed MoE checkpoints are downloaded locally and the runner is
configured with ``local_files_only=True`` to avoid an unintended
multi-GB download.

The full W.5 grid (1890 cells) will run on the 128 GB GB10 host; per
``plan.md`` G-10 the W.5 smoke is allowed to be deferred to that
environment.  The patcher implementation, unit tests, and aggregator
are validated locally via the synthetic mock adapter — see
``tests/test_moe_attn_patcher.py``.
"""
    )


# ---------------------------------------------------------------------------
# NLL evaluation
# ---------------------------------------------------------------------------


def compute_nll(model, tok, text: str, span_tokens: int = 64) -> float:
    """Mean per-token NLL on the last ``span_tokens`` tokens of ``text``."""
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
    ids = enc["input_ids"]
    if ids.size(1) < 4:
        return float("nan")
    span = min(span_tokens, ids.size(1) - 1)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]  # (T, V)
    log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
    targets = ids[0, 1:]
    nll_per_pos = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(nll_per_pos[-span:].mean().item())


def compute_inj_nll(patcher, bank, tok, text: str, alpha: float, span_tokens: int = 64) -> float:
    from deltamemory.memory.attn_native_bank import forward_with_bank  # noqa: F401

    device = next(patcher.model.parameters()).device
    enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
    ids = enc["input_ids"]
    if ids.size(1) < 4:
        return float("nan")
    span = min(span_tokens, ids.size(1) - 1)
    with patcher.patched(), patcher.injecting(bank, alpha=alpha), torch.no_grad():
        out = patcher.model(input_ids=ids, use_cache=False)
    logits = out.logits[0]
    log_probs = F.log_softmax(logits[:-1].float(), dim=-1)
    targets = ids[0, 1:]
    nll_per_pos = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(nll_per_pos[-span:].mean().item())


# ---------------------------------------------------------------------------
# Main grid loop
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="W.5 MoE per-expert column-cap runner.")
    ap.add_argument("--smoke", action="store_true",
                    help="1α × 1seed × 1cap × 5 prompts (5 cells).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        alphas = [1.0]
        seeds = [0]
        cap_modes = ["per_expert"]
        prompts = load_prompts(limit=5)
        cells_path = SMOKE_CELLS_PATH
    else:
        alphas = ALPHAS
        seeds = SEEDS
        cap_modes = CAP_MODES
        prompts = load_prompts()
        cells_path = CELLS_PATH

    # Try to load an MoE model.
    model, tok, model_name, err = try_load_moe_model()
    env = env_signature(model_name)
    ENV_PATH.write_text(json.dumps(env, indent=2))

    if model is None:
        print(f"[BLOCKED] no MoE model loaded.  See {BLOCKED_PATH.relative_to(ROOT)}", flush=True)
        write_blocked(err or "(no error trace captured)")
        return 0

    print(f"[ok] loaded MoE model: {model_name}", flush=True)

    # Build patcher.
    from deltamemory.memory.attn_native_bank import (
        AttnNativePatcher,
        fresh_bank,
        write_fact,
    )
    from deltamemory.memory.moe_attn_patcher import MoeAttnNativePatcher
    from deltamemory.memory.arch_moe_adapter import (
        Qwen3MoeAdapter,
        make_moe_adapter_from_dense,
    )

    # Pick MoE adapter; if none matches the attention class, fall back to
    # promoting the dense adapter for proxy testing.
    probe = AttnNativePatcher(model)
    try:
        moe_adapter = Qwen3MoeAdapter(num_experts=128, top_k=8) if "Qwen3Moe" in type(probe.attn_modules[0]).__name__ else make_moe_adapter_from_dense(probe.adapter, num_experts=8, top_k=2)
    except Exception:
        moe_adapter = make_moe_adapter_from_dense(probe.adapter, num_experts=8, top_k=2)

    patcher = MoeAttnNativePatcher(model, adapter=moe_adapter, cap_mode="per_expert", kappa=KAPPA)

    bank = fresh_bank(model)
    bank.mhc_shield = True  # required for the shield dispatcher to fire
    # Write a single neutral fact so the bank has at least one slot for α>0.
    write_fact(patcher, bank, tok, "Mneme stores facts in attention banks.", "moe_smoke_fact", "Mneme")

    # Loop.
    n_cells = 0
    flush_every = 100
    cells_path.unlink(missing_ok=True)
    t0 = time.time()
    for cap_mode in cap_modes:
        patcher.cap_mode = cap_mode
        for alpha in alphas:
            for seed in seeds:
                torch.manual_seed(seed)
                for p in prompts:
                    pid = p["id"]
                    text = p["text"]
                    base_nll = compute_nll(model, tok, text)
                    inj_nll = compute_inj_nll(patcher, bank, tok, text, alpha=alpha)
                    drift = inj_nll - base_nll
                    row = {
                        "cell_id": cell_id(model_name, cap_mode, alpha, seed, pid),
                        "model": model_name,
                        "cap_mode": cap_mode,
                        "alpha": alpha,
                        "seed": seed,
                        "prompt_id": pid,
                        "base_nll": base_nll,
                        "inj_nll": inj_nll,
                        "drift": drift,
                        "kappa": KAPPA,
                        "env_commit": env["env_commit"],
                    }
                    with open(cells_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                        if (n_cells + 1) % flush_every == 0:
                            f.flush()
                    n_cells += 1
                    if args.smoke:
                        print(f"[smoke] cell {n_cells}: {pid} cap={cap_mode} α={alpha} drift={drift:+.4f}", flush=True)

    dt = time.time() - t0
    print(f"[done] {n_cells} cells written to {cells_path.relative_to(ROOT)} in {dt:.1f}s", flush=True)

    # Smoke red-line: if α=0 was in the grid, drift must be ~0.
    if args.smoke and 0.0 in alphas:
        with open(cells_path) as f:
            zero_drifts = [json.loads(l)["drift"] for l in f if json.loads(l)["alpha"] == 0.0]
        max_abs = max(abs(d) for d in zero_drifts) if zero_drifts else 0.0
        if max_abs > 1e-4:
            print(f"[REDLINE] α=0 max-abs-drift={max_abs:.3e} > 1e-4", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
