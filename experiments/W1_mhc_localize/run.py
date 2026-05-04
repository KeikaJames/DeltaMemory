"""W.1 — mHC shield failure localization sweep runner.

CLI:
    python -m experiments.W1_mhc_localize.run --model gpt2 --output cells.jsonl
    python -m experiments.W1_mhc_localize.run --model all --output cells.jsonl
    python -m experiments.W1_mhc_localize.run --model gpt2 --seeds 0 --output cells.jsonl

Each output line is one JSON-serialised cell record containing all per-cell
measurements.  Append-safe: run multiple times (different --model flags) and
cat all lines into one cells.jsonl.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = [
    "gpt2",
    "gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-1B",
]

# Models skipped due to unsupported architecture or access issues (documented for REPORT)
KNOWN_SKIP_REASONS: dict[str, str] = {
    "gpt2": "unsupported architecture: GPT-2 uses transformer.h[i].attn (not self_attn); AttnNativePatcher incompatible",
    "gpt2-medium": "unsupported architecture: GPT-2-medium uses transformer.h[i].attn (not self_attn); AttnNativePatcher incompatible",
    "meta-llama/Llama-3.2-1B": "gated repo: 403 Forbidden — HuggingFace access not authorized",
}

ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SHIELDS = [False, True]
KAPPAS = [1.0, 2.0, 4.0]
SEEDS = [0, 1, 2]
V_SCALES = [False, True]  # False = "none", True = "auto_rms_cap"

GOLD_PROMPTS_PATH = ROOT / "experiments" / "datasets" / "gold_30prompts.jsonl"
BANK_N = 32  # number of K/V entries in the bank


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Kappa context manager (monkey-patch for non-1.0 kappa)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched_kappa(kappa: float):
    """Temporarily replace shield_attention_weights with kappa-parameterised version.

    The attn_native_bank.py call site does `from deltamemory.memory.mhc_shield
    import shield_attention_weights` inside the hot path, so patching the
    module-level name is picked up at each forward call.
    """
    import deltamemory.memory.mhc_shield as _mod
    original = _mod.shield_attention_weights

    def _wrapped(weights, bank_size, enabled, kappa=kappa):  # noqa: E741
        return original(weights, bank_size=bank_size, enabled=enabled, kappa=kappa)

    _mod.shield_attention_weights = _wrapped
    try:
        yield
    finally:
        _mod.shield_attention_weights = original


# ---------------------------------------------------------------------------
# Bank construction — sha-locked random K/V
# ---------------------------------------------------------------------------

def _bank_seed_int(seed: int) -> int:
    tag = f"wikitext-bank-N{BANK_N}-seed{seed}"
    digest = hashlib.sha256(tag.encode()).hexdigest()
    return int(digest[:8], 16)


def build_bank(model, patcher, seed: int, v_scale_on: bool, device, dtype) -> Any:
    """Build a bank with BANK_N sha-locked random K/V entries."""
    from deltamemory.memory.attn_native_bank import AttnNativeBank, fresh_bank

    bank = fresh_bank(model)
    bank.value_scale_mode = "auto_rms_cap" if v_scale_on else "none"

    rng_seed = _bank_seed_int(seed)
    gen = torch.Generator()
    gen.manual_seed(rng_seed)

    for layer in range(bank.num_layers):
        d = bank.head_dims[layer]
        nkv = bank.num_kv_heads
        # Random K/V; use separate seed offsets per layer to avoid correlation
        k_seed = rng_seed ^ (layer * 0x9E37_79B9)
        v_seed = rng_seed ^ (layer * 0x6C62_272E + 0x1234_5678)
        gk = torch.Generator()
        gk.manual_seed(k_seed & 0xFFFF_FFFF_FFFF_FFFF)
        gv = torch.Generator()
        gv.manual_seed(v_seed & 0xFFFF_FFFF_FFFF_FFFF)

        K = torch.randn(BANK_N, nkv, d, generator=gk, dtype=torch.float32)
        V = torch.randn(BANK_N, nkv, d, generator=gv, dtype=torch.float32)

        # Apply value scaling consistently with the bank's mode
        if v_scale_on:
            from deltamemory.memory.attn_native_bank import _scale_bank_value_capture
            # has_native_v_norm=False (random V never has native v_norm)
            V = _scale_bank_value_capture(
                V, mode="auto_rms_cap", has_native_v_norm=False,
                target_rms=bank.value_target_rms, eps=bank.value_scale_eps,
            )

        bank.M_K[layer] = K.to(device=device, dtype=dtype)
        bank.M_V[layer] = V.to(device=device, dtype=dtype)

    bank.fact_ids = [f"bank_slot_{i}" for i in range(BANK_N)]
    return bank


# ---------------------------------------------------------------------------
# NLL computation
# ---------------------------------------------------------------------------

def compute_nll(model, patcher, bank, tokenizer, text: str, alpha: float,
                device, recorder=None) -> float:
    """Compute per-token NLL for a text prompt with (or without) the bank.

    Returns NLL in nats (natural log).
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"].to(device)
    if input_ids.size(1) < 2:
        return float("nan")

    labels = input_ids.clone()

    ctx = contextlib.ExitStack()
    ctx.enter_context(patcher.patched())
    if alpha > 0.0:
        ctx.enter_context(patcher.injecting(bank, alpha=alpha))
    if recorder is not None:
        ctx.enter_context(recorder)

    with ctx, torch.no_grad():
        out = model(input_ids=input_ids, labels=labels, use_cache=False)

    return float(out.loss)  # cross-entropy = NLL per token (nats)


# ---------------------------------------------------------------------------
# Diagnostic signals extraction
# ---------------------------------------------------------------------------

def extract_signals(rec_df) -> dict:
    """Extract per-cell summary statistics from a DiagnosticRecorder DataFrame."""
    if rec_df is None or len(rec_df) == 0:
        return {}

    out = {}

    # bank_col_sum_p99
    bcs = rec_df[rec_df["signal_name"] == "bank_col_sum"]["value"]
    if len(bcs):
        out["bank_col_sum_p99"] = float(bcs.quantile(0.99))
        out["bank_col_sum_p50"] = float(bcs.quantile(0.50))
        out["bank_col_sum_mean"] = float(bcs.mean())

    # attn_entropy_bank mean
    aeb = rec_df[rec_df["signal_name"] == "attn_entropy_bank"]["value"]
    if len(aeb):
        out["attn_entropy_bank_mean"] = float(aeb.mean())

    # attn_entropy_native mean
    aen = rec_df[rec_df["signal_name"] == "attn_entropy_native"]["value"]
    if len(aen):
        out["attn_entropy_native_mean"] = float(aen.mean())

    # m_perp_energy_ratio
    mpe = rec_df[rec_df["signal_name"] == "m_perp_energy_ratio"]["value"]
    if len(mpe):
        out["m_perp_energy_ratio_mean"] = float(mpe.mean())
    else:
        out["m_perp_energy_ratio_mean"] = 0.0

    # residual_norm_p50 (median across all layers+tokens)
    rn = rec_df[rec_df["signal_name"] == "residual_norm"]["value"]
    if len(rn):
        out["residual_norm_p50"] = float(rn.quantile(0.50))
        out["residual_norm_p99"] = float(rn.quantile(0.99))

    return out


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model + tokenizer; returns (model, tokenizer, patcher)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deltamemory.memory.attn_native_bank import AttnNativePatcher

    print(f"  Loading {model_name} ...", flush=True)
    t0 = time.time()

    dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map=str(device),
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    patcher = AttnNativePatcher(model)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)
    return model, tokenizer, patcher


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_model_sweep(
    model_name: str,
    seeds: list[int],
    output_path: Path,
    alphas: list[float] = ALPHAS,
    shields: list[bool] = SHIELDS,
    kappas: list[float] = KAPPAS,
    v_scales: list[bool] = V_SCALES,
) -> list[dict]:
    device = _get_device()
    print(f"\n=== Model: {model_name} | device: {device} ===", flush=True)

    try:
        model, tokenizer, patcher = load_model_and_tokenizer(model_name, device)
    except Exception as exc:
        print(f"  SKIP {model_name}: failed to load — {exc}", flush=True)
        return []

    # Load gold prompts
    gold_texts = []
    with open(GOLD_PROMPTS_PATH) as fh:
        for line in fh:
            obj = json.loads(line)
            gold_texts.append(obj["text"])
    assert len(gold_texts) == 30, f"Expected 30 gold prompts, got {len(gold_texts)}"

    dtype = next(model.parameters()).dtype
    records = []
    cells_done = 0

    for seed in seeds:
        print(f"  Seed {seed}", flush=True)

        # Baseline NLL (no bank) — computed once per seed (shared across all cells)
        print("    Computing baseline NLL (no bank)...", flush=True)
        baseline_nlls = []
        for text in gold_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc["input_ids"].to(device)
            if input_ids.size(1) < 2:
                baseline_nlls.append(float("nan"))
                continue
            labels = input_ids.clone()
            with patcher.patched(), torch.no_grad():
                out = model(input_ids=input_ids, labels=labels, use_cache=False)
            baseline_nlls.append(float(out.loss))

        baseline_mean = float(sum(x for x in baseline_nlls if not math.isnan(x)) /
                              max(1, sum(1 for x in baseline_nlls if not math.isnan(x))))
        print(f"    Baseline NLL mean: {baseline_mean:.4f} nats", flush=True)

        # Build banks for each V-scale mode
        banks = {}
        for vs in v_scales:
            banks[vs] = build_bank(model, patcher, seed, vs, device, dtype)

        # Sweep cells
        total_cells = len(alphas) * len(shields) * len(kappas) * len(v_scales)
        cell_idx = 0

        for alpha in alphas:
            for v_scale in v_scales:
                for shield in shields:
                    for kappa in kappas:
                        # kappa irrelevant when shield=off; skip duplicates
                        if not shield and kappa != kappas[0]:
                            continue

                        cell_idx += 1
                        cell_label = (f"α={alpha:.1f} shield={int(shield)} "
                                      f"κ={kappa:.1f} V={int(v_scale)}")
                        print(f"    [{cell_idx}/{total_cells}] {cell_label}", end="", flush=True)
                        t0 = time.time()

                        bank = banks[v_scale]
                        bank.mhc_shield = shield

                        # Collect diagnostic signals across all 30 prompts
                        all_dfs = []
                        nlls_bank = []

                        for text in gold_texts:
                            from deltamemory.diagnostics import DiagnosticRecorder

                            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                            input_ids = enc["input_ids"].to(device)
                            if input_ids.size(1) < 2:
                                nlls_bank.append(float("nan"))
                                continue
                            labels = input_ids.clone()

                            rec = DiagnosticRecorder(model, patcher, enabled=(alpha > 0.0))

                            ctx_mgr = patched_kappa(kappa) if (shield and kappa != 1.0) else contextlib.nullcontext()
                            with ctx_mgr:
                                with rec:
                                    ctx2 = contextlib.ExitStack()
                                    ctx2.enter_context(patcher.patched())
                                    if alpha > 0.0:
                                        ctx2.enter_context(patcher.injecting(bank, alpha=alpha))
                                    with ctx2, torch.no_grad():
                                        out = model(input_ids=input_ids, labels=labels, use_cache=False)

                            nlls_bank.append(float(out.loss))
                            if alpha > 0.0:
                                df = rec.to_pandas()
                                if len(df):
                                    all_dfs.append(df)

                        # Aggregate NLL
                        valid_pairs = [
                            (b, baseline_nlls[i])
                            for i, b in enumerate(nlls_bank)
                            if not math.isnan(b) and not math.isnan(baseline_nlls[i])
                        ]
                        if valid_pairs:
                            drifts = [b - base for b, base in valid_pairs]
                            mean_drift = float(sum(drifts) / len(drifts))
                        else:
                            mean_drift = float("nan")

                        # Aggregate diagnostics
                        diag_signals = {}
                        if all_dfs:
                            import pandas as pd
                            combined = pd.concat(all_dfs, ignore_index=True)
                            diag_signals = extract_signals(combined)

                        elapsed = time.time() - t0
                        print(f" → drift={mean_drift:.4f} ({elapsed:.1f}s)", flush=True)

                        record = {
                            "model": model_name,
                            "seed": seed,
                            "alpha": alpha,
                            "shield": shield,
                            "kappa": kappa,
                            "v_scale": v_scale,
                            "mean_drift": mean_drift,
                            "baseline_nll": baseline_mean,
                            "n_prompts": len(valid_pairs),
                            "elapsed_s": round(elapsed, 2),
                            **diag_signals,
                        }
                        records.append(record)
                        cells_done += 1

                        # Append to output file immediately (checkpoint)
                        with open(output_path, "a") as fh:
                            fh.write(json.dumps(record) + "\n")

    # Clean up
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="W.1 mHC localization sweep")
    parser.add_argument("--model", default="gpt2",
                        help="Model name or 'all' to run all 5 models")
    parser.add_argument("--output", default="experiments/W1_mhc_localize/cells.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Random seeds (default: [0], single-seed pilot)")
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells already in output file (model+seed+alpha+shield+kappa+v_scale key)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-done cell keys if resuming
    done_keys: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    k = f"{r['model']}|{r['seed']}|{r['alpha']}|{r['shield']}|{r['kappa']}|{r['v_scale']}"
                    done_keys.add(k)
                except Exception:
                    pass
        print(f"Resume: {len(done_keys)} cells already done.", flush=True)

    model_list = MODELS if args.model == "all" else [args.model]

    skipped_models = []
    total_cells = 0
    for model_name in model_list:
        recs = run_model_sweep(
            model_name=model_name,
            seeds=args.seeds,
            output_path=output_path,
            alphas=args.alphas,
        )
        if not recs:
            skipped_models.append(model_name)
        else:
            total_cells += len(recs)

    print(f"\nDone. {total_cells} cells written to {output_path}")
    if skipped_models:
        print(f"Skipped models: {skipped_models}")


if __name__ == "__main__":
    main()
