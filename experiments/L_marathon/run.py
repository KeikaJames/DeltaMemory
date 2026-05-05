"""L Marathon runner — long-conversation bank stability harness.

Implements the experiment specified in
``experiments/L_marathon/PREREG.md``. Per checkpoint we measure:

    nll_target_new       — mean NLL of target_new on held-out probe set
    residual_norm_mu     — L2 norm of hidden_states at mu_arch layer
    mem_rss_mb           — process RSS in MB
    nan_inf_count        — NaN/Inf count in bank K + V
    kv_cache_size_bytes  — HF model KV-cache size (best-effort)

Cell key
--------
``run_id = sha1(f"{model}|{method}|{seed}|{turns}")``

Abort logic
-----------
* Any NaN/Inf in bank K/V → abort_reason set, exit non-zero.
* Any mem_rss jump >1 GB between checkpoints → abort.
* Any 10x decay in nll_target_new between checkpoints → abort.
* alpha=0 drift=0 witness at turn 1.

Author: BIRI GA, 2026-05-10.
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
from typing import Any, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory import (
    AttnNativePatcher,
    CAAInjector,
    fresh_bank,
    write_fact,
    pick_adapter,
    LOPIConfig,
)
from deltamemory.memory.caa_injector import CAAConfig
from deltamemory.memory._layer_locator import get_decoder_layers

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Constants

PREREG_VERSION = "L.v1"
CHECKPOINT_TURNS = [1, 50, 200, 500, 1000, 2000]


# ---------------------------------------------------------------------------
# Utility

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_id(model: str, method: str, seed: int, turns: int) -> str:
    key = f"{model}|{method}|{seed}|{turns}"
    return hashlib.sha1(key.encode()).hexdigest()


def append_row(path: Path, row: dict) -> None:
    with open(path, "at") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    try:
        with open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = row.get("run_id")
                turn = row.get("turn")
                if rid and turn is not None:
                    done.add(f"{rid}|{turn}")
    except Exception:
        pass
    return done


def get_mem_rss_mb() -> float:
    if not _HAS_PSUTIL:
        return float("nan")
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / (1024.0 ** 2)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Data loading

def load_facts(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_probes(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_filler(path: Path) -> str:
    with open(path) as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Residual norm helper

def _compute_residual_norm(model: Any, hidden_states: torch.Tensor, mu_layer: int) -> float:
    """Compute L2 norm of hidden_states at mu_arch layer.
    
    TODO(opus): integrate with deltamemory.memory.arch_adapter to get mu_arch
    for the given model architecture. For now, mu_layer is passed as an
    argument or computed via a heuristic (middle layer).
    """
    if hidden_states is None:
        return float("nan")
    try:
        # hidden_states is a tuple of tensors, one per layer (if output_hidden_states=True)
        if isinstance(hidden_states, (tuple, list)) and len(hidden_states) > mu_layer:
            hs = hidden_states[mu_layer]
            norm = torch.linalg.norm(hs.float(), ord=2).item()
            return float(norm)
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# Bank NaN/Inf check

def _check_bank_nan_inf(bank: Any) -> int:
    """Count NaN/Inf values in bank K + V tensors.
    
    TODO(opus): iterate over bank.mk / bank.mv (per-layer K/V tensors)
    and check for NaN/Inf. Return total count.
    """
    count = 0
    try:
        # bank.mk and bank.mv are lists of tensors (one per layer)
        if hasattr(bank, "mk") and bank.mk is not None:
            for k_tensor in bank.mk:
                if k_tensor is not None:
                    count += int(torch.isnan(k_tensor).sum().item())
                    count += int(torch.isinf(k_tensor).sum().item())
        if hasattr(bank, "mv") and bank.mv is not None:
            for v_tensor in bank.mv:
                if v_tensor is not None:
                    count += int(torch.isnan(v_tensor).sum().item())
                    count += int(torch.isinf(v_tensor).sum().item())
    except Exception:
        pass
    return count


# ---------------------------------------------------------------------------
# KV-cache size helper

def _estimate_kv_cache_size(model: Any) -> int:
    """Best-effort estimate of HF model KV-cache size in bytes.
    
    TODO(opus): if model has past_key_values in the generation cache,
    sum the tensor sizes. Otherwise return 0 or -1.
    """
    # Placeholder — HF does not expose cache size directly in all versions.
    return 0


# ---------------------------------------------------------------------------
# Abort check

def _check_abort(prev_row: Optional[dict], curr_row: dict) -> Optional[str]:
    """Check red-line abort conditions.
    
    Returns abort_reason string if abort triggered, else None.
    """
    # 1. NaN/Inf in bank
    if curr_row.get("nan_inf_count", 0) > 0:
        return "nan_inf_in_bank"
    
    # 2. RSS jump >1 GB
    if prev_row is not None:
        prev_rss = prev_row.get("mem_rss_mb")
        curr_rss = curr_row.get("mem_rss_mb")
        if prev_rss is not None and curr_rss is not None:
            if (curr_rss - prev_rss) > 1024.0:
                return "mem_rss_jump_gt_1gb"
    
    # 3. 10x decay in nll_target_new (i.e., recall gets 10x worse)
    if prev_row is not None:
        prev_nll = prev_row.get("nll_target_new")
        curr_nll = curr_row.get("nll_target_new")
        if prev_nll is not None and curr_nll is not None:
            if prev_nll > 0 and (curr_nll / prev_nll) > 10.0:
                return "recall_decay_10x"
    
    return None


# ---------------------------------------------------------------------------
# Probe evaluation

@torch.no_grad()
def probe_nll_target_new(
    model: Any,
    tok: Any,
    probes: list[dict],
    device: str,
) -> float:
    """Compute mean NLL of target_new tokens on held-out probe set.
    
    Each probe has: {"prompt": str, "target_new": str}
    """
    nlls = []
    for p in probes:
        prompt_text = p["prompt"]
        target_text = p["target_new"]
        
        prompt_ids = tok.encode(prompt_text, add_special_tokens=True)
        sep = "" if (prompt_text.endswith(" ") or not prompt_text) else " "
        full_text = prompt_text + sep + target_text
        full_ids = tok.encode(full_text, add_special_tokens=True)
        
        if len(full_ids) <= len(prompt_ids):
            nlls.append(float("nan"))
            continue
        
        ids = torch.tensor([full_ids], device=device)
        out = model(input_ids=ids, use_cache=False, output_hidden_states=False)
        logp = F.log_softmax(out.logits[0].float(), dim=-1)
        target_start = len(prompt_ids)
        token_logps = []
        for i in range(target_start, len(full_ids)):
            token_logps.append(float(logp[i - 1, full_ids[i]].item()))
        nll = -float(sum(token_logps) / len(token_logps))
        nlls.append(nll)
    
    valid = [x for x in nlls if x == x]  # filter NaN
    return float(sum(valid) / len(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Main marathon loop

def run_marathon(args: argparse.Namespace) -> None:
    out_path = Path(args.out) / "cells.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    env_path = out_path.parent / "env.json"
    facts_sha = sha1_of_file(Path(args.inject_facts))
    probes_sha = sha1_of_file(Path(args.probe_set))
    filler_sha = sha1_of_file(Path(args.filler))
    dataset_sha = hashlib.sha1((facts_sha + probes_sha + filler_sha).encode()).hexdigest()

    # Authenticity contract conformance: use tools.env_writer to populate
    # all required fields (commit, dirty, dirty_diff_sha1, host, ...).
    sys.path.insert(0, str(ROOT))
    from tools.env_writer import write_env_json  # noqa: E402

    write_env_json(
        out_dir=out_path.parent,
        prereg_version=PREREG_VERSION,
        dataset_sha1={
            "facts": facts_sha,
            "probes": probes_sha,
            "filler": filler_sha,
            "combined": dataset_sha,
        },
        device=args.device,
        dtype=args.dtype,
        cli_argv=sys.argv,
        extra={"method": args.method, "model": args.model, "seed": args.seed,
               "turns": args.turns},
    )
    print(f"[L] wrote {env_path}", flush=True)
    
    # Load data
    facts = load_facts(Path(args.inject_facts))
    probes = load_probes(Path(args.probe_set))
    filler_text = load_filler(Path(args.filler))
    
    print(f"[L] loaded {len(facts)} facts, {len(probes)} probes", flush=True)
    
    # Resume support
    done_ids = load_done_ids(out_path) if args.resume else set()
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map.get(args.dtype, torch.float32)
    
    print(f"[L] loading model {args.model} on {args.device} {args.dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device)
    model.eval()
    
    # Prepare injection context
    # TODO(opus): set up AttnNativePatcher + bank for lopi_default,
    # or CAAInjector for caa, with the correct layer targeting.
    # For skeleton, we just run the base model (no injection).
    
    inj_ctx = None
    bank = None
    patcher = None
    
    if args.method == "lopi_default":
        # TODO(opus): implement LOPI default arm
        # patcher = AttnNativePatcher(model)
        # bank = fresh_bank(model)
        # bank.lopi_cfg = LOPIConfig(enabled=True, profile_mode="auto")
        # Write facts into bank at turn 0
        # for fact in facts:
        #     write_fact(patcher, bank, tok,
        #                write_prompt=f"Fact: {fact['subject']} {fact['relation']} {fact['target_new']}.",
        #                fact_id=fact['id'], address=fact['subject'])
        # inj_ctx = patcher.patched() and patcher.injecting(bank, alpha=1.0)
        print("[L] lopi_default not yet implemented (skeleton mode)", flush=True)
    elif args.method == "caa":
        # TODO(opus): implement CAA injector
        # layers = get_decoder_layers(model)
        # mu_layer = len(layers) // 2
        # cfg = CAAConfig(inject_layer=mu_layer, alpha=1.0, use_lopi_gate=False)
        # inj = CAAInjector(model, tok, cfg)
        # inj_ctx = inj
        print("[L] caa not yet implemented (skeleton mode)", flush=True)
    else:
        raise ValueError(f"unknown method: {args.method}")
    
    # Compute mu_arch layer (middle layer heuristic)
    layers = get_decoder_layers(model)
    mu_layer = len(layers) // 2
    
    rid = run_id(args.model, args.method, args.seed, args.turns)
    torch.manual_seed(args.seed)
    
    # Checkpoint turns (skip those beyond args.turns)
    checkpoints = [t for t in CHECKPOINT_TURNS if t <= args.turns]
    
    prev_row: Optional[dict] = None
    for turn in checkpoints:
        check_key = f"{rid}|{turn}"
        if check_key in done_ids:
            print(f"[L] skip completed {check_key}", flush=True)
            continue
        
        # TODO(opus): advance conversation state with filler text
        # For turn=1, just inject facts and probe immediately.
        # For turn>1, run filler text (turn - prev_turn) times.
        
        # Probe
        try:
            nll_target_new = probe_nll_target_new(model, tok, probes, args.device)
        except Exception as exc:
            print(f"[L][ERROR] probe failed at turn {turn}: {exc!r}", file=sys.stderr)
            nll_target_new = float("nan")
        
        # Residual norm (requires output_hidden_states=True)
        # TODO(opus): run a single forward pass with output_hidden_states=True
        # and compute residual norm at mu_layer.
        residual_norm_mu = float("nan")
        
        mem_rss_mb = get_mem_rss_mb()
        nan_inf_count = _check_bank_nan_inf(bank) if bank else 0
        kv_cache_size_bytes = _estimate_kv_cache_size(model)
        
        row: dict[str, Any] = {
            "run_id": rid,
            "model": args.model,
            "method": args.method,
            "seed": args.seed,
            "turn": turn,
            "nll_target_new": nll_target_new,
            "residual_norm_mu": residual_norm_mu,
            "mem_rss_mb": mem_rss_mb,
            "nan_inf_count": nan_inf_count,
            "kv_cache_size_bytes": kv_cache_size_bytes,
            "abort_reason": None,
        }
        
        # Abort check
        abort_reason = _check_abort(prev_row, row)
        if abort_reason:
            row["abort_reason"] = abort_reason
            append_row(out_path, row)
            print(f"[L][ABORT] {abort_reason} at turn {turn}", file=sys.stderr, flush=True)
            sys.exit(1)
        
        append_row(out_path, row)
        print(f"[L] checkpoint turn={turn} nll={nll_target_new:.3f} rss={mem_rss_mb:.1f}MB", flush=True)
        prev_row = row
    
    print(f"[L] DONE {rid}  {len(checkpoints)} checkpoints", flush=True)


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="L Marathon runner")
    ap.add_argument("--model", required=True, help="HF model identifier")
    ap.add_argument("--method", required=True, choices=["lopi_default", "caa"])
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--turns", type=int, required=True, help="Total turns to run")
    ap.add_argument("--inject-facts", required=True, help="Path to facts_3.jsonl")
    ap.add_argument("--probe-set", required=True, help="Path to probes_8.jsonl")
    ap.add_argument("--filler", required=True, help="Path to filler.txt")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--resume", action="store_true", help="Skip completed checkpoints")
    args = ap.parse_args()
    
    run_marathon(args)


if __name__ == "__main__":
    main()
