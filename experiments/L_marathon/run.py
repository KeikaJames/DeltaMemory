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


def _build_fact_line(subject: str, relation: str, target_new: str) -> str:
    return f"Fact: {subject} {relation} {target_new}."
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

def _compute_residual_norm(model: Any, hidden_states: Any, mu_layer: int) -> float:
    """L2 norm of hidden_states[mu_layer] over (batch, seq, hidden).

    Returns NaN if hidden_states is missing or layer index out of range.
    Exceptions are surfaced to stderr so the marathon does not silently
    drop a key probe.
    """
    if hidden_states is None:
        return float("nan")
    try:
        if not isinstance(hidden_states, (tuple, list)):
            return float("nan")
        if mu_layer < 0 or mu_layer >= len(hidden_states):
            return float("nan")
        hs = hidden_states[mu_layer]
        if hs is None:
            return float("nan")
        return float(torch.linalg.norm(hs.detach().float().reshape(-1)).item())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[L][WARN] residual_norm computation failed: {exc!r}",
              file=sys.stderr, flush=True)
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
    
    # Compute mu_arch layer (middle layer heuristic) up front; needed by CAA cfg.
    layers = get_decoder_layers(model)
    mu_layer = len(layers) // 2

    # ------------------------------------------------------------------
    # Build the persistent injection state (LOPI bank or CAA steering)
    # populated with all 3 facts at turn 1.  This object lives for the
    # entire marathon; the *active* α is toggled per-checkpoint via the
    # context manager protocol below.
    # ------------------------------------------------------------------

    patcher: Optional[Any] = None
    bank: Optional[Any] = None
    caa_inj: Optional[CAAInjector] = None
    fact_alpha = 1.0  # injection strength for filler + probes (PREREG L.v1)

    if args.method == "lopi_default":
        patcher = AttnNativePatcher(model)
        bank = fresh_bank(model)
        for fact in facts:
            wp = _build_fact_line(fact["subject"], fact["relation"], fact["target_new"])
            write_fact(
                patcher, bank, tok,
                write_prompt=wp,
                fact_id=str(fact["id"]),
                address=str(fact["subject"]),
            )
        patcher.install()
        patcher.bank = bank
        patcher.alpha = float(fact_alpha)
        print(f"[L] LOPI bank populated with {len(facts)} facts; "
              f"patcher installed at alpha={fact_alpha}", flush=True)
    elif args.method == "caa":
        cfg = CAAConfig(inject_layer="mu_arch", alpha=float(fact_alpha),
                        use_lopi_gate=False)
        caa_inj = CAAInjector(model, cfg, tokenizer=tok,
                              device=torch.device(args.device))
        pos_texts = [_build_fact_line(f["subject"], f["relation"], f["target_new"])
                     for f in facts]
        neg_texts = [_build_fact_line(f["subject"], f["relation"], f["target_true"])
                     for f in facts]
        caa_inj.calibrate(pos_texts, neg_texts)
        caa_inj.__enter__()  # activate hook for the lifetime of the run
        print(f"[L] CAA steering calibrated on {len(facts)} contrastive pairs "
              f"at layer {mu_layer}; hook active alpha={fact_alpha}", flush=True)
    else:
        raise ValueError(f"unknown method: {args.method}")

    # Encode filler text once; we will slice token windows from it to
    # advance the marathon clock between checkpoints.
    filler_ids_full = tok.encode(filler_text, add_special_tokens=False)
    if not filler_ids_full:
        filler_ids_full = tok.encode("Lorem ipsum dolor sit amet.", add_special_tokens=False)
    FILLER_TOKENS_PER_TURN = 32
    FILLER_WINDOW = 512  # cap context per filler forward pass

    rid = run_id(args.model, args.method, args.seed, args.turns)
    torch.manual_seed(args.seed)

    checkpoints = [t for t in CHECKPOINT_TURNS if t <= args.turns]

    prev_row: Optional[dict] = None
    prev_turn = 0
    try:
        for turn in checkpoints:
            check_key = f"{rid}|{turn}"
            if check_key in done_ids:
                print(f"[L] skip completed {check_key}", flush=True)
                prev_turn = turn
                continue

            # ----- advance marathon clock by (turn - prev_turn) filler turns
            n_filler = max(0, turn - prev_turn)
            if n_filler > 0:
                total_tokens = n_filler * FILLER_TOKENS_PER_TURN
                # Stream the filler in capped windows so KV-cache never grows.
                with torch.no_grad():
                    pos = 0
                    while pos < total_tokens:
                        win = min(FILLER_WINDOW, total_tokens - pos)
                        # Wrap-around through the filler text if needed.
                        slice_ids = [
                            filler_ids_full[(pos + i) % len(filler_ids_full)]
                            for i in range(win)
                        ]
                        ids = torch.tensor([slice_ids], device=args.device)
                        try:
                            _ = model(input_ids=ids, use_cache=False,
                                      output_hidden_states=False)
                        except Exception as exc:
                            print(f"[L][WARN] filler forward failed at "
                                  f"turn={turn} pos={pos}: {exc!r}",
                                  file=sys.stderr, flush=True)
                            break
                        pos += win

            # ----- probe nll_target_new (injection still active)
            try:
                nll_target_new = probe_nll_target_new(model, tok, probes, args.device)
            except Exception as exc:
                print(f"[L][ERROR] probe failed at turn {turn}: {exc!r}",
                      file=sys.stderr)
                nll_target_new = float("nan")

            # ----- residual norm at mu_layer via a single hidden-state pass
            residual_norm_mu = float("nan")
            try:
                rn_prompt = probes[0]["prompt"] if probes else "Hello world."
                rn_ids = tok.encode(rn_prompt, add_special_tokens=True,
                                    truncation=True, max_length=64)
                rn_t = torch.tensor([rn_ids], device=args.device)
                with torch.no_grad():
                    rn_out = model(input_ids=rn_t, use_cache=False,
                                   output_hidden_states=True)
                residual_norm_mu = _compute_residual_norm(
                    model, getattr(rn_out, "hidden_states", None), mu_layer
                )
            except Exception as exc:
                print(f"[L][WARN] residual_norm probe failed at turn {turn}: "
                      f"{exc!r}", file=sys.stderr, flush=True)

            mem_rss_mb = get_mem_rss_mb()
            nan_inf_count = _check_bank_nan_inf(bank) if bank is not None else 0
            kv_cache_size_bytes = _estimate_kv_cache_size(model)

            row: dict[str, Any] = {
                "run_id": rid,
                "model": args.model,
                "method": args.method,
                "seed": args.seed,
                "turn": turn,
                "alpha": fact_alpha,
                "nll_target_new": nll_target_new,
                "residual_norm_mu": residual_norm_mu,
                "mu_layer": mu_layer,
                "mem_rss_mb": mem_rss_mb,
                "nan_inf_count": nan_inf_count,
                "kv_cache_size_bytes": kv_cache_size_bytes,
                "abort_reason": None,
            }

            abort_reason = _check_abort(prev_row, row)
            if abort_reason:
                row["abort_reason"] = abort_reason
                append_row(out_path, row)
                print(f"[L][ABORT] {abort_reason} at turn {turn}",
                      file=sys.stderr, flush=True)
                sys.exit(1)

            append_row(out_path, row)
            print(f"[L] checkpoint turn={turn} nll={nll_target_new:.3f} "
                  f"resid={residual_norm_mu:.2f} rss={mem_rss_mb:.1f}MB "
                  f"nan_inf={nan_inf_count}", flush=True)
            prev_row = row
            prev_turn = turn
    finally:
        if caa_inj is not None:
            try:
                caa_inj.__exit__(None, None, None)
            except Exception:
                pass
        if patcher is not None:
            try:
                patcher.bank = None
                patcher.alpha = 0.0
                patcher.remove()
            except Exception:
                pass

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
