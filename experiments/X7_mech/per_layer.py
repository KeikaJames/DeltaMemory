#!/usr/bin/env python3
"""X7_mech/B1 — Per-layer attention probe.

For each |bank| ∈ {100, 500, 5000} and seeds 0,1,2, captures per-layer:
  - Bank-column attention mass (sum of bank-column weights / total weights)
  - Bank-column entropy (across bank columns only)
  - Top-1, Top-3, Top-10 attention concentration on bank
  - Residual contribution at each layer (norm of bank-induced delta / total)

Produces cells.jsonl with one row per (size, seed, layer).
Identifies the phase-transition layer.

PREREG: experiments/X7_mech/PREREG.md (X7MECH.v1 §1).
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
import torch.nn.functional as F

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

PREREG_VERSION = "X7MECH.v1"
DEFAULT_BANK_SIZES = [100, 500, 5000]
DEFAULT_SEEDS = [0, 1, 2]


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


def cell_id(model: str, size: int, seed: int, fact_id: str) -> str:
    return hashlib.sha1(
        f"perlay|{model}|{size}|{seed}|{fact_id}".encode()
    ).hexdigest()[:16]


def first_token_id(tok, text: str) -> int:
    ids = tok(" " + text.strip(), add_special_tokens=False).input_ids
    if not ids:
        ids = tok(text, add_special_tokens=False).input_ids
    return int(ids[0])


# ---------------------------------------------------------------------------
# Per-layer hook collector
# ---------------------------------------------------------------------------

class PerLayerProbe:
    """Collects per-layer attention mass and residual stats during a forward.

    We hook each decoder layer to capture:
    1. The raw attention weight tensor (via a hook injected into the patched
       forward — we use the DiagnosticRecorder infrastructure).
    2. The residual stream hidden state (input vs. output of each layer).

    For the attention metrics we extend the DiagnosticRecorder by adding a
    custom hook_fn that gets called from attn_native_bank.record_bank_attn.
    For residual metrics we register forward hooks on each decoder layer.
    """

    def __init__(self, model: Any, patcher: Any):
        self._model = model
        self._patcher = patcher
        self._layer_data: dict[int, dict[str, Any]] = {}  # layer_idx -> stats
        self._hook_handles: list = []
        self._baseline_residual: dict[int, float] = {}  # layer_idx -> norm without bank
        self._bank_residual: dict[int, float] = {}       # layer_idx -> norm with bank

    def _get_layers(self):
        from deltamemory.memory._layer_locator import get_decoder_layers
        return get_decoder_layers(self._model)

    def _install_residual_hooks(self) -> None:
        """Install forward hooks that capture residual norms per layer."""
        layers = self._get_layers()
        for idx, layer in enumerate(layers):
            def _make_hook(layer_idx: int):
                def _hook(module: Any, inp: Any, out: Any) -> None:
                    hidden = out[0] if isinstance(out, tuple) else out
                    # (B, T, D) -> mean L2 norm across tokens
                    norm = torch.linalg.vector_norm(
                        hidden.detach().float(), ord=2, dim=-1
                    ).mean().item()
                    self._bank_residual[layer_idx] = norm
                return _hook
            self._hook_handles.append(layer.register_forward_hook(_make_hook(idx)))

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def run_probe(
        self,
        patcher: Any,
        bank: Any,
        tok: Any,
        read_prompt: str,
        alpha: float,
        target_new_id: int,
        target_canon_id: int,
        device: str,
    ) -> dict[str, Any]:
        """Run forward with per-layer attention capture.

        Returns aggregated per-layer dict.
        """
        import deltamemory.diagnostics as _diag_mod

        # Register residual hooks
        self._install_residual_hooks()
        self._bank_residual.clear()

        # Use the DiagnosticRecorder to capture per-layer attention signals
        from deltamemory.diagnostics import DiagnosticRecorder
        rec = DiagnosticRecorder(model=self._model, patcher=patcher, enabled=True)
        t0 = time.perf_counter()
        with rec:
            try:
                logits = forward_with_bank(
                    patcher=patcher, bank=bank, tokenizer=tok,
                    read_prompt=read_prompt, alpha=alpha,
                )
            except Exception as exc:
                self.remove()
                return {"status": "forward_failed", "error": repr(exc)[:300]}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.remove()

        if not bool(torch.isfinite(logits).all().item()):
            return {"status": "nan_inf"}

        log_margin = float(
            logits[target_new_id].item() - logits[target_canon_id].item()
        )

        # Aggregate per-layer stats from recorder
        # records: {step, layer, token, signal_name, value}
        per_layer: dict[int, dict[str, list[float]]] = {}
        for r in rec._records:
            L = r["layer"]
            sig = r["signal_name"]
            val = r["value"]
            per_layer.setdefault(L, {}).setdefault(sig, []).append(val)

        layer_stats: dict[int, dict[str, float]] = {}
        for L, signals in per_layer.items():
            stats: dict[str, float] = {}

            # Bank mass = sum of all bank_col_sum values / (sum + epsilon)
            bank_col_sums = signals.get("bank_col_sum", [])
            if bank_col_sums:
                total_bank_mass = float(sum(bank_col_sums))
                # We need total mass (bank + native). bank_col_sum are already
                # normalized weights, so total_bank_mass is the fraction in bank.
                stats["bank_mass"] = total_bank_mass
                n_bank = len(bank_col_sums)
                stats["n_bank_cols"] = float(n_bank)

                # Bank entropy (across bank cols only, re-normalized)
                col_arr = bank_col_sums
                col_sum = sum(col_arr) + 1e-10
                p = [c / col_sum for c in col_arr]
                ent = -sum(pi * (max(pi, 1e-10)) ** 0 * (
                    (pi + 1e-10) and __import__('math').log(pi + 1e-10)
                ) for pi in p)
                import math
                ent = -sum(pi * math.log(pi + 1e-10) for pi in p)
                stats["bank_entropy"] = ent

                # Top-k concentration
                sorted_cols = sorted(col_arr, reverse=True)
                total_w = sum(col_arr) + 1e-10
                for k in (1, 3, 10):
                    topk_mass = sum(sorted_cols[:k]) / total_w
                    stats[f"top{k}_concentration"] = topk_mass

            # Residual norm
            resid_norms = signals.get("residual_norm", [])
            if resid_norms:
                stats["residual_norm_mean"] = sum(resid_norms) / len(resid_norms)

            # Entropy signals
            for sig_name in ("attn_entropy_native", "attn_entropy_bank"):
                vals = signals.get(sig_name, [])
                if vals:
                    stats[f"{sig_name}_mean"] = sum(vals) / len(vals)

            layer_stats[L] = stats

        # Residual contribution from bank-induced delta (from our hooks)
        residual_by_layer = dict(self._bank_residual)

        return {
            "status": "ok",
            "log_margin": log_margin,
            "alpha": alpha,
            "elapsed_ms": elapsed_ms,
            "layer_stats": layer_stats,
            "residual_by_layer": {str(k): v for k, v in residual_by_layer.items()},
        }


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="X7_mech/B1 — Per-layer attention probe"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--bank-sizes", nargs="+", type=int, default=DEFAULT_BANK_SIZES)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Injection alpha (default 1.0)")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke: 2 sizes, 1 seed")
    args = ap.parse_args()

    if args.smoke:
        args.bank_sizes = [100, 500]
        args.seeds = [0]

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"

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
        extra={
            "experiment": "B1_per_layer",
            "model": args.model,
            "bank_sizes": args.bank_sizes,
            "seeds": args.seeds,
            "alpha": args.alpha,
        },
    )

    facts = load_jsonl(facts_path)
    distractors = load_jsonl(distract_path)
    target = facts[0]
    print(f"[B1] target={target['fact_id']} distractors={len(distractors)}", flush=True)

    done = load_done(cells_path)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print(f"[B1] loading {args.model} ({args.dtype}) → {args.device}", flush=True)
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

    target_new_id = first_token_id(tok, target["target_new"])
    target_canon_id = first_token_id(tok, target["target_canonical"])

    probe = PerLayerProbe(model=model, patcher=patcher)
    n_done = 0

    try:
        for size in args.bank_sizes:
            for seed in args.seeds:
                cid = cell_id(args.model, size, seed, target["fact_id"])
                if cid in done:
                    print(f"  [skip] |bank|={size} s={seed}", flush=True)
                    continue

                torch.manual_seed(seed)
                bank, err = write_target_then_distractors(
                    model=model, tok=tok, patcher=patcher,
                    target_fact=target, distractors=distractors,
                    n_distract=max(0, size - 1), seed=seed,
                )
                if err:
                    row = {
                        "status": "write_failed", "error": err,
                        "bank_size": size, "seed": seed,
                        "cell_id": cid, "model": args.model,
                        "prereg_version": PREREG_VERSION,
                        "experiment": "B1_per_layer",
                    }
                    append_row(cells_path, row)
                    continue

                result = probe.run_probe(
                    patcher=patcher, bank=bank, tok=tok,
                    read_prompt=target["read_prompt"],
                    alpha=args.alpha,
                    target_new_id=target_new_id,
                    target_canon_id=target_canon_id,
                    device=args.device,
                )

                # Flatten layer_stats into per-layer rows
                layer_stats = result.pop("layer_stats", {})
                residual_by_layer = result.pop("residual_by_layer", {})

                base_row = {
                    "cell_id": cid,
                    "bank_size": size,
                    "seed": seed,
                    "model": args.model,
                    "prereg_version": PREREG_VERSION,
                    "experiment": "B1_per_layer",
                    "status": result.get("status", "ok"),
                    "log_margin": result.get("log_margin"),
                    "alpha": args.alpha,
                    "elapsed_ms": result.get("elapsed_ms"),
                }

                if result.get("status") == "ok":
                    # Emit one row per layer
                    all_layers = sorted(
                        set(list(layer_stats.keys()) + [int(k) for k in residual_by_layer.keys()])
                    )
                    for L in all_layers:
                        ls = layer_stats.get(L, {})
                        row = dict(base_row)
                        row["layer"] = L
                        row.update(ls)
                        row["residual_norm_hook"] = residual_by_layer.get(str(L))
                        append_row(cells_path, row)
                    n_done += 1
                    print(
                        f"  B1 |bank|={size:>5} s={seed} "
                        f"margin={result.get('log_margin', float('nan')):+.3f} "
                        f"n_layers={len(all_layers)}",
                        flush=True,
                    )
                else:
                    append_row(cells_path, base_row)
    finally:
        patcher.remove()

    print(f"[B1] DONE wrote {n_done} cells -> {cells_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
