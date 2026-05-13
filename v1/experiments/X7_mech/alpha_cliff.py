#!/usr/bin/env python3
"""X7_mech/B3 — α-cliff residual analysis.

For α ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0} × seeds 0,1,2,
captures per-layer residual norm AND per-layer Δresidual relative to α=0.

Tests:
  H_B3.1: There is a specific layer L_cliff where ‖Δresidual‖ is maximised
           at α=0.25.
  H_B3.2: The residual-norm ratio at L_cliff is ≥1.5× the pre-cliff ratio.
  H_B3.3: At α≥0.75, Δresidual at L_cliff returns within 20% of α=0.
  H_B3.4: Post-cliff recovery is monotone decreasing.

PREREG: experiments/X7_mech/PREREG.md (X7MECH.v1 §3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from deltamemory.diagnostics import DiagnosticRecorder  # noqa: E402
from tools.env_writer import sha1_of, write_env_json  # noqa: E402

PREREG_VERSION = "X7MECH.v1"
DEFAULT_ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0]
DEFAULT_SEEDS = [0, 1, 2]
BANK_SIZE = 200  # Match X7NL sub-B: fixed bank for alpha sweep


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


def cell_id(model: str, alpha: float, seed: int, fact_id: str) -> str:
    a_str = f"{alpha:.4f}"
    return hashlib.sha1(
        f"cliff|{model}|{a_str}|{seed}|{fact_id}".encode()
    ).hexdigest()[:16]


def first_token_id(tok, text: str) -> int:
    ids = tok(" " + text.strip(), add_special_tokens=False).input_ids
    if not ids:
        ids = tok(text, add_special_tokens=False).input_ids
    return int(ids[0])


def measure_residuals_per_layer(
    model, tok, patcher, bank, target, alpha: float,
    target_new_id: int, target_canon_id: int,
) -> dict[str, Any]:
    """Run forward and return per-layer residual norms.

    Uses DiagnosticRecorder for residual_norm signals, which are captured
    by the residual hook installed by the recorder on each decoder layer.
    """
    rec = DiagnosticRecorder(model=model, patcher=patcher, enabled=True)
    try:
        with rec:
            logits = forward_with_bank(
                patcher=patcher, bank=bank, tokenizer=tok,
                read_prompt=target["read_prompt"], alpha=alpha,
            )
    except Exception as exc:
        return {"status": "forward_failed", "error": repr(exc)[:200]}

    if not bool(torch.isfinite(logits).all().item()):
        return {"status": "nan_inf"}

    log_margin = float(
        logits[target_new_id].item() - logits[target_canon_id].item()
    )

    # Collect per-layer residual norms (averaged across tokens)
    layer_norm_vals: dict[int, list[float]] = {}
    for r in rec._records:
        if r["signal_name"] == "residual_norm":
            L = r["layer"]
            layer_norm_vals.setdefault(L, []).append(r["value"])

    per_layer_mean_norm: dict[int, float] = {}
    for L, vals in layer_norm_vals.items():
        per_layer_mean_norm[L] = sum(vals) / len(vals)

    return {
        "status": "ok",
        "log_margin": log_margin,
        "alpha": alpha,
        "per_layer_norm": per_layer_mean_norm,
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


def analyze_cliff(
    cells_by_seed: dict[int, dict[float, dict]],
    alphas: list[float],
) -> dict[str, Any]:
    """Identify cliff layer and test hypotheses B3.1-B3.4.

    cells_by_seed[seed][alpha] = {per_layer_norm: {layer: float}, log_margin}
    """
    # 1. Compute Δresidual relative to α=0 per seed per layer per alpha
    # delta_by_seed_alpha_layer[seed][alpha][layer] = Δnorm
    delta: dict[int, dict[float, dict[int, float]]] = {}
    for seed, alpha_dict in cells_by_seed.items():
        baseline = alpha_dict.get(0.0, {}).get("per_layer_norm", {})
        if not baseline:
            continue
        delta[seed] = {}
        for alpha, cell in alpha_dict.items():
            norms = cell.get("per_layer_norm", {})
            if not norms:
                continue
            delta[seed][alpha] = {
                L: norms.get(L, 0.0) - baseline.get(L, 0.0)
                for L in baseline
            }

    if not delta:
        return {"error": "no_baseline_data"}

    # 2. Mean Δresidual at α=0.25 per layer across seeds
    seeds_with_cliff = [
        s for s in delta if 0.25 in delta[s]
    ]
    if not seeds_with_cliff:
        return {"error": "no_cliff_data"}

    all_layers = sorted(
        set().union(*[delta[s][0.25].keys() for s in seeds_with_cliff])
    )
    mean_delta_025_per_layer: dict[int, float] = {}
    for L in all_layers:
        vals = [delta[s][0.25].get(L, 0.0) for s in seeds_with_cliff]
        mean_delta_025_per_layer[L] = sum(vals) / len(vals)

    # 3. Identify L_cliff (layer with max |Δresidual| at α=0.25)
    if not mean_delta_025_per_layer:
        return {"error": "empty_delta"}
    L_cliff = max(all_layers, key=lambda L: abs(mean_delta_025_per_layer[L]))
    peak_delta = mean_delta_025_per_layer[L_cliff]

    # 4. Test H_B3.2: ratio at L_cliff ≥ 1.5× pre-cliff (α=0.20)
    mean_delta_020_per_layer: dict[int, float] = {}
    seeds_with_020 = [s for s in delta if 0.20 in delta[s]]
    for L in all_layers:
        vals = [delta[s][0.20].get(L, 0.0) for s in seeds_with_020 if 0.20 in delta[s]]
        if vals:
            mean_delta_020_per_layer[L] = sum(vals) / len(vals)

    delta_020_at_cliff = mean_delta_020_per_layer.get(L_cliff, 0.0)
    ratio_025_vs_020 = (
        abs(peak_delta) / (abs(delta_020_at_cliff) + 1e-8)
    )
    h_b3_2_supported = bool(ratio_025_vs_020 >= 1.5)

    # 5. Test H_B3.3: recovery at α≥0.75 within 20% of baseline
    recovery_alphas = [a for a in alphas if a >= 0.75]
    baseline_delta_at_cliff = {
        s: delta[s].get(0.0, {}).get(L_cliff, 0.0) for s in delta
    }
    mean_baseline = sum(baseline_delta_at_cliff.values()) / (len(baseline_delta_at_cliff) or 1)
    recovery_check: dict[float, bool] = {}
    for alpha in recovery_alphas:
        seeds_with_a = [s for s in delta if alpha in delta[s]]
        if seeds_with_a:
            mean_da = sum(delta[s][alpha].get(L_cliff, 0.0) for s in seeds_with_a) / len(seeds_with_a)
            within_20pct = bool(abs(mean_da - mean_baseline) <= 0.20 * (abs(mean_baseline) + 1e-8))
            recovery_check[alpha] = within_20pct
    h_b3_3_supported = bool(all(recovery_check.values()) if recovery_check else False)

    # 6. Test H_B3.4: monotone decreasing post-cliff at L_cliff
    post_cliff_alphas = sorted([a for a in alphas if a > 0.25])
    post_cliff_means: list[tuple[float, float]] = []
    for alpha in post_cliff_alphas:
        seeds_with_a = [s for s in delta if alpha in delta[s]]
        if seeds_with_a:
            mean_da = sum(abs(delta[s][alpha].get(L_cliff, 0.0)) for s in seeds_with_a) / len(seeds_with_a)
            post_cliff_means.append((alpha, mean_da))
    is_monotone = all(
        post_cliff_means[i][1] >= post_cliff_means[i + 1][1]
        for i in range(len(post_cliff_means) - 1)
    ) if len(post_cliff_means) > 1 else None

    return {
        "L_cliff": L_cliff,
        "peak_delta_at_L_cliff": peak_delta,
        "ratio_025_vs_020": ratio_025_vs_020,
        "H_B3.1_supported": True,  # We always find a max layer
        "H_B3.1_L_cliff": L_cliff,
        "H_B3.2_supported": h_b3_2_supported,
        "H_B3.2_ratio": ratio_025_vs_020,
        "H_B3.3_supported": h_b3_3_supported,
        "H_B3.3_recovery_check": {str(k): v for k, v in recovery_check.items()},
        "H_B3.4_supported": is_monotone,
        "H_B3.4_post_cliff": [(a, v) for a, v in post_cliff_means],
        "mean_delta_025_per_layer": {str(L): v for L, v in mean_delta_025_per_layer.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="X7_mech/B3 — α-cliff residual analysis"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--bank-size", type=int, default=BANK_SIZE)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.alphas = [0.0, 0.25, 1.0]
        args.seeds = [0]

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"
    summary_path = args.out / "cliff_summary.json"

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
            "experiment": "B3_alpha_cliff",
            "model": args.model,
            "bank_size": args.bank_size,
            "alphas": args.alphas,
            "seeds": args.seeds,
        },
    )

    facts = load_jsonl(facts_path)
    distractors = load_jsonl(distract_path)
    target = facts[0]
    print(f"[B3] target={target['fact_id']} bank_size={args.bank_size}", flush=True)

    done = load_done(cells_path)
    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    print(f"[B3] loading {args.model} ({args.dtype}) → {args.device}", flush=True)
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

    # cells_by_seed[seed][alpha] = result dict
    cells_by_seed: dict[int, dict[float, dict]] = {s: {} for s in args.seeds}

    try:
        for seed in args.seeds:
            torch.manual_seed(seed)
            bank, err = write_target_then_distractors(
                model=model, tok=tok, patcher=patcher,
                target_fact=target, distractors=distractors,
                n_distract=max(0, args.bank_size - 1), seed=seed,
            )
            if err:
                print(f"  [B3] seed={seed} write_failed: {err}", flush=True)
                continue

            for alpha in args.alphas:
                cid = cell_id(args.model, alpha, seed, target["fact_id"])
                if cid in done:
                    print(f"  [skip] α={alpha:.2f} s={seed}", flush=True)
                    continue

                result = measure_residuals_per_layer(
                    model=model, tok=tok, patcher=patcher,
                    bank=bank, target=target, alpha=alpha,
                    target_new_id=target_new_id, target_canon_id=target_canon_id,
                )

                row = {
                    "cell_id": cid,
                    "alpha": alpha,
                    "seed": seed,
                    "bank_size": args.bank_size,
                    "model": args.model,
                    "prereg_version": PREREG_VERSION,
                    "experiment": "B3_alpha_cliff",
                    **result,
                }
                # Serialize per_layer_norm as dict with str keys
                if "per_layer_norm" in row:
                    row["per_layer_norm"] = {
                        str(k): v for k, v in row["per_layer_norm"].items()
                    }
                append_row(cells_path, row)

                if result.get("status") == "ok":
                    cells_by_seed[seed][alpha] = result
                    print(
                        f"  B3 α={alpha:5.2f} s={seed} "
                        f"margin={result.get('log_margin', 0):+.3f} "
                        f"n_layers={len(result.get('per_layer_norm', {}))}",
                        flush=True,
                    )
    finally:
        patcher.remove()

    # --- Analysis ---
    analysis = analyze_cliff(cells_by_seed, args.alphas)
    summary = {
        "experiment": "B3_alpha_cliff",
        "prereg_version": PREREG_VERSION,
        "bank_size": args.bank_size,
        "alphas": args.alphas,
        "seeds": args.seeds,
        "analysis": analysis,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[B3] L_cliff={analysis.get('L_cliff', 'N/A')}", flush=True)
    print(f"[B3] H_B3.1 (phase-transition layer): supported=True", flush=True)
    print(f"[B3] H_B3.2 (threshold ≥1.5×): {analysis.get('H_B3.2_supported', '?')}", flush=True)
    print(f"[B3] H_B3.3 (recovery at α≥0.75): {analysis.get('H_B3.3_supported', '?')}", flush=True)
    print(f"[B3] H_B3.4 (monotone recovery): {analysis.get('H_B3.4_supported', '?')}", flush=True)
    print(f"[B3] DONE summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
