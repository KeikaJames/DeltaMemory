#!/usr/bin/env python3
"""Stage 6 orchestrator: token-preserving writer + 3-channel sweep.

Runs the locked Stage 6 configuration with one CLI entry. Phase 1 is the
synthetic 256/256 sweep. Phase 2 (LAMA factual transfer) reuses Phase 1
config and points at ``factual_capital_binding``.

The orchestrator:

- iterates over (writer_pool x swap_loss) cells and seeds;
- writes per-cell ``summary.json`` and ``report.md`` under ``--report-root``;
- regenerates the README AUTOGEN block at the end via
  ``scripts/update_readme_charts.py``.

Examples:

    python3 scripts/run_stage6.py --phase 1 --model mock-gemma --device cpu \\
        --train-samples 8 --eval-samples 8 --steps 32 --seeds 0 \\
        --writer-pools attn --swap-options on \\
        --report-root reports/experiments/stage6_pilot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rcvhc.engine.delta_experiment import (  # noqa: E402
    DeltaExperimentConfig,
    run_delta_experiment,
    write_delta_experiment_report,
)


def _build_config(
    *,
    args: argparse.Namespace,
    seed: int,
    writer_pool: str,
    swap_on: bool,
    suite: str,
    report_dir: Path,
) -> DeltaExperimentConfig:
    return DeltaExperimentConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        steps=args.steps,
        lr=args.lr,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        seed=seed,
        task_suite=suite,
        block_size=args.block_size,
        memory_dim=args.memory_dim,
        top_k=args.top_k,
        layers=args.layers,
        alpha_scale=args.alpha_scale,
        gate_bias=args.gate_bias,
        conflict_margins=True,
        oracle_span_writer=True,
        payload_probe_layer_strategy="first_layer",
        payload_answer_loss_weight=args.payload_answer_loss_weight,
        payload_answer_loss_warmup_frac=0.1,
        payload_embedding_loss_weight=args.payload_embedding_loss_weight,
        stage2_swap_loss_weight=args.stage2_swap_loss_weight if swap_on else 0.0,
        stage2_swap_margin=2.0,
        stage2_swap_mode=args.stage2_swap_mode,
        lm_head_lora_loss_weight=args.lm_head_lora_loss_weight,
        lm_head_lora_rank=args.lm_head_lora_rank,
        lm_head_lora_scale=args.lm_head_lora_scale,
        logit_bias_loss_weight=args.logit_bias_loss_weight,
        logit_bias_scale=args.logit_bias_scale,
        eval_injection_modes="no_memory,delta_qv,payload_probe,logit_bias,lm_head_lora,oracle_logit_answer_embedding",
        control_margin_min=0.05,
        writer_pool=writer_pool,
        report_dir=str(report_dir),
    )


def _check_frozen_base(summary: dict[str, Any]) -> None:
    trainable_base = summary.get("trainable_base_params", 0)
    if trainable_base != 0:
        raise RuntimeError(
            f"frozen-base invariant violated: trainable_base_params = {trainable_base}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--model", default="mock-gemma")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--alpha-scale", type=float, default=0.2)
    parser.add_argument("--gate-bias", type=float, default=-1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--writer-pools", nargs="+", default=["mean", "attn"], choices=["mean", "attn"])
    parser.add_argument("--swap-options", nargs="+", default=["off", "on"], choices=["off", "on"])
    parser.add_argument("--payload-answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--payload-embedding-loss-weight", type=float, default=0.5)
    parser.add_argument("--stage2-swap-loss-weight", type=float, default=0.5)
    parser.add_argument("--stage2-swap-mode", default="lm_head_lora", choices=["payload_probe","logit_bias","lm_head_lora"])
    parser.add_argument("--lm-head-lora-loss-weight", type=float, default=0.5)
    parser.add_argument("--lm-head-lora-rank", type=int, default=4)
    parser.add_argument("--lm-head-lora-scale", type=float, default=50.0)
    parser.add_argument("--logit-bias-loss-weight", type=float, default=0.5)
    parser.add_argument("--logit-bias-scale", type=float, default=50.0)
    parser.add_argument("--report-root", default="reports/experiments/stage6")
    parser.add_argument("--update-readme", action="store_true", default=True)
    parser.add_argument("--no-update-readme", dest="update_readme", action="store_false")
    args = parser.parse_args()

    suite = "address_token_binding_single_token" if args.phase == 1 else "factual_capital_binding"
    report_root = (REPO_ROOT / args.report_root).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for pool in args.writer_pools:
        for swap_label in args.swap_options:
            swap_on = swap_label == "on"
            for seed in args.seeds:
                cell = f"phase{args.phase}_pool-{pool}_swap-{swap_label}_seed-{seed}"
                report_dir = report_root / cell
                cfg = _build_config(
                    args=args,
                    seed=seed,
                    writer_pool=pool,
                    swap_on=swap_on,
                    suite=suite,
                    report_dir=report_dir,
                )
                print(f"[stage6] running {cell}", flush=True)
                summary = run_delta_experiment(cfg)
                _check_frozen_base(summary)
                paths = write_delta_experiment_report(summary, str(report_dir))
                runs.append({
                    "cell": cell,
                    "pool": pool,
                    "swap": swap_label,
                    "seed": seed,
                    "report_dir": str(report_dir),
                    "diagnosis": summary.get("diagnosis"),
                    "paths": paths,
                })

    manifest_path = report_root / "stage6_manifest.json"
    manifest_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print(f"[stage6] manifest -> {manifest_path}")

    if args.update_readme:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "update_readme_charts.py")]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print("[stage6] WARNING: update_readme_charts.py exited non-zero", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
