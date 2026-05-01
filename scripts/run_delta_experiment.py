#!/usr/bin/env python3
"""Run a small multi-example Delta Q/V experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcvhc.engine.delta_experiment import DeltaExperimentConfig, run_delta_experiment, write_delta_experiment_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mock-gemma")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--eval-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-suite", default="single_fact_late_reference")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--alpha-scale", type=float, default=0.2)
    parser.add_argument("--gate-bias", type=float, default=-1.0)
    parser.add_argument("--report-dir", default="reports/cleanroom/delta_experiment")
    args = parser.parse_args()
    cfg = DeltaExperimentConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        steps=args.steps,
        lr=args.lr,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        seed=args.seed,
        task_suite=args.task_suite,
        block_size=args.block_size,
        memory_dim=args.memory_dim,
        top_k=args.top_k,
        layers=args.layers,
        alpha_scale=args.alpha_scale,
        gate_bias=args.gate_bias,
        report_dir=args.report_dir,
    )
    summary = run_delta_experiment(cfg)
    paths = write_delta_experiment_report(summary, args.report_dir)
    print(json.dumps(paths | {"diagnosis": summary["diagnosis"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
