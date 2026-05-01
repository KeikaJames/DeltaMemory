#!/usr/bin/env python3
"""Create reproducible Delta Memory experiment matrix commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASK_SUITES = [
    "single_fact_late_reference",
    "multi_hop_binding",
    "temporal_overwrite",
    "paraphrase_nolima_style",
    "adversarial_negative",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/cleanroom/delta_memory_matrix.json")
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--memory-dims", default="256,512")
    parser.add_argument("--top-ks", default="2,4")
    parser.add_argument("--layers", default="all,max_exposed")
    args = parser.parse_args()

    rows = []
    for task_suite in TASK_SUITES:
        for seed in _ints(args.seeds):
            for memory_dim in _ints(args.memory_dims):
                for top_k in _ints(args.top_ks):
                    for layers in _strings(args.layers):
                        report_dir = (
                            f"reports/cleanroom/delta_memory/{task_suite}/"
                            f"seed{seed}_mem{memory_dim}_top{top_k}_{layers}"
                        )
                        command = [
                            "python",
                            "scripts/run_delta_experiment.py",
                            "--model",
                            args.model,
                            "--device",
                            args.device,
                            "--dtype",
                            args.dtype,
                            "--task-suite",
                            task_suite,
                            "--seed",
                            str(seed),
                            "--train-samples",
                            str(args.train_samples),
                            "--eval-samples",
                            str(args.eval_samples),
                            "--steps",
                            str(args.steps),
                            "--memory-dim",
                            str(memory_dim),
                            "--top-k",
                            str(top_k),
                            "--layers",
                            layers,
                            "--report-dir",
                            report_dir,
                        ]
                        rows.append(
                            {
                                "task_suite": task_suite,
                                "seed": seed,
                                "memory_dim": memory_dim,
                                "top_k": top_k,
                                "layers": layers,
                                "report_dir": report_dir,
                                "command": command,
                            }
                        )
    payload = {
        "description": "Delta Memory scaling and ablation matrix. `layers=all` is the main path; `max_exposed` is last-layer ablation.",
        "num_runs": len(rows),
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "num_runs": len(rows)}, indent=2, sort_keys=True))
    return 0


def _ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
