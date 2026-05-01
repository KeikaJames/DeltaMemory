#!/usr/bin/env python3
"""Run the Delta Memory attention-injection demo.

The default uses the local mock Gemma-style model so CI and mock tests do
not download weights. Pass ``--model google/gemma-4-E2B`` to try the real base
model when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcvhc.core.config import RCVHCCleanConfig
from rcvhc.engine.attention_memory_engine import AttentionMemoryEngine
from rcvhc.gemma.model_adapter import load_model_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mock-gemma")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--input", default="examples/delta_attention_demo.txt")
    parser.add_argument("--store", default="runs/rcvhc_attention_store")
    parser.add_argument("--report-dir", default="reports/experiments")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)
    cfg = RCVHCCleanConfig(model_name=args.model, device=args.device, dtype=args.dtype, block_size=32, memory_dim=128)
    engine = AttentionMemoryEngine(bundle, cfg)
    ingest_summary = engine.ingest(text, layers="all")
    engine.save_store(args.store)
    result = engine.ask(
        "What is the secret code for unit XJQ-482?",
        answer="tulip-91",
        modes=list(cfg.main_modes),
        top_k=4,
        alpha_scale=0.2,
        gate_bias=-1.0,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "engineering_status": "success",
        "scientific_signal": "wiring_only_under_mock_or_cpu_constraints",
        "ingest": ingest_summary,
        "ask": result,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (report_dir / "demo_report.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"report": str(report_dir / "demo_report.md"), "summary": str(report_dir / "summary.json")}, indent=2))
    return 0


def _markdown(summary: dict) -> str:
    lines = [
        "# Delta Memory Experiment Demo Report",
        "",
        "## Status",
        "",
        f"- `engineering_status`: `{summary['engineering_status']}`",
        f"- `scientific_signal`: `{summary['scientific_signal']}`",
        f"- `model`: `{summary['model']}`",
        "",
        "This demo exercises externalized attention memory and Q/K/V intervention.",
        "It is not evidence of effectiveness unless aligned Delta beats zero, random, and shuffled controls.",
        "",
        "## Ingest",
        "",
    ]
    for key, value in summary["ingest"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Ask Comparisons", "", "| mode | nll | rank | top10 | q_delta | v_delta | gate_v |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for mode, row in summary["ask"]["comparisons"].items():
        metrics = row["metrics"]
        trace = row["qkv_trace"]
        lines.append(
            "| {mode} | {nll} | {rank} | {top10} | {q} | {v} | {g} |".format(
                mode=mode,
                nll=_fmt(metrics.get("answer_nll")),
                rank=_fmt(metrics.get("answer_rank")),
                top10=_fmt(metrics.get("top10")),
                q=_fmt(trace.get("q_delta_norm", 0.0)),
                v=_fmt(trace.get("v_delta_norm", 0.0)),
                g=_fmt(trace.get("gate_v", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Retrieved Memory",
            "",
            "Source snippets below are debug metadata only; they were not inserted into the prompt.",
            "",
        ]
    )
    for record in summary["ask"]["retrieved_memory"]:
        lines.append(
            f"- layer `{record['layer_id']}`, block `{record['block_id']}`, score `{record['score']:.3f}`, "
            f"range `{record['token_range']}`"
        )
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
