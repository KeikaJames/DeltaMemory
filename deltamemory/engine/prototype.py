"""Reusable runner for the Gemma4 Delta Memory prototype."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deltamemory.core.config import RCVHCCleanConfig
from deltamemory.engine.attention_memory_engine import AttentionMemoryEngine
from deltamemory.gemma.model_adapter import load_model_bundle


DEFAULT_MODES = [
    "no_memory",
    "raw_memory",
    "delta_qv",
    "delta_qv_zero",
    "delta_qv_random",
    "delta_qv_shuffled",
    "delta_qv_force_gate",
]


@dataclass
class PrototypeRunConfig:
    model: str = "google/gemma-4-E2B"
    device: str = "auto"
    dtype: str = "bfloat16"
    input_path: str = "examples/delta_attention_demo.txt"
    store_path: str = "runs/gemma4_attention_store"
    report_dir: str = "reports/experiments"
    question: str = "What is the secret code for unit XJQ-482?"
    answer: str = "tulip-91"
    block_size: int = 128
    memory_dim: int = 512
    top_k: int = 4
    layers: str = "all"
    alpha_scale: float = 0.2
    gate_bias: float = -1.0
    modes: tuple[str, ...] = tuple(DEFAULT_MODES)


def run_attention_memory_prototype(cfg: PrototypeRunConfig) -> dict[str, Any]:
    text = Path(cfg.input_path).read_text(encoding="utf-8")
    bundle = load_model_bundle(cfg.model, device=cfg.device, dtype=cfg.dtype)
    clean_cfg = RCVHCCleanConfig(
        model_name=cfg.model,
        memory_dim=cfg.memory_dim,
        block_size=cfg.block_size,
        top_k=cfg.top_k,
        layers=cfg.layers,
        alpha_scale=cfg.alpha_scale,
        gate_bias=cfg.gate_bias,
        device=cfg.device,
        dtype=cfg.dtype,
        main_modes=cfg.modes,
    )
    engine = AttentionMemoryEngine(bundle, clean_cfg)
    ingest = engine.ingest(text, layers=cfg.layers)
    engine.save_store(cfg.store_path)
    ask = engine.ask(
        cfg.question,
        answer=cfg.answer,
        modes=list(cfg.modes),
        top_k=cfg.top_k,
        alpha_scale=cfg.alpha_scale,
        gate_bias=cfg.gate_bias,
    )
    summary = {
        "config": asdict(cfg),
        "path": "external_attention_memory_to_gemma_qkv",
        "prompt_insertion_used": False,
        "source_text_debug_only": True,
        "claim_boundary": _claim_boundary(cfg.model),
        "ingest": ingest,
        "ask": ask,
        "diagnosis": diagnose_controls(ask),
    }
    return summary


def write_prototype_report(summary: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "gemma4_prototype_summary.json"
    report_path = report_dir / "gemma4_prototype_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return {"summary": str(summary_path), "report": str(report_path)}


def diagnose_controls(ask: dict[str, Any]) -> dict[str, Any]:
    comparisons = ask.get("comparisons", {})
    delta = comparisons.get("delta_qv", {})
    zero = comparisons.get("delta_qv_zero", {})
    random = comparisons.get("delta_qv_random", {})
    shuffled = comparisons.get("delta_qv_shuffled", {})
    forced = comparisons.get("delta_qv_force_gate", {})

    def nll(row):
        return (((row or {}).get("metrics") or {}).get("answer_nll"))

    def trace(row, key):
        return (((row or {}).get("qkv_trace") or {}).get(key, 0.0))

    delta_nll = nll(delta)
    zero_nll = nll(zero)
    random_nll = nll(random)
    shuffled_nll = nll(shuffled)
    forced_nll = nll(forced)
    beats_zero_random = (
        delta_nll is not None
        and zero_nll is not None
        and random_nll is not None
        and delta_nll < zero_nll
        and delta_nll < random_nll
    )
    beats_shuffled = delta_nll is not None and shuffled_nll is not None and delta_nll < shuffled_nll
    force_gate_stronger = forced_nll is not None and delta_nll is not None and forced_nll < delta_nll
    q_nonzero = trace(delta, "q_delta_norm") > 0.0
    v_nonzero = trace(delta, "v_delta_norm") > 0.0
    if beats_zero_random and beats_shuffled:
        signal = "aligned_delta_signal"
    elif beats_zero_random:
        signal = "non_aligned_delta_signal"
    elif q_nonzero and v_nonzero:
        signal = "wiring_signal_only"
    else:
        signal = "underpowered_or_no_signal"
    return {
        "delta_qv_q_nonzero": q_nonzero,
        "delta_qv_v_nonzero": v_nonzero,
        "delta_beats_zero_random": beats_zero_random,
        "delta_beats_shuffled": beats_shuffled,
        "force_gate_stronger": force_gate_stronger,
        "signal_status": signal,
    }


def _claim_boundary(model: str) -> str:
    if model in {"mock-gemma", "mock"}:
        return "engineering_wiring_only_mock_model"
    return "real_model_prototype_controls_required_for_scientific_claim"


def _markdown(summary: dict[str, Any]) -> str:
    cfg = summary["config"]
    diagnosis = summary["diagnosis"]
    lines = [
        "# Delta Memory Gemma4 Layerwise Injection Prototype",
        "",
        "## Config",
        "",
    ]
    for key in ["model", "device", "dtype", "block_size", "memory_dim", "top_k", "layers", "alpha_scale", "gate_bias"]:
        lines.append(f"- `{key}`: `{cfg[key]}`")
    lines.extend(
        [
            f"- `prompt_insertion_used`: `{summary['prompt_insertion_used']}`",
            f"- `claim_boundary`: `{summary['claim_boundary']}`",
            "",
            "## Ingest",
            "",
        ]
    )
    for key, value in summary["ingest"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "| mode | nll | rank | top10 | q_delta | v_delta | gate_v |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, row in summary["ask"]["comparisons"].items():
        metrics = row.get("metrics", {})
        trace = row.get("qkv_trace", {})
        lines.append(
            "| {mode} | {nll} | {rank} | {top10} | {q} | {v} | {gate} |".format(
                mode=mode,
                nll=_fmt(metrics.get("answer_nll")),
                rank=_fmt(metrics.get("answer_rank")),
                top10=_fmt(metrics.get("top10")),
                q=_fmt(trace.get("q_delta_norm", 0.0)),
                v=_fmt(trace.get("v_delta_norm", 0.0)),
                gate=_fmt(trace.get("gate_v", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Retrieved Memory",
            "",
            "Retrieved source snippets are debug metadata only and were not inserted into the prompt.",
            "",
        ]
    )
    for record in summary["ask"].get("retrieved_memory", []):
        lines.append(
            f"- layer `{record['layer_id']}`, block `{record['block_id']}`, score `{record['score']:.4f}`, "
            f"range `{record['token_range']}`, usage `{record['usage_mass']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
        ]
    )
    for key, value in diagnosis.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is the Gemma4-oriented Delta Memory path: external memory, per-layer top-k retrieval, and Q/K/V residual injection inside a frozen decoder LM.",
            "A scientific claim requires aligned Delta to beat zero, random, and shuffled controls on a real model run. Otherwise the result should be treated as engineering progress only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
