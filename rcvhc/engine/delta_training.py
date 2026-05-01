"""Train the frozen-backbone Delta Memory Q/V adapter on a single cleanroom task.

This is deliberately small. It answers the practical question: the base Gemma
model does not know how to use Delta Memory by itself, so we train only
the external writer and Q/K/V intervention modules while keeping all model
weights frozen.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rcvhc.core.config import RCVHCCleanConfig, resolve_layer_policy
from rcvhc.core.types import AttentionMemoryItem
from rcvhc.engine.attention_memory_engine import compute_answer_metrics
from rcvhc.gemma.attention_injector import GemmaAttentionInjector, QKVDeltaProjector
from rcvhc.gemma.model_adapter import exposed_qkv_layers, get_hidden_size, load_model_bundle, trainable_base_params
from rcvhc.memory.writer import RCVHCWriter, fit_memory_dim, split_source_snippets


TRAIN_EVAL_MODES = [
    "no_memory",
    "raw_memory",
    "delta_qv",
    "delta_qv_zero",
    "delta_qv_random",
    "delta_qv_shuffled",
    "delta_qv_wrong_layer",
    "delta_qv_force_gate",
]


@dataclass
class DeltaTrainingConfig:
    model: str = "google/gemma-4-E2B"
    device: str = "auto"
    dtype: str = "bfloat16"
    input_path: str = "examples/delta_attention_demo.txt"
    report_dir: str = "reports/cleanroom/gemma4_training"
    question: str = "What is the secret code for unit XJQ-482?"
    answer: str = "tulip-91"
    steps: int = 5
    lr: float = 1e-3
    block_size: int = 128
    memory_dim: int = 512
    top_k: int = 1
    layers: str = "all"
    alpha_scale: float = 0.2
    gate_bias: float = -1.0


def run_delta_training(cfg: DeltaTrainingConfig) -> dict[str, Any]:
    torch.manual_seed(7)
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
    )
    hidden_size = get_hidden_size(bundle.model)
    writer = RCVHCWriter(hidden_size, cfg.memory_dim, cfg.block_size).to(bundle.device, dtype=bundle.dtype)
    projector = QKVDeltaProjector(cfg.memory_dim, hidden_size, cfg.alpha_scale, cfg.gate_bias).to(
        bundle.device, dtype=bundle.dtype
    )
    injector = GemmaAttentionInjector(bundle.model, projector)

    context = _encode(bundle.tokenizer, text, bundle.device)
    with torch.no_grad():
        context_out = bundle.model(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
    if context_out.hidden_states is None or context_out.attentions is None:
        raise RuntimeError("Delta training requires hidden_states and attentions")
    exposed = exposed_qkv_layers(bundle.model) or list(range(max(0, len(context_out.hidden_states) - 1)))
    layer_ids = resolve_layer_policy(cfg.layers, exposed)
    injector.materialize_for_layers(layer_ids, bundle.device, bundle.dtype)

    prompt_text, answer_start, answer_ids = _metric_prompt(bundle.tokenizer, cfg.question, cfg.answer)
    prompt = _encode(bundle.tokenizer, prompt_text, bundle.device)
    with torch.no_grad():
        base_prompt = bundle.model(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
    query = fit_memory_dim(base_prompt.hidden_states[-1].mean(dim=(0, 1)).detach().float().cpu(), cfg.memory_dim)
    snippets = split_source_snippets(bundle.tokenizer, context["input_ids"], cfg.block_size)

    optimizer = torch.optim.AdamW(list(writer.parameters()) + list(projector.parameters()), lr=cfg.lr)
    initial = _evaluate_modes(
        writer,
        injector,
        context_out,
        prompt,
        base_prompt.logits,
        query,
        snippets,
        layer_ids,
        cfg,
        answer_start,
        answer_ids,
    )

    train_rows = []
    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        selected = _select_topk_by_layer(_live_memories(writer, context_out, snippets, layer_ids, cfg), query, cfg.top_k)
        result = injector.forward_layers(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            memories_by_layer=selected,
            mode="delta_qv",
        )
        loss = _answer_loss(result.logits, answer_start, answer_ids)
        loss.backward()
        grad_norm = _grad_norm(list(writer.parameters()) + list(projector.parameters()))
        optimizer.step()
        train_rows.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "q_delta_norm": result.trace.q_delta_norm,
                "v_delta_norm": result.trace.v_delta_norm,
                "gate_v": result.trace.gate_v,
            }
        )

    final = _evaluate_modes(
        writer,
        injector,
        context_out,
        prompt,
        base_prompt.logits,
        query,
        snippets,
        layer_ids,
        cfg,
        answer_start,
        answer_ids,
    )
    summary = {
        "config": asdict(cfg),
        "layer_ids": layer_ids,
        "trainable_base_params": trainable_base_params(bundle.model),
        "prompt_insertion_used": False,
        "source_text_debug_only": True,
        "training_scope": "writer_and_qkv_projector_only",
        "initial": initial,
        "train": train_rows,
        "final": final,
        "diagnosis": _diagnose_training(initial, final),
    }
    return summary


def write_training_report(summary: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "delta_training_summary.json"
    report_path = report_dir / "delta_training_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_training_markdown(summary), encoding="utf-8")
    return {"summary": str(summary_path), "report": str(report_path)}


def _evaluate_modes(
    writer: RCVHCWriter,
    injector: GemmaAttentionInjector,
    context_out,
    prompt: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    query: torch.Tensor,
    snippets: list[str],
    layer_ids: list[int],
    cfg: DeltaTrainingConfig,
    answer_start: int,
    answer_ids: list[int],
) -> dict[str, Any]:
    memories = _select_topk_by_layer(_live_memories(writer, context_out, snippets, layer_ids, cfg), query, cfg.top_k)
    rows: dict[str, Any] = {}
    for mode in TRAIN_EVAL_MODES:
        if mode == "no_memory":
            logits = base_logits
            trace = {}
        elif mode == "raw_memory":
            logits, trace = _raw_late_readout(base_logits, prompt["input_ids"], memories, injector)
        else:
            result = injector.forward_layers(
                input_ids=prompt["input_ids"],
                attention_mask=prompt["attention_mask"],
                memories_by_layer=memories,
                mode=mode,
            )
            logits = result.logits
            trace = result.trace.as_dict()
        rows[mode] = {
            "metrics": compute_answer_metrics(logits, prompt["input_ids"], answer_start, answer_ids),
            "qkv_trace": trace,
        }
    return rows


def _live_memories(
    writer: RCVHCWriter,
    context_out,
    snippets: list[str],
    layer_ids: list[int],
    cfg: DeltaTrainingConfig,
) -> list[AttentionMemoryItem]:
    items: list[AttentionMemoryItem] = []
    for layer_id in layer_ids:
        items.extend(
            writer.write_layer(
                layer_id=layer_id,
                h_in=context_out.hidden_states[layer_id].detach(),
                h_out=context_out.hidden_states[layer_id + 1].detach(),
                attn=context_out.attentions[layer_id].detach(),
                token_offset=0,
                source_text_by_block=snippets,
            )
        )
    return items


def _select_topk_by_layer(memories: list[AttentionMemoryItem], query: torch.Tensor, top_k: int) -> dict[int, list[AttentionMemoryItem]]:
    selected: dict[int, list[AttentionMemoryItem]] = {}
    for layer_id in sorted({item.layer_id for item in memories}):
        layer_memories = [item for item in memories if item.layer_id == layer_id]
        selected[layer_id] = _select_topk(layer_memories, query, top_k)
    return selected


def _select_topk(memories: list[AttentionMemoryItem], query: torch.Tensor, top_k: int) -> list[AttentionMemoryItem]:
    if not memories:
        return []
    q = query.float().to(memories[0].raw_key.device)
    keys = torch.stack([item.raw_key.float() for item in memories], dim=0)
    scores = F.normalize(keys, dim=-1).matmul(F.normalize(q, dim=0))
    _, idx = scores.topk(min(top_k, len(memories)))
    selected = []
    for rank, item_idx in enumerate(idx.tolist()):
        item = memories[int(item_idx)]
        item.metadata = dict(item.metadata)
        item.metadata["retrieval_rank"] = rank
        selected.append(item)
    return selected


def _raw_late_readout(
    base_logits: torch.Tensor,
    input_ids: torch.Tensor,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    injector: GemmaAttentionInjector,
) -> tuple[torch.Tensor, dict[str, float]]:
    memories = [item for layer_memories in memories_by_layer.values() for item in layer_memories]
    if not memories:
        return base_logits, {}
    model = injector.model
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_hidden_states=True, output_attentions=False, use_cache=False)
    hidden = out.hidden_states[-1]
    raw = torch.stack([item.raw_value.to(hidden.device, hidden.dtype) for item in memories], dim=0).mean(dim=0)
    update = fit_memory_dim(raw, hidden.shape[-1]).to(hidden.device, hidden.dtype)
    hidden_prime = hidden + 0.05 * update.view(1, 1, -1)
    logits = model.get_output_embeddings()(hidden_prime)
    return logits, {"hidden_delta_norm": float((hidden_prime - hidden).detach().float().norm().cpu())}


def _answer_loss(logits: torch.Tensor, answer_start: int, answer_ids: list[int]) -> torch.Tensor:
    losses = []
    for offset, token_id in enumerate(answer_ids):
        logit_pos = answer_start + offset - 1
        if 0 <= logit_pos < logits.shape[1]:
            target = torch.tensor([int(token_id)], device=logits.device)
            losses.append(F.cross_entropy(logits[0, logit_pos].float().view(1, -1), target))
    if not losses:
        raise ValueError("answer span produced no supervised positions")
    return torch.stack(losses).mean()


def _metric_prompt(tokenizer, question: str, answer: str) -> tuple[str, int, list[int]]:
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    return prompt + " " + answer, len(prompt_ids), answer_ids


def _encode(tokenizer, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return {"input_ids": encoded["input_ids"].long().to(device), "attention_mask": encoded["attention_mask"].long().to(device)}


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().cpu())
    return total**0.5


def _diagnose_training(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    def nll(rows, mode):
        return rows[mode]["metrics"]["answer_nll"]

    delta_drop = nll(initial, "delta_qv") - nll(final, "delta_qv")
    final_delta = nll(final, "delta_qv")
    final_zero = nll(final, "delta_qv_zero")
    final_random = nll(final, "delta_qv_random")
    final_shuffled = nll(final, "delta_qv_shuffled")
    q_nonzero = final["delta_qv"]["qkv_trace"].get("q_delta_norm", 0.0) > 0.0
    v_nonzero = final["delta_qv"]["qkv_trace"].get("v_delta_norm", 0.0) > 0.0
    return {
        "delta_qv_nll_drop": delta_drop,
        "trained_delta_beats_zero": final_delta < final_zero,
        "trained_delta_beats_random": final_delta < final_random,
        "trained_delta_beats_shuffled": final_delta < final_shuffled,
        "q_delta_nonzero": q_nonzero,
        "v_delta_nonzero": v_nonzero,
        "adapter_learned_to_use_delta": delta_drop > 0.0 and q_nonzero and v_nonzero,
    }


def _training_markdown(summary: dict[str, Any]) -> str:
    cfg = summary["config"]
    lines = [
        "# Delta Memory Q/V Adapter Training Report",
        "",
        "## Config",
        "",
    ]
    for key in ["model", "device", "dtype", "steps", "lr", "block_size", "memory_dim", "top_k", "alpha_scale", "gate_bias"]:
        lines.append(f"- `{key}`: `{cfg[key]}`")
    lines.extend(
        [
            f"- `layer_ids`: `{summary['layer_ids']}`",
            f"- `trainable_base_params`: `{summary['trainable_base_params']}`",
            f"- `training_scope`: `{summary['training_scope']}`",
            f"- `prompt_insertion_used`: `{summary['prompt_insertion_used']}`",
            "",
            "## Before / After",
            "",
            "| mode | initial_nll | final_nll | final_rank | q_delta | v_delta | gate_v |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in TRAIN_EVAL_MODES:
        initial = summary["initial"][mode]
        final = summary["final"][mode]
        lines.append(
            "| {mode} | {initial_nll:.4f} | {final_nll:.4f} | {rank:.4f} | {q:.4f} | {v:.4f} | {gate:.4f} |".format(
                mode=mode,
                initial_nll=float(initial["metrics"]["answer_nll"]),
                final_nll=float(final["metrics"]["answer_nll"]),
                rank=float(final["metrics"]["answer_rank"]),
                q=float(final["qkv_trace"].get("q_delta_norm", 0.0)),
                v=float(final["qkv_trace"].get("v_delta_norm", 0.0)),
                gate=float(final["qkv_trace"].get("gate_v", 0.0)),
            )
        )
    lines.extend(["", "## Diagnosis", ""])
    for key, value in summary["diagnosis"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The frozen base model is not trained. This run trains only the Delta Memory writer and Q/V intervention adapter, showing whether the external Delta path can be optimized.",
            "A stronger scientific claim still requires trained Delta to beat zero, random, and shuffled controls on held-out examples.",
        ]
    )
    return "\n".join(lines) + "\n"
