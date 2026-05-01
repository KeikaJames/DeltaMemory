"""Multi-example Delta Q/V experiment runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from rcvhc.core.config import resolve_layer_policy
from rcvhc.core.types import AttentionMemoryItem
from rcvhc.engine.attention_memory_engine import compute_answer_metrics
from rcvhc.engine.delta_dataset import DeltaExample, make_delta_memory_examples
from rcvhc.engine.delta_training import TRAIN_EVAL_MODES, _answer_loss, _grad_norm
from rcvhc.engine.statistics import primary_delta_memory_statistics
from rcvhc.gemma.attention_injector import GemmaAttentionInjector, QKVDeltaProjector
from rcvhc.gemma.model_adapter import exposed_qkv_layers, get_hidden_size, get_vocab_size, load_model_bundle, trainable_base_params
from rcvhc.memory.writer import RCVHCWriter, fit_memory_dim, split_source_snippets


EXPERIMENT_EVAL_MODES = [*TRAIN_EVAL_MODES, "logit_bias", "payload_probe", "oracle_logit_answer_embedding"]


@dataclass
class DeltaExperimentConfig:
    model: str = "mock-gemma"
    device: str = "cpu"
    dtype: str = "float32"
    steps: int = 8
    lr: float = 1e-3
    train_samples: int = 4
    eval_samples: int = 4
    seed: int = 0
    task_suite: str = "single_fact_late_reference"
    block_size: int = 64
    memory_dim: int = 128
    top_k: int = 2
    layers: str = "all"
    alpha_scale: float = 0.2
    gate_bias: float = -1.0
    conflict_margins: bool = False
    contrastive_margin_weight: float = 0.0
    contrastive_margin: float = 0.5
    oracle_contrastive_weight: float = 0.0
    address_margin_weight: float = 0.0
    address_margin: float = 0.1
    address_score_scale: float = 16.0
    shared_memory_retrieval: bool = False
    identity_gate_beta: float = 64.0
    identity_gate_tau: float = 0.01
    oracle_span_writer: bool = False
    logit_bias_loss_weight: float = 0.0
    logit_bias_scale: float = 1.0
    payload_answer_loss_weight: float = 0.0
    payload_probe_layer_strategy: str = "mean_all"  # mean_all | last_layer | first_layer | best_layer
    payload_embedding_loss_weight: float = 0.0
    stage2_swap_loss_weight: float = 0.0
    stage2_swap_margin: float = 2.0
    stage2_swap_mode: str = "payload_probe"  # payload_probe | logit_bias
    eval_injection_modes: str = "all"
    control_margin_min: float = 0.05
    report_dir: str = "reports/experiments/delta_experiment"


@dataclass
class PreparedDeltaExample:
    example: DeltaExample
    context_out: Any
    prompt: dict[str, torch.Tensor]
    base_logits: torch.Tensor
    query_hidden: torch.Tensor
    snippets: list[str]
    answer_start: int
    answer_ids: list[int]
    address_token_range: tuple[int, int] | None = None
    value_token_range: tuple[int, int] | None = None


def run_delta_experiment(cfg: DeltaExperimentConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    bundle = load_model_bundle(cfg.model, device=cfg.device, dtype=cfg.dtype)
    hidden_size = get_hidden_size(bundle.model)
    writer = RCVHCWriter(hidden_size, cfg.memory_dim, cfg.block_size).to(bundle.device, dtype=bundle.dtype)
    logit_projector = PayloadLogitProjector(cfg.memory_dim, get_vocab_size(bundle.model), cfg.logit_bias_scale).to(
        bundle.device, dtype=bundle.dtype
    )
    payload_probe = PayloadAnswerProbe(cfg.memory_dim, hidden_size, bundle.model.get_output_embeddings()).to(
        bundle.device, dtype=bundle.dtype
    )
    projector = QKVDeltaProjector(cfg.memory_dim, hidden_size, cfg.alpha_scale, cfg.gate_bias).to(
        bundle.device, dtype=bundle.dtype
    )
    injector = GemmaAttentionInjector(bundle.model, projector)

    train_examples = make_delta_memory_examples(cfg.task_suite, cfg.train_samples, seed=cfg.seed, start_id=0)
    eval_examples = make_delta_memory_examples(cfg.task_suite, cfg.eval_samples, seed=cfg.seed + 10_000, start_id=10_000)
    first_context = _encode(bundle.tokenizer, train_examples[0].text, bundle.device)
    with torch.no_grad():
        first_out = bundle.model(
            input_ids=first_context["input_ids"],
            attention_mask=first_context["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
    exposed = exposed_qkv_layers(bundle.model) or list(range(max(0, len(first_out.hidden_states) - 1)))
    layer_ids = resolve_layer_policy(cfg.layers, exposed)
    injector.materialize_for_layers(layer_ids, bundle.device, bundle.dtype)

    train_prepared = [_prepare_example(bundle, example, cfg) for example in train_examples]
    eval_prepared = [_prepare_example(bundle, example, cfg) for example in eval_examples]
    optimizer = torch.optim.AdamW(
        _trainable_parameters(writer, projector, logit_projector, payload_probe),
        lr=cfg.lr,
    )

    initial_eval = evaluate_prepared(writer, injector, logit_projector, payload_probe, eval_prepared, layer_ids, cfg)
    train_rows = []
    for step in range(1, cfg.steps + 1):
        sample_idx = (step - 1) % len(train_prepared)
        sample = train_prepared[sample_idx]
        optimizer.zero_grad(set_to_none=True)
        train_pool = _memory_pool(writer, train_prepared, layer_ids, cfg) if cfg.shared_memory_retrieval else None
        sample_query = _query_key(writer, sample)
        memories = _select_topk_by_layer(
            train_pool if train_pool is not None else _live_memories(writer, sample, layer_ids, cfg),
            sample_query,
            cfg.top_k,
            cfg.identity_gate_beta,
            cfg.identity_gate_tau,
        )
        result = injector.forward_layers(
            input_ids=sample.prompt["input_ids"],
            attention_mask=sample.prompt["attention_mask"],
            memories_by_layer=memories,
            mode="delta_qv",
        )
        answer_loss = _answer_loss(result.logits, sample.answer_start, sample.answer_ids)
        contrastive_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        oracle_contrastive_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        oracle_margin_advantage = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        margin_advantage = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        address_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        address_margin_value = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        logit_bias_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        payload_answer_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        payload_embedding_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        stage2_swap_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        stage2_binding_margin_value = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        stage2_swap_margin_value = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        foreign = _foreign_sample(train_prepared, sample_idx) if len(train_prepared) > 1 else sample
        if cfg.logit_bias_loss_weight > 0.0:
            logit_bias_logits, _ = _logit_bias_readout(sample, memories, logit_projector, cfg.payload_probe_layer_strategy)
            logit_bias_loss = _answer_loss(logit_bias_logits, sample.answer_start, sample.answer_ids)
        if cfg.payload_answer_loss_weight > 0.0:
            payload_answer_loss = _payload_answer_loss(payload_probe, memories, sample.answer_ids, cfg.payload_probe_layer_strategy)
        if cfg.payload_embedding_loss_weight > 0.0:
            payload_embedding_loss = _payload_embedding_loss(
                payload_probe,
                memories,
                sample.answer_ids,
                cfg.payload_probe_layer_strategy,
            )
        if cfg.stage2_swap_loss_weight > 0.0 and len(train_prepared) > 1:
            foreign_query = _query_key(writer, foreign)
            foreign_memories = _select_topk_by_layer(
                train_pool if train_pool is not None else _live_memories(writer, foreign, layer_ids, cfg),
                foreign_query,
                cfg.top_k,
                cfg.identity_gate_beta,
                cfg.identity_gate_tau,
            )
            stage2_swap_loss, stage2_binding_margin_value, stage2_swap_margin_value = _stage2_swap_loss(
                sample,
                foreign,
                memories,
                foreign_memories,
                payload_probe,
                logit_projector,
                cfg.stage2_swap_mode,
                cfg.stage2_swap_margin,
                cfg.payload_probe_layer_strategy,
            )
        if cfg.address_margin_weight > 0.0 and len(train_prepared) > 1:
            address_loss, address_margin_value = _address_ranking_loss(writer, sample, train_prepared, layer_ids, cfg)
        if cfg.contrastive_margin_weight > 0.0 and len(train_prepared) > 1:
            foreign_query = _query_key(writer, foreign)
            foreign_memories = _select_topk_by_layer(
                train_pool if train_pool is not None else _live_memories(writer, foreign, layer_ids, cfg),
                foreign_query,
                cfg.top_k,
                cfg.identity_gate_beta,
                cfg.identity_gate_tau,
            )
            foreign_answer_with_correct_memory = _candidate_answer_loss(
                bundle,
                injector,
                sample.example.question,
                foreign.example.answer,
                memories,
                "delta_qv",
            )
            correct_answer_with_foreign_memory = _candidate_answer_loss(
                bundle,
                injector,
                sample.example.question,
                sample.example.answer,
                foreign_memories,
                "delta_qv",
            )
            foreign_answer_with_foreign_memory = _candidate_answer_loss(
                bundle,
                injector,
                sample.example.question,
                foreign.example.answer,
                foreign_memories,
                "delta_qv",
            )
            correct_margin = foreign_answer_with_correct_memory - answer_loss
            foreign_margin = foreign_answer_with_foreign_memory - correct_answer_with_foreign_memory
            margin_advantage = correct_margin - foreign_margin
            target_margin = torch.as_tensor(cfg.contrastive_margin, device=answer_loss.device, dtype=answer_loss.dtype)
            contrastive_loss = F.relu(target_margin - margin_advantage)
        if cfg.oracle_contrastive_weight > 0.0 and len(train_prepared) > 1:
            oracle_correct = _select_sample_topk_by_layer(
                train_pool if train_pool is not None else _live_memories(writer, sample, layer_ids, cfg),
                sample_query,
                sample.example.sample_id,
                cfg.top_k,
            )
            paired_id = sample.example.paired_sample_id if sample.example.paired_sample_id is not None else foreign.example.sample_id
            oracle_foreign = _select_sample_topk_by_layer(
                train_pool if train_pool is not None else _live_memories(writer, foreign, layer_ids, cfg),
                sample_query,
                paired_id,
                cfg.top_k,
            )
            correct_with_correct = _candidate_answer_loss(
                bundle, injector, sample.example.question, sample.example.answer, oracle_correct, "delta_qv"
            )
            foreign_with_correct = _candidate_answer_loss(
                bundle, injector, sample.example.question, foreign.example.answer, oracle_correct, "delta_qv"
            )
            correct_with_foreign = _candidate_answer_loss(
                bundle, injector, sample.example.question, sample.example.answer, oracle_foreign, "delta_qv"
            )
            foreign_with_foreign = _candidate_answer_loss(
                bundle, injector, sample.example.question, foreign.example.answer, oracle_foreign, "delta_qv"
            )
            correct_memory_margin = foreign_with_correct - correct_with_correct
            foreign_memory_margin = correct_with_foreign - foreign_with_foreign
            target_margin = torch.as_tensor(cfg.contrastive_margin, device=answer_loss.device, dtype=answer_loss.dtype)
            oracle_contrastive_loss = F.relu(target_margin - correct_memory_margin) + F.relu(
                target_margin - foreign_memory_margin
            )
            oracle_margin_advantage = correct_memory_margin + foreign_memory_margin
        loss = (
            answer_loss
            + cfg.contrastive_margin_weight * contrastive_loss
            + cfg.oracle_contrastive_weight * oracle_contrastive_loss
            + cfg.address_margin_weight * address_loss
            + cfg.logit_bias_loss_weight * logit_bias_loss
            + cfg.payload_answer_loss_weight * payload_answer_loss
            + cfg.payload_embedding_loss_weight * payload_embedding_loss
            + cfg.stage2_swap_loss_weight * stage2_swap_loss
        )
        loss.backward()
        grad_norm = _grad_norm(_trainable_parameters(writer, projector, logit_projector, payload_probe))
        optimizer.step()
        train_rows.append(
            {
                "step": step,
                "sample_id": sample.example.sample_id,
                "loss": float(loss.detach().cpu()),
                "answer_loss": float(answer_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "contrastive_margin_advantage": float(margin_advantage.detach().cpu()),
                "oracle_contrastive_loss": float(oracle_contrastive_loss.detach().cpu()),
                "oracle_margin_advantage": float(oracle_margin_advantage.detach().cpu()),
                "address_loss": float(address_loss.detach().cpu()),
                "address_margin": float(address_margin_value.detach().cpu()),
                "logit_bias_loss": float(logit_bias_loss.detach().cpu()),
                "payload_answer_loss": float(payload_answer_loss.detach().cpu()),
                "payload_embedding_loss": float(payload_embedding_loss.detach().cpu()),
                "stage2_swap_loss": float(stage2_swap_loss.detach().cpu()),
                "stage2_binding_margin": float(stage2_binding_margin_value.detach().cpu()),
                "stage2_swap_margin": float(stage2_swap_margin_value.detach().cpu()),
                "grad_norm": grad_norm,
                "q_delta_norm": result.trace.q_delta_norm,
                "v_delta_norm": result.trace.v_delta_norm,
                "gate_v": result.trace.gate_v,
            }
        )

    final_train = evaluate_prepared(writer, injector, logit_projector, payload_probe, train_prepared, layer_ids, cfg)
    final_eval = evaluate_prepared(writer, injector, logit_projector, payload_probe, eval_prepared, layer_ids, cfg)
    conflict_margins = (
        evaluate_conflict_margins(bundle, writer, injector, logit_projector, payload_probe, eval_prepared, layer_ids, cfg)
        if cfg.conflict_margins
        else None
    )
    summary = {
        "config": asdict(cfg),
        "layer_ids": layer_ids,
        "trainable_base_params": trainable_base_params(bundle.model),
        "prompt_insertion_used": False,
        "retrieval_query_uses_answer": False,
        "retrieval_key": "address_query_to_address_key",
        "address_key_separate_from_payload": True,
        "oracle_span_writer": cfg.oracle_span_writer,
        "logit_bias_payload": True,
        "source_text_debug_only": True,
        "train_examples": [example.as_dict() for example in train_examples],
        "eval_examples": [example.as_dict() for example in eval_examples],
        "initial_eval": initial_eval,
        "train": train_rows,
        "final_train": final_train,
        "final_eval": final_eval,
        "conflict_margins": conflict_margins,
        "stage2_binding_summary": _stage2_binding_summary(final_eval, conflict_margins),
        "statistics": primary_delta_memory_statistics(final_eval, seed=cfg.seed),
        "diagnosis": diagnose_experiment(initial_eval, final_eval, min_control_gap=cfg.control_margin_min),
    }
    return summary


class PayloadLogitProjector(nn.Module):
    def __init__(self, memory_dim: int, vocab_size: int, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)
        self.norm = nn.LayerNorm(memory_dim)
        self.to_logits = nn.Linear(memory_dim, vocab_size)
        nn.init.zeros_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        return self.scale * self.to_logits(self.norm(payload))


class PayloadAnswerProbe(nn.Module):
    def __init__(self, memory_dim: int, hidden_size: int, output_embeddings: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(memory_dim)
        self.to_hidden = nn.Linear(memory_dim, hidden_size)
        self.output_embeddings = output_embeddings
        for param in self.output_embeddings.parameters():
            param.requires_grad_(False)

    def project_hidden(self, payload: torch.Tensor) -> torch.Tensor:
        return self.to_hidden(self.norm(payload))

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        return self.output_embeddings(self.project_hidden(payload))


def _trainable_parameters(*modules: nn.Module) -> list[torch.nn.Parameter]:
    return [param for module in modules for param in module.parameters() if param.requires_grad]


def evaluate_prepared(
    writer: RCVHCWriter,
    injector: GemmaAttentionInjector,
    logit_projector: PayloadLogitProjector,
    payload_probe: PayloadAnswerProbe,
    prepared: list[PreparedDeltaExample],
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> dict[str, Any]:
    per_sample = []
    eval_modes = _eval_modes(cfg)
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in eval_modes}
    shared_pool = _memory_pool(writer, prepared, layer_ids, cfg) if cfg.shared_memory_retrieval else None
    for sample_idx, sample in enumerate(prepared):
        sample_query = _query_key(writer, sample)
        memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, sample, layer_ids, cfg),
            sample_query,
            cfg.top_k,
            cfg.identity_gate_beta,
            cfg.identity_gate_tau,
        )
        wrong_query_sample = _foreign_sample(prepared, sample_idx)
        wrong_query = _query_key(writer, wrong_query_sample)
        wrong_query_memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, wrong_query_sample, layer_ids, cfg),
            wrong_query,
            cfg.top_k,
            cfg.identity_gate_beta,
            cfg.identity_gate_tau,
        )
        sample_rows = {}
        for mode in eval_modes:
            if mode == "no_memory":
                logits = sample.base_logits
                trace = {}
            elif mode in {"raw_memory", "hidden_retrieval"}:
                logits, trace = _raw_late_readout(sample, memories, injector)
            elif mode == "retrieved_attention":
                logits, trace = _retrieved_attention_readout(sample, memories, injector)
            elif mode == "logit_bias":
                logits, trace = _logit_bias_readout(sample, memories, logit_projector, cfg.payload_probe_layer_strategy)
            elif mode == "payload_probe":
                logits, trace = _payload_probe_logits(memories, payload_probe, cfg.payload_probe_layer_strategy)
                row = {
                    "metrics": _payload_probe_metrics(logits, sample.answer_ids),
                    "qkv_trace": trace,
                }
                sample_rows[mode] = row
                by_mode[mode].append(row)
                continue
            elif mode == "oracle_logit_answer_embedding":
                logits, trace = _oracle_answer_embedding_bias(
                    sample.base_logits,
                    sample.answer_ids,
                    payload_probe.output_embeddings,
                    cfg.logit_bias_scale,
                )
            else:
                selected_memories = wrong_query_memories if mode == "delta_qv_wrong_query" else memories
                result = injector.forward_layers(
                    input_ids=sample.prompt["input_ids"],
                    attention_mask=sample.prompt["attention_mask"],
                    memories_by_layer=selected_memories,
                    mode=mode,
                )
                logits = result.logits
                trace = result.trace.as_dict()
            row = {
                "metrics": compute_answer_metrics(logits, sample.prompt["input_ids"], sample.answer_start, sample.answer_ids),
                "qkv_trace": trace,
            }
            sample_rows[mode] = row
            by_mode[mode].append(row)
        per_sample.append({"sample_id": sample.example.sample_id, "unit": sample.example.unit, "answer": sample.example.answer, "modes": sample_rows})
    return {"aggregate": _aggregate_by_mode(by_mode), "samples": per_sample}


def evaluate_conflict_margins(
    bundle,
    writer: RCVHCWriter,
    injector: GemmaAttentionInjector,
    logit_projector: PayloadLogitProjector,
    payload_probe: PayloadAnswerProbe,
    prepared: list[PreparedDeltaExample],
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> dict[str, Any]:
    if len(prepared) < 2:
        return {"aggregate": {}, "samples": []}
    modes = [
        "no_memory",
        "delta_qv",
        "delta_qv_identity_gate",
        "delta_qv_wrong_query",
        "delta_qv_oracle_correct",
        "delta_qv_oracle_paired",
        "delta_qv_oracle_correct_address_paired_payload",
        "delta_qv_oracle_paired_address_correct_payload",
        "logit_bias_oracle_correct",
        "logit_bias_oracle_paired",
        "logit_bias_correct_address_paired_payload",
        "logit_bias_paired_address_correct_payload",
        "payload_probe_oracle_correct",
        "payload_probe_oracle_paired",
        "payload_probe_correct_address_paired_payload",
        "payload_probe_paired_address_correct_payload",
    ]
    by_mode: dict[str, list[float]] = {mode: [] for mode in modes}
    token_by_mode: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
    address_rows: list[dict[str, float]] = []
    samples: list[dict[str, Any]] = []
    shared_pool = _memory_pool(writer, prepared, layer_ids, cfg) if cfg.shared_memory_retrieval else None
    for sample_idx, sample in enumerate(prepared):
        foreign = _foreign_sample(prepared, sample_idx)
        diagnostic_pool = shared_pool if shared_pool is not None else (
            _live_memories(writer, sample, layer_ids, cfg) + _live_memories(writer, foreign, layer_ids, cfg)
        )
        sample_query = _query_key(writer, sample)
        foreign_query = _query_key(writer, foreign)
        address_diagnostics = _address_diagnostics(diagnostic_pool, sample_query, sample, foreign)
        address_rows.append(address_diagnostics)
        correct_memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, sample, layer_ids, cfg),
            sample_query,
            cfg.top_k,
            cfg.identity_gate_beta,
            cfg.identity_gate_tau,
        )
        foreign_memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, foreign, layer_ids, cfg),
            foreign_query,
            cfg.top_k,
            cfg.identity_gate_beta,
            cfg.identity_gate_tau,
        )
        oracle_correct_memories = _select_sample_topk_by_layer(diagnostic_pool, sample_query, sample.example.sample_id, cfg.top_k)
        paired_id = sample.example.paired_sample_id if sample.example.paired_sample_id is not None else foreign.example.sample_id
        oracle_paired_memories = _select_sample_topk_by_layer(diagnostic_pool, sample_query, paired_id, cfg.top_k)
        oracle_correct_address_paired_payload = _swap_payload_by_layer(oracle_correct_memories, oracle_paired_memories)
        oracle_paired_address_correct_payload = _swap_payload_by_layer(oracle_paired_memories, oracle_correct_memories)
        sample_modes: dict[str, dict[str, float]] = {}
        for mode in modes:
            if mode == "no_memory":
                memories = {}
                injection_mode = mode
            elif mode == "delta_qv_wrong_query":
                memories = foreign_memories
                injection_mode = "delta_qv"
            elif mode == "delta_qv_oracle_correct":
                memories = oracle_correct_memories
                injection_mode = "delta_qv"
            elif mode == "delta_qv_oracle_paired":
                memories = oracle_paired_memories
                injection_mode = "delta_qv"
            elif mode == "delta_qv_oracle_correct_address_paired_payload":
                memories = oracle_correct_address_paired_payload
                injection_mode = "delta_qv"
            elif mode == "delta_qv_oracle_paired_address_correct_payload":
                memories = oracle_paired_address_correct_payload
                injection_mode = "delta_qv"
            elif mode == "logit_bias_oracle_correct":
                memories = oracle_correct_memories
                injection_mode = "logit_bias"
            elif mode == "logit_bias_oracle_paired":
                memories = oracle_paired_memories
                injection_mode = "logit_bias"
            elif mode == "logit_bias_correct_address_paired_payload":
                memories = oracle_correct_address_paired_payload
                injection_mode = "logit_bias"
            elif mode == "logit_bias_paired_address_correct_payload":
                memories = oracle_paired_address_correct_payload
                injection_mode = "logit_bias"
            elif mode == "payload_probe_oracle_correct":
                memories = oracle_correct_memories
                injection_mode = "payload_probe"
            elif mode == "payload_probe_oracle_paired":
                memories = oracle_paired_memories
                injection_mode = "payload_probe"
            elif mode == "payload_probe_correct_address_paired_payload":
                memories = oracle_correct_address_paired_payload
                injection_mode = "payload_probe"
            elif mode == "payload_probe_paired_address_correct_payload":
                memories = oracle_paired_address_correct_payload
                injection_mode = "payload_probe"
            else:
                memories = correct_memories
                injection_mode = mode
            correct = _score_candidate_answer(bundle, injector, logit_projector, payload_probe, sample.example.question, sample.example.answer, memories, injection_mode, cfg.payload_probe_layer_strategy)
            foreign_score = _score_candidate_answer(bundle, injector, logit_projector, payload_probe, sample.example.question, foreign.example.answer, memories, injection_mode, cfg.payload_probe_layer_strategy)
            margin = foreign_score["answer_nll"] - correct["answer_nll"]
            by_mode[mode].append(margin)
            token_row = _answer_token_discrimination(
                bundle,
                injector,
                logit_projector,
                payload_probe,
                sample.example.question,
                sample.example.answer,
                foreign.example.answer,
                memories,
                injection_mode,
                cfg.payload_probe_layer_strategy,
            )
            token_by_mode[mode].append(token_row)
            sample_modes[mode] = {
                "correct_answer_nll": correct["answer_nll"],
                "foreign_answer_nll": foreign_score["answer_nll"],
                "foreign_minus_correct_nll": margin,
                "answer_token": token_row,
            }
        samples.append(
            {
                "sample_id": sample.example.sample_id,
                "unit": sample.example.unit,
                "answer": sample.example.answer,
                "foreign_sample_id": foreign.example.sample_id,
                "foreign_answer": foreign.example.answer,
                "paired_sample_id": sample.example.paired_sample_id,
                "collision_group_id": sample.example.collision_group_id,
                "address_diagnostics": address_diagnostics,
                "modes": sample_modes,
            }
        )
    aggregate = {mode: {"foreign_minus_correct_nll": _mean(values)} for mode, values in by_mode.items()}
    for mode, rows in token_by_mode.items():
        aggregate[mode].update(_aggregate_answer_token_rows(rows))
    aggregate["address"] = _aggregate_address_diagnostics(address_rows)
    if by_mode["delta_qv"] and by_mode["delta_qv_wrong_query"]:
        aggregate["delta_qv"]["margin_advantage_vs_wrong_query"] = (
            aggregate["delta_qv"]["foreign_minus_correct_nll"]
            - aggregate["delta_qv_wrong_query"]["foreign_minus_correct_nll"]
        )
    return {"aggregate": aggregate, "samples": samples}


def _swap_payload_by_layer(
    address_memories: dict[int, list[AttentionMemoryItem]],
    payload_memories: dict[int, list[AttentionMemoryItem]],
) -> dict[int, list[AttentionMemoryItem]]:
    swapped: dict[int, list[AttentionMemoryItem]] = {}
    for layer_id, address_items in address_memories.items():
        payload_items = payload_memories.get(layer_id, [])
        if not payload_items:
            swapped[layer_id] = list(address_items)
            continue
        layer_swapped = []
        for idx, address_item in enumerate(address_items):
            payload_item = payload_items[min(idx, len(payload_items) - 1)]
            metadata = dict(address_item.metadata)
            metadata.update(
                {
                    "payload_swapped": True,
                    "address_sample_id": address_item.metadata.get("sample_id"),
                    "payload_sample_id": payload_item.metadata.get("sample_id"),
                    "payload_answer": payload_item.metadata.get("answer"),
                }
            )
            layer_swapped.append(
                replace(
                    address_item,
                    raw_value=payload_item.raw_value,
                    delta_q=payload_item.delta_q,
                    delta_k=payload_item.delta_k,
                    delta_v=payload_item.delta_v,
                    usage_mass=payload_item.usage_mass,
                    metadata=metadata,
                )
            )
        swapped[layer_id] = layer_swapped
    return swapped


def _address_diagnostics(
    memories: list[AttentionMemoryItem],
    query: torch.Tensor,
    sample: PreparedDeltaExample,
    foreign: PreparedDeltaExample,
) -> dict[str, float]:
    if not memories:
        return {
            "correct_address_rank": 0.0,
            "paired_negative_rank": 0.0,
            "top1_score": 0.0,
            "top2_score": 0.0,
            "address_margin": 0.0,
            "correct_vs_paired_score_margin": 0.0,
        }
    q = F.normalize(query.float().to(memories[0].address_key.device), dim=0)
    scored = []
    for item in memories:
        score = float(F.normalize(item.address_key.float(), dim=0).matmul(q).detach().cpu())
        scored.append((score, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    top1 = scored[0][0]
    top2 = scored[1][0] if len(scored) > 1 else top1
    correct_id = sample.example.sample_id
    paired_id = sample.example.paired_sample_id if sample.example.paired_sample_id is not None else foreign.example.sample_id
    correct_scores = [(rank, score) for rank, (score, item) in enumerate(scored, start=1) if item.metadata.get("sample_id") == correct_id]
    paired_scores = [(rank, score) for rank, (score, item) in enumerate(scored, start=1) if item.metadata.get("sample_id") == paired_id]
    correct_rank, correct_score = correct_scores[0] if correct_scores else (0, 0.0)
    paired_rank, paired_score = paired_scores[0] if paired_scores else (0, 0.0)
    return {
        "correct_address_rank": float(correct_rank),
        "paired_negative_rank": float(paired_rank),
        "top1_score": top1,
        "top2_score": top2,
        "address_margin": top1 - top2,
        "correct_vs_paired_score_margin": correct_score - paired_score,
    }


def _aggregate_address_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: _mean([float(row.get(key, 0.0)) for row in rows]) for key in keys}


def _foreign_sample(prepared: list[PreparedDeltaExample], sample_idx: int) -> PreparedDeltaExample:
    sample = prepared[sample_idx]
    if len(prepared) <= 1:
        return sample
    for offset in range(1, len(prepared)):
        candidate = prepared[(sample_idx + offset) % len(prepared)]
        if candidate.example.unit == sample.example.unit:
            return candidate
    return prepared[(sample_idx + 1) % len(prepared)]


def _address_ranking_loss(
    writer: RCVHCWriter,
    sample: PreparedDeltaExample,
    prepared: list[PreparedDeltaExample],
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    pool = _memory_pool(writer, prepared, layer_ids, cfg)
    if not pool:
        device = next(writer.parameters()).device
        zero = torch.zeros((), device=device)
        return zero, zero
    query = _query_key(writer, sample)
    query = F.normalize(query, dim=0)
    losses = []
    margins = []
    target = torch.as_tensor(cfg.address_margin, device=query.device, dtype=query.dtype)
    sample_ids = sorted({int(item.metadata["sample_id"]) for item in pool})
    if sample.example.sample_id not in sample_ids:
        zero = torch.zeros((), device=query.device, dtype=query.dtype)
        return zero, zero
    target_idx = torch.tensor([sample_ids.index(sample.example.sample_id)], device=query.device)
    for layer_id in layer_ids:
        per_sample_scores = []
        for sample_id in sample_ids:
            layer_items = [
                item
                for item in pool
                if item.layer_id == layer_id and int(item.metadata.get("sample_id", -1)) == sample_id
            ]
            if not layer_items:
                per_sample_scores.append(torch.tensor(-1e4, device=query.device, dtype=query.dtype))
                continue
            keys = torch.stack([item.address_key.float() for item in layer_items], dim=0)
            per_sample_scores.append(F.normalize(keys, dim=-1).matmul(query).max())
        if not per_sample_scores:
            continue
        scores = torch.stack(per_sample_scores)
        correct_score = scores[target_idx.item()]
        negative_scores = torch.cat([scores[: target_idx.item()], scores[target_idx.item() + 1 :]])
        if negative_scores.numel() == 0:
            continue
        margin = correct_score - negative_scores.max()
        margins.append(margin)
        margin_loss = F.relu(target - margin)
        class_loss = F.cross_entropy((cfg.address_score_scale * scores).view(1, -1), target_idx)
        losses.append(class_loss + margin_loss)
    if not losses:
        zero = torch.zeros((), device=query.device, dtype=query.dtype)
        return zero, zero
    return torch.stack(losses).mean(), torch.stack(margins).mean()


def _candidate_answer_loss(
    bundle,
    injector: GemmaAttentionInjector,
    question: str,
    answer: str,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    mode: str,
) -> torch.Tensor:
    prompt_text, answer_start, answer_ids = _metric_prompt(bundle.tokenizer, question, answer)
    prompt = _encode(bundle.tokenizer, prompt_text, bundle.device)
    result = injector.forward_layers(
        input_ids=prompt["input_ids"],
        attention_mask=prompt["attention_mask"],
        memories_by_layer=memories_by_layer,
        mode=mode,
    )
    return _answer_loss(result.logits, answer_start, answer_ids)


def _payload_answer_loss(
    payload_probe: PayloadAnswerProbe,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    answer_ids: list[int],
    layer_strategy: str = "mean_all",
) -> torch.Tensor:
    logits, _ = _payload_probe_logits(memories_by_layer, payload_probe, layer_strategy)
    if not answer_ids:
        raise ValueError("payload answer loss requires at least one answer token")
    target = torch.tensor([int(answer_ids[0])], device=logits.device)
    return F.cross_entropy(logits.view(1, -1).float(), target)


def _payload_embedding_loss(
    payload_probe: PayloadAnswerProbe,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    answer_ids: list[int],
    layer_strategy: str = "mean_all",
) -> torch.Tensor:
    payload, _ = _payload_vector(memories_by_layer, payload_probe, layer_strategy)
    if not answer_ids:
        raise ValueError("payload embedding loss requires at least one answer token")
    hidden = payload_probe.project_hidden(payload)
    target = payload_probe.output_embeddings.weight[int(answer_ids[0])].to(hidden.device, hidden.dtype)
    return 1.0 - F.cosine_similarity(hidden.float(), target.float(), dim=0).mean()


def _stage2_swap_loss(
    sample: PreparedDeltaExample,
    foreign: PreparedDeltaExample,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    foreign_memories_by_layer: dict[int, list[AttentionMemoryItem]],
    payload_probe: PayloadAnswerProbe,
    logit_projector: PayloadLogitProjector,
    mode: str,
    margin: float,
    layer_strategy: str = "mean_all",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not sample.answer_ids or not foreign.answer_ids:
        device = next(payload_probe.parameters()).device
        zero = torch.zeros((), device=device)
        return zero, zero, zero
    correct_id = int(sample.answer_ids[0])
    paired_id = int(foreign.answer_ids[0])
    if mode == "payload_probe":
        correct_logits, _ = _payload_probe_logits(memories_by_layer, payload_probe, layer_strategy)
        paired_logits, _ = _payload_probe_logits(foreign_memories_by_layer, payload_probe, layer_strategy)
        correct_log_probs = torch.log_softmax(correct_logits.view(-1).float(), dim=-1)
        paired_log_probs = torch.log_softmax(paired_logits.view(-1).float(), dim=-1)
    elif mode == "logit_bias":
        correct_logits, _ = _apply_logit_bias(sample.base_logits, memories_by_layer, logit_projector, layer_strategy)
        paired_logits, _ = _apply_logit_bias(sample.base_logits, foreign_memories_by_layer, logit_projector, layer_strategy)
        logit_pos = sample.answer_start - 1
        correct_log_probs = torch.log_softmax(correct_logits[0, logit_pos].float(), dim=-1)
        paired_log_probs = torch.log_softmax(paired_logits[0, logit_pos].float(), dim=-1)
    else:
        raise ValueError(f"unsupported stage2 swap mode: {mode}")
    binding_margin = correct_log_probs[correct_id] - correct_log_probs[paired_id]
    swap_margin = paired_log_probs[paired_id] - paired_log_probs[correct_id]
    target = torch.as_tensor(float(margin), device=binding_margin.device, dtype=binding_margin.dtype)
    loss = F.relu(target - binding_margin) + F.relu(target - swap_margin)
    return loss, binding_margin.detach(), swap_margin.detach()


def _score_candidate_answer(
    bundle,
    injector: GemmaAttentionInjector,
    logit_projector: PayloadLogitProjector,
    payload_probe: PayloadAnswerProbe,
    question: str,
    answer: str,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    mode: str,
    layer_strategy: str = "mean_all",
) -> dict[str, float]:
    prompt_text, answer_start, answer_ids = _metric_prompt(bundle.tokenizer, question, answer)
    prompt = _encode(bundle.tokenizer, prompt_text, bundle.device)
    if mode == "no_memory":
        with torch.no_grad():
            out = bundle.model(
                input_ids=prompt["input_ids"],
                attention_mask=prompt["attention_mask"],
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
        logits = out.logits
    elif mode == "logit_bias":
        with torch.no_grad():
            out = bundle.model(
                input_ids=prompt["input_ids"],
                attention_mask=prompt["attention_mask"],
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
        logits, _ = _apply_logit_bias(out.logits, memories_by_layer, logit_projector, layer_strategy)
    elif mode == "payload_probe":
        logits, _ = _payload_probe_logits(memories_by_layer, payload_probe, layer_strategy)
        if not answer_ids:
            return {"answer_nll": 0.0, "answer_rank": 0.0, "top10": 0.0, "answer_logprob": 0.0}
        target_id = int(answer_ids[0])
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        rank = int((log_probs > log_probs[target_id]).sum().detach().cpu()) + 1
        return {
            "answer_nll": float((-log_probs[target_id]).detach().cpu()),
            "answer_rank": float(rank),
            "top10": float(rank <= 10),
            "answer_logprob": float(log_probs[target_id].detach().cpu()),
        }
    else:
        result = injector.forward_layers(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            memories_by_layer=memories_by_layer,
            mode=mode,
        )
        logits = result.logits
    return compute_answer_metrics(logits, prompt["input_ids"], answer_start, answer_ids)


def _answer_token_discrimination(
    bundle,
    injector: GemmaAttentionInjector,
    logit_projector: PayloadLogitProjector,
    payload_probe: PayloadAnswerProbe,
    question: str,
    correct_answer: str,
    paired_answer: str,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    mode: str,
    layer_strategy: str = "mean_all",
) -> dict[str, float]:
    prompt_text, answer_start, correct_ids = _metric_prompt(bundle.tokenizer, question, correct_answer)
    _, _, paired_ids = _metric_prompt(bundle.tokenizer, question, paired_answer)
    prompt = _encode(bundle.tokenizer, prompt_text, bundle.device)
    if mode == "no_memory":
        with torch.no_grad():
            out = bundle.model(
                input_ids=prompt["input_ids"],
                attention_mask=prompt["attention_mask"],
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
        logits = out.logits
    elif mode == "logit_bias":
        with torch.no_grad():
            out = bundle.model(
                input_ids=prompt["input_ids"],
                attention_mask=prompt["attention_mask"],
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
        logits, _ = _apply_logit_bias(out.logits, memories_by_layer, logit_projector, layer_strategy)
    elif mode == "payload_probe":
        logits, _ = _payload_probe_logits(memories_by_layer, payload_probe, layer_strategy)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        if not correct_ids or not paired_ids:
            return {
                "correct_token_logprob": 0.0,
                "paired_token_logprob": 0.0,
                "binding_margin": 0.0,
                "top1_is_correct": 0.0,
                "top1_is_paired": 0.0,
                "correct_token_id": 0.0,
                "paired_token_id": 0.0,
            }
        correct_id = int(correct_ids[0])
        paired_id = int(paired_ids[0])
        top1 = int(torch.argmax(log_probs).detach().cpu())
        correct_lp = float(log_probs[correct_id].detach().cpu())
        paired_lp = float(log_probs[paired_id].detach().cpu())
        return {
            "correct_token_logprob": correct_lp,
            "paired_token_logprob": paired_lp,
            "binding_margin": correct_lp - paired_lp,
            "top1_is_correct": float(top1 == correct_id),
            "top1_is_paired": float(top1 == paired_id),
            "correct_token_id": float(correct_id),
            "paired_token_id": float(paired_id),
        }
    else:
        result = injector.forward_layers(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            memories_by_layer=memories_by_layer,
            mode=mode,
        )
        logits = result.logits
    logit_pos = answer_start - 1
    if logit_pos < 0 or logit_pos >= logits.shape[1] or not correct_ids or not paired_ids:
        return {
            "correct_token_logprob": 0.0,
            "paired_token_logprob": 0.0,
            "binding_margin": 0.0,
            "top1_is_correct": 0.0,
            "top1_is_paired": 0.0,
            "correct_token_id": 0.0,
            "paired_token_id": 0.0,
        }
    correct_id = int(correct_ids[0])
    paired_id = int(paired_ids[0])
    log_probs = torch.log_softmax(logits[0, logit_pos].float(), dim=-1)
    top1 = int(torch.argmax(log_probs).detach().cpu())
    correct_lp = float(log_probs[correct_id].detach().cpu())
    paired_lp = float(log_probs[paired_id].detach().cpu())
    return {
        "correct_token_logprob": correct_lp,
        "paired_token_logprob": paired_lp,
        "binding_margin": correct_lp - paired_lp,
        "top1_is_correct": float(top1 == correct_id),
        "top1_is_paired": float(top1 == paired_id),
        "correct_token_id": float(correct_id),
        "paired_token_id": float(paired_id),
    }


def diagnose_experiment(initial_eval: dict[str, Any], final_eval: dict[str, Any], min_control_gap: float = 0.05) -> dict[str, Any]:
    required = {"delta_qv", "delta_qv_zero", "delta_qv_random", "delta_qv_shuffled"}
    missing = sorted(mode for mode in required if mode not in initial_eval.get("aggregate", {}) or mode not in final_eval.get("aggregate", {}))
    if missing:
        return {
            "control_margin_min": float(min_control_gap),
            "diagnosis_skipped": True,
            "missing_modes": missing,
            "mechanism_supported_on_eval": False,
        }
    initial_delta = _mode_metric(initial_eval, "delta_qv", "answer_nll")
    final_delta = _mode_metric(final_eval, "delta_qv", "answer_nll")
    final_zero = _mode_metric(final_eval, "delta_qv_zero", "answer_nll")
    final_random = _mode_metric(final_eval, "delta_qv_random", "answer_nll")
    final_shuffled = _mode_metric(final_eval, "delta_qv_shuffled", "answer_nll")
    zero_gap = final_zero - final_delta
    random_gap = final_random - final_delta
    shuffled_gap = final_shuffled - final_delta
    return {
        "control_margin_min": float(min_control_gap),
        "eval_delta_nll_drop": initial_delta - final_delta,
        "eval_delta_zero_nll_gap": zero_gap,
        "eval_delta_random_nll_gap": random_gap,
        "eval_delta_shuffled_nll_gap": shuffled_gap,
        "eval_delta_beats_zero": zero_gap > min_control_gap,
        "eval_delta_beats_random": random_gap > min_control_gap,
        "eval_delta_beats_shuffled": shuffled_gap > min_control_gap,
        "mechanism_supported_on_eval": zero_gap > min_control_gap and random_gap > min_control_gap and shuffled_gap > min_control_gap,
    }


def _eval_modes(cfg: DeltaExperimentConfig) -> list[str]:
    if cfg.eval_injection_modes.strip().lower() == "all":
        return list(EXPERIMENT_EVAL_MODES)
    requested = [item.strip() for item in cfg.eval_injection_modes.split(",") if item.strip()]
    valid = set(EXPERIMENT_EVAL_MODES)
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"unknown eval injection modes: {unknown}")
    modes = []
    for mode in ["no_memory", *requested]:
        if mode in valid and mode not in modes:
            modes.append(mode)
    return modes


def _stage2_binding_summary(final_eval: dict[str, Any], conflict_margins: dict[str, Any] | None) -> dict[str, Any]:
    """Compact Stage 2 answer-token view for channel diagnostics.

    Sequence-level NLL is not enough for binding. This summary keeps the
    top-level report focused on answer-token rank and payload-swap metrics.
    """

    mode_rows: dict[str, dict[str, float]] = {}
    samples = final_eval.get("samples") or []
    for mode, aggregate in (final_eval.get("aggregate") or {}).items():
        sample_metrics = [
            sample.get("modes", {}).get(mode, {}).get("metrics", {})
            for sample in samples
            if mode in sample.get("modes", {})
        ]
        top1_values = [1.0 if float(metrics.get("answer_rank", 0.0)) == 1.0 else 0.0 for metrics in sample_metrics]
        mode_rows[mode] = {
            "answer_nll": float(aggregate.get("answer_nll", 0.0)),
            "answer_rank": float(aggregate.get("answer_rank", 0.0)),
            "top1_correct_rate": _mean(top1_values),
            "top10": float(aggregate.get("top10", 0.0)),
            "answer_logprob": float(aggregate.get("answer_logprob", 0.0)),
        }
    swap_rows: dict[str, dict[str, float]] = {}
    if conflict_margins:
        for mode, row in (conflict_margins.get("aggregate") or {}).items():
            if mode == "address":
                continue
            swap_rows[mode] = {
                "foreign_minus_correct_nll": float(row.get("foreign_minus_correct_nll", 0.0)),
                "binding_margin": float(row.get("answer_token_binding_margin", 0.0)),
                "top1_is_correct": float(row.get("answer_token_top1_is_correct", 0.0)),
                "top1_is_paired": float(row.get("answer_token_top1_is_paired", 0.0)),
            }
    return {
        "eval_modes": mode_rows,
        "swap_controls": swap_rows,
        "oracle_channel_pass": bool(
            mode_rows.get("oracle_logit_answer_embedding", {}).get("top1_correct_rate", 0.0) >= 0.85
            and mode_rows.get("oracle_logit_answer_embedding", {}).get("answer_nll", 1e9) <= 1.0
        ),
        "payload_probe_layer_strategy": (
            samples[0].get("modes", {}).get("payload_probe", {}).get("qkv_trace", {}).get("payload_probe_layer_strategy")
            if samples
            else None
        ),
    }


def write_delta_experiment_report(summary: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "delta_experiment_summary.json"
    report_path = report_dir / "delta_experiment_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(_markdown(summary), encoding="utf-8")
    return {"summary": str(summary_path), "report": str(report_path)}


def _prepare_example(bundle, example: DeltaExample, cfg: DeltaExperimentConfig) -> PreparedDeltaExample:
    context = _encode(bundle.tokenizer, example.text, bundle.device)
    with torch.no_grad():
        out = bundle.model(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )
    if out.hidden_states is None or out.attentions is None:
        raise RuntimeError("Delta experiment requires hidden_states and attentions")
    prompt_text, answer_start, answer_ids = _metric_prompt(bundle.tokenizer, example.question, example.answer)
    prompt = _encode(bundle.tokenizer, prompt_text, bundle.device)
    with torch.no_grad():
        base = bundle.model(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
    query_prompt = _encode(bundle.tokenizer, _query_prompt(example.question), bundle.device)
    query_prompt_text = _query_prompt(example.question)
    with torch.no_grad():
        query_base = bundle.model(
            input_ids=query_prompt["input_ids"],
            attention_mask=query_prompt["attention_mask"],
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
    address_token_range = _token_range_for_example(bundle.tokenizer, example.text, example.address_text, example.address_char_range)
    value_token_range = _token_range_for_example(bundle.tokenizer, example.text, example.value_text, example.value_char_range)
    query_hidden = query_base.hidden_states[-1].mean(dim=(0, 1)).detach().float().cpu()
    if cfg.oracle_span_writer and example.address_text is not None:
        query_address_range = _token_range_for_example(bundle.tokenizer, query_prompt_text, example.address_text, None)
        if query_address_range is not None:
            query_hidden = _mean_hidden_span(query_base.hidden_states[-1], query_address_range).detach().float().cpu()
    snippets = split_source_snippets(bundle.tokenizer, context["input_ids"], cfg.block_size)
    detached = SimpleNamespace(
        hidden_states=tuple(t.detach() for t in out.hidden_states),
        attentions=tuple(t.detach() for t in out.attentions),
    )
    return PreparedDeltaExample(
        example=example,
        context_out=detached,
        prompt=prompt,
        base_logits=base.logits.detach(),
        query_hidden=query_hidden,
        snippets=snippets,
        answer_start=answer_start,
        answer_ids=answer_ids,
        address_token_range=address_token_range,
        value_token_range=value_token_range,
    )


def _live_memories(
    writer: RCVHCWriter,
    sample: PreparedDeltaExample,
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> list[AttentionMemoryItem]:
    items: list[AttentionMemoryItem] = []
    for layer_id in layer_ids:
        if cfg.oracle_span_writer:
            if sample.address_token_range is None or sample.value_token_range is None:
                raise ValueError("oracle_span_writer requires address/value span metadata")
            layer_items = writer.write_oracle_span_layer(
                layer_id=layer_id,
                h_out=sample.context_out.hidden_states[layer_id + 1],
                address_token_range=sample.address_token_range,
                value_token_range=sample.value_token_range,
                token_offset=0,
                source_text=sample.example.text,
            )
        else:
            layer_items = writer.write_layer(
                layer_id=layer_id,
                h_in=sample.context_out.hidden_states[layer_id],
                h_out=sample.context_out.hidden_states[layer_id + 1],
                attn=sample.context_out.attentions[layer_id],
                token_offset=0,
                source_text_by_block=sample.snippets,
            )
        for item in layer_items:
            item.metadata.update(
                {
                    "sample_id": sample.example.sample_id,
                    "unit": sample.example.unit,
                    "answer": sample.example.answer,
                    "paired_sample_id": sample.example.paired_sample_id,
                    "collision_group_id": sample.example.collision_group_id,
                    "foreign_answer": sample.example.foreign_answer,
                    "address_text": sample.example.address_text,
                    "value_text": sample.example.value_text,
                    "foreign_address_text": sample.example.foreign_address_text,
                    "foreign_value_text": sample.example.foreign_value_text,
                }
            )
        items.extend(layer_items)
    return items


def _memory_pool(
    writer: RCVHCWriter,
    prepared: list[PreparedDeltaExample],
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> list[AttentionMemoryItem]:
    items: list[AttentionMemoryItem] = []
    for sample in prepared:
        items.extend(_live_memories(writer, sample, layer_ids, cfg))
    return items


def _query_key(writer: RCVHCWriter, sample: PreparedDeltaExample) -> torch.Tensor:
    param = next(writer.parameters())
    query_hidden = sample.query_hidden.to(device=param.device, dtype=param.dtype)
    return writer.address_query(query_hidden).float()


def _token_range_for_example(
    tokenizer,
    text: str,
    needle: str | None,
    char_range: list[int] | None,
) -> tuple[int, int] | None:
    if needle is None:
        return None
    if char_range is None:
        start = text.find(needle)
        if start < 0:
            return None
        char_range = [start, start + len(needle)]
    return _char_range_to_token_range(tokenizer, text, tuple(char_range))


def _char_range_to_token_range(tokenizer, text: str, char_range: tuple[int, int]) -> tuple[int, int]:
    start_char, end_char = int(char_range[0]), int(char_range[1])
    try:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"][0].tolist()
        token_indices = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > start_char and tok_start < end_char
        ]
        if token_indices:
            return token_indices[0], token_indices[-1] + 1
    except (TypeError, NotImplementedError):
        pass
    except KeyError:
        pass
    prefix = text[:start_char]
    span = text[start_char:end_char]
    start = _encoded_len(tokenizer, prefix)
    end = start + _encoded_len(tokenizer, span)
    return start, max(start + 1, end)


def _mean_hidden_span(hidden: torch.Tensor, token_range: tuple[int, int]) -> torch.Tensor:
    start = max(0, min(int(token_range[0]), hidden.shape[1] - 1))
    end = max(start + 1, min(int(token_range[1]), hidden.shape[1]))
    return hidden[0, start:end].mean(dim=0)


def _encoded_len(tokenizer, text: str) -> int:
    if not text.strip():
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _select_topk_by_layer(
    memories: list[AttentionMemoryItem],
    query: torch.Tensor,
    top_k: int,
    identity_gate_beta: float = 64.0,
    identity_gate_tau: float = 0.01,
) -> dict[int, list[AttentionMemoryItem]]:
    selected: dict[int, list[AttentionMemoryItem]] = {}
    for layer_id in sorted({item.layer_id for item in memories}):
        layer_memories = [item for item in memories if item.layer_id == layer_id]
        selected[layer_id] = _select_topk(layer_memories, query, top_k, identity_gate_beta, identity_gate_tau)
    return selected


def _select_sample_topk_by_layer(
    memories: list[AttentionMemoryItem],
    query: torch.Tensor,
    sample_id: int,
    top_k: int,
) -> dict[int, list[AttentionMemoryItem]]:
    selected: dict[int, list[AttentionMemoryItem]] = {}
    scoped = [item for item in memories if int(item.metadata.get("sample_id", -1)) == int(sample_id)]
    for layer_id in sorted({item.layer_id for item in scoped}):
        layer_memories = [item for item in scoped if item.layer_id == layer_id]
        selected[layer_id] = _select_topk(layer_memories, query, top_k)
    return selected


def _select_topk(
    memories: list[AttentionMemoryItem],
    query: torch.Tensor,
    top_k: int,
    identity_gate_beta: float = 64.0,
    identity_gate_tau: float = 0.01,
) -> list[AttentionMemoryItem]:
    if not memories:
        return []
    q = query.float().to(memories[0].address_key.device)
    keys = torch.stack([item.address_key.float() for item in memories], dim=0)
    scores = F.normalize(keys, dim=-1).matmul(F.normalize(q, dim=0))
    _, idx = scores.topk(min(top_k, len(memories)))
    top_scores, _ = scores.topk(min(2, len(memories)))
    top1 = float(top_scores[0].detach().cpu())
    top2 = float(top_scores[1].detach().cpu()) if len(top_scores) > 1 else top1
    margin = top1 - top2
    gate = float(torch.sigmoid(torch.tensor(identity_gate_beta * (margin - identity_gate_tau))).item())
    selected = []
    for rank, item_idx in enumerate(idx.tolist()):
        item = memories[int(item_idx)]
        item.metadata = dict(item.metadata)
        item.metadata["retrieval_rank"] = rank
        item.metadata["address_score"] = float(scores[int(item_idx)].detach().cpu())
        item.metadata["address_top1_score"] = top1
        item.metadata["address_top2_score"] = top2
        item.metadata["address_margin"] = margin
        item.metadata["identity_gate"] = gate
        selected.append(item)
    return selected


def _raw_late_readout(
    sample: PreparedDeltaExample,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    injector: GemmaAttentionInjector,
) -> tuple[torch.Tensor, dict[str, float]]:
    memories = [item for layer_memories in memories_by_layer.values() for item in layer_memories]
    if not memories:
        return sample.base_logits, {}
    model = injector.model
    with torch.no_grad():
        out = model(
            input_ids=sample.prompt["input_ids"],
            attention_mask=sample.prompt["attention_mask"],
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
    hidden = out.hidden_states[-1]
    raw = torch.stack([item.raw_value.to(hidden.device, hidden.dtype) for item in memories], dim=0).mean(dim=0)
    update = fit_memory_dim(raw, hidden.shape[-1]).to(hidden.device, hidden.dtype)
    hidden_prime = hidden + 0.05 * update.view(1, 1, -1)
    logits = model.get_output_embeddings()(hidden_prime)
    return logits, {"hidden_delta_norm": float((hidden_prime - hidden).detach().float().norm().cpu())}


def _retrieved_attention_readout(
    sample: PreparedDeltaExample,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    injector: GemmaAttentionInjector,
) -> tuple[torch.Tensor, dict[str, float]]:
    memories = [item for layer_memories in memories_by_layer.values() for item in layer_memories]
    if not memories:
        return sample.base_logits, {}
    model = injector.model
    with torch.no_grad():
        out = model(
            input_ids=sample.prompt["input_ids"],
            attention_mask=sample.prompt["attention_mask"],
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
    hidden = out.hidden_states[-1]
    keys = torch.stack([item.address_key.to(hidden.device, hidden.dtype) for item in memories], dim=0)
    values = torch.stack([item.raw_value.to(hidden.device, hidden.dtype) for item in memories], dim=0)
    query = _fit_last_dim(hidden, keys.shape[-1])
    scores = torch.matmul(F.normalize(query.float(), dim=-1), F.normalize(keys.float(), dim=-1).transpose(0, 1))
    weights = torch.softmax(scores, dim=-1).to(values.dtype)
    context = torch.matmul(weights, values)
    update = _fit_last_dim(context, hidden.shape[-1]).to(hidden.device, hidden.dtype)
    hidden_prime = hidden + 0.2 * update
    logits = model.get_output_embeddings()(hidden_prime)
    return logits, {
        "retrieved_attention_delta_norm": float((hidden_prime - hidden).detach().float().norm().cpu()),
        "retrieved_attention_entropy": float((-(weights.float() * weights.float().clamp_min(1e-8).log()).sum(dim=-1).mean()).detach().cpu()),
    }


def _logit_bias_readout(
    sample: PreparedDeltaExample,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    logit_projector: PayloadLogitProjector,
    layer_strategy: str = "mean_all",
) -> tuple[torch.Tensor, dict[str, float]]:
    return _apply_logit_bias(sample.base_logits, memories_by_layer, logit_projector, layer_strategy)


def _apply_logit_bias(
    base_logits: torch.Tensor,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    logit_projector: PayloadLogitProjector,
    layer_strategy: str = "mean_all",
) -> tuple[torch.Tensor, dict[str, float]]:
    payload, payload_trace = _payload_vector(memories_by_layer, logit_projector, layer_strategy)
    if payload.numel() == 0:
        return base_logits, {}
    payload = payload.to(base_logits.device, base_logits.dtype)
    bias = logit_projector(payload).to(base_logits.device, base_logits.dtype).view(1, 1, -1)
    logits = base_logits + bias
    return logits, {
        **payload_trace,
        "logit_bias_norm": float(bias.detach().float().norm().cpu()),
        "logit_bias_max": float(bias.detach().float().max().cpu()),
    }


def _oracle_answer_embedding_bias(
    base_logits: torch.Tensor,
    answer_ids: list[int],
    output_embeddings: nn.Module,
    scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage 0 upper-bound: feed the answer token's own output embedding as a logit bias.

    This bypasses writer and projector. It is purely diagnostic and tests whether the
    LM-head readout path can map ``W[answer]`` back to ``argmax = answer``. If this fails,
    the bug is in the metric/readout pipeline; if it passes, the bottleneck is upstream.
    """

    if not answer_ids:
        return base_logits, {}
    weight = output_embeddings.weight  # (vocab, hidden)
    target_emb = weight[int(answer_ids[0])].to(base_logits.device, base_logits.dtype)
    bias = (weight.to(base_logits.device, base_logits.dtype) @ target_emb) * float(scale)
    logits = base_logits + bias.view(1, 1, -1)
    return logits, {
        "oracle_answer_embedding_bias_norm": float(bias.detach().float().norm().cpu()),
        "oracle_answer_embedding_bias_max": float(bias.detach().float().max().cpu()),
    }


def _payload_probe_logits(
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    payload_probe: PayloadAnswerProbe,
    layer_strategy: str = "mean_all",
) -> tuple[torch.Tensor, dict[str, float]]:
    payload, trace = _payload_vector(memories_by_layer, payload_probe, layer_strategy)
    logits = payload_probe(payload)
    trace["payload_probe_logit_norm"] = float(logits.detach().float().norm().cpu())
    return logits, trace


def _payload_vector(
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    payload_probe: PayloadAnswerProbe,
    layer_strategy: str = "mean_all",
) -> tuple[torch.Tensor, dict[str, float]]:
    if not memories_by_layer:
        param = next(payload_probe.parameters())
        return torch.zeros(payload_probe.norm.normalized_shape[0], device=param.device, dtype=param.dtype), {}
    sorted_layers = sorted(memories_by_layer.keys())
    if layer_strategy == "last_layer":
        layers_to_use = [sorted_layers[-1]]
    elif layer_strategy == "first_layer":
        layers_to_use = [sorted_layers[0]]
    else:
        layers_to_use = sorted_layers
    memories = [item for lid in layers_to_use for item in memories_by_layer[lid]]
    if not memories:
        param = next(payload_probe.parameters())
        return torch.zeros(payload_probe.norm.normalized_shape[0], device=param.device, dtype=param.dtype), {}
    device = next(payload_probe.parameters()).device
    dtype = next(payload_probe.parameters()).dtype
    payload = torch.stack([item.raw_value.to(device, dtype) for item in memories], dim=0).mean(dim=0)
    return payload, {
        "payload_probe_layer_strategy": layer_strategy,
        "payload_probe_layers_used": float(len(layers_to_use)),
    }


def _payload_probe_metrics(logits: torch.Tensor, answer_ids: list[int]) -> dict[str, float]:
    if not answer_ids:
        return {"answer_nll": 0.0, "answer_rank": 0.0, "top10": 0.0, "answer_logprob": 0.0}
    target_id = int(answer_ids[0])
    log_probs = torch.log_softmax(logits.view(-1).float(), dim=-1)
    rank = int((log_probs > log_probs[target_id]).sum().detach().cpu()) + 1
    answer_logprob = float(log_probs[target_id].detach().cpu())
    return {
        "answer_nll": float(-answer_logprob),
        "answer_rank": float(rank),
        "top10": float(rank <= 10),
        "answer_logprob": answer_logprob,
    }


def _aggregate_by_mode(by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for mode, rows in by_mode.items():
        metrics = {"answer_nll": [], "answer_rank": [], "top10": [], "answer_logprob": []}
        traces = {"q_delta_norm": [], "v_delta_norm": [], "gate_v": [], "identity_gate": [], "injected_layers": []}
        for row in rows:
            for key in metrics:
                value = row["metrics"].get(key)
                if value is not None:
                    metrics[key].append(float(value))
            for key in traces:
                traces[key].append(float(row["qkv_trace"].get(key, 0.0)))
        aggregate[mode] = {key: _mean(values) for key, values in (metrics | traces).items()}
    return aggregate


def _aggregate_answer_token_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [
        "correct_token_logprob",
        "paired_token_logprob",
        "binding_margin",
        "top1_is_correct",
        "top1_is_paired",
    ]
    return {f"answer_token_{key}": _mean([float(row.get(key, 0.0)) for row in rows]) for key in keys}


def _fit_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.shape[-1] == target_dim:
        return tensor
    if tensor.shape[-1] > target_dim:
        return tensor[..., :target_dim]
    return F.pad(tensor, (0, target_dim - tensor.shape[-1]))


def _mode_metric(summary: dict[str, Any], mode: str, metric: str) -> float:
    return float(summary["aggregate"][mode][metric])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_prompt(tokenizer, question: str, answer: str) -> tuple[str, int, list[int]]:
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    return prompt + " " + answer, len(prompt_ids), answer_ids


def _query_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def _encode(tokenizer, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    if "attention_mask" not in encoded:
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
    return {"input_ids": encoded["input_ids"].long().to(device), "attention_mask": encoded["attention_mask"].long().to(device)}


def _markdown(summary: dict[str, Any]) -> str:
    cfg = summary["config"]
    lines = [
        "# Delta Memory Q/V Multi-Example Experiment",
        "",
        "## Config",
        "",
    ]
    for key in [
        "model",
        "device",
        "dtype",
        "steps",
        "lr",
        "task_suite",
        "train_samples",
        "eval_samples",
        "block_size",
        "memory_dim",
        "top_k",
        "address_margin_weight",
        "address_margin",
        "address_score_scale",
        "oracle_contrastive_weight",
        "identity_gate_beta",
        "identity_gate_tau",
        "oracle_span_writer",
        "logit_bias_loss_weight",
        "logit_bias_scale",
        "payload_answer_loss_weight",
        "payload_probe_layer_strategy",
        "payload_embedding_loss_weight",
        "stage2_swap_loss_weight",
        "stage2_swap_margin",
        "stage2_swap_mode",
        "eval_injection_modes",
        "control_margin_min",
    ]:
        lines.append(f"- `{key}`: `{cfg.get(key)}`")
    lines.extend(
        [
            f"- `layer_ids`: `{summary['layer_ids']}`",
            f"- `trainable_base_params`: `{summary['trainable_base_params']}`",
            f"- `prompt_insertion_used`: `{summary['prompt_insertion_used']}`",
            f"- `retrieval_key`: `{summary.get('retrieval_key', 'raw_key')}`",
            "",
            "## Eval Aggregate",
            "",
            "| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v | identity_gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    initial = summary["initial_eval"]["aggregate"]
    final = summary["final_eval"]["aggregate"]
    for mode in [mode for mode in EXPERIMENT_EVAL_MODES if mode in initial and mode in final]:
        if mode not in initial or mode not in final:
            continue
        lines.append(
            "| {mode} | {initial_nll:.4f} | {final_nll:.4f} | {rank:.4f} | {top10:.4f} | {q:.4f} | {v:.4f} | {gate:.4f} | {identity:.4f} |".format(
                mode=mode,
                initial_nll=float(initial[mode]["answer_nll"]),
                final_nll=float(final[mode]["answer_nll"]),
                rank=float(final[mode]["answer_rank"]),
                top10=float(final[mode]["top10"]),
                q=float(final[mode]["q_delta_norm"]),
                v=float(final[mode]["v_delta_norm"]),
                gate=float(final[mode]["gate_v"]),
                identity=float(final[mode].get("identity_gate", 0.0)),
            )
        )
    lines.extend(["", "## Diagnosis", ""])
    for key, value in summary["diagnosis"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Paired Statistics", ""])
    stats = summary.get("statistics", {})
    lines.append(f"- `strongest_non_prompt_baseline`: `{stats.get('strongest_non_prompt_baseline')}`")
    for baseline, row in (stats.get("comparisons") or {}).items():
        ci = row.get("bootstrap_ci95", [0.0, 0.0])
        lines.append(
            "- `{baseline}`: mean_delta=`{mean:.4f}`, ci95=`[{lo:.4f}, {hi:.4f}]`, win_rate=`{win:.4f}`, permutation_p=`{perm:.4f}`".format(
                baseline=baseline,
                mean=float(row.get("mean_delta", 0.0)),
                lo=float(ci[0]),
                hi=float(ci[1]),
                win=float(row.get("win_rate", 0.0)),
                perm=float(row.get("permutation_p", 1.0)),
            )
        )
    margins = summary.get("conflict_margins")
    if margins:
        lines.extend(["", "## Conflict Margins", ""])
        lines.append("| mode | foreign_minus_correct_nll |")
        lines.append("| --- | ---: |")
        for mode, row in (margins.get("aggregate") or {}).items():
            if mode == "address":
                continue
            lines.append(f"| {mode} | {float(row.get('foreign_minus_correct_nll', 0.0)):.4f} |")
        lines.extend(["", "## Answer Token Discrimination", ""])
        lines.append("| mode | binding_margin | top1_correct | top1_paired |")
        lines.append("| --- | ---: | ---: | ---: |")
        for mode, row in (margins.get("aggregate") or {}).items():
            if mode == "address":
                continue
            lines.append(
                "| {mode} | {margin:.4f} | {correct:.4f} | {paired:.4f} |".format(
                    mode=mode,
                    margin=float(row.get("answer_token_binding_margin", 0.0)),
                    correct=float(row.get("answer_token_top1_is_correct", 0.0)),
                    paired=float(row.get("answer_token_top1_is_paired", 0.0)),
                )
            )
        address = (margins.get("aggregate") or {}).get("address")
        if address:
            lines.extend(
                [
                    "",
                    "## Address Diagnostics",
                    "",
                    f"- `correct_address_rank`: `{float(address.get('correct_address_rank', 0.0)):.4f}`",
                    f"- `paired_negative_rank`: `{float(address.get('paired_negative_rank', 0.0)):.4f}`",
                    f"- `address_margin`: `{float(address.get('address_margin', 0.0)):.4f}`",
                    f"- `correct_vs_paired_score_margin`: `{float(address.get('correct_vs_paired_score_margin', 0.0)):.4f}`",
                ]
            )
        delta_row = (margins.get("aggregate") or {}).get("delta_qv", {})
        if "margin_advantage_vs_wrong_query" in delta_row:
            lines.append("")
            lines.append(f"- `delta_qv_margin_advantage_vs_wrong_query`: `{float(delta_row['margin_advantage_vs_wrong_query']):.4f}`")
    stage2 = summary.get("stage2_binding_summary") or {}
    if stage2:
        lines.extend(["", "## Stage 2 Binding Summary", ""])
        lines.append("| mode | answer_nll | top1_correct | top10 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for mode, row in (stage2.get("eval_modes") or {}).items():
            lines.append(
                "| {mode} | {nll:.4f} | {top1:.4f} | {top10:.4f} |".format(
                    mode=mode,
                    nll=float(row.get("answer_nll", 0.0)),
                    top1=float(row.get("top1_correct_rate", 0.0)),
                    top10=float(row.get("top10", 0.0)),
                )
            )
        lines.append("")
        lines.append(f"- `oracle_channel_pass`: `{stage2.get('oracle_channel_pass')}`")
        lines.append(f"- `payload_probe_layer_strategy`: `{stage2.get('payload_probe_layer_strategy')}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment trains only the Delta Memory writer and Q/V adapter while the base model remains frozen.",
            "A positive result requires held-out `delta_qv` to beat zero, random, and shuffled controls.",
        ]
    )
    return "\n".join(lines) + "\n"
