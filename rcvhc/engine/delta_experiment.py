"""Multi-example Delta Q/V experiment runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from rcvhc.core.config import resolve_layer_policy
from rcvhc.core.types import AttentionMemoryItem
from rcvhc.engine.attention_memory_engine import compute_answer_metrics
from rcvhc.engine.delta_dataset import DeltaExample, make_delta_memory_examples
from rcvhc.engine.delta_training import TRAIN_EVAL_MODES, _answer_loss, _grad_norm
from rcvhc.engine.statistics import primary_delta_memory_statistics
from rcvhc.gemma.attention_injector import GemmaAttentionInjector, QKVDeltaProjector
from rcvhc.gemma.model_adapter import exposed_qkv_layers, get_hidden_size, load_model_bundle, trainable_base_params
from rcvhc.memory.writer import RCVHCWriter, fit_memory_dim, split_source_snippets


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
    shared_memory_retrieval: bool = False
    report_dir: str = "reports/experiments/delta_experiment"


@dataclass
class PreparedDeltaExample:
    example: DeltaExample
    context_out: Any
    prompt: dict[str, torch.Tensor]
    base_logits: torch.Tensor
    query: torch.Tensor
    snippets: list[str]
    answer_start: int
    answer_ids: list[int]


def run_delta_experiment(cfg: DeltaExperimentConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    bundle = load_model_bundle(cfg.model, device=cfg.device, dtype=cfg.dtype)
    hidden_size = get_hidden_size(bundle.model)
    writer = RCVHCWriter(hidden_size, cfg.memory_dim, cfg.block_size).to(bundle.device, dtype=bundle.dtype)
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
    optimizer = torch.optim.AdamW(list(writer.parameters()) + list(projector.parameters()), lr=cfg.lr)

    initial_eval = evaluate_prepared(writer, injector, eval_prepared, layer_ids, cfg)
    train_rows = []
    for step in range(1, cfg.steps + 1):
        sample_idx = (step - 1) % len(train_prepared)
        sample = train_prepared[sample_idx]
        optimizer.zero_grad(set_to_none=True)
        train_pool = _memory_pool(writer, train_prepared, layer_ids, cfg) if cfg.shared_memory_retrieval else None
        memories = _select_topk_by_layer(
            train_pool if train_pool is not None else _live_memories(writer, sample, layer_ids, cfg),
            sample.query,
            cfg.top_k,
        )
        result = injector.forward_layers(
            input_ids=sample.prompt["input_ids"],
            attention_mask=sample.prompt["attention_mask"],
            memories_by_layer=memories,
            mode="delta_qv",
        )
        answer_loss = _answer_loss(result.logits, sample.answer_start, sample.answer_ids)
        contrastive_loss = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        margin_advantage = torch.zeros((), device=answer_loss.device, dtype=answer_loss.dtype)
        if cfg.contrastive_margin_weight > 0.0 and len(train_prepared) > 1:
            foreign = _foreign_sample(train_prepared, sample_idx)
            foreign_memories = _select_topk_by_layer(
                train_pool if train_pool is not None else _live_memories(writer, foreign, layer_ids, cfg),
                foreign.query,
                cfg.top_k,
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
        loss = answer_loss + cfg.contrastive_margin_weight * contrastive_loss
        loss.backward()
        grad_norm = _grad_norm(list(writer.parameters()) + list(projector.parameters()))
        optimizer.step()
        train_rows.append(
            {
                "step": step,
                "sample_id": sample.example.sample_id,
                "loss": float(loss.detach().cpu()),
                "answer_loss": float(answer_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "contrastive_margin_advantage": float(margin_advantage.detach().cpu()),
                "grad_norm": grad_norm,
                "q_delta_norm": result.trace.q_delta_norm,
                "v_delta_norm": result.trace.v_delta_norm,
                "gate_v": result.trace.gate_v,
            }
        )

    final_train = evaluate_prepared(writer, injector, train_prepared, layer_ids, cfg)
    final_eval = evaluate_prepared(writer, injector, eval_prepared, layer_ids, cfg)
    conflict_margins = (
        evaluate_conflict_margins(bundle, writer, injector, eval_prepared, layer_ids, cfg)
        if cfg.conflict_margins
        else None
    )
    summary = {
        "config": asdict(cfg),
        "layer_ids": layer_ids,
        "trainable_base_params": trainable_base_params(bundle.model),
        "prompt_insertion_used": False,
        "retrieval_query_uses_answer": False,
        "source_text_debug_only": True,
        "train_examples": [example.as_dict() for example in train_examples],
        "eval_examples": [example.as_dict() for example in eval_examples],
        "initial_eval": initial_eval,
        "train": train_rows,
        "final_train": final_train,
        "final_eval": final_eval,
        "conflict_margins": conflict_margins,
        "statistics": primary_delta_memory_statistics(final_eval, seed=cfg.seed),
        "diagnosis": diagnose_experiment(initial_eval, final_eval),
    }
    return summary


def evaluate_prepared(
    writer: RCVHCWriter,
    injector: GemmaAttentionInjector,
    prepared: list[PreparedDeltaExample],
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> dict[str, Any]:
    per_sample = []
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in TRAIN_EVAL_MODES}
    shared_pool = _memory_pool(writer, prepared, layer_ids, cfg) if cfg.shared_memory_retrieval else None
    for sample_idx, sample in enumerate(prepared):
        memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, sample, layer_ids, cfg),
            sample.query,
            cfg.top_k,
        )
        wrong_query_sample = _foreign_sample(prepared, sample_idx)
        wrong_query_memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, wrong_query_sample, layer_ids, cfg),
            wrong_query_sample.query,
            cfg.top_k,
        )
        sample_rows = {}
        for mode in TRAIN_EVAL_MODES:
            if mode == "no_memory":
                logits = sample.base_logits
                trace = {}
            elif mode in {"raw_memory", "hidden_retrieval"}:
                logits, trace = _raw_late_readout(sample, memories, injector)
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
    prepared: list[PreparedDeltaExample],
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> dict[str, Any]:
    if len(prepared) < 2:
        return {"aggregate": {}, "samples": []}
    modes = ["no_memory", "delta_qv", "delta_qv_wrong_query"]
    by_mode: dict[str, list[float]] = {mode: [] for mode in modes}
    address_rows: list[dict[str, float]] = []
    samples: list[dict[str, Any]] = []
    shared_pool = _memory_pool(writer, prepared, layer_ids, cfg) if cfg.shared_memory_retrieval else None
    for sample_idx, sample in enumerate(prepared):
        foreign = _foreign_sample(prepared, sample_idx)
        diagnostic_pool = shared_pool if shared_pool is not None else (
            _live_memories(writer, sample, layer_ids, cfg) + _live_memories(writer, foreign, layer_ids, cfg)
        )
        address_diagnostics = _address_diagnostics(diagnostic_pool, sample.query, sample, foreign)
        address_rows.append(address_diagnostics)
        correct_memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, sample, layer_ids, cfg),
            sample.query,
            cfg.top_k,
        )
        foreign_memories = _select_topk_by_layer(
            shared_pool if shared_pool is not None else _live_memories(writer, foreign, layer_ids, cfg),
            foreign.query,
            cfg.top_k,
        )
        sample_modes: dict[str, dict[str, float]] = {}
        for mode in modes:
            if mode == "no_memory":
                memories = {}
            elif mode == "delta_qv_wrong_query":
                memories = foreign_memories
            else:
                memories = correct_memories
            correct = _score_candidate_answer(bundle, injector, sample.example.question, sample.example.answer, memories, mode)
            foreign_score = _score_candidate_answer(bundle, injector, sample.example.question, foreign.example.answer, memories, mode)
            margin = foreign_score["answer_nll"] - correct["answer_nll"]
            by_mode[mode].append(margin)
            sample_modes[mode] = {
                "correct_answer_nll": correct["answer_nll"],
                "foreign_answer_nll": foreign_score["answer_nll"],
                "foreign_minus_correct_nll": margin,
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
    aggregate["address"] = _aggregate_address_diagnostics(address_rows)
    if by_mode["delta_qv"] and by_mode["delta_qv_wrong_query"]:
        aggregate["delta_qv"]["margin_advantage_vs_wrong_query"] = (
            aggregate["delta_qv"]["foreign_minus_correct_nll"]
            - aggregate["delta_qv_wrong_query"]["foreign_minus_correct_nll"]
        )
    return {"aggregate": aggregate, "samples": samples}


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
    q = F.normalize(query.float().to(memories[0].raw_key.device), dim=0)
    scored = []
    for item in memories:
        score = float(F.normalize(item.raw_key.float(), dim=0).matmul(q).detach().cpu())
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


def _score_candidate_answer(
    bundle,
    injector: GemmaAttentionInjector,
    question: str,
    answer: str,
    memories_by_layer: dict[int, list[AttentionMemoryItem]],
    mode: str,
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
    else:
        result = injector.forward_layers(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            memories_by_layer=memories_by_layer,
            mode=mode,
        )
        logits = result.logits
    return compute_answer_metrics(logits, prompt["input_ids"], answer_start, answer_ids)


def diagnose_experiment(initial_eval: dict[str, Any], final_eval: dict[str, Any]) -> dict[str, Any]:
    initial_delta = _mode_metric(initial_eval, "delta_qv", "answer_nll")
    final_delta = _mode_metric(final_eval, "delta_qv", "answer_nll")
    final_zero = _mode_metric(final_eval, "delta_qv_zero", "answer_nll")
    final_random = _mode_metric(final_eval, "delta_qv_random", "answer_nll")
    final_shuffled = _mode_metric(final_eval, "delta_qv_shuffled", "answer_nll")
    return {
        "eval_delta_nll_drop": initial_delta - final_delta,
        "eval_delta_beats_zero": final_delta < final_zero,
        "eval_delta_beats_random": final_delta < final_random,
        "eval_delta_beats_shuffled": final_delta < final_shuffled,
        "mechanism_supported_on_eval": final_delta < final_zero and final_delta < final_random and final_delta < final_shuffled,
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
    with torch.no_grad():
        query_base = bundle.model(
            input_ids=query_prompt["input_ids"],
            attention_mask=query_prompt["attention_mask"],
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
    query = fit_memory_dim(query_base.hidden_states[-1].mean(dim=(0, 1)).detach().float().cpu(), cfg.memory_dim)
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
        query=query,
        snippets=snippets,
        answer_start=answer_start,
        answer_ids=answer_ids,
    )


def _live_memories(
    writer: RCVHCWriter,
    sample: PreparedDeltaExample,
    layer_ids: list[int],
    cfg: DeltaExperimentConfig,
) -> list[AttentionMemoryItem]:
    items: list[AttentionMemoryItem] = []
    for layer_id in layer_ids:
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


def _aggregate_by_mode(by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for mode, rows in by_mode.items():
        metrics = {"answer_nll": [], "answer_rank": [], "top10": [], "answer_logprob": []}
        traces = {"q_delta_norm": [], "v_delta_norm": [], "gate_v": [], "injected_layers": []}
        for row in rows:
            for key in metrics:
                value = row["metrics"].get(key)
                if value is not None:
                    metrics[key].append(float(value))
            for key in traces:
                traces[key].append(float(row["qkv_trace"].get(key, 0.0)))
        aggregate[mode] = {key: _mean(values) for key, values in (metrics | traces).items()}
    return aggregate


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
    for key in ["model", "device", "dtype", "steps", "lr", "task_suite", "train_samples", "eval_samples", "block_size", "memory_dim", "top_k"]:
        lines.append(f"- `{key}`: `{cfg[key]}`")
    lines.extend(
        [
            f"- `layer_ids`: `{summary['layer_ids']}`",
            f"- `trainable_base_params`: `{summary['trainable_base_params']}`",
            f"- `prompt_insertion_used`: `{summary['prompt_insertion_used']}`",
            "",
            "## Eval Aggregate",
            "",
            "| mode | initial_nll | final_nll | final_rank | top10 | q_delta | v_delta | gate_v |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    initial = summary["initial_eval"]["aggregate"]
    final = summary["final_eval"]["aggregate"]
    for mode in TRAIN_EVAL_MODES:
        lines.append(
            "| {mode} | {initial_nll:.4f} | {final_nll:.4f} | {rank:.4f} | {top10:.4f} | {q:.4f} | {v:.4f} | {gate:.4f} |".format(
                mode=mode,
                initial_nll=float(initial[mode]["answer_nll"]),
                final_nll=float(final[mode]["answer_nll"]),
                rank=float(final[mode]["answer_rank"]),
                top10=float(final[mode]["top10"]),
                q=float(final[mode]["q_delta_norm"]),
                v=float(final[mode]["v_delta_norm"]),
                gate=float(final[mode]["gate_v"]),
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
            lines.append(f"| {mode} | {float(row.get('foreign_minus_correct_nll', 0.0)):.4f} |")
        delta_row = (margins.get("aggregate") or {}).get("delta_qv", {})
        if "margin_advantage_vs_wrong_query" in delta_row:
            lines.append("")
            lines.append(f"- `delta_qv_margin_advantage_vs_wrong_query`: `{float(delta_row['margin_advantage_vs_wrong_query']):.4f}`")
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
