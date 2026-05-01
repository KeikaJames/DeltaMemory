"""Unified ingest/ask engine for the cleanroom prototype."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rcvhc.core.config import RCVHCCleanConfig, resolve_layer_policy
from rcvhc.core.types import AttentionMemoryItem
from rcvhc.gemma.attention_injector import GemmaAttentionInjector, QKVDeltaProjector
from rcvhc.gemma.model_adapter import ModelBundle, exposed_qkv_layers, get_hidden_size, trainable_base_params
from rcvhc.memory.attention_store import AttentionMemoryStore
from rcvhc.memory.writer import RCVHCWriter, fit_memory_dim, split_source_snippets


@dataclass
class EngineBundle:
    bundle: ModelBundle
    cfg: RCVHCCleanConfig
    store: AttentionMemoryStore


class AttentionMemoryEngine:
    def __init__(self, bundle: ModelBundle, cfg: RCVHCCleanConfig, store: AttentionMemoryStore | None = None) -> None:
        cfg.validate()
        self.bundle = bundle
        self.cfg = cfg
        hidden = get_hidden_size(bundle.model)
        self.writer = RCVHCWriter(hidden, cfg.memory_dim, cfg.block_size).to(bundle.device, dtype=bundle.dtype)
        self.projector = QKVDeltaProjector(cfg.memory_dim, hidden, alpha_scale=cfg.alpha_scale, gate_bias=cfg.gate_bias).to(
            bundle.device, dtype=bundle.dtype
        )
        self.store = store or AttentionMemoryStore(cfg.memory_dim)
        self.store.metadata.update(
            {
                "model_name": bundle.model_name,
                "device": str(bundle.device),
                "dtype": str(bundle.dtype).removeprefix("torch."),
                "block_size": cfg.block_size,
                "local_window": cfg.local_window,
                "source_text_debug_only": True,
            }
        )

    def ingest(self, text: str, layers: str | None = None) -> dict[str, Any]:
        encoded = self._encode(text)
        input_ids = encoded["input_ids"].to(self.bundle.device)
        attention_mask = encoded["attention_mask"].to(self.bundle.device)
        with torch.no_grad():
            out = self.bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
        if out.hidden_states is None:
            raise RuntimeError("model did not return hidden_states")
        exposed = exposed_qkv_layers(self.bundle.model) or list(range(max(0, len(out.hidden_states) - 1)))
        enabled = resolve_layer_policy(layers or self.cfg.layers, exposed)
        snippets = split_source_snippets(self.bundle.tokenizer, input_ids, self.cfg.block_size)
        all_items: list[AttentionMemoryItem] = []
        for layer_id in enabled:
            h_in = out.hidden_states[layer_id]
            h_out = out.hidden_states[layer_id + 1]
            attn = out.attentions[layer_id] if out.attentions is not None and layer_id < len(out.attentions) else None
            all_items.extend(
                self.writer.write_layer(
                    layer_id=layer_id,
                    h_in=h_in,
                    h_out=h_out,
                    attn=attn,
                    token_offset=0,
                    source_text_by_block=snippets,
                )
            )
        self.store.append(all_items)
        self.store.metadata.update(
            {
                "total_tokens_ingested": int(input_ids.shape[1]),
                "enabled_layers": enabled,
                "attentions_available": out.attentions is not None,
            }
        )
        return {
            "total_tokens_ingested": int(input_ids.shape[1]),
            "enabled_layers": enabled,
            "memory_blocks": self.store.memory_count(),
            "storage_bytes": self.store.storage_bytes(),
            "trainable_base_params": trainable_base_params(self.bundle.model),
        }

    def ask(
        self,
        question: str,
        *,
        answer: str | None = None,
        modes: list[str] | None = None,
        top_k: int | None = None,
        alpha_scale: float | None = None,
        gate_bias: float | None = None,
    ) -> dict[str, Any]:
        modes = modes or list(self.cfg.main_modes)
        top_k = top_k or self.cfg.top_k
        if alpha_scale is not None or gate_bias is not None:
            alpha = self.cfg.alpha_scale if alpha_scale is None else alpha_scale
            gate = self.cfg.gate_bias if gate_bias is None else gate_bias
            self.projector = QKVDeltaProjector(self.cfg.memory_dim, get_hidden_size(self.bundle.model), alpha, gate).to(
                self.bundle.device, dtype=self.bundle.dtype
            )
        prompt_text, answer_start, answer_ids = self._metric_prompt(question, answer)
        encoded = self._encode(prompt_text)
        input_ids = encoded["input_ids"].to(self.bundle.device)
        attention_mask = encoded["attention_mask"].to(self.bundle.device)
        with torch.no_grad():
            base = self.bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )
        query = fit_memory_dim(base.hidden_states[-1].mean(dim=(0, 1)).detach().float().cpu(), self.cfg.memory_dim)
        layer_ids = _memory_layers(self.store)
        retrieved_by_layer = {
            layer_id: self.store.retrieve_topk(query, layer_id=layer_id, k=top_k)
            for layer_id in layer_ids
        }
        retrieved = [record for layer_records in retrieved_by_layer.values() for record in layer_records]
        memory_ids_by_layer = {
            layer_id: [record.memory_id for record in layer_records]
            for layer_id, layer_records in retrieved_by_layer.items()
        }
        memories_by_layer = {
            layer_id: self.store.load_topk_to_device(memory_ids, self.bundle.device)
            for layer_id, memory_ids in memory_ids_by_layer.items()
        }
        memories = [item for layer_memories in memories_by_layer.values() for item in layer_memories]

        comparisons: dict[str, dict[str, Any]] = {}
        injector = GemmaAttentionInjector(self.bundle.model, self.projector)
        for mode in modes:
            if mode == "raw_memory":
                logits, trace = self._raw_late_readout(base.logits, base.hidden_states[-1], memories)
            elif mode == "no_memory":
                logits, trace = base.logits, {}
            else:
                result = injector.forward_layers(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    memories_by_layer=memories_by_layer,
                    mode=mode,
                )
                logits, trace = result.logits, result.trace.as_dict()
            comparisons[mode] = {
                "answer": _greedy_answer(self.bundle.tokenizer, logits),
                "metrics": compute_answer_metrics(logits, input_ids, answer_start, answer_ids),
                "qkv_trace": trace,
            }

        return {
            "question": question,
            "answer_expected": answer,
            "prompt_used": prompt_text,
            "source_text_used_in_prompt": False,
            "comparisons": comparisons,
            "retrieved_memory": [
                {
                    "memory_id": record.memory_id,
                    "score": record.score,
                    "layer_id": record.item.layer_id,
                    "block_id": record.item.block_id,
                    "token_range": [record.item.token_start, record.item.token_end],
                    "usage_mass": record.item.usage_mass,
                    "source_snippet_debug_only": str(record.item.metadata.get("source_text", ""))[:240],
                }
                for record in retrieved
            ],
            "system_stats": {
                "total_tokens_ingested": self.store.metadata.get("total_tokens_ingested", 0),
                "local_window": self.cfg.local_window,
                "memory_blocks": self.store.memory_count(),
                "storage_bytes": self.store.storage_bytes(),
                "top_k": top_k,
                "injection_layers": layer_ids,
                "device": str(self.bundle.device),
                "dtype": str(self.bundle.dtype).removeprefix("torch."),
                "trainable_base_params": trainable_base_params(self.bundle.model),
            },
        }

    def save_store(self, path: str | Path) -> None:
        self.store.save(path)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        return {"input_ids": encoded["input_ids"].long(), "attention_mask": encoded["attention_mask"].long()}

    def _metric_prompt(self, question: str, answer: str | None) -> tuple[str, int | None, list[int] | None]:
        prompt = f"Question: {question}\nAnswer:"
        if answer is None:
            return prompt, None, None
        prompt_ids = self.bundle.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.bundle.tokenizer.encode(" " + answer, add_special_tokens=False)
        return prompt + " " + answer, len(prompt_ids), answer_ids

    def _raw_late_readout(
        self,
        base_logits: torch.Tensor,
        hidden: torch.Tensor,
        memories: list[AttentionMemoryItem],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not memories:
            return base_logits, {}
        raw = torch.stack([item.raw_value.to(hidden.device, hidden.dtype) for item in memories], dim=0).mean(dim=0)
        update = fit_memory_dim(raw, hidden.shape[-1]).to(hidden.device, hidden.dtype)
        hidden_prime = hidden + 0.05 * update.view(1, 1, -1)
        logits = self.bundle.model.get_output_embeddings()(hidden_prime)
        return logits, {"hidden_delta_norm": float((hidden_prime - hidden).detach().float().norm().cpu())}


def compute_answer_metrics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    answer_start: int | None,
    answer_ids: list[int] | None,
) -> dict[str, float | None]:
    keys = {
        "answer_nll": None,
        "answer_rank": None,
        "top10": None,
        "answer_logprob": None,
    }
    if answer_start is None or not answer_ids:
        return keys
    ranks = []
    logprobs = []
    nlls = []
    for offset, token_id in enumerate(answer_ids):
        logit_pos = answer_start + offset - 1
        if logit_pos < 0 or logit_pos >= logits.shape[1]:
            continue
        step_logits = logits[0, logit_pos].float()
        logprob = F.log_softmax(step_logits, dim=-1)[int(token_id)]
        rank = int((step_logits > step_logits[int(token_id)]).sum().item()) + 1
        ranks.append(rank)
        logprobs.append(float(logprob.detach().cpu()))
        nlls.append(float((-logprob).detach().cpu()))
    if not ranks:
        return keys
    return {
        "answer_nll": sum(nlls) / len(nlls),
        "answer_rank": sum(ranks) / len(ranks),
        "top10": sum(1.0 for rank in ranks if rank <= 10) / len(ranks),
        "answer_logprob": sum(logprobs),
    }


def _greedy_answer(tokenizer, logits: torch.Tensor) -> str:
    token_id = int(logits[0, -1].argmax().detach().cpu())
    try:
        return tokenizer.decode([token_id], skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode([token_id])


def _default_layer(store: AttentionMemoryStore) -> int:
    layers = [item.layer_id for item in store._items]
    return max(layers) if layers else 0


def _memory_layers(store: AttentionMemoryStore) -> list[int]:
    return sorted({int(item.layer_id) for item in store._items})
