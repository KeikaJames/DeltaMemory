"""CPU/disk storage for cleanroom attention memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from rcvhc.core.types import AttentionMemoryItem, RetrievalRecord


class AttentionMemoryStore:
    """Storage-backed memory bank.

    The store keeps all memory tensors on CPU. Retrieval scans CPU keys and only
    selected top-k records are moved to the model device by the caller.
    """

    def __init__(self, memory_dim: int, temperature: float = 8.0, eps: float = 1e-6) -> None:
        self.memory_dim = int(memory_dim)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self._items: list[AttentionMemoryItem] = []
        self._next_id = 0
        self.metadata: dict[str, object] = {
            "memory_dim": self.memory_dim,
            "temperature": self.temperature,
            "created_by": "rcvhc.cleanroom.AttentionMemoryStore",
        }

    def append(self, items: Iterable[AttentionMemoryItem]) -> list[int]:
        written: list[int] = []
        for item in items:
            stored = item.to_cpu_detached()
            stored.memory_id = self._next_id
            stored.metadata = dict(stored.metadata)
            stored.metadata.setdefault("memory_id", self._next_id)
            stored.metadata.setdefault("layer_id", stored.layer_id)
            stored.metadata.setdefault("block_id", stored.block_id)
            stored.metadata.setdefault("token_range", [stored.token_start, stored.token_end])
            self._items.append(stored)
            written.append(self._next_id)
            self._next_id += 1
        return written

    def retrieve_topk(self, query: torch.Tensor, layer_id: int, k: int) -> list[RetrievalRecord]:
        if k <= 0:
            raise ValueError("k must be positive")
        candidates = [item for item in self._items if item.layer_id == layer_id]
        if not candidates:
            return []
        q = query.detach().cpu().float()
        if q.dim() == 3:
            q = q.mean(dim=(0, 1))
        elif q.dim() == 2:
            q = q.mean(dim=0)
        if q.numel() != self.memory_dim:
            q = _fit_dim(q, self.memory_dim)
        keys = torch.stack([item.raw_key.float() for item in candidates], dim=0)
        scores = self.temperature * F.normalize(keys, dim=-1, eps=self.eps).matmul(
            F.normalize(q, dim=0, eps=self.eps)
        )
        top_scores, top_idx = scores.topk(min(k, len(candidates)))
        records: list[RetrievalRecord] = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            item = candidates[int(idx)]
            records.append(RetrievalRecord(memory_id=int(item.memory_id), score=float(score), item=item))
        return records

    def load_topk_to_device(self, memory_ids: list[int], device: torch.device | str) -> list[AttentionMemoryItem]:
        device = torch.device(device)
        by_id = {item.memory_id: item for item in self._items}
        loaded: list[AttentionMemoryItem] = []
        for rank, memory_id in enumerate(memory_ids):
            item = by_id.get(memory_id)
            if item is None:
                continue
            metadata = dict(item.metadata)
            metadata["retrieval_rank"] = rank
            loaded.append(
                AttentionMemoryItem(
                    memory_id=item.memory_id,
                    layer_id=item.layer_id,
                    block_id=item.block_id,
                    token_start=item.token_start,
                    token_end=item.token_end,
                    raw_key=item.raw_key.to(device),
                    raw_value=item.raw_value.to(device),
                    delta_q=item.delta_q.to(device),
                    delta_k=item.delta_k.to(device),
                    delta_v=item.delta_v.to(device),
                    usage_mass=item.usage_mass,
                    metadata=metadata,
                )
            )
        return loaded

    def metadata_lookup(self, memory_ids: list[int]) -> list[dict[str, object]]:
        wanted = set(memory_ids)
        return [dict(item.metadata) for item in self._items if item.memory_id in wanted]

    def clear_gpu_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def to_cpu(self) -> "AttentionMemoryStore":
        self._items = [item.to_cpu_detached() for item in self._items]
        return self

    def memory_count(self) -> int:
        return len(self._items)

    def storage_bytes(self, path: str | Path | None = None) -> int:
        if path is not None and Path(path).exists():
            return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        tensor_bytes = 0
        for item in self._items:
            for tensor in (item.raw_key, item.raw_value, item.delta_q, item.delta_k, item.delta_v):
                tensor_bytes += tensor.numel() * tensor.element_size()
        metadata_bytes = sum(len(json.dumps(item.metadata, sort_keys=True).encode("utf-8")) for item in self._items)
        return int(tensor_bytes + metadata_bytes)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "memory_dim": self.memory_dim,
            "temperature": self.temperature,
            "eps": self.eps,
            "next_id": self._next_id,
            "metadata": self.metadata,
            "records": [
                {
                    "memory_id": item.memory_id,
                    "layer_id": item.layer_id,
                    "block_id": item.block_id,
                    "token_start": item.token_start,
                    "token_end": item.token_end,
                    "raw_key": item.raw_key,
                    "raw_value": item.raw_value,
                    "delta_q": item.delta_q,
                    "delta_k": item.delta_k,
                    "delta_v": item.delta_v,
                    "usage_mass": item.usage_mass,
                    "metadata": item.metadata,
                }
                for item in self._items
            ],
        }
        torch.save(payload, path / "attention_memory.pt")
        with (path / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(self.metadata | {"memory_count": self.memory_count()}, fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "AttentionMemoryStore":
        path = Path(path)
        payload = torch.load(path / "attention_memory.pt", map_location="cpu", weights_only=False)
        store = cls(int(payload["memory_dim"]), float(payload["temperature"]), float(payload["eps"]))
        store._next_id = int(payload["next_id"])
        store.metadata = dict(payload.get("metadata", {}))
        store._items = [
            AttentionMemoryItem(
                memory_id=int(record["memory_id"]),
                layer_id=int(record["layer_id"]),
                block_id=int(record["block_id"]),
                token_start=int(record["token_start"]),
                token_end=int(record["token_end"]),
                raw_key=record["raw_key"].float().cpu(),
                raw_value=record["raw_value"].float().cpu(),
                delta_q=record["delta_q"].float().cpu(),
                delta_k=record["delta_k"].float().cpu(),
                delta_v=record["delta_v"].float().cpu(),
                usage_mass=float(record["usage_mass"]),
                metadata=dict(record["metadata"]),
            )
            for record in payload.get("records", [])
        ]
        return store


def _fit_dim(vector: torch.Tensor, dim: int) -> torch.Tensor:
    flat = vector.flatten().float()
    if flat.numel() == dim:
        return flat
    if flat.numel() > dim:
        return flat[:dim]
    return F.pad(flat, (0, dim - flat.numel()))
