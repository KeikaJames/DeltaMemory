"""Small retrieval helpers for attention memory."""

from __future__ import annotations

import torch

from deltamemory.legacy.core.types import RetrievalRecord
from deltamemory.memory.attention_store import AttentionMemoryStore


def retrieve_layer_topk(
    store: AttentionMemoryStore,
    query: torch.Tensor,
    layer_id: int,
    top_k: int,
) -> list[RetrievalRecord]:
    return store.retrieve_topk(query=query, layer_id=layer_id, k=top_k)
