from __future__ import annotations

import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

import torch

from deltamemory.core.types import AttentionMemoryItem
from deltamemory.memory.attention_store import AttentionMemoryStore


def _item(block_id: int, score_bias: float = 0.0, *, address: torch.Tensor | None = None) -> AttentionMemoryItem:
    key = torch.zeros(8)
    key[0] = 1.0 + score_bias
    address_key = key if address is None else address
    value = torch.ones(8) * (block_id + 1)
    return AttentionMemoryItem(
        memory_id=None,
        layer_id=0,
        block_id=block_id,
        token_start=block_id * 4,
        token_end=block_id * 4 + 4,
        raw_key=key,
        address_key=address_key,
        raw_value=value,
        delta_q=value,
        delta_k=value,
        delta_v=value,
        usage_mass=1.0,
        metadata={"source_text": f"block {block_id}", "source_text_debug_only": True},
    )


def test_attention_store_save_load_topk(tmp_path):
    store = AttentionMemoryStore(memory_dim=8)
    store.append([_item(0), _item(1, score_bias=0.5)])
    records = store.retrieve_topk(torch.tensor([1.0] + [0.0] * 7), layer_id=0, k=1)
    assert len(records) == 1
    assert records[0].item.layer_id == 0
    loaded = store.load_topk_to_device([records[0].memory_id], "cpu")
    assert len(loaded) == 1
    assert loaded[0].raw_value.device.type == "cpu"
    assert store.storage_bytes() > 0

    store.save(tmp_path / "store")
    restored = AttentionMemoryStore.load(tmp_path / "store")
    assert restored.memory_count() == 2
    assert restored.storage_bytes(tmp_path / "store") > 0


def test_attention_store_retrieves_with_address_key_not_raw_key():
    store = AttentionMemoryStore(memory_dim=8)
    wrong_address = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    correct_address = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    store.append([_item(0, score_bias=10.0, address=wrong_address), _item(1, score_bias=0.0, address=correct_address)])

    records = store.retrieve_topk(correct_address, layer_id=0, k=1)

    assert records[0].item.block_id == 1
