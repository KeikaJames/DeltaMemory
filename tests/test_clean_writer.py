from __future__ import annotations

import torch

from rcvhc.memory.writer import RCVHCWriter


def test_writer_outputs_raw_and_delta_memory():
    torch.manual_seed(0)
    writer = RCVHCWriter(hidden_size=16, memory_dim=8, block_size=4)
    h_in = torch.randn(1, 12, 16)
    h_out = torch.randn(1, 12, 16)
    attn = torch.rand(1, 2, 12, 12)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    items = writer.write_layer(layer_id=1, h_in=h_in, h_out=h_out, attn=attn)
    assert len(items) == 3
    assert items[0].raw_key.shape == (8,)
    assert items[0].address_key.shape == (8,)
    assert items[0].delta_q.shape == (8,)
    assert items[0].metadata["source_text_debug_only"] is True
    assert all(torch.isfinite(item.delta_v).all() for item in items)
