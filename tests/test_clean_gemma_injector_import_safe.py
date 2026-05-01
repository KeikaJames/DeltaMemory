from __future__ import annotations

import torch

from rcvhc.core.types import AttentionMemoryItem
from rcvhc.gemma.attention_injector import GemmaAttentionInjector, QKVDeltaProjector
from rcvhc.gemma.model_adapter import MockGemmaModel, freeze_model


def test_mock_gemma_qkv_injector_nonzero_delta_and_frozen_base():
    torch.manual_seed(0)
    model = MockGemmaModel(vocab_size=128, hidden_size=16, num_layers=2)
    freeze_model(model)
    projector = QKVDeltaProjector(memory_dim=8, projection_dim=16, alpha_scale=0.2, gate_bias=-1.0)
    injector = GemmaAttentionInjector(model, projector)
    memory = AttentionMemoryItem(
        memory_id=0,
        layer_id=1,
        block_id=0,
        token_start=0,
        token_end=4,
        raw_key=torch.randn(8),
        raw_value=torch.randn(8),
        delta_q=torch.randn(8),
        delta_k=torch.randn(8),
        delta_v=torch.randn(8),
        usage_mass=1.0,
        metadata={},
    )
    ids = torch.randint(0, 100, (1, 6))
    out = injector.forward(ids, torch.ones_like(ids), layer_id=1, memories=[memory], mode="delta_qv")
    assert out.logits.shape[:2] == (1, 6)
    assert out.trace.q_delta_norm > 0
    assert out.trace.v_delta_norm > 0
    assert all(param.grad is None for param in model.parameters())


def test_mock_gemma_qkv_injector_injects_multiple_attention_layers():
    torch.manual_seed(1)
    model = MockGemmaModel(vocab_size=128, hidden_size=16, num_layers=3)
    freeze_model(model)
    projector = QKVDeltaProjector(memory_dim=8, projection_dim=16, alpha_scale=0.2, gate_bias=-1.0)
    injector = GemmaAttentionInjector(model, projector)
    memories_by_layer = {
        layer_id: [
            AttentionMemoryItem(
                memory_id=layer_id,
                layer_id=layer_id,
                block_id=0,
                token_start=0,
                token_end=4,
                raw_key=torch.randn(8),
                raw_value=torch.randn(8),
                delta_q=torch.randn(8),
                delta_k=torch.randn(8),
                delta_v=torch.randn(8),
                usage_mass=1.0,
                metadata={},
            )
        ]
        for layer_id in range(3)
    }
    ids = torch.randint(0, 100, (1, 6))
    out = injector.forward_layers(ids, torch.ones_like(ids), memories_by_layer=memories_by_layer, mode="delta_qv")
    assert out.logits.shape[:2] == (1, 6)
    assert out.trace.injected_layers == 3
    assert out.trace.q_delta_norm > 0
    assert out.trace.v_delta_norm > 0
    assert all(param.grad is None for param in model.parameters())
