from __future__ import annotations

from rcvhc.core.config import RCVHCCleanConfig
from rcvhc.engine.attention_memory_engine import AttentionMemoryEngine
from rcvhc.gemma.model_adapter import load_model_bundle, trainable_base_params


def test_engine_ingest_ask_keeps_source_out_of_prompt(demo_text):
    bundle = load_model_bundle("mock-gemma", device="cpu", dtype="float32")
    cfg = RCVHCCleanConfig(model_name="mock-gemma", memory_dim=32, block_size=8, top_k=2)
    engine = AttentionMemoryEngine(bundle, cfg)
    summary = engine.ingest(demo_text)
    assert summary["memory_blocks"] > 0
    assert summary["enabled_layers"] == [0, 1, 2, 3]
    result = engine.ask(
        "What is the secret code for unit XJQ-482?",
        answer="tulip-91",
        modes=["no_memory", "delta_qv", "delta_qv_zero", "delta_qv_random", "delta_qv_shuffled", "delta_qv_force_gate"],
    )
    assert result["source_text_used_in_prompt"] is False
    assert "The secret code for unit XJQ-482 is tulip-91" not in result["prompt_used"]
    assert result["comparisons"]["delta_qv"]["qkv_trace"]["q_delta_norm"] > 0
    assert result["comparisons"]["delta_qv"]["qkv_trace"]["v_delta_norm"] > 0
    assert result["comparisons"]["delta_qv"]["qkv_trace"]["injected_layers"] == 4
    assert result["system_stats"]["injection_layers"] == [0, 1, 2, 3]
    assert result["comparisons"]["delta_qv_zero"]["qkv_trace"]["q_delta_norm"] >= 0
    assert trainable_base_params(bundle.model) == 0
