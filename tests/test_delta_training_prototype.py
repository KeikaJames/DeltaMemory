from __future__ import annotations

from pathlib import Path

from rcvhc.engine.delta_training import DeltaTrainingConfig, run_delta_training, write_training_report


def test_delta_training_prototype_mock(tmp_path):
    input_path = tmp_path / "demo.txt"
    input_path.write_text(
        "The secret code for unit XJQ-482 is tulip-91. "
        "The secret code for unit XJQ-483 is tulip-19. "
        "The unit XJQ-482 was later selected for verification. "
        "The verifier checked unit XJQ-482 again.",
        encoding="utf-8",
    )
    cfg = DeltaTrainingConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        input_path=str(input_path),
        report_dir=str(tmp_path / "report"),
        steps=2,
        lr=1e-3,
        block_size=8,
        memory_dim=32,
        top_k=1,
    )
    summary = run_delta_training(cfg)
    assert summary["trainable_base_params"] == 0
    assert len(summary["train"]) == 2
    assert summary["train"][-1]["grad_norm"] > 0.0
    assert summary["final"]["delta_qv"]["qkv_trace"]["q_delta_norm"] > 0.0
    assert summary["final"]["delta_qv"]["qkv_trace"]["v_delta_norm"] > 0.0
    paths = write_training_report(summary, tmp_path / "report")
    assert Path(paths["report"]).exists()
