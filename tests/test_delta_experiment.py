from __future__ import annotations

from pathlib import Path

from rcvhc.engine.delta_dataset import make_later_reference_examples
from rcvhc.engine.delta_experiment import DeltaExperimentConfig, run_delta_experiment, write_delta_experiment_report


def test_later_reference_examples_have_required_fields():
    examples = make_later_reference_examples(3, seed=3)
    assert len(examples) == 3
    for example in examples:
        assert example.unit in example.text
        assert example.answer in example.text
        assert example.unit in example.question
        assert example.answer not in example.question


def test_delta_experiment_mock_smoke(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=2,
        train_samples=2,
        eval_samples=2,
        block_size=16,
        memory_dim=32,
        top_k=2,
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert summary["trainable_base_params"] == 0
    assert len(summary["train"]) == 2
    assert "delta_qv" in summary["final_eval"]["aggregate"]
    assert summary["final_eval"]["aggregate"]["delta_qv"]["q_delta_norm"] > 0.0
    paths = write_delta_experiment_report(summary, tmp_path / "report")
    assert Path(paths["report"]).exists()
