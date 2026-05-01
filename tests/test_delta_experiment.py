from __future__ import annotations

from pathlib import Path

from rcvhc.engine.delta_dataset import DELTA_TASK_SUITES, make_delta_memory_examples, make_later_reference_examples
from rcvhc.engine.delta_experiment import DeltaExperimentConfig, run_delta_experiment, write_delta_experiment_report


def test_later_reference_examples_have_required_fields():
    examples = make_later_reference_examples(3, seed=3)
    assert len(examples) == 3
    for example in examples:
        assert example.unit in example.text
        assert example.answer in example.text
        assert example.unit in example.question
        assert example.answer not in example.question


def test_delta_memory_task_suites_have_answerable_held_out_questions():
    for task_suite in DELTA_TASK_SUITES:
        examples = make_delta_memory_examples(task_suite, 2, seed=11)
        assert len(examples) == 2
        for example in examples:
            assert example.task_type == task_suite
            assert example.answer in example.text
            assert example.answer not in example.question
            assert example.unit in example.question


def test_delta_memory_answers_are_seed_dependent():
    for task_suite in DELTA_TASK_SUITES:
        train_like = make_delta_memory_examples(task_suite, 8, seed=4)
        eval_like = make_delta_memory_examples(task_suite, 8, seed=10_004)
        assert [example.answer for example in train_like] != [example.answer for example in eval_like]


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
    assert summary["retrieval_query_uses_answer"] is False
    assert len(summary["train"]) == 2
    assert "delta_qv" in summary["final_eval"]["aggregate"]
    assert "raw_memory" in summary["final_eval"]["aggregate"]
    assert "delta_qv_wrong_layer" in summary["final_eval"]["aggregate"]
    assert "delta_qv_wrong_query" in summary["final_eval"]["aggregate"]
    assert "statistics" in summary
    assert "no_memory" in summary["statistics"]["comparisons"]
    assert summary["final_eval"]["aggregate"]["delta_qv"]["q_delta_norm"] > 0.0
    paths = write_delta_experiment_report(summary, tmp_path / "report")
    assert Path(paths["report"]).exists()
