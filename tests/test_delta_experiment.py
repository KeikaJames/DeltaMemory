from __future__ import annotations

from pathlib import Path
from collections import Counter

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


def test_paired_conflict_binding_has_same_unit_conflicts():
    examples = make_delta_memory_examples("paired_conflict_binding", 4, seed=13)
    unit_counts = Counter(example.unit for example in examples)
    assert sorted(unit_counts.values()) == [2, 2]
    for first, second in zip(examples[0::2], examples[1::2]):
        assert first.unit == second.unit
        assert first.answer != second.answer
        assert first.question != second.question
        assert first.paired_sample_id == second.sample_id
        assert second.paired_sample_id == first.sample_id
        assert first.collision_group_id == second.collision_group_id


def test_address_token_binding_has_explicit_paired_addresses():
    examples = make_delta_memory_examples("address_token_binding", 4, seed=19)
    for first, second in zip(examples[0::2], examples[1::2]):
        assert first.unit == second.unit
        assert first.answer != second.answer
        assert first.paired_sample_id == second.sample_id
        assert second.paired_sample_id == first.sample_id
        assert "ADDRESS:" in first.text
        assert "ADDR::" in first.question
        assert first.answer not in first.question
        assert first.address_text is not None
        assert first.value_text == f"secret-code = {first.answer}"
        assert first.foreign_address_text == second.address_text
        assert first.foreign_value_text == second.value_text
        assert first.address_char_range is not None
        assert first.value_char_range is not None


def test_address_token_binding_single_token_answers():
    examples = make_delta_memory_examples("address_token_binding_single_token", 4, seed=23)
    assert {example.task_type for example in examples} == {"address_token_binding_single_token"}
    for example in examples:
        assert " " not in example.answer
        assert "-" not in example.answer
        assert example.value_text == f"secret-code = {example.answer}"


def test_long_distance_nolima_has_large_gap():
    example = make_delta_memory_examples("long_distance_nolima_style", 1, seed=17)[0]
    assert example.task_type == "long_distance_nolima_style"
    assert example.answer in example.text
    assert example.answer not in example.question
    assert len(example.text.split()) > 350


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
    assert "hidden_retrieval" in summary["final_eval"]["aggregate"]
    assert "retrieved_attention" in summary["final_eval"]["aggregate"]
    assert "logit_bias" in summary["final_eval"]["aggregate"]
    assert "payload_probe" in summary["final_eval"]["aggregate"]
    assert "delta_qv_wrong_layer" in summary["final_eval"]["aggregate"]
    assert "delta_qv_wrong_query" in summary["final_eval"]["aggregate"]
    assert "payload_probe" in summary["stage2_binding_summary"]["eval_modes"]
    assert "statistics" in summary
    assert "no_memory" in summary["statistics"]["comparisons"]
    assert "retrieved_attention" in summary["statistics"]["comparisons"]
    assert "logit_bias" in summary["statistics"]["comparisons"]
    assert summary["final_eval"]["aggregate"]["delta_qv"]["q_delta_norm"] > 0.0
    paths = write_delta_experiment_report(summary, tmp_path / "report")
    assert Path(paths["report"]).exists()


def test_delta_experiment_conflict_margins(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=1,
        train_samples=2,
        eval_samples=2,
        block_size=16,
        memory_dim=32,
        top_k=2,
        conflict_margins=True,
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    margins = summary["conflict_margins"]
    assert margins is not None
    assert "delta_qv" in margins["aggregate"]
    assert "delta_qv_wrong_query" in margins["aggregate"]
    assert "margin_advantage_vs_wrong_query" in margins["aggregate"]["delta_qv"]
    assert "address" in margins["aggregate"]
    assert "correct_address_rank" in margins["samples"][0]["address_diagnostics"]
    assert "delta_qv_oracle_correct_address_paired_payload" in margins["aggregate"]
    assert "delta_qv_oracle_paired_address_correct_payload" in margins["aggregate"]
    assert "logit_bias_oracle_correct" in margins["aggregate"]
    assert "payload_probe_oracle_correct" in margins["aggregate"]
    assert "answer_token_binding_margin" in margins["aggregate"]["logit_bias_oracle_correct"]


def test_delta_experiment_oracle_span_writer_conflict_controls(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=1,
        train_samples=2,
        eval_samples=2,
        task_suite="address_token_binding",
        block_size=16,
        memory_dim=32,
        top_k=1,
        oracle_span_writer=True,
        conflict_margins=True,
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert summary["oracle_span_writer"] is True
    assert summary["eval_examples"][0]["address_text"] is not None
    sample = summary["conflict_margins"]["samples"][0]
    assert "delta_qv_oracle_correct_address_paired_payload" in sample["modes"]
    assert "delta_qv_oracle_paired_address_correct_payload" in sample["modes"]
    assert "logit_bias_oracle_correct" in sample["modes"]
    assert "payload_probe_oracle_correct" in sample["modes"]
    assert "answer_token" in sample["modes"]["logit_bias_oracle_correct"]


def test_delta_experiment_logit_bias_training_records_loss(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=1,
        train_samples=2,
        eval_samples=2,
        task_suite="address_token_binding_single_token",
        block_size=16,
        memory_dim=32,
        top_k=1,
        oracle_span_writer=True,
        conflict_margins=True,
        logit_bias_loss_weight=1.0,
        payload_answer_loss_weight=1.0,
        payload_embedding_loss_weight=0.1,
        stage2_swap_loss_weight=0.1,
        stage2_swap_mode="payload_probe",
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert summary["train"][0]["logit_bias_loss"] >= 0.0
    assert summary["train"][0]["payload_answer_loss"] >= 0.0
    assert summary["train"][0]["payload_embedding_loss"] >= 0.0
    assert summary["train"][0]["stage2_swap_loss"] >= 0.0
    assert "stage2_binding_margin" in summary["train"][0]
    assert "stage2_swap_margin" in summary["train"][0]
    assert "logit_bias" in summary["final_eval"]["aggregate"]
    assert "payload_probe" in summary["final_eval"]["aggregate"]
    assert "logit_bias_oracle_correct" in summary["conflict_margins"]["aggregate"]
    assert "payload_probe_oracle_correct" in summary["conflict_margins"]["aggregate"]
    assert summary["stage2_binding_summary"]["payload_probe_layer_strategy"] is not None


def test_delta_experiment_restricted_eval_modes_include_payload_probe(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=1,
        train_samples=2,
        eval_samples=2,
        task_suite="address_token_binding_single_token",
        block_size=16,
        memory_dim=32,
        top_k=1,
        oracle_span_writer=True,
        eval_injection_modes="payload_probe,logit_bias",
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert list(summary["final_eval"]["aggregate"]) == ["no_memory", "payload_probe", "logit_bias"]
    assert summary["diagnosis"]["diagnosis_skipped"] is True
    assert "payload_probe" in summary["stage2_binding_summary"]["eval_modes"]


def test_delta_experiment_lm_head_lora_training_records_loss(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=1,
        train_samples=2,
        eval_samples=2,
        task_suite="address_token_binding_single_token",
        block_size=16,
        memory_dim=32,
        top_k=1,
        oracle_span_writer=True,
        conflict_margins=True,
        lm_head_lora_loss_weight=1.0,
        stage2_swap_loss_weight=0.1,
        stage2_swap_mode="lm_head_lora",
        eval_injection_modes="lm_head_lora,payload_probe",
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert summary["train"][0]["lm_head_lora_loss"] >= 0.0
    assert summary["train"][0]["stage2_swap_loss"] >= 0.0
    assert "lm_head_lora" in summary["final_eval"]["aggregate"]
    assert "lm_head_lora_update_norm" in summary["final_eval"]["aggregate"]["lm_head_lora"]
    assert "lm_head_lora_oracle_correct" in summary["conflict_margins"]["aggregate"]
    assert "lm_head_lora" in summary["stage2_binding_summary"]["eval_modes"]


def test_delta_experiment_contrastive_training_records_margin(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=1,
        train_samples=2,
        eval_samples=2,
        task_suite="paired_conflict_binding",
        block_size=16,
        memory_dim=32,
        top_k=2,
        contrastive_margin_weight=0.1,
        contrastive_margin=0.2,
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert summary["train"][0]["contrastive_loss"] >= 0.0
    assert "contrastive_margin_advantage" in summary["train"][0]


def test_delta_experiment_shared_memory_retrieval(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=1,
        train_samples=2,
        eval_samples=2,
        task_suite="paired_conflict_binding",
        block_size=16,
        memory_dim=32,
        top_k=2,
        shared_memory_retrieval=True,
        conflict_margins=True,
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert summary["config"]["shared_memory_retrieval"] is True
    assert summary["conflict_margins"]["aggregate"]["delta_qv"]


def test_factual_capital_examples_have_paired_metadata():
    examples = make_delta_memory_examples(num_examples=8, seed=0, task_suite="factual_capital_binding")
    assert len(examples) == 8
    for example in examples:
        assert example.task_type == "factual_capital_binding"
        assert example.address_text and example.address_text in example.text
        assert example.value_text and example.value_text in example.text
        assert example.foreign_answer and example.foreign_answer != example.answer
        assert example.unit in example.question


def test_writer_attn_pool_runs_in_mock_smoke(tmp_path):
    cfg = DeltaExperimentConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        steps=2,
        train_samples=2,
        eval_samples=2,
        block_size=16,
        memory_dim=32,
        top_k=1,
        task_suite="address_token_binding_single_token",
        oracle_span_writer=True,
        payload_probe_layer_strategy="first_layer",
        payload_answer_loss_weight=1.0,
        payload_answer_loss_warmup_frac=0.5,
        writer_pool="attn",
        eval_injection_modes="no_memory,payload_probe",
        report_dir=str(tmp_path / "report"),
    )
    summary = run_delta_experiment(cfg)
    assert summary["trainable_base_params"] == 0
    assert summary["config"]["writer_pool"] == "attn"
    assert "payload_probe" in summary["final_eval"]["aggregate"]


def test_update_readme_charts_round_trip(tmp_path):
    import json as _json
    import sys as _sys
    import importlib.util
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "update_readme_charts", repo_root / "scripts" / "update_readme_charts.py"
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    reports_dir = tmp_path / "reports"
    run_dir = reports_dir / "stage6_dummy"
    run_dir.mkdir(parents=True)
    (run_dir / "delta_experiment_summary.json").write_text(_json.dumps({
        "config": {"oracle_span_writer": True, "task_suite": "address_token_binding_single_token",
                   "writer_pool": "attn", "train_samples": 8, "eval_samples": 8, "seed": 0},
        "final_eval": {"aggregate": {
            "delta_qv": {"answer_nll": 4.5, "top1_correct_rate": 0.3},
            "payload_probe": {"answer_nll": 3.1, "top1_correct_rate": 0.85},
        }},
        "stage2_binding_summary": {"eval_modes": {
            "payload_probe": {"top1_correct_rate": 0.85},
        }},
    }))
    rows = mod.collect_summaries(reports_dir)
    block = mod.render_block(rows)
    assert "stage6_dummy" in block
    assert "payload_probe" in block
    readme = tmp_path / "README.md"
    readme.write_text("Existing\n", encoding="utf-8")
    assert mod.rewrite_readme(readme, block) is True
    text = readme.read_text(encoding="utf-8")
    assert mod.BEGIN_MARK in text and mod.END_MARK in text
    assert mod.rewrite_readme(readme, block) is False  # idempotent
