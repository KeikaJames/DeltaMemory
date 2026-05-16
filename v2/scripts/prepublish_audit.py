"""Pre-publication consistency audit for v2.

This script is intentionally skeptical: it fails on stale ambiguous outputs,
paper-facing result JSONs that disagree with their logs, or cross-model JSONs
whose filename/model metadata do not match. It does not prove the science; it
protects the evidence chain from repository/paper inconsistencies.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fail(errors: list[str], msg: str) -> None:
    errors.append(f"FAIL: {msg}")


def warn(warnings: list[str], msg: str) -> None:
    warnings.append(f"WARN: {msg}")


def check_e21(errors: list[str], warnings: list[str]) -> None:
    result_path = ROOT / "v2/experiments/e21_counterfactual_injection/results.json"
    log_path = ROOT / "v2/experiments/e21_counterfactual_injection/_run.log"
    data = load_json(result_path)
    log = log_path.read_text(errors="replace")

    m = re.search(r"\[e21\] device=\S+ model=(\S+) layer=(\d+)", log)
    if not m:
        fail(errors, "E21 log does not expose model/layer header")
        return
    log_model, log_layer = m.group(1), int(m.group(2))
    if data.get("model") != log_model:
        fail(errors, f"E21 results model {data.get('model')} != log model {log_model}")
    if data.get("bank_layer") != log_layer:
        fail(errors, f"E21 results bank_layer {data.get('bank_layer')} != log layer {log_layer}")

    expected = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "bank_layer": 9,
        "steps": 200,
        "n_flips": 5,
        "n_facts_eval": 5,
        "n_cross_truth_preserved": 19,
        "n_cross_total": 20,
        "overall_pass": True,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            fail(errors, f"E21 results {key}={data.get(key)!r}, expected {value!r}")
    if any(x.get("train_time_s") is None for x in data.get("per_fact", [])):
        warn(warnings, "E21 results.json was reconstructed from _run.log; train_time_s is unavailable")


def check_e21b(errors: list[str], warnings: list[str]) -> None:
    root = ROOT / "v2/experiments/e21b_crossmodel"
    ambiguous = root / "results.json"
    if ambiguous.exists():
        fail(errors, "Ambiguous e21b_crossmodel/results.json exists; use model-specific filenames")

    expected = {
        "qwen3_1p7b_L18_500.json": ("Qwen/Qwen3-1.7B", 18, 500, 5, 5, 16, 20, True),
        "gemma2_2b_L13_500.json": ("google/gemma-2-2b", 13, 500, 2, 2, 1, 2, True),
        "qwen25_0p5b_L12_500.json": ("Qwen/Qwen2.5-0.5B-Instruct", 12, 500, 1, 1, 0, 0, True),
        "tinyllama_L14_500.json": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 14, 500, 5, 5, 13, 20, True),
    }
    allowed_extra = {
        "qwen3_1p7b_L14_default.json",
    }
    allowed_files = set(expected) | allowed_extra
    for path in sorted(root.glob("*.json")):
        if path.name not in allowed_files:
            fail(errors, f"Unregistered E21b JSON {path.name}; update verdict/README or quarantine it")

    for filename, values in expected.items():
        path = root / filename
        if not path.exists():
            fail(errors, f"Missing trusted E21b result {filename}")
            continue
        data = load_json(path)
        keys = (
            "model", "bank_layer", "steps", "n_flips", "n_facts_eval",
            "n_cross_truth_preserved", "n_cross_total", "overall_pass",
        )
        for key, expected_value in zip(keys, values):
            if data.get(key) != expected_value:
                fail(errors, f"{filename} {key}={data.get(key)!r}, expected {expected_value!r}")

    for filename in ["gemma3_1b_L13_500.json", "deepseek_32b_L30_300.json"]:
        path = root / filename
        if not path.exists():
            continue
        data = load_json(path)
        model = str(data.get("model", ""))
        if "gemma3" in filename and "gemma" not in model.lower():
            fail(errors, f"{filename} exists but model metadata is {model!r}")
        if "deepseek" in filename and "deepseek" not in model.lower():
            fail(errors, f"{filename} exists but model metadata is {model!r}")


def check_json_parse(errors: list[str]) -> None:
    for path in sorted((ROOT / "v2").rglob("*.json")):
        try:
            load_json(path)
        except Exception as exc:
            fail(errors, f"Invalid JSON {path.relative_to(ROOT)}: {exc}")


def collect_e10(warnings: list[str]) -> None:
    root = ROOT / "v2/experiments/e10_topk_retrieval"
    real = sorted(root.glob("e10_topk_cosine_real_K8_seed*.json"))
    random = sorted(root.glob("e10_topk_cosine_random_K8_seed*.json"))
    all_random = sorted(root.glob("e10_all_attend_random_renorm15_seed*.json"))
    if len(real) >= 3 and len(random) >= 3 and len(all_random) >= 3:
        def mean_delta(paths: list[Path]) -> float:
            vals = []
            for p in paths:
                data = load_json(p)
                vals.append(float(data.get("delta_signed", data.get("delta_real"))))
            return sum(vals) / len(vals)

        warnings.append(
            "E10 ablation summary: "
            f"real topK8 mean Δ={mean_delta(real):+.3f}, "
            f"random topK8 mean Δ={mean_delta(random):+.3f}, "
            f"all-random mean Δ={mean_delta(all_random):+.3f}. "
            "Comparable random-bank lift argues against content retrieval."
        )
    else:
        warn(warnings, "E10 does not have >=3 seeds for real/random/all-random comparison")


def check_phase_d(errors: list[str], warnings: list[str]) -> None:
    root = ROOT / "v2/experiments/e_phase_d_lora"
    expected_methods = {
        "plain_adapter": {
            "params": 327680,
            "max_mean_delta": -7.0,
            "expected_files": [
                "phase_d_lora_plain_adapter_seed0_r64.json",
                "phase_d_lora_plain_adapter_seed1_r64.json",
                "phase_d_lora_plain_adapter_seed2_r64.json",
            ],
        },
        "lora_q": {
            "params": 425984,
            "max_mean_delta": -0.1,
            "expected_files": [
                "phase_d_lora_lora_q_seed0_r64.json",
                "phase_d_lora_lora_q_seed1_r64.json",
                "phase_d_lora_lora_q_seed2_r64.json",
            ],
        },
        "lora_qk": {
            "params": 327680,
            "max_mean_delta": -0.1,
            "expected_files": [
                "phase_d_lora_lora_qk_seed0_r64.json",
                "phase_d_lora_lora_qk_seed1_r64.json",
                "phase_d_lora_lora_qk_seed2_r64.json",
            ],
        },
    }
    expected_files = {
        filename
        for spec in expected_methods.values()
        for filename in spec["expected_files"]
    }
    for path in sorted(root.glob("*.json")):
        if path.name not in expected_files:
            fail(errors, f"Unregistered Phase-D JSON {path.name}; update verdict/README or quarantine it")

    summaries = []
    for method, spec in expected_methods.items():
        rows = []
        for filename in spec["expected_files"]:
            path = root / filename
            if not path.exists():
                fail(errors, f"Missing Phase-D result {filename}")
                continue
            data = load_json(path)
            rows.append(data)
            if data.get("experiment") != "phase_d_lora":
                fail(errors, f"{filename} experiment={data.get('experiment')!r}")
            if data.get("method") != method:
                fail(errors, f"{filename} method={data.get('method')!r}, expected {method!r}")
            if data.get("model") != "Qwen/Qwen3-4B-Instruct-2507":
                fail(errors, f"{filename} model={data.get('model')!r}")
            if data.get("bank_layer") != 9:
                fail(errors, f"{filename} bank_layer={data.get('bank_layer')!r}, expected 9")
            if data.get("steps") != 200:
                fail(errors, f"{filename} steps={data.get('steps')!r}, expected 200")
            if data.get("n_params_trainable") != spec["params"]:
                fail(errors, f"{filename} n_params_trainable={data.get('n_params_trainable')!r}, expected {spec['params']}")
            if abs(float(data.get("test_nll_uninstalled")) - float(data.get("test_nll_before"))) > 1e-6:
                fail(errors, f"{filename} uninstalled NLL does not return to base")

        if rows:
            mean_delta = sum(float(row["delta_nll"]) for row in rows) / len(rows)
            summaries.append(f"{method} mean Δ={mean_delta:+.3f}")
            if mean_delta > spec["max_mean_delta"]:
                fail(errors, f"Phase-D {method} mean Δ={mean_delta:+.3f}, expected <= {spec['max_mean_delta']:+.3f}")
    if summaries:
        warn(warnings, "Phase-D PEFT ablation summary: " + ", ".join(summaries))


def check_e15(errors: list[str], warnings: list[str]) -> None:
    root = ROOT / "v2/experiments/e15_ponder"
    improvements = []
    for seed in [0, 1, 2]:
        path = root / f"e15_summary_seed{seed}.json"
        if not path.exists():
            fail(errors, f"Missing E15 summary for seed {seed}")
            continue
        data = load_json(path)
        verdict = data.get("verdict", {})
        if verdict.get("passes") is not False:
            fail(errors, f"E15 seed {seed} expected passes=False, got {verdict.get('passes')!r}")
        improvement = float(verdict.get("improvement_over_k2", 999.0))
        improvements.append(improvement)
        if abs(improvement) > 1e-6:
            fail(errors, f"E15 seed {seed} expected zero K>2 improvement, got {improvement:+.6f}")
        cells = data.get("cells", {})
        k2 = cells.get("K2_modecumulative", {})
        for key in ["K3_modecumulative", "K4_modecumulative"]:
            cell = cells.get(key, {})
            if cell.get("delta") != k2.get("delta"):
                fail(errors, f"E15 seed {seed} {key} delta {cell.get('delta')} != K2 {k2.get('delta')}")
    if improvements:
        warn(warnings, "E15 ponder ablation: K>2 improvement over K2 is zero for seeds 0/1/2")


def check_e11_n7(errors: list[str], warnings: list[str]) -> None:
    path = ROOT / "v2/experiments/e11_noise_robustness/e11_n7_real_bank_K0_pure_proj_seed0.json"
    if not path.exists():
        warn(warnings, "E11 n7 K=0 pure-projector control has not been run yet")
        return
    data = load_json(path)
    if data.get("n_preload") != 0:
        fail(errors, "E11 n7 expected n_preload=0")
    if data.get("n_train_params") != 0:
        fail(errors, "E11 n7 should report n_train_params=0 active trainable path")
    verdict = data.get("verdict", {})
    if verdict.get("pass") is not True:
        fail(errors, f"E11 n7 expected pass=True, got {verdict}")
    delta = float(verdict.get("delta_signed", 999.0))
    if abs(delta) > 1e-4:
        fail(errors, f"E11 n7 expected near-zero Δ_signed, got {delta:+.6f}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    check_json_parse(errors)
    check_e21(errors, warnings)
    check_e21b(errors, warnings)
    collect_e10(warnings)
    check_phase_d(errors, warnings)
    check_e15(errors, warnings)
    check_e11_n7(errors, warnings)

    for item in warnings:
        print(item)
    for item in errors:
        print(item)
    if errors:
        print(f"\nprepublish audit: FAIL ({len(errors)} errors, {len(warnings)} warnings)")
        return 1
    print(f"\npublish audit: PASS ({len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
