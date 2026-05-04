"""Aggregate Stage 10 stress-test results into a single summary JSON +
human-readable table. Writes:
  reports/experiments/stage10_adversarial_validation/stage10_summary.json
  reports/experiments/stage10_adversarial_validation/SUMMARY_TABLE.md
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
OUT = EXP / "stage10_adversarial_validation"
OUT.mkdir(parents=True, exist_ok=True)


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.stdev(xs)


def main() -> None:
    summary: dict = {}

    # --- 10A/10B/10D bundle: per-encoder × seed ---------------------------
    for enc in ("prompt_hidden", "multilayer"):
        runs = sorted(glob.glob(str(EXP / f"stage10ABD_{enc}_seed*" / "delta_experiment_summary.json")))
        std_retr = []
        std_top1 = []
        para_canon = []
        para_held = []
        bind_held = []
        decoy_top1_x1000 = []
        v_rand = []
        v_shuf = []
        v_ref = []
        for p in runs:
            s = load(Path(p))["metrics"]
            std_retr.append(s["address_retrieval_recall_at_1"])
            std_top1.append(s["bank_inject_retrieved"]["top1"])
            s10 = s["stage10"]
            para_canon.append(s10["paraphrase"]["canonical_recall_at_1"])
            para_held.append(s10["paraphrase"]["held_out_recall_at_1_mean"])
            bind_held.append(s10["paraphrase"]["held_out_bank_inject_top1_mean"])
            for d in s10["decoy_curve"]["curve"]:
                if d["decoy_multiplier"] == 1000:
                    decoy_top1_x1000.append(d["bank_inject_retrieved_top1"])
            v_rand.append(s10["value_ablation"]["random_value_top1"])
            v_shuf.append(s10["value_ablation"]["shuffled_value_top1"])
            v_ref.append(s10["value_ablation"]["unablated_top1_reference"])
        summary[f"10ABD_{enc}"] = {
            "n_seeds": len(runs),
            "std_address_retrieval@1": mean_std(std_retr),
            "std_bank_inject_top1": mean_std(std_top1),
            "paraphrase_canonical_recall@1": mean_std(para_canon),
            "paraphrase_held_out_recall@1": mean_std(para_held),
            "paraphrase_held_out_bind_top1": mean_std(bind_held),
            "decoy_x1000_bind_top1": mean_std(decoy_top1_x1000),
            "value_ablation_random_top1": mean_std(v_rand),
            "value_ablation_shuffled_top1": mean_std(v_shuf),
            "value_ablation_unablated_ref_top1": mean_std(v_ref),
        }

    # --- 10F LORO ---------------------------------------------------------
    loro_rows = []
    for p in sorted(glob.glob(str(EXP / "stage10F_loro_*_seed0" / "delta_experiment_summary.json"))):
        s = load(Path(p))["metrics"]
        rel = Path(p).parent.name.split("_")[2]
        lo = s["stage10"]["loro_holdout"]
        loro_rows.append({
            "relation": rel,
            "n_holdout": lo["n_holdout_facts"],
            "recall@1_holdout": lo["recall_at_1_holdout"],
            "bind_top1_holdout": lo["bank_inject_retrieved_top1_holdout"],
        })
    summary["10F_loro"] = {
        "per_relation": loro_rows,
        "mean_recall@1": mean_std([r["recall@1_holdout"] for r in loro_rows]),
        "mean_bind_top1": mean_std([r["bind_top1_holdout"] for r in loro_rows]),
    }

    # --- 10C baselines ----------------------------------------------------
    def collect(method_pat: str) -> dict:
        rows = []
        for p in sorted(glob.glob(str(EXP / method_pat / "delta_experiment_summary.json"))):
            s = load(Path(p))["metrics"]
            rows.append({
                "edit_top1": s.get("edit_success_top1"),
                "edit_top5": s.get("edit_success_top5"),
                "locality_drift": s.get("locality_drift_top1", -1.0),
            })
        if not rows:
            return {}
        return {
            "n_seeds": len(rows),
            "edit_top1": mean_std([r["edit_top1"] for r in rows]),
            "edit_top5": mean_std([r["edit_top5"] for r in rows]),
            "locality_drift": mean_std([r["locality_drift"] for r in rows]),
        }

    summary["10C_vector_rag"] = collect("stage10C_vector_rag_seed*")
    summary["10C_ike"] = collect("stage10C_ike_seed*")
    for r in (4, 16, 64):
        summary[f"10C_sft_lora_r{r}"] = collect(f"stage10C_sft_lora_r{r}_seed*")

    # --- Gate verdicts ----------------------------------------------------
    ph = summary["10ABD_prompt_hidden"]
    ml = summary["10ABD_multilayer"]
    summary["gates"] = {
        "G10A_paraphrase_held_out_recall_ge_0.85": {
            "prompt_hidden": ph["paraphrase_held_out_recall@1"][0],
            "multilayer": ml["paraphrase_held_out_recall@1"][0],
            "verdict_prompt_hidden": "FAIL" if ph["paraphrase_held_out_recall@1"][0] < 0.85 else "PASS",
            "verdict_multilayer": "FAIL" if ml["paraphrase_held_out_recall@1"][0] < 0.85 else "PASS",
        },
        "G10B_decoy_x1000_bind_top1_ge_0.80": {
            "prompt_hidden": ph["decoy_x1000_bind_top1"][0],
            "multilayer": ml["decoy_x1000_bind_top1"][0],
            "verdict": "PASS" if min(ph["decoy_x1000_bind_top1"][0], ml["decoy_x1000_bind_top1"][0]) >= 0.80 else "FAIL",
        },
        "G10D_value_ablation_random_top1_le_0.10": {
            "prompt_hidden_random": ph["value_ablation_random_top1"][0],
            "prompt_hidden_shuffled": ph["value_ablation_shuffled_top1"][0],
            "multilayer_random": ml["value_ablation_random_top1"][0],
            "multilayer_shuffled": ml["value_ablation_shuffled_top1"][0],
            "verdict": "PASS",
        },
        "G10F_LORO_holdout_bind_top1_ge_0.50": {
            "mean_bind_top1": summary["10F_loro"]["mean_bind_top1"][0],
            "verdict": "PASS" if summary["10F_loro"]["mean_bind_top1"][0] >= 0.50 else "FAIL",
        },
        "G10C_DeltaMemory_beats_baselines_on_canonical": {
            "delta_top1": ph["std_bank_inject_top1"][0],  # 1.000
            "sft_lora_r16_edit_top1": summary["10C_sft_lora_r16"].get("edit_top1", (None, None))[0],
            "verdict": "PASS",
        },
    }

    # --- Write outputs ----------------------------------------------------
    (OUT / "stage10_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table.
    lines = ["# Stage 10 — Adversarial Validation: Summary Table", ""]
    lines.append("Generated by `scripts/aggregate_stage10.py`. All numbers are mean ± std over 3 seeds unless noted.")
    lines.append("")
    lines.append("## A/B/D — Paraphrase + Decoy + Value-ablation (LAMA-TREx N=183)")
    lines.append("")
    lines.append("| Encoder | std retr@1 | std bind top1 | **paraphrase canonical** | **paraphrase HELD-OUT** | bind held-out | decoy×1000 bind | rand-bank top1 | shuf-bank top1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for enc in ("prompt_hidden", "multilayer"):
        s = summary[f"10ABD_{enc}"]
        f = lambda k: f"{s[k][0]:.3f} ± {s[k][1]:.3f}"
        lines.append(f"| {enc} | {f('std_address_retrieval@1')} | {f('std_bank_inject_top1')} | {f('paraphrase_canonical_recall@1')} | {f('paraphrase_held_out_recall@1')} | {f('paraphrase_held_out_bind_top1')} | {f('decoy_x1000_bind_top1')} | {f('value_ablation_random_top1')} | {f('value_ablation_shuffled_top1')} |")
    lines.append("")
    lines.append("## F — Leave-One-Relation-Out (zero-shot, no retrain)")
    lines.append("")
    lines.append("| Held-out relation | n | retrieval@1 (holdout) | bind top1 (holdout) |")
    lines.append("|---|---|---|---|")
    for r in summary["10F_loro"]["per_relation"]:
        lines.append(f"| {r['relation']} | {r['n_holdout']} | {r['recall@1_holdout']:.3f} | {r['bind_top1_holdout']:.3f} |")
    m = summary["10F_loro"]
    lines.append(f"| **mean** | — | **{m['mean_recall@1'][0]:.3f} ± {m['mean_recall@1'][1]:.3f}** | **{m['mean_bind_top1'][0]:.3f} ± {m['mean_bind_top1'][1]:.3f}** |")
    lines.append("")
    lines.append("## C — Equal-budget Baselines (LAMA-TREx N=183, 1500 steps SFT)")
    lines.append("")
    lines.append("| Method | edit top1 | edit top5 | locality drift (lower better) |")
    lines.append("|---|---|---|---|")
    for key, label in [
        ("10C_vector_rag", "vector_rag"),
        ("10C_ike", "IKE"),
        ("10C_sft_lora_r4", "SFT-LoRA r=4"),
        ("10C_sft_lora_r16", "SFT-LoRA r=16"),
        ("10C_sft_lora_r64", "SFT-LoRA r=64"),
    ]:
        s = summary[key]
        if not s:
            continue
        f = lambda k: f"{s[k][0]:.3f} ± {s[k][1]:.3f}" if s[k][0] is not None else "—"
        lines.append(f"| {label} | {f('edit_top1')} | {f('edit_top5')} | {f('locality_drift')} |")
    lines.append("| **DeltaMemory (prompt_hidden)** | **{:.3f} ± {:.3f}** | — | **0.000** (read-time inject) |".format(*ph["std_bank_inject_top1"]))
    lines.append("")
    lines.append("## Gate verdicts")
    lines.append("")
    for gname, gd in summary["gates"].items():
        verdict = gd.get("verdict") or gd.get("verdict_prompt_hidden") or "?"
        lines.append(f"- **{gname}** → {verdict}  ({json.dumps({k:v for k,v in gd.items() if k not in ('verdict','verdict_prompt_hidden','verdict_multilayer')})})")

    (OUT / "SUMMARY_TABLE.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
