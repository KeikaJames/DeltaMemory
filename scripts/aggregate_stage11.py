#!/usr/bin/env python3
"""Stage 11 aggregator + paired-bootstrap CIs.

Reads all Stage 11 run summaries from reports/experiments/stage11*/, plus
the conv summaries, and produces:
  - reports/experiments/stage11_grand_evaluation/stage11_summary.json
  - reports/experiments/stage11_grand_evaluation/SUMMARY_TABLE.md

Every headline metric gets a 10k paired bootstrap 95% CI. A gate is reported
PASS only if the CI lower bound exceeds the threshold (NOT just the mean).
"""
from __future__ import annotations
import json
import hashlib
import math
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP_DIR = REPO / "reports/experiments"
OUT_DIR = EXP_DIR / "stage11_grand_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bootstrap_ci(values, n_resamples=10000, q_low=0.025, q_high=0.975, seed=0):
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(q_low * n_resamples)]
    hi = means[int(q_high * n_resamples)]
    return (sum(values) / n, lo, hi)


def _load(pattern: str) -> list[dict]:
    out = []
    for d in sorted(EXP_DIR.glob(pattern)):
        for fn in ("delta_experiment_summary.json", "stage11_conv_summary.json"):
            p = d / fn
            if p.exists():
                try:
                    out.append({"name": d.name, "path": str(p), **json.loads(p.read_text())})
                except Exception as e:
                    print(f"[warn] {p}: {e}")
                break
    return out


def _stable_hash(summary: dict) -> str:
    """SHA-256 of stable subset (gate metrics, retrieval@1, payload norms)."""
    keys_to_hash = {}
    m = summary.get("metrics", {})
    for k in ("address_retrieval_recall_at_1", "swap_paired"):
        if k in m:
            keys_to_hash[k] = m[k]
    if "bank_inject_retrieved" in m:
        keys_to_hash["bank_inject_retrieved_top1"] = m["bank_inject_retrieved"]["top1"]
    if "no_memory" in m:
        keys_to_hash["no_memory_top1"] = m["no_memory"]["top1"]
    blob = json.dumps(keys_to_hash, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def main():
    s11A = _load("stage11A_*")
    s11B = _load("stage11B_*")
    s11D = _load("stage11D_*")
    s11E = _load("stage11E_*")
    print(f"loaded: 11A={len(s11A)} 11B={len(s11B)} 11D={len(s11D)} 11E={len(s11E)}")

    # ---- 11A: paraphrase robustness on held-out templates ----
    a_by_enc: dict[str, list[float]] = {}
    a_decoy_by_enc: dict[str, list[float]] = {}
    a_value_random: list[float] = []
    a_value_shuffled: list[float] = []
    for r in s11A:
        enc = r["args"]["encoder"]
        ho = r["metrics"].get("stage10", {}).get("paraphrase", {}).get("held_out_recall_at_1_mean")
        if ho is not None:
            a_by_enc.setdefault(enc, []).append(ho)
        decoy_curve = r["metrics"].get("stage10", {}).get("decoy_curve", {}).get("curve", [])
        for entry in decoy_curve:
            if entry.get("decoy_multiplier") == 1000:
                a_decoy_by_enc.setdefault(enc, []).append(entry["bank_inject_retrieved_top1"])
        va = r["metrics"].get("stage10", {}).get("value_ablation", {})
        if "random_value_top1" in va:
            a_value_random.append(va["random_value_top1"])
        if "shuffled_value_top1" in va:
            a_value_shuffled.append(va["shuffled_value_top1"])

    # ---- 11B: train-time LORO bind on held-out relation ----
    b_by_rel: dict[str, list[float]] = {}
    for r in s11B:
        rel = r["args"].get("stage11_loro_exclude_relation")
        bind = r["metrics"].get("stage10", {}).get("loro_holdout", {}).get("bank_inject_retrieved_top1_holdout")
        if rel and bind is not None:
            b_by_rel.setdefault(rel, []).append(bind)

    # ---- 11D: conversational ----
    d1_k = {1: [], 3: [], 5: [], 10: []}
    d1_leak = {1: [], 3: [], 5: [], 10: []}
    d2_dm, d2_rag, d2_adv = [], [], []
    d3_overwrite, d3_benign, d3_recall = [], [], []
    for r in s11D:
        d1 = r.get("d1_multi_turn_convqa", {})
        for k in (1, 3, 5, 10):
            v = d1.get(f"k_{k}", {})
            if "recall_at_1" in v:
                d1_k[k].append(v["recall_at_1"])
                d1_leak[k].append(v["no_leakage_recall"])
        d2 = r.get("d2_chat_write_api", {})
        if "dm_chat_api_top1" in d2:
            d2_dm.append(d2["dm_chat_api_top1"])
            d2_rag.append(d2["rag_baseline_top1"])
            d2_adv.append(d2["advantage_dm_minus_rag"])
        d3 = r.get("d3_poisoning", {})
        if "protected_overwrite_rate" in d3:
            d3_overwrite.append(d3["protected_overwrite_rate"])
            d3_benign.append(d3["benign_accept_rate"])
            d3_recall.append(d3["original_answer_recall_after_attack"])

    # ---- 11E: bit-exact reproduction ----
    e_hashes = []
    for r in s11E:
        e_hashes.append({"run": r["name"], "hash": _stable_hash(r)})
    e_match = len(e_hashes) >= 2 and len({h["hash"] for h in e_hashes}) == 1

    def _entry(values, gate=None, gate_op=">="):
        mean, lo, hi = _bootstrap_ci(values)
        ent = {"mean": mean, "ci95_low": lo, "ci95_high": hi, "n": len(values), "values": values}
        if gate is not None:
            if gate_op == ">=":
                ent["gate"] = gate
                ent["pass"] = lo >= gate  # CI lower bound (strict)
            elif gate_op == "<=":
                ent["gate"] = gate
                ent["pass"] = hi <= gate
        return ent

    summary = {
        "stage": "11_grand_evaluation",
        "n_runs": {"11A": len(s11A), "11B": len(s11B), "11D": len(s11D), "11E": len(s11E)},
        "11A_paraphrase_holdout_recall_at_1": {
            enc: _entry(vals, gate=0.85, gate_op=">=") for enc, vals in a_by_enc.items()
        },
        "11A_decoy_x1000_top1": {
            enc: _entry(vals, gate=0.80, gate_op=">=") for enc, vals in a_decoy_by_enc.items()
        },
        "11A_value_ablation_random_top1": _entry(a_value_random, gate=0.10, gate_op="<="),
        "11A_value_ablation_shuffled_top1": _entry(a_value_shuffled, gate=0.10, gate_op="<="),
        "11B_loro_bind_top1_by_relation": {
            rel: _entry(vals, gate=0.50, gate_op=">=") for rel, vals in b_by_rel.items()
        },
        "11B_loro_bind_top1_overall": _entry(
            [v for vs in b_by_rel.values() for v in vs], gate=0.50, gate_op=">="
        ),
        "11D_convqa": {
            f"k_{k}_recall": _entry(d1_k[k], gate=(0.85 if k == 10 else None), gate_op=">=")
            for k in (1, 3, 5, 10)
        },
        "11D_convqa_no_leakage": {
            f"k_{k}_no_leakage": _entry(d1_leak[k]) for k in (1, 3, 5, 10)
        },
        "11D_chat_api_dm_top1": _entry(d2_dm),
        "11D_chat_api_rag_top1": _entry(d2_rag),
        "11D_chat_api_advantage": _entry(d2_adv, gate=0.0, gate_op=">="),
        "11D_poisoning_overwrite_rate": _entry(d3_overwrite, gate=0.05, gate_op="<="),
        "11D_poisoning_benign_accept": _entry(d3_benign, gate=0.90, gate_op=">="),
        "11D_poisoning_original_recall": _entry(d3_recall, gate=0.95, gate_op=">="),
        "11E_bit_exact_hashes": e_hashes,
        "11E_bit_exact_match": e_match,
    }

    out_json = OUT_DIR / "stage11_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    # --- Markdown table ---
    md = ["# Stage 11 Summary (paired bootstrap 95% CI)\n"]
    md.append("## 11A — paraphrase-augmented InfoNCE encoder\n")
    md.append("| encoder | held-out paraphrase recall@1 | 95% CI | n | gate G11A ≥ 0.85 |")
    md.append("|---|---:|---|---:|---|")
    for enc, e in summary["11A_paraphrase_holdout_recall_at_1"].items():
        md.append(f"| `{enc}` | {e['mean']:.3f} | [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}] | {e['n']} | {'✅ PASS' if e['pass'] else '❌ FAIL'} |")
    md.append("\n## 11A — decoy ×1000 (regression check on G10B)\n")
    md.append("| encoder | top1 | 95% CI | n | gate ≥ 0.80 |")
    md.append("|---|---:|---|---:|---|")
    for enc, e in summary["11A_decoy_x1000_top1"].items():
        md.append(f"| `{enc}` | {e['mean']:.3f} | [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}] | {e['n']} | {'✅' if e['pass'] else '❌'} |")
    md.append("\n## 11A — value ablation (regression check on G10D)\n")
    md.append("| ablation | top1 | 95% CI | n | gate ≤ 0.10 |")
    md.append("|---|---:|---|---:|---|")
    for k in ("11A_value_ablation_random_top1", "11A_value_ablation_shuffled_top1"):
        e = summary[k]
        md.append(f"| {k.split('_')[-2]} | {e['mean']:.3f} | [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}] | {e['n']} | {'✅' if e.get('pass') else '❌'} |")
    md.append("\n## 11B — train-time LORO + adversary\n")
    md.append("| relation | bind top1 | 95% CI | n | gate ≥ 0.50 |")
    md.append("|---|---:|---|---:|---|")
    for rel, e in summary["11B_loro_bind_top1_by_relation"].items():
        md.append(f"| {rel} | {e['mean']:.3f} | [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}] | {e['n']} | {'✅' if e['pass'] else '❌'} |")
    eo = summary["11B_loro_bind_top1_overall"]
    md.append(f"| **overall** | **{eo['mean']:.3f}** | [{eo['ci95_low']:.3f}, {eo['ci95_high']:.3f}] | {eo['n']} | {'✅ PASS' if eo['pass'] else '❌ FAIL'} |")
    md.append("\n## 11D — Conversational benchmarks\n")
    md.append("### D1 multi-turn ConvQA (recall vs filler turns)\n")
    md.append("| k filler turns | recall@1 | 95% CI | no-leakage |")
    md.append("|---:|---:|---|---:|")
    for k in (1, 3, 5, 10):
        e = summary["11D_convqa"][f"k_{k}_recall"]
        leak = summary["11D_convqa_no_leakage"][f"k_{k}_no_leakage"]
        gate = " ✅" if e.get("pass") else (" ❌" if k == 10 else "")
        md.append(f"| {k} | {e['mean']:.3f}{gate} | [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}] | {leak['mean']:.3f} |")
    md.append("\n### D2 chat-as-write-API vs RAG\n")
    md.append("| method | top1 | 95% CI |")
    md.append("|---|---:|---|")
    for k, label in [("11D_chat_api_dm_top1", "DM"), ("11D_chat_api_rag_top1", "RAG"), ("11D_chat_api_advantage", "DM − RAG")]:
        e = summary[k]
        md.append(f"| {label} | {e['mean']:.3f} | [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}] |")
    md.append("\n### D3 prompt-injection / poisoning\n")
    md.append("| metric | value | 95% CI | gate |")
    md.append("|---|---:|---|---|")
    for k, gate in [("11D_poisoning_overwrite_rate", "≤ 0.05"),
                    ("11D_poisoning_benign_accept", "≥ 0.90"),
                    ("11D_poisoning_original_recall", "≥ 0.95")]:
        e = summary[k]
        md.append(f"| {k.replace('11D_poisoning_', '')} | {e['mean']:.3f} | [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}] | {gate} {'✅' if e.get('pass') else '❌'} |")
    md.append("\n## 11E — bit-exact reproduction\n")
    for h in e_hashes:
        md.append(f"- `{h['run']}`: `{h['hash'][:16]}…`")
    md.append(f"\n**bit-exact match across runs: {'✅ YES' if e_match else '❌ NO'}**")
    (OUT_DIR / "SUMMARY_TABLE.md").write_text("\n".join(md))
    print(f"wrote {OUT_DIR / 'SUMMARY_TABLE.md'}")


if __name__ == "__main__":
    main()
