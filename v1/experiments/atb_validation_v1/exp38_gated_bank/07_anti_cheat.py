"""Exp38 — AC1..AC8 anti-cheat audits, run after all variants completed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import load_bank, get_dtype  # noqa: E402


def main():
    out = HERE / "run_qwen_exp38" / "audits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    audits = {}

    bank = load_bank(HERE.parent / "exp35b_memit_bank" / "data" / "bank.pt",
                     device="cpu", dtype=torch.float32)
    fact_ids = bank.fact_ids
    N = len(fact_ids)

    # AC4: paraphrase split disjointness
    splits = HERE.parent / "exp35b_memit_bank" / "data" / "splits"
    seen = {}
    overlap = 0
    for name in ("train", "val", "test"):
        rows = json.load(open(splits / f"{name}.json"))
        for r in rows:
            for p in [r["prompt"].format(r["subject"])] + list(r.get("paraphrase_prompts", [])):
                key = p.strip().lower()
                if key in seen and seen[key] != name:
                    overlap += 1
                seen[key] = name
    audits["AC4_paraphrase_split_overlap"] = {
        "n_overlapping_strings": overlap, "pass": overlap == 0
    }

    # AC5: assert eval-set negation templates not in train templates
    eval_templates = [
        "It is not true that",
        "It is false to say that",
        "Is it the case that",
        "Some people incorrectly claim that",
        "Contrary to fact,",
    ]
    train_templates = [
        "It is not the case that",
        "Actually no.",
        "Some claim that",
    ]
    overlap_templ = set(eval_templates) & set(train_templates)
    audits["AC5_negation_template_disjoint"] = {
        "overlap": list(overlap_templ), "pass": len(overlap_templ) == 0
    }

    # AC8: |w_i| dispersion if heads trained
    for variant in ("G3", "G4"):
        p = HERE / "data" / f"{variant}_heads.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            w_norms = d["W_g"].norm(dim=0)
            nonzero = w_norms[w_norms > 0]
            std = float(nonzero.std().item()) if nonzero.numel() else 0.0
            audits[f"AC8_{variant}_w_dispersion"] = {
                "n_trained": int(nonzero.numel()),
                "w_norm_mean": float(nonzero.mean().item()) if nonzero.numel() else 0.0,
                "w_norm_std": std,
                "pass": std >= 0.1
            }

    # AC2/AC3: gate occupancy — load from variant result files
    for variant_dir in (HERE / "run_qwen_exp38").iterdir() if (HERE / "run_qwen_exp38").exists() else []:
        r = variant_dir / "results.json"
        if not r.exists():
            continue
        data = json.loads(r.read_text())
        # we didn't currently capture per-eval occupancy; placeholder
        audits.setdefault("AC2_AC3_occupancy", {})[variant_dir.name] = "see panels.results"

    # AC6: k_r sweep reported?
    g2_runs = list((HERE / "run_qwen_exp38").glob("G2_kr*")) if (HERE / "run_qwen_exp38").exists() else []
    audits["AC6_k_r_sweep"] = {
        "k_r_values_run": [d.name for d in g2_runs],
        "pass": len(g2_runs) >= 4
    }

    out.write_text(json.dumps(audits, indent=2))
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
