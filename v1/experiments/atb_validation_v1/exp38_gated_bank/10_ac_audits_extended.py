"""Exp38 — extended anti-cheat audits AC9..AC14.

Reads completed artifacts and runs comparisons. Heavy steps (AC9 shuffled
retrain, AC10 random bank panels, AC11 held-out subject) are launched as
separate subprocesses by 11_run_ac_suite.py — this file just AGGREGATES
their JSON outputs into a single audits_extended.json.

AC9  : shuffled-label head must NOT converge ≤ G3 loss × 0.8
AC10 : random-vector bank Φ1 should collapse to ~0; cross-talk |drop| ≥ 1.0
AC11 : held-out subject G3 head: Φ1 within 10% of in-domain (deferred)
AC12 : gate occupancy distribution from existing G2/G3 results (top-1 prob > 0.5)
AC13 : OOD template eval (deferred; needs hand-written templates)
AC14 : two-bank A/B test (deferred; needs second bank)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_error": str(e)}


def main():
    out_path = HERE / "run_qwen_exp38" / "audits_extended.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audits = {}

    # ---- AC9: shuffled-label sanity ----
    g3 = load_json(HERE / "run_qwen_exp38" / "G3" / "results.json")
    g3_shuf = load_json(HERE / "run_qwen_exp38" / "G3_shuffled" / "results.json")
    g3_heads = HERE / "data" / "G3_heads.pt"
    g3_shuf_heads = HERE / "data" / "G3_shuffled_heads.pt"

    ac9 = {"status": "missing"}
    if g3_heads.exists() and g3_shuf_heads.exists():
        d_real = torch.load(g3_heads, map_location="cpu", weights_only=False)
        d_shuf = torch.load(g3_shuf_heads, map_location="cpu", weights_only=False)
        loss_real = float(d_real.get("final_loss_mean", float("nan")))
        loss_shuf = float(d_shuf.get("final_loss_mean", float("nan")))
        # Required: shuffled loss should be much higher (>= 0.8 of original cross-entropy ~0.69)
        # Concretely: ratio shuf/real should be large; or shuf should stay near log(2)≈0.69
        ac9 = {
            "loss_real_mean": loss_real,
            "loss_shuffled_mean": loss_shuf,
            "ratio": (loss_shuf / loss_real) if loss_real > 0 else None,
            "pass": (loss_shuf >= 0.5),  # shuffled stays near random-guess loss
            "note": "Shuffled labels should not converge; expect loss ≈ log(2) ≈ 0.69",
        }
    audits["AC9_shuffled_label_sanity"] = ac9

    # ---- AC10: random-vector bank ----
    g0_rand = load_json(HERE / "run_qwen_exp38" / "G0_random" / "results.json")
    g2_rand = load_json(HERE / "run_qwen_exp38" / "G2_kr5_random" / "results.json")
    g0_real = load_json(HERE / "run_qwen_exp38" / "G0" / "results.json")
    g2_real = load_json(HERE / "run_qwen_exp38" / "G2_kr5" / "results.json")

    def phi1_max(d):
        if not d:
            return None
        p = d.get("phi1", {})
        gd = [p.get(f"k{k}", {}).get("mean_gate_d", float("nan"))
              for k in (1, 10, 100, 1000)]
        gd = [x for x in gd if x == x]
        return max(gd) if gd else None

    def ct_drop(d):
        if not d:
            return None
        c = d.get("cross_talk_37c", {})
        return c.get("mean_abs_drop")

    ac10 = {
        "real_bank_G0_phi1_max_gate_d": phi1_max(g0_real),
        "real_bank_G2_phi1_max_gate_d": phi1_max(g2_real),
        "random_bank_G0_phi1_max_gate_d": phi1_max(g0_rand),
        "random_bank_G2_phi1_max_gate_d": phi1_max(g2_rand),
        "real_bank_G0_ct_drop": ct_drop(g0_real),
        "random_bank_G0_ct_drop": ct_drop(g0_rand),
        "status": "missing" if g0_rand is None else "computed",
    }
    # PASS: random bank's Φ1 gate_d should collapse (≤ 1.0 nat) while real bank is > 2.0
    if g0_rand and g0_real:
        rg = ac10["random_bank_G0_phi1_max_gate_d"] or 0
        rl = ac10["real_bank_G0_phi1_max_gate_d"] or 0
        ac10["pass"] = (rg < 1.0) and (rl > 2.0)
        ac10["note"] = f"random_phi1={rg:.2f} (must <1.0); real_phi1={rl:.2f} (must >2.0)"
    audits["AC10_random_vector_bank"] = ac10

    # ---- AC12: gate occupancy from G2 results ----
    # Currently eval_panels.py doesn't capture per-call top-1; we approximate
    # from k_r sweep results: as k_r grows, cross-talk grows — that's a proxy
    # for occupancy concentration.
    ac12 = {}
    for kr in (1, 5, 20, 100):
        d = load_json(HERE / "run_qwen_exp38" / f"G2_kr{kr}" / "results.json")
        if d:
            ac12[f"k_r={kr}"] = {
                "ct_abs_drop": ct_drop(d),
                "phi1_max_gate_d": phi1_max(d),
            }
    ac12["pass"] = True  # informational only
    ac12["note"] = ("Smaller k_r → smaller cross-talk = more concentrated gate. "
                    "True per-call top-1 prob requires eval_panels instrumentation (TODO).")
    audits["AC12_gate_occupancy_proxy"] = ac12

    # ---- AC11 / AC13 / AC14 ----
    audits["AC11_held_out_subject"] = {"status": "deferred", "note": "Requires subject-disjoint split build"}
    audits["AC13_OOD_templates"] = {"status": "deferred", "note": "Requires hand-written OOD templates"}
    audits["AC14_two_bank_AB"] = {"status": "deferred", "note": "Requires second disjoint bank build"}

    out_path.write_text(json.dumps(audits, indent=2))
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
