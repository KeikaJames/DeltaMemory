#!/usr/bin/env python3
"""A.8 aggregate: H_A8.* verdicts on the (model × task) cell grid.

Computes per-model:
  - mu_arch_by_task spread (max - min) -> H_A8.1
  - delta_default[task] = |mu_arch_default - mu_arch_task| -> H_A8.2
  - kendall_tau(sigma_base_default, sigma_base_task) -> H_A8.3
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) == (db > 0):
                concord += 1
            else:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.cells.read_text().splitlines() if l.strip()]
    by_model = defaultdict(dict)
    for r in rows:
        if r.get("status") != "ok":
            continue
        by_model[r["model"]][r["task"]] = r

    models = sorted(by_model.keys())
    per_model: dict[str, dict] = {}
    redline_violations = []

    for m in models:
        cells = by_model[m]
        if "default" not in cells:
            continue
        default = cells["default"]
        if default["state_sha_pre"] != default["state_sha_post"]:
            redline_violations.append(f"{m}/default: state_sha mismatch")

        mu_archs = {t: cells[t]["mu_arch"] for t in cells}
        spread = max(mu_archs.values()) - min(mu_archs.values())
        deltas = {t: abs(mu_archs[t] - mu_archs["default"])
                  for t in mu_archs if t != "default"}
        sigma_def = default["sigma_base"]
        taus = {t: kendall_tau(sigma_def, cells[t]["sigma_base"])
                for t in cells if t != "default"}
        per_model[m] = {
            "model_name": default["model_name"],
            "num_layers": default["num_layers"],
            "mu_arch_by_task": mu_archs,
            "spread_max_minus_min": spread,
            "delta_default_by_task": deltas,
            "kendall_tau_sigma_vs_default": taus,
            "h_a8_1_intra_model_invariance": spread <= 2,
            "h_a8_2_default_representative": all(d <= 2 for d in deltas.values()),
            "h_a8_3_sigma_ranking_stable": all(t >= 0.6 for t in taus.values()),
        }

    n_models = len(per_model)
    n_h1 = sum(1 for v in per_model.values() if v["h_a8_1_intra_model_invariance"])
    n_h2_pairs = 0
    n_h2_pass = 0
    n_h3_pairs = 0
    n_h3_pass = 0
    for v in per_model.values():
        for d in v["delta_default_by_task"].values():
            n_h2_pairs += 1
            if d <= 2:
                n_h2_pass += 1
        for t in v["kendall_tau_sigma_vs_default"].values():
            n_h3_pairs += 1
            if t >= 0.6:
                n_h3_pass += 1

    out = {
        "n_models": n_models,
        "n_cells_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "redline_violations": redline_violations,
        "H_A8_0_redline": "passed" if not redline_violations else "FAILED",
        "H_A8_1_intra_model": {
            "verdict": "supported" if n_h1 >= 4 else "not_supported",
            "n_pass": n_h1, "n_total": n_models,
            "criterion": "≥4/5 models with mu_arch spread ≤ 2 (5-model PREREG; running 4-model Tier A)",
        },
        "H_A8_2_default_representative": {
            "verdict": "supported" if n_h2_pass / max(n_h2_pairs, 1) >= 0.90 else "not_supported",
            "n_pass": n_h2_pass, "n_total": n_h2_pairs,
            "criterion": "≥90% of (model, task) pairs with |Δ default| ≤ 2",
        },
        "H_A8_3_kendall_tau": {
            "verdict": "supported" if n_h3_pass / max(n_h3_pairs, 1) >= 0.80 else "not_supported",
            "n_pass": n_h3_pass, "n_total": n_h3_pairs,
            "criterion": "≥80% of (model, task) pairs with Kendall-τ ≥ 0.6",
        },
        "per_model": per_model,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[A8][aggregate] -> {args.out}")
    print(f"  H_A8.0 redline:     {out['H_A8_0_redline']}")
    print(f"  H_A8.1 intra-model: {out['H_A8_1_intra_model']['verdict']} "
          f"({n_h1}/{n_models})")
    print(f"  H_A8.2 default-rep: {out['H_A8_2_default_representative']['verdict']} "
          f"({n_h2_pass}/{n_h2_pairs})")
    print(f"  H_A8.3 kendall-τ:   {out['H_A8_3_kendall_tau']['verdict']} "
          f"({n_h3_pass}/{n_h3_pairs})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
