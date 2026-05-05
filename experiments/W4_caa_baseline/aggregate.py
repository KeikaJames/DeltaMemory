"""W.4 aggregate: paired Wilcoxon, Holm-Bonferroni, bootstrap median CI.

Reads cells.jsonl from the W.4 runner and produces:
  * aggregate.csv         per (model, method, alpha) summary
  * verdicts.json         per (model, alpha, contrast) test result + family verdict
  * summary printed to stdout

Test family (PREREG section 4): 5 models * 7 alpha * 2 contrasts = 70 tests.
Contrasts: (lopi_default - none) and (caa - none), paired by (alpha, seed, prompt_id).
Multiple-comparison correction: Holm-Bonferroni at corrected p < 0.01.
Effect size: median paired diff with 95 percent bootstrap CI (B=1000).

Usage
-----
    python experiments/W4_caa_baseline/aggregate.py \\
        --cells experiments/W4_caa_baseline/cells.jsonl \\
        --out   experiments/W4_caa_baseline/
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REDLINE_TOL = 1e-4
ALPHA_THRESHOLD = 0.01
BOOTSTRAP_B = 1000


def _stable_seed(*parts: object) -> int:
    blob = json.dumps([str(p) for p in parts], separators=(",", ":"), sort_keys=False).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Loaders


def load_cells(path: Path) -> list[dict]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    rows: list[dict] = []
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_data(cells: list[dict]) -> list[dict]:
    """Drop sentinel rows (substitution, unsupported)."""
    out = []
    for c in cells:
        if c.get("model_substituted"):
            continue
        if c.get("method_unsupported"):
            continue
        if c.get("drift") is None:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Aggregate table


def build_aggregate(cells: list[dict]):
    import pandas as pd

    rows: dict[tuple, list[float]] = {}
    redlines: dict[tuple, int] = {}
    for c in cells:
        key = (c["model"], c["method"], float(c["alpha"]))
        rows.setdefault(key, []).append(float(c["drift"]))
        if c.get("redline_violation"):
            redlines[key] = redlines.get(key, 0) + 1
    out_rows = []
    for (model, method, alpha), drifts in sorted(rows.items()):
        out_rows.append({
            "model": model,
            "method": method,
            "alpha": alpha,
            "n": len(drifts),
            "mean_drift": float(np.mean(drifts)),
            "median_drift": float(np.median(drifts)),
            "std_drift": float(np.std(drifts)),
            "redline_count": int(redlines.get((model, method, alpha), 0)),
        })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Statistics


def paired_diffs(cells: list[dict], model: str, alpha: float,
                  method: str, baseline: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """Return aligned arrays (drift_method, drift_baseline) for the
    (model, alpha) cell, paired on (seed, prompt_id)."""
    a: dict[tuple, float] = {}
    b: dict[tuple, float] = {}
    for c in cells:
        if c["model"] != model or float(c["alpha"]) != alpha:
            continue
        key = (int(c["seed"]), c["prompt_id"])
        if c["method"] == method:
            a[key] = float(c["drift"])
        elif c["method"] == baseline:
            b[key] = float(c["drift"])
    keys = sorted(set(a) & set(b))
    return (np.array([a[k] for k in keys], dtype=float),
            np.array([b[k] for k in keys], dtype=float))


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    diffs = x - y
    if len(diffs) == 0 or np.allclose(diffs, 0.0):
        return float("nan")
    try:
        res = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue)
    except ValueError:
        return float("nan")


def bootstrap_median_ci(x: np.ndarray, y: np.ndarray, B: int = BOOTSTRAP_B,
                          rng_seed: int = 0) -> tuple[float, float, float]:
    diffs = x - y
    if len(diffs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    n = len(diffs)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.median(diffs[idx]))
    med = float(np.median(diffs))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return (med, lo, hi)


def holm_bonferroni(pvals: list[float], alpha: float) -> list[bool]:
    """Return list of significance flags after Holm-Bonferroni at 'alpha'.

    NaN p-values are treated as not-significant.
    """
    n = len(pvals)
    order = sorted(range(n), key=lambda i: (np.isnan(pvals[i]), pvals[i]))
    sig = [False] * n
    for rank, i in enumerate(order):
        p = pvals[i]
        if np.isnan(p):
            continue
        threshold = alpha / max(1, n - rank)
        if p < threshold:
            sig[i] = True
        else:
            break
    return sig


# ---------------------------------------------------------------------------
# Verdicts


def compute_verdicts(cells: list[dict]) -> dict:
    models = sorted({c["model"] for c in cells})
    alphas = sorted({float(c["alpha"]) for c in cells})
    contrasts = ["lopi_default", "caa"]

    tests: list[dict] = []
    for model in models:
        for alpha in alphas:
            for method in contrasts:
                x, y = paired_diffs(cells, model, alpha, method, baseline="none")
                p = wilcoxon_p(x, y) if len(x) > 0 else float("nan")
                med, lo, hi = bootstrap_median_ci(x, y, B=BOOTSTRAP_B,
                                                    rng_seed=_stable_seed(model, alpha, method))
                tests.append({
                    "model": model,
                    "alpha": alpha,
                    "contrast": f"{method}_minus_none",
                    "n_pairs": int(len(x)),
                    "p_value": p,
                    "median_diff": med,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                })

    pvals = [t["p_value"] for t in tests]
    sig = holm_bonferroni(pvals, ALPHA_THRESHOLD)
    for t, s in zip(tests, sig):
        t["significant_holm_p<0.01"] = bool(s)

    redline_violations = [
        {"model": c["model"], "method": c["method"], "seed": c["seed"],
          "prompt_id": c["prompt_id"], "drift": c["drift"]}
        for c in cells if c.get("redline_violation")
    ]

    # Family verdict per PREREG section 5
    n_caa_wins = 0
    n_lopi_wins = 0
    caa_per_model = {m: 0 for m in models}
    lopi_per_model = {m: 0 for m in models}
    for t in tests:
        if not t["significant_holm_p<0.01"]:
            continue
        # "Better" = lower drift => median_diff < 0
        if t["median_diff"] is None or np.isnan(t["median_diff"]):
            continue
        if t["median_diff"] >= 0:
            continue
        if t["contrast"] == "caa_minus_none":
            n_caa_wins += 1
            caa_per_model[t["model"]] += 1
        elif t["contrast"] == "lopi_default_minus_none":
            n_lopi_wins += 1
            lopi_per_model[t["model"]] += 1

    caa_models_with_wins = sum(1 for m in models if caa_per_model[m] >= 3)
    lopi_models_with_wins = sum(1 for m in models if lopi_per_model[m] >= 3)

    if caa_models_with_wins >= 3 and caa_models_with_wins > lopi_models_with_wins:
        family_verdict = "CAA_PROMOTED_TO_DEFAULT"
    elif caa_models_with_wins >= 3 and lopi_models_with_wins >= 3:
        family_verdict = "CAA_AND_LOPI_BOTH_VIABLE"
    elif caa_models_with_wins == 0 and lopi_models_with_wins == 0:
        family_verdict = "NEITHER_BEATS_NONE"
    else:
        family_verdict = "MIXED"

    return {
        "alpha_threshold": ALPHA_THRESHOLD,
        "family_size": len(tests),
        "n_significant_caa_wins": n_caa_wins,
        "n_significant_lopi_wins": n_lopi_wins,
        "caa_models_with_3plus_wins": caa_models_with_wins,
        "lopi_models_with_3plus_wins": lopi_models_with_wins,
        "family_verdict": family_verdict,
        "redline_violations": redline_violations,
        "tests": tests,
    }


# ---------------------------------------------------------------------------
# Main


def main():
    ap = argparse.ArgumentParser(description="W.4 aggregate")
    ap.add_argument("--cells", default="experiments/W4_caa_baseline/cells.jsonl")
    ap.add_argument("--out", default="/tmp/deltamemory/W4_caa_baseline/")
    args = ap.parse_args()

    cells_path = Path(args.cells)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cells_path.exists():
        gz = cells_path.with_suffix(".jsonl.gz")
        if gz.exists():
            cells_path = gz
        else:
            print(f"[agg] cells file not found: {cells_path}", file=sys.stderr)
            sys.exit(1)

    print(f"[agg] loading {cells_path}", flush=True)
    raw_cells = load_cells(cells_path)
    cells = filter_data(raw_cells)
    print(f"[agg] {len(raw_cells)} raw rows, {len(cells)} usable "
          f"(after filtering substitutions / unsupported)", flush=True)

    if not cells:
        print("[agg] no usable cells", file=sys.stderr)
        sys.exit(1)

    df = build_aggregate(cells)
    csv_out = out_dir / "aggregate.csv"
    df.to_csv(csv_out, index=False)
    print(f"[agg] saved {csv_out}")
    print(df.to_string(index=False))

    verdicts = compute_verdicts(cells)
    verdicts_out = out_dir / "verdicts.json"
    with open(verdicts_out, "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"[agg] saved {verdicts_out}")

    print("\n=== W.4 family verdict ===")
    print(f"  family size           : {verdicts['family_size']}")
    print(f"  significant CAA wins  : {verdicts['n_significant_caa_wins']}")
    print(f"  significant LOPI wins : {verdicts['n_significant_lopi_wins']}")
    print(f"  CAA models (>=3 wins) : {verdicts['caa_models_with_3plus_wins']}")
    print(f"  LOPI models (>=3 wins): {verdicts['lopi_models_with_3plus_wins']}")
    print(f"  redline violations    : {len(verdicts['redline_violations'])}")
    print(f"  family verdict        : {verdicts['family_verdict']}")


if __name__ == "__main__":
    main()
