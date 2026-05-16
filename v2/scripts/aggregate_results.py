#!/usr/bin/env python3
"""Aggregate v2 experiment JSON results into a master Markdown table."""
import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = REPO_ROOT / "v2" / "experiments"
OUT_MD = REPO_ROOT / "v2" / "scripts" / "all_results.md"
OUT_CSV = REPO_ROOT / "v2" / "scripts" / "all_results.csv"

MISSING = "—"


def fnum(x, digits=3):
    if x is None:
        return MISSING
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _pick_nested(d, key):
    """Return d[key] or first nested .{key} (e.g. e06's before.test_rel_ood.real)."""
    if not isinstance(d, dict):
        return None
    if key in d and not isinstance(d[key], dict):
        return d[key]
    # Prefer test/ood-shaped sub-block, else first dict child with the key
    preferred = [k for k in d if "test" in k.lower() or "ood" in k.lower()]
    for k in preferred + [k for k in d if k not in preferred]:
        v = d.get(k)
        if isinstance(v, dict) and key in v and not isinstance(v[key], dict):
            return v[key]
    return None


def load_records():
    paths = sorted(
        set(
            glob.glob(str(EXP_ROOT / "e*" / "*.json"))
            + glob.glob(str(EXP_ROOT / "e*" / "cells" / "*.json"))
        )
    )
    records = []
    for p in paths:
        path = Path(p)
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict):
            continue

        exp_id = path.parent.name if path.parent.name != "cells" else path.parent.parent.name
        variant = d.get("variant") or d.get("mode") or d.get("tag") or path.stem
        seed = d.get("seed")
        layer = d.get("bank_layer")
        if layer is None:
            layers = d.get("layers")
            if isinstance(layers, list) and layers:
                layer = layers[0] if len(layers) == 1 else ",".join(str(x) for x in layers)
        n_train = d.get("n_train")
        n_preload = d.get("n_preload")
        steps = d.get("steps")

        before = d.get("before") or {}
        after = d.get("after") or {}
        rec = {
            "path": str(path.relative_to(REPO_ROOT)),
            "experiment": exp_id,
            "variant": variant,
            "seed": seed,
            "layer": layer,
            "n_train": n_train,
            "n_preload": n_preload,
            "steps": steps,
            "before_base": _pick_nested(before, "base"),
            "before_real": _pick_nested(before, "real"),
            "before_rand": _pick_nested(before, "rand"),
            "after_base": _pick_nested(after, "base"),
            "after_real": _pick_nested(after, "real"),
            "after_rand": _pick_nested(after, "rand"),
            "after_zero": _pick_nested(after, "zero"),
            "after_off": _pick_nested(after, "off"),
        }
        delta = d.get("delta_real")
        if delta is None:
            verdict = d.get("verdict") or {}
            delta = verdict.get("delta_real") or verdict.get("delta_test_ood") or verdict.get("delta")
        if delta is None and rec["after_real"] is not None and rec["before_real"] is not None:
            try:
                delta = float(rec["after_real"]) - float(rec["before_real"])
            except (TypeError, ValueError):
                delta = None
        rec["delta_real"] = delta

        verdict = d.get("verdict") or {}
        rec["verdict_pass"] = verdict.get("pass")
        rec["verdict_rule"] = verdict.get("rule")
        records.append(rec)
    return records


def _sort_key(r):
    layer = r.get("layer")
    try:
        lk = (0, float(str(layer).split(",")[0]))
    except (TypeError, ValueError):
        lk = (1, str(layer))
    seed = r.get("seed")
    try:
        sk = (0, int(seed))
    except (TypeError, ValueError):
        sk = (1, str(seed))
    return (lk, sk, str(r.get("variant") or ""))


def fmt_row(r):
    verdict = MISSING
    if r["verdict_pass"] is True:
        verdict = "PASS"
    elif r["verdict_pass"] is False:
        verdict = "FAIL"
    return (
        f"| {r['variant'] or MISSING} "
        f"| {r['seed'] if r['seed'] is not None else MISSING} "
        f"| {r['layer'] if r['layer'] is not None else MISSING} "
        f"| {r['n_preload'] if r['n_preload'] is not None else MISSING} "
        f"| {r['n_train'] if r['n_train'] is not None else MISSING}/{r['steps'] if r['steps'] is not None else MISSING} "
        f"| {fnum(r['before_base'])} "
        f"| {fnum(r['after_real'])} "
        f"| {fnum(r['delta_real'])} "
        f"| {verdict} |"
    )


HEADER = "| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |"
SEP = "|---|---|---|---|---|---|---|---|---|"


def render_group(records):
    out = [HEADER, SEP]
    for r in sorted(records, key=_sort_key):
        out.append(fmt_row(r))
    return "\n".join(out)


def find(records, exp, predicate=lambda r: True):
    return [r for r in records if r["experiment"] == exp and predicate(r)]


def render_headline(records):
    lines = ["## Headline\n"]

    # e01 canonical (any seed)
    rows = find(records, "e01_anticheat_b2",
                lambda r: "canonical" in (r["variant"] or "") and "v2/experiments/e01_anticheat_b2/e01_canonical" in r["path"])
    if rows:
        lines.append("**e01 canonical (B2 reproduce)**\n")
        lines.append(render_group(rows) + "\n")

    # e01 h6 layer sweep — by filename pattern
    rows = [r for r in records if r["experiment"] == "e01_anticheat_b2" and "e01_h6_layer" in r["path"]]
    if rows:
        lines.append("**e01 h6 layer sweep (L3 / L9 / L21 / L33)**\n")
        lines.append(render_group(rows) + "\n")

    # e11 noise variants
    rows = find(records, "e11_noise_robustness")
    if rows:
        lines.append("**e11 noise variants (falsification)**\n")
        lines.append(render_group(rows) + "\n")

    # e02 scale cells
    rows = find(records, "e02_scale_matrix")
    if rows:
        lines.append("**e02 scale matrix**\n")
        lines.append(render_group(rows) + "\n")

    # e05 cross-model
    rows = find(records, "e05_cross_model")
    if rows:
        lines.append("**e05 cross-model**\n")
        lines.append(render_group(rows) + "\n")

    # e06 OOD
    rows = find(records, "e06_relation_disjoint_ood")
    if rows:
        lines.append("**e06 relation disjoint OOD (train vs test_ood — Δ shown is OOD)**\n")
        lines.append(render_group(rows) + "\n")

    # e19 seed replication summary
    rows = find(records, "e19_seed_replication")
    if rows:
        lines.append("**e19 seed replication summary (Δ_real mean ± stdev by layer)**\n")
        lines.append("| layer | n_seeds | Δ_real mean | Δ_real stdev |")
        lines.append("|---|---|---|---|")
        by_layer = {}
        for r in rows:
            if r["delta_real"] is None:
                continue
            by_layer.setdefault(r["layer"], []).append(float(r["delta_real"]))
        for layer in sorted(by_layer, key=lambda x: float(x) if x is not None else 1e9):
            vals = by_layer[layer]
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"| {layer} | {len(vals)} | {mean:.3f} | {stdev:.3f} |")
        lines.append("")

    # e09 v1_orig vs v2_kproj
    rows = find(records, "e09_v1_anb_resurrect")
    if rows:
        lines.append("**e09 v1_orig vs v2_kproj**\n")
        lines.append(render_group(rows) + "\n")

    return "\n".join(lines)


def render_md(records):
    out = ["# v2 experiment results — aggregated\n", render_headline(records), "\n## All experiments\n"]
    by_exp = {}
    for r in records:
        by_exp.setdefault(r["experiment"], []).append(r)
    for exp in sorted(by_exp):
        out.append(f"### {exp}\n")
        out.append(render_group(by_exp[exp]))
        out.append("")
    return "\n".join(out)


CSV_COLS = [
    "experiment", "path", "variant", "seed", "layer", "n_train", "n_preload", "steps",
    "before_base", "before_real", "before_rand",
    "after_base", "after_real", "after_rand", "after_zero", "after_off",
    "delta_real", "verdict_pass", "verdict_rule",
]


def write_csv(records, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in CSV_COLS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write all_results.csv")
    args = ap.parse_args()

    records = load_records()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(records))
    if args.csv:
        write_csv(records, OUT_CSV)

    n_exp = len({r["experiment"] for r in records})
    print(f"Wrote {len(records)} rows across {n_exp} experiments")


if __name__ == "__main__":
    main()
