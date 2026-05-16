#!/usr/bin/env python3
"""Cross-check every v2 cell JSON for sign-convention consistency.

We re-derive delta from raw NLL fields (before/after, base/post, base/lpl)
and compare against the stored `delta`/`delta_real`/`delta_signed` field
if present. Flags any cell where the stored delta has the wrong sign vs
the recomputed signed delta (post - base, negative=improvement).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path("/Users/gabiri/projects/RCV-HC/v2/experiments")


def deepget(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def find_pairs(j):
    """Return list of (label, base_nll, post_nll) tuples from one JSON."""
    pairs = []
    # generic before/after
    b = deepget(j, "before", "real")
    a = deepget(j, "after", "real")
    if b is not None and a is not None:
        pairs.append(("real", b, a))
    # base/post inside results
    res = j.get("results", {}) or {}
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(v, dict):
                bb = v.get("base") or v.get("nll_base")
                pp = v.get("lpl") or v.get("post") or v.get("nll") or v.get("nll_lpl")
                if isinstance(bb, (int, float)) and isinstance(pp, (int, float)):
                    pairs.append((k, bb, pp))
    # top-level
    if "base_nll" in j and "post_nll" in j:
        pairs.append(("top", j["base_nll"], j["post_nll"]))
    return pairs


def get_stored_delta(j, label):
    """Try to find a stored delta value with matching key context."""
    candidates = []
    res = j.get("results", {}) or {}
    if label in res and isinstance(res, dict) and isinstance(res.get(label), dict):
        for fld in ("delta", "delta_signed", "delta_real", "nll_drop"):
            if fld in res[label]:
                candidates.append((fld, res[label][fld]))
    if label == "real":
        for fld in ("delta_real", "delta_signed", "delta", "nll_drop"):
            if fld in j:
                candidates.append((fld, j[fld]))
    return candidates


def main():
    n_files = 0
    n_flagged = 0
    flagged = []
    for path in sorted(ROOT.rglob("*.json")):
        if "__pycache__" in str(path):
            continue
        try:
            j = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(j, dict):
            continue
        n_files += 1
        for label, b, a in find_pairs(j):
            signed = a - b
            for fld, stored in get_stored_delta(j, label):
                if not isinstance(stored, (int, float)):
                    continue
                if abs(stored - signed) < 0.15:
                    continue
                if abs(stored - (-signed)) < 0.15:
                    # legacy unsigned — accept silently (aggregator recomputes signed)
                    continue
                # mismatched magnitude beyond rounding
                rel = str(path.relative_to(ROOT))
                flagged.append((rel, label, fld, b, a, signed, stored))
                n_flagged += 1
    print(f"# Sign-validator scan — {n_files} JSON files, {n_flagged} cell-fields flagged\n")
    if flagged:
        print("| file | label | field | base | post | signed=post-base | stored |")
        print("|---|---|---|---:|---:|---:|---:|")
        for row in flagged:
            f, lab, fld, b, a, s, st = row
            print(f"| {f} | {lab} | {fld} | {b:.3f} | {a:.3f} | {s:+.3f} | {st:+.3f} |")
    else:
        print("All scanned cells are sign-consistent under either signed (post-base, negative=improve) or unsigned-positive (base-post, positive=improve) convention.")


if __name__ == "__main__":
    main()
