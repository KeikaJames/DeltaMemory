"""Cross-architecture aggregator + paired-bootstrap CIs for Exp26/26b/27.

Reads cells.jsonl from multiple run dirs, computes the four gates:
  A = topk1_full_bank − topk1_minus_correct      (correct fact contribution)
  B = retrieval_accuracy                         (selects correct slot)
  C = topk1_full_bank − topk1_meanV              (V carries content)
  D = topk1_full_bank − topk1_shuffled_factids   (K/V identity bound to fact)

Each gate is computed per (seed, alpha) cell with paired bootstrap CI over
fact_id. Output: pretty-printed table + summary.json per run dir, plus a
cross-arch CSV.

Usage:
  python analyze_cross_arch.py <run_dir1> [<run_dir2> ...] [--out summary.csv]
"""
from __future__ import annotations
import argparse, json, math, random, statistics
from pathlib import Path
from collections import defaultdict


def load_cells(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out


def paired_bootstrap(diffs: list[float], B: int = 2000, alpha: float = 0.05,
                     seed: int = 0):
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(B):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return mean, lo, hi


def analyze(run_dir: Path, *, label: str | None = None):
    cells_path = run_dir / "cells.jsonl"
    if not cells_path.exists():
        return None
    cells = load_cells(cells_path)
    label = label or run_dir.name

    # Group: (seed, fact_id, alpha) -> {variant: margin}
    rec = defaultdict(dict)
    retrieval_correct_by_alpha = defaultdict(list)
    bank_sizes = set()
    for c in cells:
        key = (c["seed"], c["fact_id"], c.get("alpha", 0.0))
        rec[key][c["variant"]] = c.get("margin", float("nan"))
        if "retrieval_correct" in c and c["retrieval_correct"] is not None:
            retrieval_correct_by_alpha[(c["seed"], c["alpha"])].append(
                c["retrieval_correct"])
        if "bank_size" in c:
            bank_sizes.add(c["bank_size"])

    # Per-alpha gates A/C/D = topk1_full_bank − {minus_correct, meanV, shuffled}
    # Use full_bank_topk1 as the reference.
    alphas = sorted({k[2] for k in rec.keys() if k[2] > 0})
    results = []
    for a in alphas:
        diffs_A, diffs_C, diffs_D = [], [], []
        retr_correct = []
        seeds_seen = set()
        for (seed, fid, alpha), v in rec.items():
            if alpha != a: continue
            seeds_seen.add(seed)
            full = v.get("full_bank_topk1")
            if full is None or math.isnan(full): continue
            mc = v.get("full_bank_topk1_minus_correct")
            mv = v.get("full_bank_topk1_meanV")
            sf = v.get("full_bank_topk1_shuffled_factids")
            if mc is not None and not math.isnan(mc):
                diffs_A.append(full - mc)
            if mv is not None and not math.isnan(mv):
                diffs_C.append(full - mv)
            if sf is not None and not math.isnan(sf):
                diffs_D.append(full - sf)
        mean_A, lo_A, hi_A = paired_bootstrap(diffs_A, seed=int(a * 1e6))
        mean_C, lo_C, hi_C = paired_bootstrap(diffs_C, seed=int(a * 1e6) + 1)
        mean_D, lo_D, hi_D = paired_bootstrap(diffs_D, seed=int(a * 1e6) + 2)
        all_retr = []
        for s in seeds_seen:
            all_retr.extend(retrieval_correct_by_alpha.get((s, a), []))
        retr_acc = (sum(all_retr) / len(all_retr)) if all_retr else float("nan")
        n_facts = len(diffs_A)
        results.append({
            "alpha": a,
            "n_facts": n_facts,
            "n_retrieval": len(all_retr),
            "A_mean": mean_A, "A_lo": lo_A, "A_hi": hi_A,
            "C_mean": mean_C, "C_lo": lo_C, "C_hi": hi_C,
            "D_mean": mean_D, "D_lo": lo_D, "D_hi": hi_D,
            "retrieval_accuracy": retr_acc,
        })

    bank_size = max(bank_sizes) if bank_sizes else 0
    chance = 1.0 / bank_size if bank_size > 0 else float("nan")
    print(f"\n== {label}  (bank={bank_size}, chance={chance:.4f}) ==")
    print(f"{'alpha':>8} {'A':>22} {'C':>22} {'D':>22} {'retr_acc':>10} {'×chance':>8}")
    for r in results:
        rx = r["retrieval_accuracy"] / chance if chance > 0 else float("nan")
        print(f"{r['alpha']:>8.4f}  "
              f"{r['A_mean']:>+7.3f} [{r['A_lo']:+.2f},{r['A_hi']:+.2f}]  "
              f"{r['C_mean']:>+7.3f} [{r['C_lo']:+.2f},{r['C_hi']:+.2f}]  "
              f"{r['D_mean']:>+7.3f} [{r['D_lo']:+.2f},{r['D_hi']:+.2f}]  "
              f"{r['retrieval_accuracy']:>9.4f}  {rx:>7.2f}")
    summary = {"label": label, "bank_size": bank_size, "chance": chance,
               "cells_total": len(cells), "alphas": results}
    (run_dir / "cross_arch_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None, help="optional CSV path for aggregate")
    args = ap.parse_args()

    all_summaries = []
    for d in args.dirs:
        p = Path(d)
        s = analyze(p)
        if s:
            all_summaries.append(s)

    if args.out and all_summaries:
        with open(args.out, "w") as f:
            f.write("label,bank_size,chance,alpha,n_facts,n_retr,"
                    "A_mean,A_lo,A_hi,C_mean,C_lo,C_hi,D_mean,D_lo,D_hi,"
                    "retrieval_accuracy,x_chance\n")
            for s in all_summaries:
                for r in s["alphas"]:
                    rx = r["retrieval_accuracy"] / s["chance"] if s["chance"] > 0 else float("nan")
                    f.write(f"{s['label']},{s['bank_size']},{s['chance']:.4f},"
                            f"{r['alpha']:.4f},{r['n_facts']},{r['n_retrieval']},"
                            f"{r['A_mean']:.4f},{r['A_lo']:.4f},{r['A_hi']:.4f},"
                            f"{r['C_mean']:.4f},{r['C_lo']:.4f},{r['C_hi']:.4f},"
                            f"{r['D_mean']:.4f},{r['D_lo']:.4f},{r['D_hi']:.4f},"
                            f"{r['retrieval_accuracy']:.4f},{rx:.4f}\n")
        print(f"\nWrote aggregate CSV to {args.out}")


if __name__ == "__main__":
    main()
