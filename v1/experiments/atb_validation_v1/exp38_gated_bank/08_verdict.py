"""Exp38 — assemble cross-variant verdict table and write EXP38_VERDICT.md."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "run_qwen_exp38"


def fmt(x, prec=3):
    if x is None: return "—"
    if isinstance(x, float): return f"{x:.{prec}f}"
    return str(x)


def main():
    rows = []
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        r = d / "results.json"
        if not r.exists(): continue
        try:
            data = json.loads(r.read_text())
        except Exception:
            continue
        phi1 = data.get("phi1", {})
        # pick k=10 as headline
        k10 = phi1.get("k10", {})
        ct = data.get("cross_talk_37c", {})
        neg = data.get("negation_36_4", {})
        hs = data.get("hellaswag", {})
        rows.append({
            "variant": d.name,
            "phi1_k10_gate_d": k10.get("mean_gate_d"),
            "phi1_k100_gate_d": phi1.get("k100", {}).get("mean_gate_d"),
            "phi1_k1000_gate_d": phi1.get("k1000", {}).get("mean_gate_d"),
            "cross_talk_pass": ct.get("frac_abs_drop_below_0p5"),
            "cross_talk_mean_drop": ct.get("mean_abs_drop"),
            "neg_pass": neg.get("frac_facts_neg_suppresses"),
            "neg_mean_diff": neg.get("mean_affirm_minus_neg_nats"),
            "hellaswag_acc": hs.get("acc"),
        })

    audits = {}
    af = RUNS / "audits.json"
    if af.exists():
        audits = json.loads(af.read_text())

    # Build markdown table
    header = "| variant | Φ1 k=10 gate_d | k=100 | k=1000 | 37.C drop_pass | 37.C mean|drop| | 36.4 neg_pass | 36.4 mean_diff | HellaSwag |"
    sep = "|" + "|".join(["---"] * 9) + "|"
    body = "\n".join(
        f"| {r['variant']} | {fmt(r['phi1_k10_gate_d'])} | {fmt(r['phi1_k100_gate_d'])} "
        f"| {fmt(r['phi1_k1000_gate_d'])} | {fmt(r['cross_talk_pass'])} "
        f"| {fmt(r['cross_talk_mean_drop'])} | {fmt(r['neg_pass'])} | {fmt(r['neg_mean_diff'])} "
        f"| {fmt(r['hellaswag_acc'])} |"
        for r in rows
    )

    md = f"""# Exp38 — Gated Fact-LoRA Bank Verdict

Pre-registered thresholds (from `preregister.json`):
- Φ1: mean_gate_d ≥ 2.0 @ k=10
- 37.C: frac(|drop|<0.5) ≥ 0.85
- 36.4: neg_pass ≥ 0.75
- HellaSwag: rel drop ≤ 5%

## Results

{header}
{sep}
{body}

## Anti-cheat audits

```json
{json.dumps(audits, indent=2)}
```

## Per-variant PASS/FAIL

"""
    for r in rows:
        v = r["variant"]
        gate_d = r["phi1_k10_gate_d"]
        ct = r["cross_talk_pass"]
        neg = r["neg_pass"]
        flags = []
        flags.append(f"Φ1≥2.0: {'PASS' if gate_d and gate_d >= 2.0 else 'FAIL'} ({fmt(gate_d)})")
        flags.append(f"37.C≥0.85: {'PASS' if ct and ct >= 0.85 else 'FAIL'} ({fmt(ct)})")
        flags.append(f"36.4≥0.75: {'PASS' if neg and neg >= 0.75 else 'FAIL'} ({fmt(neg)})")
        md += f"- **{v}**: " + "; ".join(flags) + "\n"

    out = HERE / "EXP38_VERDICT.md"
    out.write_text(md)
    print(md)
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
