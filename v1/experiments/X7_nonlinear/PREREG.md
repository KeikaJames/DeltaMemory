# X.7-NL — Non-linear Memory ↔ Attention Interactions (PREREG.v1)

**Author**: BIRI GA
**Branch**: `feat/v05-counterfactual-industrial`
**Whitelist target archs**: `google/gemma-4-31B-it`, `Qwen/Qwen3.6-27B`,
`openai/gpt-oss-120b` (cross-arch HARD rule).

> **Driver mandate (verbatim)**:
>
> 系统性观察 AttnNativeBank 注入外部记忆后，Attention 内部的真实非线性
> 行为 ... 验证是否存在目前未知的非线性现象（权重突然集中、残差流振荡、
> 记忆相互抑制/放大等）... 为后续"让 AI 自己检索 / 智能 memory 管理"
> 提供数据基础和设计依据。

This study is **descriptive + falsifying**, not just a sanity check.
It is **distinct** from X.7 (forget/merge) which tests *remediation*;
X.7-NL tests the *phenomenology* under a fixed-policy bank.

---

## 1. Sub-experiments and hypotheses

### A — Bank Size Scaling (records phenomenology vs. |bank|)
- Fixed prompt, fixed α=1.0, vary `|bank| ∈ {10, 50, 100, 500, 1000, 5000}`.
- Each bank entry is a synthetic Lama-fact line; entries are **distinct
  subjects** so retrieval ambiguity is held constant.
- Hypotheses:
  - **H_X7N.A1** (monotone-entropy): mean attention weight on injected
    tokens is monotone-decreasing in |bank|.
  - **H_X7N.A2** (entropy plateau): attn-entropy plateaus at large
    |bank| (≥1000).
  - **H_X7N.A3** (residual collapse): residual-stream norm at the
    injection layer drifts <5% across the |bank| sweep.

### B — Alpha Sweep (non-linear response to injection strength)
- Fixed |bank|=200, sweep α ∈ {0.0, 0.05, 0.10, …, 2.00} (41 points).
- Capture per-α: Δ logprob of target token, mean attn weight on
  injected entries, residual norm at injection layer.
- Hypotheses:
  - **H_X7N.B1** (monotone-recall): Δ logprob is monotone-increasing in
    α on [0, 1].
  - **H_X7N.B2** (saturation/inversion): Δ logprob saturates or
    decreases on (1, 2] (super-injection harms).
  - **H_X7N.B3** (norm-blowup): residual-stream norm grows
    super-linearly above α=1.5 (unstable regime).

### C — Multi-turn dynamic injection
- 50 turns; alternate between writing fact-A and a contradictory fact-A'
  on the same `(subject, relation)` pair.
- Capture per-turn: recall of A vs A' at α=1.
- Hypotheses:
  - **H_X7N.C1** (oscillation): recall winner alternates with the
    most-recent write (consistent with X.2 contradictory-facts study,
    if order matters at all).
  - **H_X7N.C2** (drift-bound): per-turn ‖residual‖₂ change is
    bounded by a constant ε independent of turn index.

### D — SCAR signal correlation
- Across all (A, B, C) cells, log SCAR `proj`, `ortho`, `alpha-drift`.
- Hypotheses:
  - **H_X7N.D1** (proj predicts recall): Pearson(proj, Δ logprob) > 0.5
    across all cells.
  - **H_X7N.D2** (alpha-drift forewarns saturation): high alpha-drift
    cells are exactly the cells where Δ logprob fails to monotone in α.

---

## 2. Hard rules

- **Cross-arch (CLAUDE.md)**: every published verdict must be reproduced
  on ≥ 2 of the whitelist archs.
- **No precision drop**: bf16 native (or model's official lower native,
  e.g. gpt-oss MXFP4); never community-quantized below.
- **No fabrication**: every plot point traces to `cells.jsonl` row;
  `env.json` captures (commit, model_dir SHA, dataset SHA, seed list,
  device, dtype, transformers version).
- **Authenticity contract**: bit-equality redline = run with α=0 must
  match no-bank baseline within 1e-7 nll spread.

## 3. Out of scope

- Capacity / eviction (covered by X.7 forget/merge).
- Long-context positional-encoding edge cases (covered by X.4 / X.4b).
- Adversarial misuse of Δ logprob (covered by X.3 redteam).

## 4. Output contract

```
runs/X7NL_full_v1_<arch>/
  cells.jsonl              # one row per (sub-exp, knob, seed) cell
  env.json
  summary.json             # per-hypothesis verdict block
  REPORT.md                # cross-cells narrative + figs links
  figs/
    A_bank_scaling.png
    B_alpha_curve.png
    C_recall_oscillation.png
    D_scar_corr.png
```

`summary.json` schema:
```json
{
  "verdicts": {
    "H_X7N.A1": "supported|not_supported|inconclusive",
    "H_X7N.A2": "...", "H_X7N.A3": "...",
    "H_X7N.B1": "...", "H_X7N.B2": "...", "H_X7N.B3": "...",
    "H_X7N.C1": "...", "H_X7N.C2": "...",
    "H_X7N.D1": "...", "H_X7N.D2": "..."
  },
  "headline": "...",
  "n_cells": <int>,
  "arch": "...",
  "commit": "..."
}
```

## 5. Cells × seed budget

| sub-exp | knob | n_levels | seeds | cells |
|---|---|---:|---:|---:|
| A bank-scaling | \|bank\| | 6 | 3 | 18 |
| B alpha-sweep | α | 41 | 3 | 123 |
| C multi-turn | turn | 50 | 3 | 150 |
| D SCAR-corr | (free; piggybacks on A+B+C) | — | — | — |
| **per arch** | | | | **291** |
| **× 2 archs** | | | | **582** |

D is computed post-hoc from (A,B,C) cell rows; no separate runs.
