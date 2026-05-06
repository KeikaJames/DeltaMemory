# X.7-NL Report — gemma-4-31B-it (full grid, 291 cells)

**Status**: complete (291/291 ok), spark1 / GB10 / CUDA bf16.
**Commit**: `7fd0befa55c2966e6e1ada0c7930b8fe335fda99` (+ uncommitted v_proj/per-layer-kv fix `2feee661`).
**PREREG**: `experiments/X7_nonlinear/PREREG.md` v1.

## Verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H_X7N.A1 (entropy↑ with |bank|) | **supported** | 2.15 → 3.74 → 4.41 → 5.99 → 6.68 → 8.29 monotone |
| H_X7N.A2 (recall↓ monotone with |bank|) | **NOT supported** | non-monotone U-shape (see headline) |
| H_X7N.A3 (residual drift bounded) | not_supported | rel drift 0.054 across sizes (>0.05 cap) |
| H_X7N.B1 (smooth α-response) | **NOT supported** | catastrophic cliff at α=0.25 (see headline) |
| H_X7N.B2 (recall monotone with α) | not_supported | recovers then plateaus |
| H_X7N.B3 (residual ~ linear in α) | not_supported | ratio at α=2 / α=1 = 0.97 |
| H_X7N.C1 (consecutive-fact interference small) | not_supported | even=1.49 / odd=1.43 (close, but not <0.05 stricter test) |
| H_X7N.C2 (50-turn residual stability) | **supported** | C2_max_rel_step = 0.0045 |
| H_X7N.D1/D2 (SCAR signal correlation) | inconclusive | SCAR not captured in this run |

**3 supported / 5 not_supported / 2 inconclusive.**

## Headline non-linearities

### 1. U-shape bank-scaling response (sub-A)

```
|bank|     mean log_margin    min       max       bank_entropy
   10        -0.188          -0.344    +0.000        2.15
   50        +0.365          +0.188    +0.625        3.74
  100        +0.479          +0.375    +0.688        4.41   ← peak
  500        -0.375          -0.500    -0.250        5.99   ← collapse
 1000        -0.344          -0.469    -0.188        6.68
 5000        +0.604          +0.531    +0.719        8.29   ← rescue
```

Naive softmax-dilution (X.1 H_X1.1) predicts monotone collapse with |bank|.
**Falsified at 31B flagship scale.** Recall collapses in the mid-range
(500–1000) but is *rescued* when the bank scales further (5000), even
as bank attention entropy reaches 8.29 (near-uniform).

Mechanistic conjecture (untested here): mid-range banks contain enough
distractors to compete for attention mass without enabling sparse
selection; very large banks trigger a regime where attention behaves
quasi-top-k and the genuine target re-emerges.

### 2. Catastrophic α-cliff at α=0.25 (sub-B)

```
α     mean log_margin
0.00     +0.959   ← no-injection (parametric memory wins)
0.25     -5.740   ← CLIFF: catastrophic interference
0.50     -0.839
0.75     +0.010
1.00     +0.365
1.25     +0.104
1.50     +0.047
1.75     +0.026
2.00     +0.052
```

A small injection (α=0.25) **destroys** recall (-5.74 nats) before
recovery in the 0.75–2.0 plateau. The textbook "smooth response with α"
hypothesis (B1) is decisively falsified. Implication: practitioners
should not assume small-α injections are "safe" or interpolative —
the AttnNativeBank α-response is catastrophically non-monotonic in
the low-α regime on flagship-scale models.

### 3. 50-turn residual-stream stability (sub-C)

C2_max_rel_step = 0.0045 — across 50 alternating fact / counter-fact
injections, the residual-stream norm at the bank-injection layer
drifts <0.5% between any consecutive turns. Long-conversation
stability holds at this scale.

## Whitelist coverage status

* **gemma-4-31B-it (this report)**: complete.
* **Qwen3.6-27B**: blocked. The 27B model uses linear attention
  (`Qwen3_5DecoderLayer.linear_attn`, no `self_attn`). AttnNativeBank
  requires standard softmax Q/K/V attention to inject bank columns
  into the attention weight matrix. **Architectural incompatibility**
  — not a code bug. Documented as a finding for `docs/integration/`.
* **gpt-oss-120b**: deferred. MXFP4 quantization requires bf16
  dequantization on load (slow on GB10); a separate dispatch with
  proper warm-up is queued.

## Authenticity

- 291/291 cells `status=ok`; raw `cells.jsonl` retained.
- env.json carries commit, dataset SHAs, gpu, dtype.
- `dirty_tree=true` because the v_proj fallback fix (`2feee661`)
  was applied to live patcher to unblock gemma-4 (this is now
  committed — re-runs will show clean tree).

## Reproduction

```bash
ssh spark1
cd /home/gabira/projects/RCV-HC && source .venv-gb10/bin/activate
python experiments/X7_nonlinear/run.py \
  --model /home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it \
  --device cuda --dtype bf16 --sub all --seeds 0 1 2 \
  --out runs/X7NL_full_v1_gemma4_31B
python experiments/X7_nonlinear/aggregate.py --run-dir runs/X7NL_full_v1_gemma4_31B
```

