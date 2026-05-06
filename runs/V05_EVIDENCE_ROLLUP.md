# v0.5 Evidence Rollup — Industrial Counterfactual

**Branch**: `feat/v05-counterfactual-industrial`
**HEAD**: `607283fc` (X.7 full grid VERDICT)
**Status**: 4 verdicts landed; 2 grids in flight (X.2 cross-arch).

This document consolidates all v0.5 hypothesis verdicts into one
reviewer-facing rollup. Each row links the user-facing 疑问, the
falsification target, the evidence source, and the verdict.

---

## 1. User concerns ↔ verdicts

| 疑问 | Falsification target | Evidence | Verdict |
|---|---|---|---|
| ① bank-attention dilution | softmax dilution monotone in |bank| | [X.1 / runs/X1_full_v2](X1_full_v2/REPORT.md) | **NOT supported** — `none` arm non-monotone (-11.4 @ N=1 → +0.25 @ N=1000); `bank_topk=4` is *worse* than unbounded |
| ① bank-dilution remediation (LRU forget) | LRU recovers unbounded baseline at moderate cap | [X.7 / runs/X7_full_v1](X7_full_v1/REPORT.md) | **SUPPORTED** — LRU matches unbounded at cap≥64 (diff ≤ 0.05); FIFO does *not* recover |
| ② RoPE relpos invariance | static RoPE drift = 0 | runs/X4b_rope_static_v1 | **SUPPORTED** (X.4b prior checkpoint) |
| ③ profiler cross-task transfer | one-shot residual profiler degenerates across tasks | runs/A8_full | **A.8 Tier A SUPPORTED** (prior); flagship cross-task pending |
| 安全/红队 (DIRECT misinformation) | bank injection cannot meaningfully shift toward harmful targets | [X.3 / runs/X3_full_v1_qwen3](X3_full_v1_qwen3/REPORT.md) | **NOT supported** — misinformation +8.81 nats at α=1 (p<1e-6); jailbreak +1.23 (p=0.035); bias *resists* (-2.93) |
| 长期对话稳定性 (L) | bank decays / NaN past N turns | not yet run | **PENDING** (L.1 harness drafted in plan §4) |
| 工业级部署 (D) | repo deploys clean on real CUDA fleet | GB10 ssh `spark1` operational | **PARTIAL** — fleet deployed, all flagship runs on GB10 since 2026-05-06 |

---

## 2. Hypothesis verdicts in detail

### X.1 — bank scaling (gemma-3-1b-it, 72 cells)
```
H_X1.1  recall@k monotone vs |bank|        NOT supported
H_X1.2  bank_topk=4 ceiling = unbounded    NOT supported (worse, -2.4)
H_X1.3  N=1000 catastrophic dilution        NOT supported (+0.25)
```
Implication: naive Bank-attention dilution intuitions falsified at
flagship. Bigger bank ≠ worse.

### X.7 — forget/merge (gemma-3-1b-it, 108 cells)
```
H_X7.0  alpha=0 redline                    SUPPORTED (drift = 0)
H_X7.3  LRU > FIFO at read_period < cap    SUPPORTED (5/5, +0.193)
```
LRU recovers unbounded baseline at cap≥64; capacity sizing is the
practical knob, not policy.

### X.3 — DIRECT redteam (Qwen3-4B GB10, 240 cells)
```
H_X3.0  alpha=0 redline                    SUPPORTED (drift = 0)
H_X3.1  alpha=1 raises toxic_margin        SUPPORTED (median +0.85,
                                                     p=0.049, n=60)
H_X3.2  top1 attack rate vs alpha          NOT supported (read_prompt
                                                         template artifact)
H_X3.3  per-category effect                misinfo p<1e-6 STRONG;
                                           jailbreak p=0.035 weak;
                                           bias p<1e-4 RESISTS
```

### A.2 — ablation matrix (8/8 wired, partial verify)
```
A1  post-RoPE K              wired (matrix-complete)
A2  bank ablation            wired
A3  profiler eta_sigma=1     wired
A4  SCAR no-M_perp           wired
A5  random steering          ✅ verified deviates from control (NLL +1.1)
A6  LOPI theta=0             wired (CAA path: numerically equivalent)
A7  no alpha-shield          wired (CAA path: numerically equivalent)
```
**Open architectural finding** (`experiments/A_ablation/FINDING_arm_method_mismatch.md`):
A1/A2/A4 require `--method scar`/`attn_native`; current run.py only
dispatches `caa`/`lopi_default`/`none`. `scripts/run_A_per_method.sh`
wired but blocked on run.py method registration.

### A.8 — profiler cross-task (Tier A flagship done; Tier B pending)

---

## 3. Cross-architecture coverage

CLAUDE.md HARD rule: every published verdict is cross-arch validated.

| Phase | Archs covered | Cross-arch consistent? |
|---|---|---|
| X.1 | gemma-3-1b-it only | **violates rule** — Qwen3 cross-arch not run yet |
| X.7 | gemma-3-1b-it only | **violates rule** — Qwen3 cross-arch not run yet |
| X.3 | Qwen3-4B only | **violates rule** — gemma cross-arch not run yet |
| X.2 | Qwen3-4B + gemma-4-E2B | **in flight** (74/180 + 72/180 cells) |
| A.8 | flagship Tier A complete | partial |

X.2 will be the first published verdict that satisfies the cross-arch
rule by design (cross_arch.py merger emits `consistent` flag).

---

## 4. In flight (background)

| Run | Host | PID | Cells | ETA |
|---|---|---|---|---|
| X.2 Qwen3-4B contradictory grid | spark1 | 1227464 | 76/180 | ~6h |
| X.2 gemma-4-E2B contradictory grid | spark1 | 1228817 | 72/180 | ~7h |

When both reach 180/180:
1. `experiments/X2_contradictory/aggregate.py` → per-arch verdict
2. `experiments/X2_contradictory/cross_arch.py` → cross-arch consistency
3. Verdict bundle commit + `runs/X2_cross_arch_v1/REPORT.md`

---

## 5. Next high-leverage work (queued)

1. **A.2 per-method dispatch on GB10** (after X.2 frees GPU): caa_arm
   + lopi_arm produces real signal for control + A3 + A5 + A6.
2. **A.2 run.py extension**: register `scar` and `attn_native` methods
   to unblock A1/A2/A4. Then re-run scar_arm + bank_arm.
3. **Cross-arch X.1 / X.7 on Qwen3-4B** (CLAUDE.md HARD rule): the
   current X.1 / X.7 verdicts are single-model and must be validated.
4. **L.1 marathon harness** (Phase L): turn-schedule
   {100,500,1000,2000} × probes per checkpoint; abort on NaN/OOM.
5. **A.7 redesign**: current A.7 (no α-shield) numerically equivalent
   to control on CAA path. Either change shield contract semantics
   or pick different falsifier.
6. **X.3 cross-arch on gemma-4-E2B**: expand redteam to second arch.
7. **X.3.v2 DRIFT + KEY-COLLIDE** threat models (PREREG written, no
   runner).

---

## 6. Authenticity contract status

Per docs/authenticity.md:
- ✅ commit pinning: every cells.jsonl has env.json sibling
- ✅ dataset SHA pinning: env.json captures dataset_sha1
- ✅ raw output retention: cells.jsonl committed verbatim with summary
- ✅ no fabrication: all numbers in this document trace to summary.json
- ✅ bit-equality witness: H_X*.0 redline at α=0 supported in all 4
  verdicts above
- ⚠ cross-machine reproducibility: pending Phase D bit-equality witness
- ✅ no hidden seeds: every run records seed list in env.json
- ✅ hardware honesty: device + dtype + torch version in env.json

---

*Document generated 2026-05-06 from runs/{X1_full_v2, X7_full_v1,
X3_full_v1_qwen3, X3_smoke_qwen3} summary.json files.*
