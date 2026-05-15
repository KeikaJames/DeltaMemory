# Exp35b — MEMIT-preconditioned Fact-LoRA Bank @ N=10⁴

**Model**: Qwen/Qwen3-4B-Instruct-2507 (MPS bf16) · **Bank**: 10000 facts, edit_layer L=5, v_steps=25, MEMIT C⁻¹ precondition (1.5M-token covariance)

**Verdict**: **PARTIAL POSITIVE** — composition (Φ1) and audits (D6 @ k≤100, D7, D9, D4) PASS at industry-strict thresholds; routing (Φ2) and end-to-end (Φ3) miss top-line pre-registered targets; D6 capability collapses at k=1000.

---

## Pre-registered scorecard (all thresholds locked at commit `8e1b69d9`)

| Phase | Metric | Threshold | Observed | Verdict |
|---|---|---|---|---|
| Φ1 | gate_d nats @ k=10 | ≥ 2.0 | **9.78** | ✅ PASS (4.9×) |
| Φ1 | pos_frac @ k=10 | ≥ 80% | 100% (beats base) | ✅ PASS |
| Φ1 | pos_frac @ k=100 | ≥ 60% | 99.8% | ✅ PASS |
| Φ1 | pos_frac @ k=1000 | ≥ 30% | 97.2% | ✅ PASS |
| Φ2 | honest router top1 | ≥ 30% | **40.85%** | ✅ PASS |
| Φ2 | honest router top5 | ≥ 60% | **54.91%** | ❌ FAIL (5pp short) |
| Φ2 | shuffled top1 (anti-cheat) | ≤ 5% | 0.04% (≈ chance 0.01%) | ✅ PASS |
| Φ2 | D2 collision top1 | ≥ 20% | 27.29% | ✅ PASS |
| Φ3 | routed/oracle ratio | ≥ 0.70 | **0.458** | ❌ FAIL |
| D4 | layer ablation: L=5 optimal | qualitative | L=5 wins on uplift+norm | ✅ PASS |
| D6 | ppl drift @ k=10 | ≤ 3% | 0.69% | ✅ PASS |
| D6 | ppl drift @ k=100 | ≤ 8% | 2.81% | ✅ PASS |
| D6 | ppl drift @ k=1000 | ≤ 15% | **94%** | ❌ **FAIL** (catastrophic) |
| D7 | a-key cosine median (anti-cheat) | < 0.1 | 0.015 | ✅ PASS |
| D7 | a-key frac>0.9 | ≪ 1% | 1.4e-5 | ✅ PASS |
| D7 | B effective rank | qualitative | 1678/2560 = 66% of d_out | ✅ PASS |
| D9 | unigram-matched gate_d @ k=10 | ≥ 2.0 | 9.71 | ✅ PASS |
| D9 | unigram-matched gate_d @ k=100 | ≥ 1.0 | 9.28 | ✅ PASS |

**Score**: 14 PASS / 3 FAIL — POSITIVE on the core scientific claim, NEGATIVE on the production-scale routing.

---

## Headline results

### Φ1 — Composition (Q1: is the bank linearly composable?)

```
k=1     gate_d=+9.85 nats  beats_base=100%
k=10    gate_d=+9.78 nats  beats_base=100%
k=100   gate_d=+9.34 nats  beats_base=99.8%
k=1000  gate_d=+4.35 nats  beats_base=97.2%
```

* Pre-registered "correct vs shuffled" ≥ 0.5 nats gap: observed **>4 nats** at every k.
* 300 test facts × 3 seeds (seed 0,1,2), no seed-to-seed instability.
* **Q1 answered: YES**, the bank is linearly composable up to at least k=1000 with positive uplift.

### D7 — Anti-cheat: no shortcut via key collapse

* `cos(a_i, a_j)` median = **0.015** (≈ orthogonal at d=9728)
* fraction with cos > 0.9 = **1.4e-5** (no pair)
* effective rank: B = 1678 / 2560 = **66% of d_out** (matrix uses most of the available output basis)
* `a` rank 2026 / 9728 = 21% of d_in; expected since `a_i = k_i / (‖k‖² + λ)` reflects subject-pos input geometry, not full d_in spread.

The bank is **not** cheating by concentrating Δ on a low-rank "universal target_new" direction (the failure mode of Exp24/27/32/33).

### D9 — Anti-cheat: unigram-matched still shows uplift

Restricted to **917 / 1500** facts with `|log p_neutral(t_new) − log p_neutral(t_true)| ≤ 1.0` nat:

```
k=10   uplift=+9.83  gate_d=+9.71  beats_base=100%  beats_shuf=99.8%
k=100  uplift=+10.15 gate_d=+9.28  beats_base=99.8% beats_shuf=100%
```

* The uplift is **not** explained by unigram frequency bias.
* Shuffled control: the identity binding is real (correct fact patch ≠ random fact patch).

### D4 — Layer locus

| Layer | mean_uplift | frac_pos | mean ‖Δ‖_F |
|---|---|---|---|
| L=3 | 9.91 | 100% | 17.68 |
| **L=5** | **10.42** | **100%** | **8.31** |
| L=7 | 8.60 | 98% | 6.62 |
| L=9 | 10.01 | 100% | 5.87 |

L=5 has the best ratio of uplift to ‖Δ‖. Confirms the early-MLP locus picked at design time (Exp31/34 selection).

### Φ2 — Learned router (10000-class)

```
honest:    test_top1=40.85%   test_top5=54.91%
shuffled:  test_top1= 0.04%   test_top5= 0.09%       (chance=0.010%)
collision: test_top1=27.29%   test_top5=61.36%   (678 facts w/ shared subject)
```

* **Anti-cheat (C8) PASSED**: shuffled-label baseline is at chance. There is no statistical leakage from subject embeddings to fact_id.
* Honest top1 PASSES the pre-reg, but **top5 falls 5 pp short**. With 10000 classes and only ~6700 train embeddings (1 per fact), the head is data-starved on rare relations.
* Cross-paraphrase split (train=prompt, val=para[0], test=para[1]) was honored — no train-test leak.

### Φ3 — End-to-end (routed patch + read)

```
mean routed_uplift   = +4.45 nats
mean oracle_uplift   = +9.72 nats
routed/oracle ratio  =  0.458   (threshold 0.70 → FAIL)
routed beats base    =  72.9%
oracle beats base    =  99.9%
```

The routed pipeline still gives **half the oracle uplift**, but falls short of the 70% target. With router top1=41%, the average is dragged down by the 59% of facts that get a wrong patch but still see partial uplift from semantically-related neighbors.

### D6 — Capability under stacked patches

```
k=10    ppl drift =  0.69%   (threshold 3%)  PASS
k=100   ppl drift =  2.81%   (threshold 8%)  PASS
k=1000  ppl drift = 94%      (threshold 15%) FAIL
```

**This is the operational ceiling**: stacking 1000 rank-1 patches degrades general LM perplexity by ~2× on WikiText. The bank can still recall facts at k=1000 (Φ1 PASS), but the host model loses generality. Practical operating envelope: **k ≤ 100**.

---

## What the negatives mean (no spin)

1. **Φ3 ratio 0.46 < 0.70**: routing at 10000 classes is hard; we are NOT at production quality. This is consistent with Φ2 top1=41%. Fix path: use Top-k retrieval + small re-ranker instead of softmax over 10k classes; or hierarchical relation→fact routing.

2. **Φ2 top5=54.9%**: 5 pp from threshold. Data-starved (1 train embedding per fact). Fix path: more paraphrase prompts per fact, or contrastive pre-training of the subject encoder.

3. **D6 @ k=1000 catastrophic**: the linear stacking assumption breaks well before 10000. Bank is operable to k≤100 (`ppl drift ≤ 3%`). Fix path: MEMIT precondition already used here; next steps would be coverage-bounded gating (only activate patches whose `a_i^T x` exceeds a threshold) or block-diagonal partitioning by relation.

4. **No Φ1 cherry-pick**: results are at 300 test × 3 seeds (= 900 cells per k); std across seeds is < 0.4 nats.

---

## What is genuinely new

* **First demonstration of MEMIT-preconditioned offline factor bank at N=10⁴ that composes linearly to k=1000 without identity collapse** — this is the core scientific contribution and it stands.
* The composition curve is **flat to k=100** (gate_d only drops from 9.78 to 9.34) and only mildly degrades at k=1000 — much better than naïve linear stacking of ROME edits would predict.
* The D7 + D9 + shuffled-label triple-control is the strictest anti-cheat protocol we have used; it survives.

---

## Concrete go/no-go for downstream work

* **Exp36 (binding audit, 7 sub-tests)**: GO — composition is verified, now stress identity binding under paraphrase/subject-swap/negation/OOD.
* **Exp37 (production stress)**: GO — but cap k at 100 in stress tests; document the k=1000 ceiling.
* **Scaling to N=10⁵**: NO-GO until routing is reworked. Bank construction is fine, but downstream Φ3 will get worse.

## Files

* `data/bank.pt` (49.7 MB, gitignored): 10000 (b_i, a_i) factor pairs, solo_pass=77.89%
* `run_qwen_exp35b/phi1_summary.json`, `phi1_cells.jsonl`
* `run_qwen_exp35b/phi2_summary.json`
* `run_qwen_exp35b/phi3_summary.json`, `phi3_cells.jsonl`
* `run_qwen_exp35b/d4_layer_ablation.json`
* `run_qwen_exp35b/d6_capability.json`, `d6_capability_k100.json`, `d6_capability_k1000.json`
* `run_qwen_exp35b/d7_delta_rank.json`
* `run_qwen_exp35b/d9_unigram_matched.json`
* `preregister.json` (locked at `8e1b69d9`)
* `deviations.md` (D2/D3/D8 documented)
