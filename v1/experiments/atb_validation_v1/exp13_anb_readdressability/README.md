# Exp13–22 — ANB attention-trace re-addressability (MPS smoke results)

Branch: `feat/exp13-anb-readdressability`
Substrate: Qwen3-4B-Instruct-2507, MPS bf16, n=21 eligible CounterFact rows × seeds {0,1,2} (Exp21 uses {0..5}).

All runs honor the authenticity contract (`env.json`, `cells.jsonl`,
`tools/check_authenticity.py` PASS).  Full pytest at branch start: **522 PASS / 10 SKIP / 0 FAIL**.

## Verdict chain

| Exp | Question | Verdict | Key evidence |
|---|---|---|---|
| 13 | Is M_K naturally re-addressable by future Q? | **ADDRESSABILITY_STRONG** | `subject_last` × `subject_only` recall@5 = 1.000, +15.23 vs shuffle_layer (CI [+12.8, +17.6]) |
| 14 | Does oracle (K,V) injection beat matched controls? | **ORACLE_DIRECTIONAL** | At α=0.005, oracle_correct_KV vs base Δ +2.23 (CI [+1.56, +3.00]); vs shuffled_layer +2.09 (CI [+1.44, +2.85]); vs random_KV +0.44 (CI [−0.29, +1.13]) |
| 15 | Is the effect K-conditional or V-only at whole stack (subject_last)? | **BINDING_DIRECTIONAL** | V-causality Kc_Vc−Kc_Vr +0.24 (CI [+0.02, +0.55]) **PASS**; K-causality Kc_Vc−Kr_Vc +0.00 (CI [−0.75, +0.66]) **null** at subject_last |
| 16 | Does K-causality recover in a layer subset? | **SITE_DIRECTIONAL** | Q3 (layers 18–26) K-causality +0.07 (CI [+0.001, +0.143]) — first stratum with CI-positive K-causality |
| 17 | Across 9 capture sites, which yield K-causality at whole stack? | **SWEEP_DIRECTIONAL** | `relation_last` & `subject_relation_pair` K-causal (CI [+0.09, +1.31]); `subject_last` V-causal (CI [+0.02, +0.55]); no site simultaneously K&V causal |
| 18 | Does multi-slot natural bank route to correct slot? | **NATURAL_FAIL** | full_bank beats base +3.58 (CI [+2.85, +4.28]) BUT does NOT beat random_K (−0.34, CI [−0.68, −0.00]) nor minus_correct (−0.40, CI [−0.85, +0.04]). Bank-softmax produces generic steering, not addressing |
| 19 | Does oracle binding survive paraphrase? | **BINDING_WEAK_DIRECTIONAL** | Kc_Vc vs base CI [+0.18, +0.84] holds; V-causality CI [−0.02, +0.07] and K-causality CI [−0.02, +0.27] both cross zero |
| 20 | How does K-causality scale with bank size? | **SCALE_K_PARTIAL** | k=1 K-causality CI [+0.10, +1.64]; k=3 CI [−1.61, −0.19] (flips); k=7 CI [+0.08, +1.35] (recovers). Non-monotonic — too noisy at n=21 |
| 21 | Does K-causality replicate with more seeds? | **BINDING_DIRECTIONAL** | 6-seed @ relation_last: K-causality CI [+0.09, +0.92] **stable**; V-causality CI [−0.12, +0.02] null. Kc_Vc vs base CI [+2.21, +3.34] |
| 22 | Qualitative synthesis (this section) | **see below** | — |

## What this rules in / out

**Ruled in (replicated across Exp13/15/17/21):**
- **Natural QK addressability is structurally real.** When capture site is chosen well (`subject_last`, `subject_only` query), the correct write-time K slot ranks at recall@5 = 1.000 from the read-time Q — without any injection.
- **K-routing is causal at `relation_last`.** Across Exp17 (3 seeds) and Exp21 (6 seeds), Kc_Vc − Kr_Vc has a paired CI strictly > 0 at the `relation_last` capture site. This is the first repeatedly-positive K-causality signal in the entire ANB program.
- **V-routing is causal at `subject_last`.** Kc_Vc − Kc_Vr has a paired CI strictly > 0 at `subject_last`. V-channel identity also matters, but at a different site than K.
- **Capture site governs which channel is causal.** No site is both K- and V-causal at whole stack: the routing function is *site-stratified*.

**Ruled out (or strongly downgraded):**
- **Whole-stack natural bank-softmax addressing (Exp18, NATURAL_FAIL).** Even with the correct slot present, the natural multi-slot bank does not route preferentially to the correct slot. Random-K and minus-correct controls match or beat the full bank in mean margin. The downstream uplift from any bank presence is **steering**, not addressing.
- **K/V binding under paraphrase (Exp19).** Once the read query is paraphrased, both K- and V-causality CIs cross zero — only the oracle vs base contrast survives. Binding is not paraphrase-robust at α=0.005.
- **Monotonic K-causality decay with bank size (Exp20).** K=3 flipped negative while K=1 and K=7 stayed positive; noise at n=21 dominates, so the bank-size scaling story remains undetermined.

## Exp22 — Qualitative synthesis

The 10-stage ladder produced one consistent and falsifiable conclusion:

> **The ATB attention trace IS re-addressable in principle (Exp13/17/21), but the natural bank-softmax injection path does NOT exercise that addressability (Exp18). At whole-stack injection, the gain comes from generic steering, and the K-channel only earns its keep at one specific capture site (`relation_last`) with a non-trivial but small effect (~+0.5 logit margin).**

Two competing hypotheses survived this round:

1. **Site-specific oracle injection** (`relation_last` K, `subject_last` V) is the real ATB pathway. The "natural" multi-slot bank fails because bank-softmax mass is dominated by distractors at sites where Q does not discriminate sharply. Remedy: per-site bank with hard-gated keys.
2. **The ATB read-side architecture is mismatched to the write-side.** Re-addressability proves the K bank holds discriminative information, but Qwen's attention softmax cannot exploit it without a stronger key-projection. Remedy: a learned key adapter at read time.

Both hypotheses are testable on GPU at full n=807. Within MPS smoke, the ladder cannot distinguish them.

## Open work (GPU-deferred)

1. **Full-n confirmation** of K-causality at `relation_last` (Exp15/17/21 re-run at n≈807 × 3 seeds).
2. **Per-site bank** at `relation_last` only — does isolating K-causal site eliminate the NATURAL_FAIL?
3. **Cross-model replication** on Gemma-4 / Llama-3 — does site-stratification hold across architectures?
4. **Learned key adapter** at read time — closes the read/write architecture gap if hypothesis 2 is correct.

## Files

```
PREREG.md                  # frozen prereg
run_addressability.py      + run_mps_smoke/         VERDICT=ADDRESSABILITY_STRONG
run_oracle_addressed.py    + run_mps_exp14_smoke/   VERDICT=ORACLE_DIRECTIONAL
run_kv_binding.py          + run_mps_exp15_smoke/   VERDICT=BINDING_DIRECTIONAL  (subject_last)
run_site_map.py            + run_mps_exp16_smoke/   VERDICT=SITE_DIRECTIONAL
run_capture_sweep.py       + run_mps_exp17_smoke/   VERDICT=SWEEP_DIRECTIONAL
run_natural_addressed.py   + run_mps_exp18_smoke/   VERDICT=NATURAL_FAIL
run_paraphrase.py          + run_mps_exp19_smoke/   VERDICT=BINDING_WEAK_DIRECTIONAL
run_bank_scaling.py        + run_mps_exp20_smoke/   VERDICT=SCALE_K_PARTIAL
                           run_mps_exp21_smoke/     VERDICT=BINDING_DIRECTIONAL  (relation_last, 6 seeds)
analyze.py / analyze_exp14.py / analyze_exp15.py / analyze_exp16.py
analyze_exp17.py / analyze_exp18.py / analyze_exp20.py / analyze_kv_2x2.py
```
