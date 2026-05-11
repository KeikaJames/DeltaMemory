# Exp13–16 — ANB attention-trace re-addressability (MPS smoke results)

Branch: `feat/exp13-anb-readdressability`
Substrate: Qwen3-4B-Instruct-2507, MPS bf16, n=21 eligible CounterFact rows × seeds {0,1,2}.

All four runs honor the authenticity contract (`env.json`, `cells.jsonl`,
`tools/check_authenticity.py` PASS).  Full pytest: **522 PASS / 10 SKIP / 0 FAIL**.

## Verdict chain

| Exp | Question | Verdict | Key evidence |
|---|---|---|---|
| 13 | Is M_K naturally re-addressable by future Q? | **ADDRESSABILITY_STRONG** | `subject_last` × `subject_only` recall@5 = 1.000, +15.23 vs shuffle_layer (CI [+12.8, +17.6]), +58.97 vs random_K |
| 14 | Does oracle (K,V) injection beat matched controls in margin? | **ORACLE_DIRECTIONAL** | At α=0.005 oracle_correct_KV beats every control in mean (Δ vs base +2.23 CI [+1.56, +3.00]); strong vs shuffled_layer (+2.09 CI [+1.44, +2.85]); weak vs random_KV (+0.44 CI [-0.29, +1.13]) |
| 15 | Is the effect K-conditional or V-only at whole stack? | **BINDING_DIRECTIONAL** | V-causality Kc_Vc−Kc_Vr = +0.24 (CI [+0.02, +0.55]) **PASS**; K-causality Kc_Vc−Kr_Vc = +0.00 (CI [−0.75, +0.66]) **null** |
| 16 | Does K-causality recover in a layer subset? | **SITE_DIRECTIONAL** | Q3 (layers 18–26) Kc_Vc−Kr_Vc = +0.07 (CI [+0.001, +0.143]) → first stratum with CI-positive K-causality; Q1 (0–8) drives most of the margin uplift but K/V are indistinguishable there |

## What this rules in / out

**Ruled in:**
- Natural QK addressing works structurally — Q at read time can rank the correct fact's K bank slot near-perfectly when the capture site is `subject_last`.
- Injecting the correct V payload at small α causes a real downstream margin shift vs base; this shift is reproducible across seeds and dominates the early-layer (Q1) effect.
- There is a non-trivial mid-late layer band (Q3, 18–26) where correct-K specifically beats random-K with paired CI > 0 — i.e. K-channel identity matters there.

**Ruled out (at this n):**
- Whole-stack K-gating: at α=0.005 injecting random-K with correct-V is indistinguishable from injecting correct-KV.  The "natural" Exp18 ANB cannot rely on whole-stack K-routing alone.

**Open:**
- Whether Q3-only K-causality is real or an n=63 artifact.  Needs GPU full-n (n≈807 × 3 seeds) to tighten CIs.
- Whether Exp17 capture sweep can find a site (e.g. `subject_relation_pair`, `object_first`) where K-causality holds at the whole stack.

## Files

```
PREREG.md                                # frozen prereg (Exp13 ladder)
run_addressability.py    + run_mps_smoke/           VERDICT=ADDRESSABILITY_STRONG
run_oracle_addressed.py  + run_mps_exp14_smoke/     VERDICT=ORACLE_DIRECTIONAL
run_kv_binding.py        + run_mps_exp15_smoke/     VERDICT=BINDING_DIRECTIONAL
run_site_map.py          + run_mps_exp16_smoke/     VERDICT=SITE_DIRECTIONAL
analyze.py               # Exp13 analyzer
analyze_exp14.py
analyze_exp15.py
analyze_exp16.py
```

## Next steps (GPU-deferred)

1. **Exp17 capture sweep** — extend Exp13 across all sites
   (`subject_first`, `subject_last`, `relation_*`, `object_*`,
   `subject_relation_pair`, `subject_object_pair`, `full_content`,
   `all_content_sparse`).  Looking for a capture site that yields
   `BINDING_PASS` at whole-stack injection.
2. **Exp18 natural addressed ANB** — only if Exp17 finds a viable site or
   the Q3-only K-causality replicates with full n.
3. **Full-n on GPU** — re-run Exp13–16 with n≈807 × 3 seeds to upgrade
   `_DIRECTIONAL` verdicts where possible.
