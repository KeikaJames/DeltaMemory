# Cleanroom Reports — Index

## How to read this index

Each row points at a single headline report file (`REPORT.md`,
`AGGREGATE.md`, `FINDINGS.md` or lower-case `report.md`) and summarises
its own conclusion in one line. The **Verdict** column is taken
verbatim from each report's stated pass/fail; when a report does not
declare a clear PASS/FAIL we use **MIXED** (some hypotheses pass, others
fail) or **N/A** (data dump / preregistration / training log without a
verdict). Dates are the latest commit touching that file. Nothing here
overrides the source report — when in doubt, follow the link.

### Phase R+ canonical (where to start reading today)

The canonical Phase R verdict lives in
[`lopi_v33/FINDINGS.md`](lopi_v33/FINDINGS.md) and
[`lopi_v33/AGGREGATE.md`](lopi_v33/AGGREGATE.md) (the 630-cell ablation
sweep). Cross-architecture follow-up is in
[`lopi_v33/R4_xarch/REPORT.md`](lopi_v33/R4_xarch/REPORT.md). The Phase
R-5 adversarial chat pilot is in
[`lopi_v33/R5_q3/REPORT.md`](lopi_v33/R5_q3/REPORT.md). There is no
`R6_storage/` artefact yet; persistent-bank work is referenced as future
work in R-5. Phase S (U-LOPI auto-calibration profiler) lands in commit
`aa3825a9` — see [`deltamemory/memory/lopi_profiler.py`](../../deltamemory/memory/lopi_profiler.py);
no Phase S report has been written yet.

---

## Phase S (U-LOPI, in progress)

* Commit: **`aa3825a9`** — `feat(S): U-LOPI auto-calibration profiler + 4 latent-bug fixes` (2026-05-04).
* Code entry point: [`deltamemory/memory/lopi_profiler.py`](../../deltamemory/memory/lopi_profiler.py).
* Status: **S-7 cross-arch re-sweep on MPS pending.** No INDEX-level
  Phase S report yet; this row exists so future readers know where the
  next set of cleanroom artefacts will land.

---

## Phase Q — Flagship verification (`flagship_v32/`, gemma4_delta-class)

| Report | Phase | Date | Headline | Verdict | Path |
|---|---|---|---|---|---|
| Phase Q — mHC-Mneme Flagship Verification (v3.2) | Q | 2026-05-04 | Shield V2 column-cap eliminates V1 collapse on 4 models, but cannot overcome per-arch V-magnitude differences; non-Gemma drift floor 1.5–3.5 nats. | MIXED | [flagship_v32/REPORT.md](flagship_v32/REPORT.md) |
| Phase Q — Preregistration (5 hypotheses H1–H5) | Q | 2026-05-04 | Frozen prereg for the Q1/Q2/Q3 verification protocol. | N/A | [flagship_v32/PREREGISTRATION.md](flagship_v32/PREREGISTRATION.md) |
| Phase Q2 — α-Safety NLL/Lift Sweep (Gemma-4-E2B) | Q2 | 2026-05-04 | Shield ON @ α=10 yields lift +2.84 nats with drift +0.17 — H1+H2 PASS on Gemma. | PASS | [flagship_v32/Q2/REPORT.md](flagship_v32/Q2/REPORT.md) |
| Phase Q2 — Aggregate (4 models × 7 α × shield) | Q2 | 2026-05-04 | Cross-arch sweep: Qwen3 / GLM-4 / DeepSeek-32B drift floors persist with shield ON. | MIXED | [flagship_v32/Q2/AGGREGATE.md](flagship_v32/Q2/AGGREGATE.md) |
| Phase Q3 — Adversarial Memory-Implant Chat (Gemma-4-E2B) | Q3 | 2026-05-04 | Logprob lift > 0 at α∈{1,5} on 5/5 facts, but greedy-decode "accurate implant" 0/5 at α=1.0 — H3 NOT YET MET. | FAIL | [flagship_v32/Q3/REPORT.md](flagship_v32/Q3/REPORT.md) |

## Phase R — LOPI v3.3/v3.4 (`lopi_v33*`)

| Report | Phase | Date | Headline | Verdict | Path |
|---|---|---|---|---|---|
| Phase R — Dynamic LOPI v3.3 Preregistration | R | 2026-05-04 | Frozen prereg: training-free orthogonal-projection injection to close the mHC drift gap. | N/A | [lopi_v33/PREREGISTRATION.md](lopi_v33/PREREGISTRATION.md) |
| Phase R-2 LOPI smoke (GPT-2 small, MPS bf16) | R-2 | 2026-05-04 | SMOKE PASS — α=0 bit-equal across all 5 LOPI variants; plumbing validated at one cell. | PASS | [lopi_v33_smoke/REPORT.md](lopi_v33_smoke/REPORT.md) |
| Phase R-3 — 630-cell LOPI ablation (Aggregate) | R-3 | 2026-05-04 | H5/H2/H3 PASS; H1 (drift collapse via M⊥) FAIL; post-hoc H1′ (Gaussian shield at high-α) PASS on 4/6 cells. | MIXED | [lopi_v33/AGGREGATE.md](lopi_v33/AGGREGATE.md) |
| Phase R-3 — Findings (executive write-up) | R-3 | 2026-05-04 | Strike 1 against as-specified Dynamic LOPI v3.3; v3.4 default flips `orthogonal=False`, retains Gaussian focusing. | MIXED | [lopi_v33/FINDINGS.md](lopi_v33/FINDINGS.md) |
| Phase R-3.5 — LOPI Gaussian Layer-Norm Probe | R-3.5 | 2026-05-04 | Mean per-layer relative perturbation drops 0.690 → 0.110 (−84.0%) under A4 (Gauss + γ). | PASS | [lopi_v33/R35_NORM_PROBE.md](lopi_v33/R35_NORM_PROBE.md) |
| Phase R-4 — Cross-arch LOPI α-Safety Sweep | R-4 | 2026-05-04 | All 12 α=0 cells bit-equal; LOPI roughly halves drift on Qwen3/GLM-4 but cannot bring Qwen3 under 0.5 nats. | MIXED | [lopi_v33/R4_xarch/REPORT.md](lopi_v33/R4_xarch/REPORT.md) |
| Phase R-4 — Cross-arch paired aggregate | R-4 | 2026-05-04 | Per-(model, α) paired tables for shield × LOPI configurations. | N/A | [lopi_v33/R4_xarch/AGGREGATE.md](lopi_v33/R4_xarch/AGGREGATE.md) |

## Phase R-5 adversarial chat (`lopi_v33/R5_q3/`, flagship_q3-class)

| Report | Phase | Date | Headline | Verdict | Path |
|---|---|---|---|---|---|
| Phase R-5.1 — Q3 Adversarial Chat × LOPI (Gemma-4-E2B pilot) | R-5.1 | 2026-05-04 | Shield+LOPI lifts a single partial implant (0/5 → 1/5) at high α, but accurate-implant threshold not crossed on the 5-fact pilot. | FAIL | [lopi_v33/R5_q3/REPORT.md](lopi_v33/R5_q3/REPORT.md) |

## Phase mHC — Manifold/HC ablations (`mHC*`)

| Report | Phase | Date | Headline | Verdict | Path |
|---|---|---|---|---|---|
| mHC Mneme α-Safety (mHC0–mHC6 synthesis) | mHC0–6 | 2026-05-04 | Multi-stream HC/mHC preserves lift advantage at depth (+4.13 nats GPT-2 medium @ α=1.0); H2 drift safety NOT delivered by mHC alone. | MIXED | [mHC_alpha_safe/REPORT.md](mHC_alpha_safe/REPORT.md) |
| Phase mHC2 — V-perturbation α-NLL stability | mHC2 | 2026-05-03 | H1 PASS (residual ΔNLL +5.08 nats @ α=1); H2 FAIL on absolute threshold but H2-revised (mHC strictly more stable than residual) PASS; H3 indeterminate at equivalence init. | MIXED | [mHC2_perturbation/REPORT.md](mHC2_perturbation/REPORT.md) |
| Phase mHC3 — Mneme bank injection into 3-Arm GPT-2 (corrected) | mHC3 | 2026-05-04 | Amendment 1: corrected sequence-NLL drift. Multi-stream preserves +0.071 nats lift @ α=1, but amplifies neutral drift (+2.26 vs Residual +0.70). | MIXED | [mHC3_bank_injection/REPORT.md](mHC3_bank_injection/REPORT.md) |
| Phase mHC3 — Legacy single-token results (superseded) | mHC3 | 2026-05-04 | Pre-amendment single-token drift metric; preserved for audit only. | N/A | [mHC3_bank_injection/REPORT_legacy_singletok.md](mHC3_bank_injection/REPORT_legacy_singletok.md) |

## Phase 0..H legacy (`stage13*` / `stage14*` / `stage15*`, pre-Q)

| Report | Phase | Date | Headline | Verdict | Path |
|---|---|---|---|---|---|
| Stage 13B robustness (full) — AttentionNative Mneme bank | 13B | 2026-05-03 | Paraphrase recall@1 = 0.003 (FAIL @ 0.7); LORO macro recall@1 = 0.000 (FAIL @ 0.5); decoy diagnostic PASS. | MIXED | [stage13b_robust/report.md](stage13b_robust/report.md) |
| Stage 13B robustness (smoke) | 13B-smoke | 2026-05-03 | Smoke run: same gate structure, paraphrase recall@1 = 0.200 (FAIL); decoy PASS. | MIXED | [stage13b_smoke/report.md](stage13b_smoke/report.md) |
| Stage 13C — Writer-layer feature decoupling (SVD/ROME) | 13C | 2026-05-03 | Best held-out recall@1 = 0.184 at r=4; pass criterion (≥0.55) FAIL. | FAIL | [stage13c_writer_decouple/report.md](stage13c_writer_decouple/report.md) |
| Stage 13D — Per-query routing fix for P3 locality drift | 13D | 2026-05-03 | Status: PASS. Drift ≤ 0.05 AND override ≥ 0.90 achieved with gated soft routing. | PASS | [stage13d_locality_fix/report.md](stage13d_locality_fix/report.md) |
| Stage 13F — Interactive transcripts (Mneme) | 13F | 2026-05-03 | Pass rate 1/6 — confirmed and refined the Stage 13B negative result; surfaced as honest evidence. | FAIL | [stage13f_interactive/REPORT.md](stage13f_interactive/REPORT.md) |
| Stage 14 dev sweep (Gemma-4-E2B, no kproj) | 14-dev | 2026-05-03 | All v3 conditions collapse to recall@1=0 vs B0=0.354. | FAIL | [stage14_dev/REPORT.md](stage14_dev/REPORT.md) |
| Stage 14 dev sweep — with trained InfoNCE K-projector | 14-dev-kproj | 2026-05-03 | v3_period_kproj recall@1 = 0.4343 vs B0 0.3535 (+8.1pp, paired-significant on dev only). | MIXED | [stage14_dev_kproj/REPORT.md](stage14_dev_kproj/REPORT.md) |
| Phase G — Held-out Test Eval (Gemma-4-E2B, FROZEN) | G | 2026-05-03 | v3_period_kproj recall@1 = 0.276 vs B0 0.359 / B1 0.658 — does not clear prompt-insertion bar; dev/test sign-flip documented. | FAIL | [stage14_test_gemma4_e2b/REPORT.md](stage14_test_gemma4_e2b/REPORT.md) |
| Phase L4-rev — v3.1 architectural sweep | L4-rev | 2026-05-03 | Baseline recall@1 = 0.5585; bank_topk / τ / cosine variants strictly worse; separate-softmax β=1 collapses (−47.1pp). | MIXED | [stage15_arch_sweep/REPORT.md](stage15_arch_sweep/REPORT.md) |
| Phase L4-rev — sub-config: baseline | L4-rev | 2026-05-03 | recall@1 = 0.5585 at τ=1.0, bank_topk=0. | N/A | [stage15_arch_sweep/baseline/REPORT.md](stage15_arch_sweep/baseline/REPORT.md) |
| Phase L4-rev — sub-config: cosine | L4-rev | 2026-05-03 | recall@1 = 0.5415 / 0.5390 / 0.5195 at τ ∈ {0.07, 0.1, 0.2}. | N/A | [stage15_arch_sweep/cosine/REPORT.md](stage15_arch_sweep/cosine/REPORT.md) |
| Phase L4-rev — sub-config: sep_b05 | L4-rev | 2026-05-03 | recall@1 = 0.5244 (separate-softmax, β=0.5). | N/A | [stage15_arch_sweep/sep_b05/REPORT.md](stage15_arch_sweep/sep_b05/REPORT.md) |
| Phase L4-rev — sub-config: sep_b1 | L4-rev | 2026-05-03 | recall@1 = 0.0878 (separate-softmax, β=1.0 — collapse). | N/A | [stage15_arch_sweep/sep_b1/REPORT.md](stage15_arch_sweep/sep_b1/REPORT.md) |
| Phase L4-rev — sub-config: sep_cos | L4-rev | 2026-05-03 | recall@1 = 0.1024 (separate-softmax, cosine, τ=0.1). | N/A | [stage15_arch_sweep/sep_cos/REPORT.md](stage15_arch_sweep/sep_cos/REPORT.md) |
| Phase L4 dev eval — Gemma-4-E2B (v3.1 K-projector) | L4 | 2026-05-03 | v3.1 period_kproj recall@1 = 0.5585 (+20.7pp over B0 = 0.351); −7.8pp vs B1 prompt-insertion oracle (0.6366). | MIXED | [stage15_dev_v31/REPORT.md](stage15_dev_v31/REPORT.md) |
| Phase L4 dev eval — SUMMARY (reading guide) | L4 | 2026-05-03 | Recovers ~73% of prompt-insertion gain without putting the fact in context; v2-only collapses. | MIXED | [stage15_dev_v31/SUMMARY.md](stage15_dev_v31/SUMMARY.md) |
| Stage 15 — v3.1 Phase L2 + N preliminary findings | L2 | 2026-05-03 | K-projector training summary: loss 3.045 → 1.052 over 8 epochs on 1,464 pairs; sha pinned. | N/A | [stage15_kproj_v31/REPORT.md](stage15_kproj_v31/REPORT.md) |
| Phase L4-rev — repro check (bank_topk = 0) | L4-rev | 2026-05-03 | recall@1 = 0.5537 (single-cell repro). | N/A | [stage15_repro_check/REPORT.md](stage15_repro_check/REPORT.md) |
| Phase L4-rev — τ sweep | L4-rev | 2026-05-03 | recall@1 vs τ ∈ {0.25 … 1.5}: peak 0.5585 at τ=1.0. | N/A | [stage15_tau_sweep/REPORT.md](stage15_tau_sweep/REPORT.md) |
| Phase L4-rev — bank_topk sweep | L4-rev | 2026-05-03 | recall@1 vs bank_topk ∈ {0,1,2,4,8}: peak 0.5585 at bank_topk=0; sharper top-k strictly worse. | N/A | [stage15_topk_sweep/REPORT.md](stage15_topk_sweep/REPORT.md) |
| Phase L4-rev — val2 gate | L4-rev | 2026-05-04 | val2_v31 recall@1 ≈ 0.50 across 3 seeds (gate sample). | N/A | [stage15_val2_gate/REPORT.md](stage15_val2_gate/REPORT.md) |

## Misc

_(No additional report files outside the groups above. `figures/` holds
PNG/SVG artefacts only; `_dl_logs/` holds raw runner logs.)_
