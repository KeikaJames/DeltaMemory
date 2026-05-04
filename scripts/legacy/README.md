# Archived legacy one-shot scripts

These scripts produced pre-Phase-Q artefacts and are kept here only for
historical / archeological reference. They are **not** invoked by:

- `tests/`
- `repro_v3.sh` / `repro_v31.sh`
- `scripts/run_v31_benchmark*.py`
- `scripts/run_mhc_flagship_sweep.py`
- `scripts/run_lopi_*` (Phase R / S)

For canonical reproduction use the top-level `repro_v3.sh` (Stage 14
v3 pipeline) or `repro_v31.sh` (v3.1 counter-prior intervention demo).

| Script | Purpose | Pinning report |
|---|---|---|
| `run_clean_demo.py` | Early Delta Memory attention-injection demo (mock-Gemma default). | _none_ (developer demo) |
| `run_stage6.py` | Stage 6 orchestrator: token-preserving writer + 3-channel sweep (synthetic + LAMA Phase 2). | `reports/experiments/stage6_phase1`, `stage6_phase2_lama` |
| `run_stage7a_diagnostic.py` | Stage 7A payload-identity diagnostic over span × layer × pool. | `reports/experiments/stage7a_*` |
| `run_stage8.py` | Stage 8 closed-book memory test (writer + address bank + read pass). | `reports/experiments/stage8_*` |
| `run_stage8_interference.py` | Stage 8.3 sequential-write interference / retention curve. | `reports/experiments/stage8_v3_interference_n1024_seed0` |
| `run_stage8_rag_baseline.py` | Stage 8.5 head-to-head vs RAG baselines. | `reports/experiments/stage8_v3_rag_n4096_seed0` |
| `run_stage9_baselines.py` | Stage 9-C head-to-head baselines vs Delta Memory. | `reports/experiments/stage9*`, `stage10C_*` |
| `run_stage9_sweep.sh` | Stage 9 sweep driver (encoder pool variants, GB10 CUDA bf16). | `reports/experiments/stage9A_*`, `stage9_grand_evaluation` |
| `run_stage10_resume.sh` | Stage 10 resume harness for 10F (LORO) + 10C baselines. | `reports/experiments/stage10F_loro_*`, `stage10C_*` |
| `run_stage10_sweep.sh` | Stage 10 adversarial-validation sweep (10A paraphrase + variants). | `reports/experiments/stage10ABD_*`, `stage10_adversarial_validation` |
| `run_stage11_conv.py` | Stage 11D conversational memory benchmarks for DeltaMemory. | `reports/experiments/stage11_grand_evaluation` |
| `run_stage11_sweep.sh` | Stage 11 sweep (11A paraphrase InfoNCE, 11B LORO, 11C re-run, 11D conv). | `reports/experiments/stage11A_*`, `stage11B_loro_*` |
| `reproduce_stage11.sh` | Stage 11 deterministic reproduction harness (bit-exact double-run + SHA-256 compare). | `reports/experiments/stage11_grand_evaluation` |
| `run_stage12_multimodel.py` | Stage 12 multi-model cross-architecture adversarial validation. | `reports/experiments/stage12_gemma4_e2b`, `stage12_cross_model_summary.json` |
| `run_stage13b_robust.py` | Stage 13B robustness benchmarks on the AttentionNative DeltaMemory bank. | `reports/cleanroom/stage13b_robust`, `stage13b_smoke` |
| `run_stage13c_writer_decouple.py` | Stage 13C writer-layer feature decoupling via SVD/ROME on AttnNativeBank M_V. | `reports/cleanroom/stage13c_writer_decouple` |
| `run_stage13d_locality_fix.py` | Stage 13D per-query routing fix for P3 locality drift. | `reports/cleanroom/stage13d_locality_fix` |
| `run_stage13f_interactive.py` | Stage 13F interactive transcripts with DeltaMemory injection. | `reports/cleanroom/stage13f_interactive` |

Stage 14 (`run_stage14_dev_sweep.py`, `run_stage14_dev_with_kproj.py`,
`run_stage14_test_eval.py`, `train_k_projector.py`) is **not** archived
because `repro_v3.sh` still drives it end-to-end.
