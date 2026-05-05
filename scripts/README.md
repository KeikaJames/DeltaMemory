# scripts/

This directory contains the **currently-used** orchestration scripts for the
Mneme project. Pre-Phase-Q one-shot scripts have been archived under
[`scripts/legacy/`](legacy/) — see that directory's README for the historical
inventory.

## Active scripts (Phase Q+ / v3 / v3.1)

| Script | Purpose |
|---|---|
| `run_mhc_flagship_sweep.py` | Phase Q+ flagship MHC sweep (canonical entry-point). |
| `run_mHC3_bank_injection.py` | MHC stage-3 bank-injection experiment. |
| `run_mHC5_layer_norms.py` | MHC stage-5 layer-norm probe. |
| `run_v31_benchmark.py` | v3.1 benchmark harness. |
| `run_v31_topk_sweep.py` | v3.1 top-k sweep. |
| `run_r7_vscale_smoke.py` | Phase R7 v-scale smoke test. |
| `run_ulopi_xarch_smoke.py` | U-LOPI cross-architecture smoke test. |
| `run_flagship_q3_adversarial_chat.py` | Phase Q3 adversarial-chat flagship. |
| `run_delta_experiment.py` | Delta-experiment runner. |
| `make_delta_experiment_matrix.py` | Build the delta-experiment configuration matrix. |
| `train_delta_qv_prototype.py` | Train the Δ-QV prototype. |
| `train_k_projector.py` | Train the K-projector head. |
| `demo_chat.py` | Interactive chat demo on top of the bank. |
| `bank_store.py` | CLI helper for inspecting / mutating saved banks. |
| `make_figures.py` | Regenerate canonical figures for the current paper. |
| `update_readme_charts.py` | Refresh README charts from the latest reports. |
| `run_mHC1_6_finetune_gb10.sh` | Shell wrapper for the MHC1.6 fine-tune on GB10. |

## Archived scripts

See [`scripts/legacy/`](legacy/) for the 40+ pre-Phase-Q one-shot scripts that
produced historical artefacts. They are kept for reference but are not invoked
by the active test suite, `repro_v3.sh`, or `repro_v31.sh`.
