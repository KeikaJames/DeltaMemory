# v2 experiment results — aggregated

## Headline

**e01 canonical (B2 reproduce)**

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| canonical | 0 | 9 | 512 | 120/200 | 11.998 | 8.102 | -3.911 | FAIL |
| canonical | 1 | 9 | 512 | 120/200 | 11.838 | 6.443 | -5.316 | PASS |
| canonical | 2 | 9 | 512 | 120/200 | 12.020 | 6.024 | -5.912 | PASS |

**e01 h6 layer sweep (L3 / L9 / L21 / L33)**

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| canonical | 0 | 3 | 512 | 120/200 | 11.998 | 10.414 | -1.642 | FAIL |
| canonical | 0 | 21 | 512 | 120/200 | 11.998 | 5.653 | -6.286 | PASS |
| canonical | 1 | 21 | 512 | 120/200 | 11.838 | 5.309 | -6.451 | PASS |
| canonical | 2 | 21 | 512 | 120/200 | 12.020 | 5.496 | -6.446 | PASS |
| canonical | 0 | 33 | 512 | 120/200 | 11.998 | 8.030 | -3.964 | FAIL |

**e11 noise variants (falsification)**

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| n1_iid_gaussian | 0 | 9 | 512 | 120/200 | 11.998 | 5.951 | -6.045 | FAIL |
| n2_uniform_sphere | 0 | 9 | 512 | 120/200 | 11.998 | 5.951 | -6.045 | FAIL |
| n3_single_row_replicated | 0 | 9 | 512 | 120/200 | 11.998 | 6.496 | -5.494 | FAIL |
| n4_single_random_replicated | 0 | 9 | 512 | 120/200 | 11.998 | 6.108 | -5.914 | FAIL |
| n5_constant_vector | 0 | 9 | 512 | 120/200 | 11.998 | 9.290 | -2.717 | FAIL |
| n6_real_bank_K1 | 0 | 9 | 1 | 120/200 | 11.998 | 6.237 | -5.762 | FAIL |
| n1_iid_gaussian | 0 | 21 | 512 | 120/200 | 11.998 | 5.683 | -6.257 | FAIL |
| n3_single_row_replicated | 0 | 21 | 512 | 120/200 | 11.998 | 5.642 | -6.344 | FAIL |
| n5_constant_vector | 0 | 21 | 512 | 120/200 | 11.998 | 5.519 | -6.403 | FAIL |

**e02 scale matrix**

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| n1024_t200_single_s500_seed0 | 0 | 9 | 1024 | 200/500 | 11.327 | 6.269 | -5.107 | — |
| n2048_t1000_single_s1000_seed0 | 0 | 9 | 2048 | 1000/1000 | 12.674 | 13.302 | 0.605 | — |
| n2048_t200_single_s500_seed0 | 0 | 9 | 2048 | 200/500 | 11.327 | 7.310 | -4.095 | — |
| n2048_t500_single_s500_seed0 | 0 | 9 | 2048 | 500/500 | 12.668 | 10.680 | -1.999 | — |
| n512_t200_single_s200_seed0 | 0 | 9 | 512 | 200/200 | 11.327 | 6.564 | -4.777 | — |

**e05 cross-model**

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e05_qwen3_1p7B_seed0 | 0 | 9 | 512 | 120/200 | 12.647 | 6.644 | -5.942 | PASS |

**e06 relation disjoint OOD (train vs test_ood — Δ shown is OOD)**

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e06_seed0 | 0 | 9 | 512 | 120/200 | 12.439 | 8.069 | -4.289 | PASS |

**e19 seed replication summary (Δ_real mean ± stdev by layer)**

| layer | n_seeds | Δ_real mean | Δ_real stdev |
|---|---|---|---|
| 9 | 5 | -4.845 | 0.330 |
| 21 | 5 | -6.707 | 0.247 |

**e09 v1_orig vs v2_kproj**

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| v1_orig | 0 | 9 | 512 | 120/200 | 12.033 | 12.019 | 0.002 | PASS |
| v2_kproj | 0 | 9 | 512 | 120/200 | 12.033 | 7.009 | -5.007 | PASS |


## All experiments

### e01_anticheat_b2

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| canonical | 0 | 3 | 512 | 120/200 | 11.998 | 10.414 | -1.642 | FAIL |
| canonical | 0 | 9 | 512 | 120/200 | 11.998 | 8.102 | -3.911 | FAIL |
| h1_bank_off | 0 | 9 | 512 | 120/200 | 11.998 | 8.086 | -3.927 | PASS |
| h2_shuffle_b | 0 | 9 | 512 | 120/200 | 11.998 | 7.383 | -4.626 | FAIL |
| h2c_collapsed_bank | 0 | 9 | 512 | 120/200 | 11.998 | 7.189 | -4.841 | FAIL |
| h3_disjoint_split | 0 | 9 | 512 | 120/200 | 12.722 | 10.028 | -2.776 | PASS |
| h4_zero_bank | 0 | 9 | 512 | 120/200 | 11.998 | 8.101 | -3.912 | PASS |
| h7_rand_train | 0 | 9 | 512 | 120/200 | 11.998 | 11.711 | -0.301 | PASS |
| canonical | 1 | 9 | 512 | 120/200 | 11.838 | 6.443 | -5.316 | PASS |
| canonical | 2 | 9 | 512 | 120/200 | 12.020 | 6.024 | -5.912 | PASS |
| canonical | 0 | 21 | 512 | 120/200 | 11.998 | 5.653 | -6.286 | PASS |
| canonical | 1 | 21 | 512 | 120/200 | 11.838 | 5.309 | -6.451 | PASS |
| canonical | 2 | 21 | 512 | 120/200 | 12.020 | 5.496 | -6.446 | PASS |
| canonical | 0 | 33 | 512 | 120/200 | 11.998 | 8.030 | -3.964 | FAIL |

### e02_scale_matrix

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| n1024_t200_single_s500_seed0 | 0 | 9 | 1024 | 200/500 | 11.327 | 6.269 | -5.107 | — |
| n2048_t1000_single_s1000_seed0 | 0 | 9 | 2048 | 1000/1000 | 12.674 | 13.302 | 0.605 | — |
| n2048_t200_single_s500_seed0 | 0 | 9 | 2048 | 200/500 | 11.327 | 7.310 | -4.095 | — |
| n2048_t500_single_s500_seed0 | 0 | 9 | 2048 | 500/500 | 12.668 | 10.680 | -1.999 | — |
| n512_t200_single_s200_seed0 | 0 | 9 | 512 | 200/200 | 11.327 | 6.564 | -4.777 | — |

### e03_capability_drift

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e03_drift_t8000 | — | 9 | 512 | —/— | — | — | — | — |

### e05_cross_model

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e05_qwen3_1p7B_seed0 | 0 | 9 | 512 | 120/200 | 12.647 | 6.644 | -5.942 | PASS |

### e06_relation_disjoint_ood

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e06_seed0 | 0 | 9 | 512 | 120/200 | 12.439 | 8.069 | -4.289 | PASS |

### e07_perlayer_kproj

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e07_seed0 | 0 | — | — | —/— | — | — | — | — |

### e08_interrupt_api_demo

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e08_demo | — | 9 | 512 | —/— | — | — | — | — |

### e09_v1_anb_resurrect

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| v1_orig | 0 | 9 | 512 | 120/200 | 12.033 | 12.019 | 0.002 | PASS |
| v2_kproj | 0 | 9 | 512 | 120/200 | 12.033 | 7.009 | -5.007 | PASS |

### e10_topk_retrieval

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| all_attend_random_renorm15 | 0 | 9 | 512 | 120/200 | 11.998 | 6.283 | -5.724 | — |
| all_attend_real | 0 | 9 | 512 | 120/200 | 11.998 | 7.949 | -4.063 | — |
| topk_cosine_random_K8 | 0 | 9 | 512 | 120/200 | 11.998 | 7.572 | -4.427 | — |
| topk_cosine_real_K1 | 0 | 9 | 512 | 120/200 | 11.998 | 11.311 | -0.683 | — |
| topk_cosine_real_K64 | 0 | 9 | 512 | 120/200 | 11.998 | 8.792 | -3.206 | — |
| topk_cosine_real_K8 | 0 | 9 | 512 | 120/200 | 11.998 | 9.460 | -2.540 | PASS |
| topk_random_indices_K8 | 0 | 9 | 512 | 120/200 | 11.998 | 10.509 | -1.486 | — |

### e11_noise_robustness

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| n1_iid_gaussian | 0 | 9 | 512 | 120/200 | 11.998 | 5.951 | -6.045 | FAIL |
| n2_uniform_sphere | 0 | 9 | 512 | 120/200 | 11.998 | 5.951 | -6.045 | FAIL |
| n3_single_row_replicated | 0 | 9 | 512 | 120/200 | 11.998 | 6.496 | -5.494 | FAIL |
| n4_single_random_replicated | 0 | 9 | 512 | 120/200 | 11.998 | 6.108 | -5.914 | FAIL |
| n5_constant_vector | 0 | 9 | 512 | 120/200 | 11.998 | 9.290 | -2.717 | FAIL |
| n6_real_bank_K1 | 0 | 9 | 1 | 120/200 | 11.998 | 6.237 | -5.762 | FAIL |
| n1_iid_gaussian | 0 | 21 | 512 | 120/200 | 11.998 | 5.683 | -6.257 | FAIL |
| n3_single_row_replicated | 0 | 21 | 512 | 120/200 | 11.998 | 5.642 | -6.344 | FAIL |
| n5_constant_vector | 0 | 21 | 512 | 120/200 | 11.998 | 5.519 | -6.403 | FAIL |

### e17_negation_robustness

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e17_seed0 | 0 | 9 | 512 | 120/200 | — | — | — | FAIL |

### e18_2hop

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| e18_seed0 | 0 | — | — | —/— | — | — | — | — |

### e19_seed_replication

| variant | seed | L | n_prl | t/steps | base | real_after | Δ_real | verdict |
|---|---|---|---|---|---|---|---|---|
| L9_s0 | 0 | 9 | 512 | 120/200 | 12.033 | 7.239 | -4.777 | — |
| L9_s1 | 1 | 9 | 512 | 120/200 | 11.971 | 6.806 | -5.120 | — |
| L9_s2 | 2 | 9 | 512 | 120/200 | 11.939 | 7.159 | -4.781 | — |
| L9_s3 | 3 | 9 | 512 | 120/200 | 11.903 | 7.506 | -4.360 | — |
| L9_s4 | 4 | 9 | 512 | 120/200 | 12.209 | 7.002 | -5.186 | — |
| L21_s0 | 0 | 21 | 512 | 120/200 | 12.033 | 4.826 | -7.128 | — |
| L21_s1 | 1 | 21 | 512 | 120/200 | 11.971 | 5.324 | -6.594 | — |
| L21_s2 | 2 | 21 | 512 | 120/200 | 11.939 | 5.165 | -6.730 | — |
| L21_s3 | 3 | 21 | 512 | 120/200 | 11.903 | 5.292 | -6.554 | — |
| L21_s4 | 4 | 21 | 512 | 120/200 | 12.209 | 5.621 | -6.530 | — |
