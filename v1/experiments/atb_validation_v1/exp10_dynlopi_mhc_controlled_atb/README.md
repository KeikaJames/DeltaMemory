# Exp10: Dynamic LOPI + mHC Controlled ATB

Tests whether Dynamic LOPI v3.4 + mHC + beta gate convert raw ATB injection into
controlled memory readout, following Exp8's negative result (mHC-only insufficient).

## Setup

Requires: Qwen3-4B-Instruct-2507 on CUDA.

```bash
cd /root/projects/RCV-HC
CUDA_VISIBLE_DEVICES=1 \
.venv-a10/bin/python3 \
  experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/run.py \
  --model /root/models/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --dtype bf16 \
  --seeds 0,1,2 \
  --phase AB \
  --alpha-grid 0.05,0.10,0.20 \
  --kappa 0.25 \
  --bank-size 200 \
  --n-prompts-smoke 50 \
  --n-prompts-confirm 200 \
  --out experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/run_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee exp10_smoke.log
```

## Qualitative Generation

```bash
CUDA_VISIBLE_DEVICES=1 \
.venv-a10/bin/python3 \
  experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/run_qualitative.py \
  --model /root/models/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --dtype bf16 \
  --alpha 0.05 --kappa 0.25 --beta 0.05 \
  --cases experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/qualitative_cases \
  --out experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/qual_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee exp10_qual.log
```

## Analysis

```bash
python experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/analyze.py \
  --run-dir experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/run_TIMESTAMP
```

## Arms

| Arm | mHC | LOPI | beta |
|-----|-----|------|------|
| A0_raw_atb | off | off | 1.0 |
| A1_mhc_only | on κ=0.25 | off | 1.0 |
| A2_dynlopi_only | off | on | 1.0 |
| A3_mhc_dynlopi | on κ=0.25 | on | 1.0 |
| A4_mhc_dynlopi_beta | on κ=0.25 | on | 0.05 |

## Pre-registration

See [PREREG.md](PREREG.md).
