# R-7 V-scale calibration smoke

**Model**: `Qwen/Qwen2.5-0.5B-Instruct`
**Generated**: 2026-05-04T10:36:37Z
**Cmdline**: `scripts/run_r7_vscale_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype bfloat16 --alphas 0,2 --seeds 0 --n-prompts 2 --value-scale-modes none,auto_rms_cap --out reports/cleanroom/r7_vscale_smoke`

| value_scale_mode | alpha | n | mean drift | mean bank V RMS |
|---|---:|---:|---:|---:|
| `auto_rms_cap` | 0.00 | 1 | +0.0000 | 0.3991 |
| `auto_rms_cap` | 2.00 | 1 | +3.7219 | 0.3991 |
| `none` | 0.00 | 1 | +0.0000 | 0.5172 |
| `none` | 2.00 | 1 | +4.0869 | 0.5172 |

## Interpretation

`auto_rms_cap` should keep alpha=0 bit-equal and cap no-v_norm family bank values at the configured per-head RMS without amplifying already small V activations. Drift is a smoke signal only; full R-7 resweep still needs the R-4/R-5.2 grids.

Raw cells: `reports/cleanroom/r7_vscale_smoke/cells.json`.
