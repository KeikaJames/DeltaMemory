# S-7 — U-LOPI cross-arch MPS pilot

**Model loaded**: `Qwen/Qwen2.5-0.5B-Instruct`

**Cmdline**: `python scripts/run_ulopi_xarch_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype bfloat16 --alpha 0,1,2,4 --seeds 0,1,2 --n-prompts 8 --out experiments/W_T3_6_ulopi_profiler/qwen05/`

**Git rev**: `57aeeac1094440b9d7f9773c96528ba1d0a250b4`

**Generated**: 2026-05-04T16:37:36Z


## Candidate models (load attempts)

- `Qwen/Qwen2.5-0.5B-Instruct` → **LOADED** 

## Profile artifact (auto mode)

- `mu_arch` = **5**
- `eta_sigma` = **0.7000**
- `num_layers` = 24
- `profile_corpus_sha` = `a84cd30069732834`
- See `profile.json` for full `mu_base` / `sigma_base` arrays.

## Paired drift table (mean over seeds)

| alpha | n | static_nll | auto_nll | drift_static | drift_auto | Δ(auto-static) |
|------:|--:|-----------:|---------:|-------------:|-----------:|---------------:|
| 0.00 | 3 | +1.9035 | +1.9035 | +0.0000 | +0.0000 | +0.0000 |
| 1.00 | 3 | +4.1591 | +3.5993 | +2.2557 | +1.6958 | -0.5599 |
| 2.00 | 3 | +5.0455 | +3.2156 | +3.1420 | +1.3121 | -1.8299 |
| 4.00 | 3 | +6.3299 | +6.1043 | +4.4264 | +4.2009 | -0.2255 |

## Verdict

- mean drift static (alpha>0) = **+3.2747 nats**
- mean drift auto   (alpha>0) = **+2.4029 nats**
- **Verdict**: `AUTO<STATIC`

## Limitations

- MPS small-model pilot; flagship cross-arch on GB10 deferred.
- 8 prompts × ≤3 seeds × ≤3 alphas — small-N; treated as a
  smoke / hypothesis check, not a publication-grade result.
- The `LlamaAdapter` claims Qwen2/Qwen2.5/Llama/Mistral. 
  Bit-equal at α=0 is asserted at runtime as the only 
  cross-mode safety check.