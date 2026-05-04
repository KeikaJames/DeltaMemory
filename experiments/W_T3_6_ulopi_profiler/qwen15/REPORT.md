# S-7 — U-LOPI cross-arch MPS pilot

**Model loaded**: `Qwen/Qwen2.5-1.5B`

**Cmdline**: `python scripts/run_ulopi_xarch_smoke.py --model Qwen/Qwen2.5-1.5B --device mps --dtype bfloat16 --alpha 0,1,2,4 --seeds 0,1,2 --n-prompts 8 --out experiments/W_T3_6_ulopi_profiler/qwen15/`

**Git rev**: `57aeeac1094440b9d7f9773c96528ba1d0a250b4`

**Generated**: 2026-05-04T16:38:26Z


## Candidate models (load attempts)

- `Qwen/Qwen2.5-1.5B` → **LOADED** 

## Profile artifact (auto mode)

- `mu_arch` = **5**
- `eta_sigma` = **1.0000**
- `num_layers` = 28
- `profile_corpus_sha` = `a84cd30069732834`
- See `profile.json` for full `mu_base` / `sigma_base` arrays.

## Paired drift table (mean over seeds)

| alpha | n | static_nll | auto_nll | drift_static | drift_auto | Δ(auto-static) |
|------:|--:|-----------:|---------:|-------------:|-----------:|---------------:|
| 0.00 | 3 | +1.5669 | +1.5669 | +0.0000 | +0.0000 | +0.0000 |
| 1.00 | 3 | +3.7147 | +2.9171 | +2.1477 | +1.3502 | -0.7976 |
| 2.00 | 3 | +3.7389 | +7.8773 | +2.1720 | +6.3104 | +4.1384 |
| 4.00 | 3 | +3.2592 | +10.7191 | +1.6922 | +9.1522 | +7.4600 |

## Verdict

- mean drift static (alpha>0) = **+2.0040 nats**
- mean drift auto   (alpha>0) = **+5.6042 nats**
- **Verdict**: `STATIC<=AUTO`

## Limitations

- MPS small-model pilot; flagship cross-arch on GB10 deferred.
- 8 prompts × ≤3 seeds × ≤3 alphas — small-N; treated as a
  smoke / hypothesis check, not a publication-grade result.
- The `LlamaAdapter` claims Qwen2/Qwen2.5/Llama/Mistral. 
  Bit-equal at α=0 is asserted at runtime as the only 
  cross-mode safety check.