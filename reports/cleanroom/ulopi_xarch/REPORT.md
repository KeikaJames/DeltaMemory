# S-7 — U-LOPI cross-arch MPS pilot

**Model loaded**: `Qwen/Qwen2.5-0.5B-Instruct`

**Cmdline**: `python scripts/run_ulopi_xarch_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps --dtype bfloat16 --alpha 0,2,5 --seeds 0,1,2 --n-prompts 8 --out reports/cleanroom/ulopi_xarch/`

**Git rev**: `8bc54724a1994789d4848537b66a7e6853748aa3`

**Generated**: 2026-05-04T09:25:35Z


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
| 2.00 | 3 | +11.7267 | +15.0244 | +9.8232 | +13.1209 | +3.2977 |
| 5.00 | 3 | +14.7138 | +12.2019 | +12.8103 | +10.2984 | -2.5119 |

## Verdict

- mean drift static (alpha>0) = **+11.3168 nats**
- mean drift auto   (alpha>0) = **+11.7097 nats**
- **Verdict**: `STATIC<=AUTO`

## Limitations

- MPS small-model pilot; flagship cross-arch on GB10 deferred.
- 8 prompts × ≤3 seeds × ≤3 alphas — small-N; treated as a
  smoke / hypothesis check, not a publication-grade result.
- The `LlamaAdapter` claims Qwen2/Qwen2.5/Llama/Mistral. 
  Bit-equal at α=0 is asserted at runtime as the only 
  cross-mode safety check.