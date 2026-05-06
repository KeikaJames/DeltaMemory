# A3 — X.7-NL Cross-Arch: DeepSeek-V4-Flash — SKIPPED

**Date:** 2025-05-06  
**Reason:** Model does not exist; nearest alternative substituted.

"DeepSeek-V4-Flash" as specified in the Track A plan was a forward-looking name and
does not correspond to a released model as of 2025-05-06. The available DeepSeek
models on spark1's HF cache are:
- `DeepSeek-R1-Distill-Qwen-32B` (distillation of DeepSeek-R1 reasoning model)

**Alternative executed:** Qwen3.6-27B (native MoE, on-whitelist) was run instead
as A3-bonus, providing equivalent cross-arch coverage for an MoE architecture.
Results in: `runs/X7NL_full_v1_qwen3_27B/`

**Notes:**
- `DeepSeek-R1-Distill-Qwen-32B` is architecturally a Qwen model (not a native DeepSeek
  decoder), so adding it would not provide distinct architecture signal beyond Qwen3.
- When a true DeepSeek MLA (Multi-head Latent Attention) model is available on spark1,
  a `DeepSeekMLAAdapter` will be needed in `arch_adapter.py`.

**Mitigation:** See `runs/X7NL_full_v1_qwen3_27B/` for A3-bonus results.
