# A2 — X.7-NL Cross-Arch: Llama-4-Scout-17B-16E-Instruct — SKIPPED

**Date:** 2025-05-06  
**Reason:** Model not available on spark1.

spark1 is air-gapped (inet errno 101: Network is unreachable) and does not have
`Llama-4-Scout-17B-16E-Instruct` on the local whitelist or in the HF hub cache.

**Available models on spark1:**
- `/home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it` (completed, see X7NL_full_v1_gemma4_31B)
- `/home/gabira/Desktop/workspace/models/whitelist/gpt-oss-120b` (A1, running)
- `/home/gabira/Desktop/workspace/models/whitelist/Qwen3.6-27B` (bonus, see X7NL_full_v1_qwen3_27B)

**Notes:**
- Llama-4-Scout-17B-16E-Instruct requires Meta access approval via Hugging Face.
- A bf16 load would require ~34 GB, which fits the 128 GB GB10 unified memory pool.
- A `Llama4Adapter` already exists in `deltamemory/memory/arch_adapter.py`.
- This run can be retroactively executed if the model becomes available on spark1.

**Mitigation:** Qwen3.6-27B (on-whitelist, MoE decoder) run as A3-bonus for additional
cross-arch coverage. Cross-arch comparison in A4 uses gemma-4 (baseline) + gpt-oss-120b (A1).
