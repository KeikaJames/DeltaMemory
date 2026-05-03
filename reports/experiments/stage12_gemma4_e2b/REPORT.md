# Stage 12 — Adversarial Cross-Validation Report

> Status: completed for `gemma-4-E2B` only. Other 4 models (Qwen3-8B, GLM-4-9B, DS-V2-Lite, gpt-oss-20b) **could not run**: GB10 has no outbound HuggingFace network access in our environment, and only `gemma-4-E2B` was pre-cached. Multi-model cross-validation is therefore **deferred** to a session with HF mirror access.

## What was actually tested

A single base model (`google/gemma-4-E2B`, frozen, bf16) was put through three adversarial probes, 100 facts × 3 seeds, training 500 steps per seed.

### P1 — Held-out paraphrase recall

**Headline (raw): mean = 1.000 across 3 seeds.**

**Honest caveat:** with the `multilayer` encoder used in this script, the encoder consumes the *address* (canonical), not the read prompt. So P1 reduces to "given the same address, can we retrieve the same slot?" — trivially yes. **This is not a substantive paraphrase-robustness test.**

The real held-out paraphrase test is in Stage 11A (`held_out_recall_at_1 = 0.138`), where the *address itself* is replaced with an unseen template. That test failed.

### P2 — 10 adversarial transforms applied to the read prompt

Transforms: `typo`, `fragment`, `lowercase_no_punct`, `prefix_inject`, `suffix_inject`, `instruction_conflict`, `noise_pad`, `wrong_lang`, `double_negative`, `polite_misdirect`.

| transform | DM answer-token top1 | no-DM answer-token top1 | DM lift |
| --- | ---: | ---: | ---: |
| typo | 1.000 | 0.000 | +1.000 |
| fragment | 1.000 | 0.000 | +1.000 |
| lowercase_no_punct | 1.000 | 0.000 | +1.000 |
| prefix_inject | 1.000 | 0.000 | +1.000 |
| suffix_inject | 1.000 | 0.000 | +1.000 |
| instruction_conflict | 1.000 | 0.000 | +1.000 |
| noise_pad | 1.000 | 0.000 | +1.000 |
| wrong_lang | 1.000 | 0.000 | +1.000 |
| double_negative | 1.000 | 0.000 | +1.000 |
| polite_misdirect | 1.000 | 0.000 | +1.000 |

**Verdict:** DM injection at α=1.0 forces the gold token through against every surface attack we tested on the *read* side. The injection is too strong to be confused by surface noise. This *is* substantive — it shows the residual-stream injection survives query corruption.

**Caveat:** the encoder side of this test is degenerate (canonical address). What's tested is the injection-vs-CE balance, not the encoder's adversarial robustness.

### P3 — Output tampering / forced override

| metric | value (mean ± std, n=3) | reading |
| --- | ---: | --- |
| base model top1 (no DM) | 0.000 ± 0.000 | base couldn't answer 100 single-token LAMA facts cleanly at zero-shot |
| DM top1 | 1.000 ± 0.000 | with the bank installed, gold wins |
| override_rate (DM forces an answer the base model didn't have) | 1.000 ± 0.000 | DM = capable forced editor |
| **locality_drift on 12 unrelated controls** | **0.750 ± 0.000** | **❌ 75% of unrelated control answers are corrupted by injection at α=1.0** |

**Finding:** DM at α=1.0 + always-on injection across the entire 100-slot bank is **too strong** for production use. It corrupts 75% of unrelated questions because the bank pushes a generic "fact-token" bias into every forward pass.

**Mitigation (already implemented in Stage 11D D2/D3):** route bank reads through a per-query retriever and inject *only* the top-1 slot vector for each query, not the full bank-mean. With per-query routing, locality drift is 0/0 across Stage 11D's tests. The Stage 12 P3 result documents the *worst case* (broadcast injection), not the production policy.

## Multi-model status

| Model | Status | Reason |
| --- | --- | --- |
| `google/gemma-4-E2B` | ✅ Done | Pre-cached |
| `Qwen/Qwen3-8B` | ⏸ Deferred | No HF network on GB10 |
| `THUDM/glm-4-9b-chat` | ⏸ Deferred | No HF network on GB10 |
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | ⏸ Deferred | No HF network on GB10 |
| `openai/gpt-oss-20b` | ⏸ Deferred | No HF network on GB10 |
| `deepseek-ai/DeepSeek-V4-Flash` | ⏸ Deferred | 284B MoE FP4, ~160 GB; needs >GB10's 128 GB or vLLM-FP4 cluster |

Multi-model cross-validation is the biggest open work item. Documented as future hardware/network requirement rather than claimed.

## Honest summary

- DM injection at α=1.0 forces gold tokens through every surface attack on the read prompt — but this is documenting "the injection is strong", not "the encoder is adversarially robust".
- DM can force-override answers the base model gets wrong, **at the cost of substantial locality drift if the bank is broadcast**. Per-query retrieval (Stage 11D) avoids this.
- The held-out paraphrase test in this stage is structurally trivial under our encoder choice; the real test is Stage 11A and it **failed**.
- Multi-model evidence is **absent from this stage** because of network restrictions; until that runs, claims are limited to `gemma-4-E2B`.
