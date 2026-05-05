# L Marathon Pre-Registration: Long-Conversation Bank Stability

**Status**: draft.
**Authored**: 2026-05-10 (post W.6 counter-prior verdict).
**Depends on**: W.6 must ship its method-winner verdict before this runs.
**Hardware target**: single CUDA GPU with >=24 GB or MPS device with >=32 GB unified memory.

---

## 1. Question

Does the attn-native bank with per-turn injection maintain stable recall and bounded resource consumption across N=2000 turns of dialogue?

Failure modes to detect:
- NaN/Inf in bank K or V tensors
- OOM or >1 GB RSS jump between consecutive checkpoints
- Recall decay (10x drop in target_new NLL)
- Residual-stream drift at the model's mu_arch layer
- KV-cache leak

## 2. Hypotheses

**H_L (recall stability)**:
  `recall(turn=2000) >= 0.5 * recall(turn=1)`
  paired across 3 seeds per model. Statistical test: paired Wilcoxon (n=3, two-sided, alpha=0.05).
  Small-N caveat documented.

**Red lines (abort criteria)**:
1. Any NaN or Inf on bank K or V tensors at any turn → ABORT.
2. Any process-RSS jump >1 GB between consecutive checkpoints → ABORT.
3. Any 10x decay in target_new recall between consecutive checkpoints → ABORT.
4. alpha=0 bit-equality witness at turn 1: drift must be 0.

## 3. Grid

Total cells: **3 models × 2 methods × 4 turn-counts × 3 seeds × 3 facts = 216 runs**
(each run produces multiple checkpoint rows).

- **Models**: `gpt2-medium`, `Qwen/Qwen2.5-1.5B`, `google/gemma-3-1b-it`.
- **Methods**: `lopi_default`, `caa` (skip AttnNativeBank — bank K/V state evolves and requires separate test).
- **Turn counts (N)**: `{100, 500, 1000, 2000}`.
- **Probe checkpoints**: turns `{1, 50, 200, 500, 1000, 2000}` (skip checkpoints beyond N).
- **Seeds**: `{0, 1, 2}`.

## 4. Injection content

At turn 1, inject 3 counterfactual facts:
- (subject, relation, target_new, target_true) from `facts_3.jsonl`.
- Each fact uses a subject that does not exist in the pretraining corpus (e.g., "The capital of Mars is Olympus City").

Held-out probe set:
- 8 prompts per fact (mix of direct and oblique queries) in `probes_8.jsonl`.

## 5. Filler turns

Between probe checkpoints, advance the conversation state with deterministic filler text from `filler.txt`:
- 200-word neutral chitchat snippet looped to fill turns.
- Filler does NOT mention the injected subjects.
- Same filler text across all runs (deterministic).

## 6. Probes per checkpoint

At each checkpoint turn, measure:
1. **nll_target_new**: mean NLL of target_new tokens on the held-out 8-prompt probe set.
2. **residual_norm_mu**: L2 norm of hidden_states at mu_arch layer (architecture-dependent spike layer).
3. **mem_rss_mb**: process RSS in MB (via psutil.Process().memory_info().rss / 1024**2).
4. **nan_inf_count**: count of NaN or Inf values in bank K + bank V tensors.
5. **kv_cache_size_bytes**: best-effort measure of HF model KV-cache state size.

## 7. Statistics

- **H_L**: paired Wilcoxon signed-rank, two-sided, `zero_method='wilcox'`, on `nll_target_new(turn=2000) - nll_target_new(turn=1)` paired by seed. Per model, n=3.
- **Effect size**: median paired diff with 95% bootstrap CI (B=1000, seed=0).

## 8. Abort conditions

1. **NaN/Inf**: any non-zero nan_inf_count at any checkpoint → write final row with abort_reason, exit non-zero.
2. **OOM spike**: any mem_rss_mb jump >1024 MB between consecutive checkpoints → abort.
3. **Recall decay**: any 10x increase in nll_target_new between consecutive checkpoints → abort.
4. **alpha=0 witness**: at turn 1, with alpha=0, residual_norm drift must be 0.

## 9. Deliverables

- `cells.jsonl` — one row per probe checkpoint, fields: run_id, model, method, seed, turn, nll_target_new, residual_norm_mu, mem_rss_mb, nan_inf_count, kv_cache_size_bytes, abort_reason (null unless aborted).
- `env.json` — git_commit, dataset_sha (hash of facts + probes + filler), prereg_version, torch/transformers versions, device, dtype.
- `summary.json` — H_L Wilcoxon verdicts per model, plus flat table for plotting.
- `REPORT.md` — narrative, no emoji.

## 10. Out of scope

- Multi-fact interference within the same turn.
- Non-deterministic filler (future work: adversarial probing of bank state).
- Bank K/V evolution under write-heavy workloads (requires separate AttnNativeBank test).

---

**prereg_version**: `"L.v1"`

End of pre-registration. After this point, no parameter in §3, §6, §8 may change without recording the change in REPORT.md §Deviations and bumping prereg_version in env.json.
