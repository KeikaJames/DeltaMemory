# X.1 — Bank-Size Scaling and Softmax Dilution Witness

**Status**: locked
**Version**: `X1.v1`
**Owner**: opus

## 1. Question (user 疑问①)

> "Bank 增大 → softmax 分母爆炸 → 早期重要记忆的权重被严重稀释。
> 即使有 LOPI 门控，长期来看是否会出现『记忆被自己埋葬』的现象？
> SCAR 能监控 alpha-drift，但没有看到自动压缩 / 遗忘 / 合并机制的完整方案。"

This PREREG measures **how target-fact recall degrades as the bank
grows**, and **how each existing defense knob (bank_topk,
bank_separate_softmax, bank_temperature) restores fidelity**.

## 2. Theoretical baseline

Joint softmax over (sequence ∥ bank) of size (T_seq + N) means each
score competes against (T_seq + N − 1) others. As N grows, the bank
mass for any single fact drops as ~1/N if scores are roughly uniform,
~exp(margin)/N if margins are bounded. The defenses:

* `bank_topk = K` — keep only top-K bank entries pre-softmax;
  caps the dilution at K regardless of N (Stage 15A).
* `bank_separate_softmax = True` — run independent softmaxes over
  sequence and bank, combine additively with `bank_merge_beta`;
  mathematically eliminates seq-vs-bank dilution (Stage 15C).
* `bank_temperature = τ` — divide bank scores by τ; τ < 1 sharpens.

We hypothesize:

* **H_X1.0 (red-line)**: at any (N, alpha=0) the model's logits equal
  the unpatched model's logits. Tolerance: max_abs_diff ≤ 1e-4 (fp32),
  ≤ 5e-3 (bf16).
* **H_X1.1 (dilution exists)**: with all defenses off (default
  `bank_topk=0`, `bank_separate_softmax=False`, `tau=1.0`), the
  target-fact log-margin
  `log p(target_new) − log p(target_canonical)` decays
  monotonically as N grows from 1 → 10000, with at least 2× drop
  by N=100 vs N=1.
* **H_X1.2 (top-k restores)**: with `bank_topk=4`, the log-margin at
  N=10000 is within 20% of the log-margin at N=1.
* **H_X1.3 (separate-softmax restores)**: with
  `bank_separate_softmax=True`, log-margin is N-invariant within
  10% across 1 ≤ N ≤ 10000.

## 3. Grid

| factor | levels |
|---|---|
| model | `Qwen/Qwen2.5-0.5B` (CPU/MPS friendly) |
| bank_size N | 1, 10, 100, 1000, 10000 |
| defense arm | `none`, `topk_4`, `separate_softmax` |
| target fact | one fixed counterfact (`Mount Everest is on Antarctica`) |
| distractor facts | (N − 1) random unrelated subject/relation/target_new triples |
| alpha | 0.0 (red-line), 1.0 (main) |
| seed | 0, 1, 2 |

Cells: 1 × 5 × 3 × 1 × 2 × 3 = **90 cells**.

Latency probe (separate, single-seed):
* per-arm wall-clock for the read forward, repeated 5×, median.

## 4. Procedure

1. Load model; install AttnNativePatcher; fresh_bank.
2. Configure defense arm via bank attribute setters.
3. Write the target fact at write-position 0; write `N − 1` distractor
   facts. Distractor facts use the `distractors.jsonl` pack
   (deterministic, sha-pinned).
4. Forward unpatched (no patcher) → reference logits.
5. Forward with bank, alpha=0 → logits_a0 → red-line check.
6. Forward with bank, alpha=1 → logits_a1.
7. Compute `log_margin = logits_a1[id_target_new] − logits_a1[id_target_canonical]`.
8. Append cell row.

## 5. Authenticity

Per `docs/authenticity.md`. cells.jsonl + env.json + dataset_sha
(distractors.jsonl + facts.jsonl). Cell-id keying:
`sha1(model|N|arm|alpha|seed|target_fact_id)[:16]`. Resume-safe.

## 6. Out of scope

* Bank persistence (covered by I.2).
* Forget/merge/compaction policies (X.7 — separate PR).
* Multi-target retrieval (X.7 may revisit).

## 7. Abort conditions

* Red-line fail at any cell → flag, do not aggregate that arm.
* OOM at N=10000 on 0.5B model → reduce to N ∈ {1,10,100,1000} and
  document.
* NaN/Inf at α=1, N=10000, arm=`none` → expected; record `status="nan_inf"`.

## 8. Deliverable

* `experiments/X1_bank_scaling/cells.jsonl`
* `experiments/X1_bank_scaling/summary.json`
* `experiments/X1_bank_scaling/REPORT.md`
* `docs/figures/x1_dilution_vs_n.png`
