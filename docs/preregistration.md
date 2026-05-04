# Mneme v3 — Pre-registration

This document is **signed by commit before any test-set evaluation runs**.
Any change after the signing commit must be appended as an `Amendment` block
with its own date and rationale. The signing commit hash is added to the top
of this file in the same commit (self-reference via `git log`).

The purpose is to make post-hoc tweaks detectable: if a hypothesis or gate
number is silently relaxed after the test set is touched, the diff will show
it.

---

## 1. Held-out split SHA-256 (frozen)

Source: `scripts/data/lama_trex_paraphrase.jsonl`
Split script: `eval/holdout_split.py` (seed = 42, stratified by relation,
60/20/20 ratio).

| split | count | sha256 |
|---|---|---|
| train | 104 | `1ccecdc7712586ac27f8835c8ea5a69469fb38823e09aa3a98f3e17d89072b85` |
| dev   | 33  | `26724a2106a91b31d37b4100e98aa92bff8bb6459eed42422ddc307a8f3ec156` |
| test  | 39  | `ff6a818d85a39dc9fe2584c912a19fa6b891db0b97add048a33ee1096b14e025` |
| source | 176 | `fb365a4b73fc84f50b97e025112d036e4d3c88b2e0d7a85620f9efbc18070fbd` |

Run `python eval/holdout_split.py --check` to verify reproducibility.

**Test-set discipline**: dev is used freely for tuning; test is touched
*exactly once* per (model × method × seed) cell during Phase G.

## 2. Models under test

Five frozen LLMs, each loaded in bf16 and never fine-tuned:

| family | model | device | role |
|---|---|---|---|
| Gemma-4 | `google/gemma-4-E2B` | Mac MPS / GB10 | primary |
| Gemma-4 | `google/gemma-4-31B-it` | GB10 only | scale check |
| Qwen-3 | `Qwen/Qwen3-4B-Instruct-2507` | Mac MPS / GB10 | architecture diversity |
| Llama-family | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | GB10 only | reasoning model |
| GLM-4 | `THUDM/glm-4-9b-chat` | GB10 only | architecture diversity |

## 3. Methods under test

| id | description | trainable params |
|---|---|---|
| B0 | no-memory (model alone) | 0 |
| B1 | prompt-insertion (fact text prepended) | 0 |
| B2 | RAG-BM25 top-3 (sparse retrieval into context) | 0 |
| B3 | MEMIT (weight-edit, EleutherAI memit) | 1500 step edit |
| B4 | Mneme v1 (stitched encoder) | KeyProjector + Q/K/V deltas |
| B5 | Mneme v2 (zero-param attn-native, current PR) | 0 |
| B6 | Mneme v3 = v2 + Stage 14 (this work) | 1 `Linear(d,d)` per layer (InfoNCE projector) |

## 4. Hypotheses

Each hypothesis declares the test, the metric, the gate, the statistical
procedure, and the failure-handling rule.

### H1 — Paraphrase robustness (the headline claim)

**Statement**: On Gemma-4-E2B, DM-v3 has higher paraphrase recall@1 than
B0/B1/B2/B5, on the test split.

- **Metric**: recall@1 of target-token decoding under 5 paraphrase templates,
  averaged per fact across templates.
- **Gate**:
  - DM-v3 recall@1 ≥ 0.55, **and**
  - DM-v3 - B0 ≥ +0.30 absolute, **and**
  - DM-v3 - B1 ≥ +0.05 absolute, **and**
  - DM-v3 - B2 ≥ +0.05 absolute, **and**
  - DM-v3 - B5 ≥ +0.10 absolute.
- **Stat**: Wilcoxon signed-rank on per-fact recall difference; Holm-
  Bonferroni over the 4 baseline comparisons; family-wise α = 0.05.
- **Seeds**: 3 (must hold on majority and mean must hit the gate).

### H2 — Decoy curve flatness

**Statement**: As the number of decoy facts injected with the target rises
from 1 → 10 → 50, DM-v3 recall@1 decays no faster than 0.40·log(N) compared
to B5 (whose 13B-2 curve was 0.67 → ~0 at N=50).

- **Gate**: At N=50 decoys, DM-v3 recall@1 ≥ 0.30 (vs. B5 baseline near 0).
- **Stat**: Bootstrap 95% CI on recall@1 at each N; CI lower bound at N=50
  must exceed 0.20.

### H3 — LORO (locality of rewrite)

**Statement**: Locality probes (questions whose target was *not* injected)
suffer no measurable degradation under DM-v3.

- **Metric**: target-token rank shift on 30 LAMA-TREx probes whose value is
  not in the bank.
- **Gate**: Mean rank shift ≤ +1, p > 0.05 (Wilcoxon two-sided, no
  significant degradation).

### H4 — Writer subspace decoupling (Stage 14E)

**Statement**: Adding two facts that share an address ("X is Y_old" then
"X is Y_new") leaves the rest of the bank untouched while resolving X
to Y_new with margin ≥ 1.0 nat.

- **Gate**: Held-out interference ≤ 0.05 absolute recall@1 drop on
  unrelated facts.

### H5 — Multi-position write helps (Stage 14C)

**Statement**: With `policy="multi"`, paraphrase recall@1 strictly exceeds
`policy="period"` on dev.

- **Gate**: dev paraphrase recall@1 (multi) − (period) ≥ +0.10. Dev only —
  this hypothesis tunes the v3-frozen config.

### H6 — Cross-architecture validity

**Statement**: DM-v3 ArchAdapter passes all 4 unit gates on Gemma-4 / Qwen3
/ Llama / GLM-4 (empty bit-equal, α=0 bit-equal, α>0 rank-lift, state-dict
round-trip), and recall@1 lift over B0 ≥ +0.20 on each model.

- **Gate**: All 4 gates green per model; cross-model lift ≥ +0.20.

### H7 — Blind judge faithfulness

**Statement**: On 30 open-ended chat probes, a Qwen3-4B judge rates DM-v3
faithfulness higher than B0 with Cohen's d ≥ 0.5 (medium effect),
double-blind (no model identifiers, hash labels, randomized order). A
secondary GLM-4-9B judge must agree with d ≥ 0.3 (sanity).

- **Gate**: Primary judge d ≥ 0.5; secondary judge d ≥ 0.3; both p < 0.05
  (Mann–Whitney U).

## 5. Statistical procedure (binding)

- **Per-fact pairing**: identical facts evaluated under different methods
  → Wilcoxon signed-rank.
- **Multiple comparisons**: Holm–Bonferroni over the family of baseline
  comparisons within each hypothesis. Family-wise α = 0.05.
- **Bootstrap**: 1000 resamples for any reported CI; 95% percentile
  interval.
- **Effect size**: Cohen's d for between-method scores; rank-biserial r for
  Wilcoxon.
- **Seeds**: 3 random seeds per (model × method) cell. The gate must hold
  on the mean **and** on at least 2/3 seeds individually.

## 6. Contamination check (binding)

For every test-set query string, we run a 5-gram exact-match search in a
public sample of The Pile / RedPajama. Facts whose paraphrase 5-grams hit
> 5% of the corpus are flagged. **Hits stay in the main result table** but
are also reported in a separate ablation table with the flagged facts
removed; if the gate flips on either table, both numbers are reported and
the conclusion is hedged accordingly.

## 7. Honest-failure rule (binding)

If any gate fails on test, the result is published as FAIL. The numbers are
not silently dropped, the gate is not relaxed, the seed is not re-rolled,
and the test set is **not** evaluated again under a different config in the
same paper. A follow-up Stage 15 may attempt a new fix, with its own
preregistration.

## 8. Amendments

(none yet — append below this line; do not edit any text above)

