# X.7 — Bank Forget / Merge / Compact (PREREG.v1)

**Author**: BIRI GA
**Branch**: `feat/v05-counterfactual-industrial`
**Locked at commit**: (filled at lock-time)
**Driver mandate (verbatim)**:

> 目前 SCAR 能监控 alpha-drift，但没有看到自动压缩 / 遗忘 / 合并机制的完整方案。

X.1 (bank-size scaling) is the *witness* for dilution. X.7 is the
*remediation*: a bank-lifecycle mechanism with capacity, eviction,
merge, and decay so that the live bank does not grow unboundedly nor
let early important facts be buried.

---

## 1. Hypotheses and red-lines

| ID | Statement | Test |
|----|-----------|------|
| H_X7.0 (red-line) | At `bank_capacity = 0` (default), every behavior of `AttnNativeBank.append` / `bulk_append` / forward is bit-equal to v0.4. | Existing 322-test suite + new `test_x7_capacity_default_bit_equal`. |
| H_X7.1 | At `bank_capacity = K > 0`, after writing `N > K` entries, the bank holds exactly `K` entries. | `test_x7_lru_evicts_to_capacity`. |
| H_X7.2 | LRU eviction (policy = `"lru"`) preserves the most-recently-read or most-recently-written entry. | `test_x7_lru_preserves_recent`. |
| H_X7.3 | At `bank_capacity = K`, recall@1 of a *target* fact written at t=0 and re-read at every step beats the no-capacity baseline once N > K (on the gemma-3-1b-it × 1000-distractor stream we use in X.1). | X.7 experimental run (separate). |

Any failure of H_X7.0 is a release blocker; H_X7.1/2 are correctness;
H_X7.3 is the empirical claim and is allowed to land as a *negative*
finding (which would itself be publishable).

---

## 2. Mechanism — three policies, one capacity

A new attribute on `AttnNativeBank`:

```python
self.bank_capacity: int = 0           # 0 ⇒ unbounded (default; bit-equal to v0.4)
self.bank_evict_policy: str = "lru"   # "lru" | "fifo" (v1 ships LRU; FIFO is the smoke control)
```

Per-entry metadata (only allocated when `bank_capacity > 0`):

```python
self._write_step:    list[int]    # monotonic step at which entry was appended
self._last_access:   list[int]    # last step at which any head's attention weight on this entry was the per-head max
self._access_count:  list[int]    # # of forwards in which the above happened
self._global_step:   int          # incremented on every append AND every forward
```

### 2.1 LRU evict (v1 — shipped this PR)

On `append` / `bulk_append`, after the new entries are concatenated:

```
while len(self) > bank_capacity:
    drop index i* = argmin( last_access[i] + λ * access_count[i] )
    # tie-break on smallest write_step (oldest wins eviction)
```

with default `λ = 0` so the policy is pure-LRU (most-recently-read
wins). Per-layer `M_K[layer]` and `M_V[layer]` are sliced together
with the metadata lists so indices stay aligned.

**Read-path hook**: at each forward, *if* `bank_capacity > 0`:
```
i*_per_head = argmax over bank-axis of weights[..., T_orig:]
mark unique(i*_per_head) as accessed at the current global step
```
This is gated by `bank_capacity > 0` so the default path is bit-equal.

### 2.2 EMA-merge (v2 — design only; PR follow-up)

When two entries `(K_a, V_a)`, `(K_b, V_b)` satisfy
`cos(K_a, K_b) > τ_merge` (default `0.95`), merge:
```
w_a, w_b = access_count[a] + 1, access_count[b] + 1
K* = (w_a K_a + w_b K_b) / (w_a + w_b)
V* = (w_a V_a + w_b V_b) / (w_a + w_b)
```
keeping the elder `write_step` and the newer `last_access`. This is
**not in v1** — pre-registered for the follow-up PR so its rollout
remains under the authenticity contract.

### 2.3 Time-decay (v2 — design only; PR follow-up)

Each entry's `V` is multiplied by `exp(-Δt / τ_decay)` on a periodic
sweep. Default τ_decay = ∞ ⇒ off. Pre-registered here so the API
shape is committed before any runs.

---

## 3. Experimental design (run script ships in v1, results-run is its own PR)

* Base model: `google/gemma-3-1b-it`, MPS bf16 (matches X.1).
* Distractor stream: 10 000 paraphrased "filler" facts written at
  steps 1…10 000.
* Target fact: written at step 0; re-queried at steps {1, 100, 1000,
  5 000, 10 000}.
* Capacity sweep: `bank_capacity ∈ {0, 100, 1 000}`.
* Policy sweep: `policy ∈ {"none", "lru", "fifo"}` (with policy =
  none ↔ `bank_capacity = 0`).
* Seeds: 0, 1, 2.
* Cells: 3 caps × 3 policies (effectively 1 + 2×2 = 5 valid combos)
  × 3 seeds × 5 query-step probes = 75 active cells.
* Metric: target-fact recall margin, target-token NLL, KL-unrelated.

### Red-line abort conditions

* Any `bank_capacity > 0` cell that produces NaN/Inf in K or V → abort
  before scoring.
* Any policy producing the *wrong* number of entries post-eviction →
  abort.
* Any `bank_capacity = 0` cell whose target-fact NLL differs from the
  v0.4 baseline by more than `1e-4` per token → abort (H_X7.0 broken).

---

## 4. Out of scope for X.7.v1

* Adaptive capacity (auto-tuning K from observed stream rate).
* Cross-session bank persistence with policy state.
* RBAC over evicted entries (G4 follow-up).
* Saliency from gradient (we only use forward attention weights).

---

## 5. Deviations log

(empty — locked)
