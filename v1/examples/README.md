# Mneme / Δ-Memory — Examples

**Mneme** equips a frozen language model with an external attention-side
*key/value bank* (Δ-Memory).  Facts are written by capturing the model's
own K/V activations on a single forward pass, then injected back at read
time with a tunable strength α.  At α=0 the patched model is bit-equal to
the unpatched baseline; at α>0 retrieved facts steer next-token logits.

## Install

From the repo root:

```bash
pip install -e .
```

The examples auto-detect the best device (CUDA → MPS → CPU) and use
`Qwen/Qwen2.5-0.5B-Instruct` (~500 MB) which downloads on first run.

---

## 1 · `01_quickstart.py` — α=0 bit-equal sanity + 1-fact recall

What it shows:

* wrapping a HF model with `AttnNativePatcher`;
* asserting that α=0 (with or without facts) is bit-equal to the
  unpatched baseline (max-abs-diff < 1e-5);
* writing one fact (`"Zorblax is a planet in the Krell system."`);
* α=0 vs α=1 greedy generation and top-5 token shift at the first
  generated position.

Expected tail:

```
[sanity] α=0 max-abs-diff = 0.000e+00  (must be < 1e-5)
[bank ] wrote 1 fact, bank size = 1
[gen  ] α=0.0  →  ...
[gen  ] α=1.0  →  ... Krell ...
[done ] OK
```

Runtime: ~30 s on M-series MPS, ~10 s on a recent CUDA GPU.

---

## 2 · `02_multifact_recall.py` — 5-fact recall@20 across α sweep

What it shows:

* writing 5 facts about distinct fictional entities;
* per-fact rank of the expected continuation token at α=0 vs α=1;
* recall@20 across α ∈ {0, 0.5, 1, 2}.

Expected tail:

```
[sweep] recall@20 across α:
  α=0.0  recall@20 = 0/5
  α=0.5  recall@20 = 3/5
  α=1.0  recall@20 = 5/5
  α=2.0  recall@20 = 5/5
[done ] OK
```

Runtime: ~45 s on M-series MPS.

---

## 3 · `03_diagnostics.py` — `DiagnosticRecorder` + JSON dump

What it shows:

* attaching `DiagnosticRecorder` alongside `AttnNativePatcher`;
* per-layer signals: bank-attn entropy, native-attn entropy, residual
  norms, plus LOPI gates if enabled;
* aggregating with pandas and dumping the full long-format record to
  `/tmp/dm_diag.json`.

Expected tail:

```
[diag ] captured N rows; signals = ['attn_entropy_bank', 'attn_entropy_native', 'bank_col_sum', 'residual_norm']
[dump ] wrote N records → /tmp/dm_diag.json
[done ] OK
```

Runtime: ~30 s on M-series MPS.

---

## Where to go next

* `docs/` — theory of the bank-injection operator, LOPI/U-LOPI, mHC shield.
* `experiments/` — reproducible runs from the paper (counterfact, LAMA,
  multi-fact stress).
* `tests/` — the canonical α=0 bit-equal contract is exercised in
  `tests/test_attn_native_bank.py`.
