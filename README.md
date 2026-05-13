# RCV-HC

All code, docs, experiments and reports are archived under [`v1/`](v1/).

This corresponds to the **ATB v1 (Anchor-N-Bank) investigation**,
**now closed** after Exp1 → Exp34:

- Exp24 / 27 / 31 / 32 / 33 — five independent bank-shaped architectures
  (cosine ATB, sparse joint-softmax, learned K-adapter, MLP-side gated,
  re-attention readout), all REJECTED on the same 125-fact Qwen3-4B
  test split. Gate B = 0/375 across all five.
- **Exp34** — ROME-style rank-1 `down_proj` edit on the same split passes
  Gate B at **125 / 125 (100 %)** with +9.55-nat per-fact uplift, plus
  Gate D identity binding at **98.4 %**. Positive control validates the
  test framework and locks in the negative verdicts.
- LS diagnostic (closed-form ridge + CCA): rank-64 routing reaches 76 %
  honest test top-1 vs 0.8 % on shuffled pairs — refutes data-scarcity
  hypothesis; the failure mode is the bank-readout protocol itself.

**Architectural conclusion**: on Qwen3-4B, fact-identity binding
requires editing the *parameter manifold* of `mlp.down_proj`. Bank
architectures, regardless of routing quality or readout site
(residual-additive or joint-softmax), do not write into that manifold.

See [`v1/README.md`](v1/README.md) for the project overview, current state
and falsification history.
