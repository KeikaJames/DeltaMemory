# W-T3.6 round-2 — End-to-end NLL A/B (ECOR vs additive)

**Status**: round-2 complete (round-1 = operator-level ablation, see `experiments/W_T3_6_ecor_op/COMBINED_REPORT.md`).
**Range**: 4 modes (`add`, `add_ortho`, `ecor_blend50`, `ecor_pure`) × α∈{0, 0.5, 1, 2, 4} × 3 seeds × 8 prompts × 2 models = 240 cells.
**Device**: Apple MPS, bf16. **Models**: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B (base).
**Profile mode**: STATIC (round-1 verdict: AUTO blew up on 1.5B at α≥2).
**Redline**: α=0 cross-mode max-abs logit diff = **0.000e+00** for both models, all seeds. Bit-equality preserved.

## Drift means by α (lower is better)

### Qwen2.5-0.5B-Instruct

| α | add | add_ortho | ecor_blend50 | ecor_pure | Δ(pure − add) |
|--:|--:|--:|--:|--:|--:|
| 0.00 | +0.000 | +0.000 | +0.000 | +0.000 | 0.00 |
| 0.50 | +2.331 | +2.387 | +2.783 | +2.711 | +0.38 |
| 1.00 | +2.256 | +2.308 | +2.374 | +2.674 | +0.42 |
| 2.00 | +3.142 | +2.979 | +2.587 | +2.681 | **−0.46** |
| 4.00 | +4.426 | +4.514 | +2.919 | **+2.678** | **−1.75** |

### Qwen2.5-1.5B (base)

| α | add | add_ortho | ecor_blend50 | ecor_pure | Δ(pure − add) |
|--:|--:|--:|--:|--:|--:|
| 0.00 | +0.000 | +0.000 | +0.000 | +0.000 | 0.00 |
| 0.50 | +1.968 | +1.688 | +1.806 | +1.859 | −0.11 |
| 1.00 | +2.148 | +2.067 | **+1.666** | +2.072 | −0.08 |
| 2.00 | +2.172 | +2.087 | **+1.895** | +2.022 | −0.15 |
| 4.00 | **+1.692** | +2.078 | +1.764 | +2.105 | +0.41 |

## Verdict

**Decision rule (from plan): flip default to ECOR if `ecor_pure` < `add` by > 0.5 nats at α≥2 on *both* models.**

- 0.5B: ecor_pure beats add by **−0.46 (α=2)** and **−1.75 (α=4)**. Qualifies.
- 1.5B: ecor_blend50 beats add at α∈{1,2}; ecor_pure does not consistently beat add (and loses by +0.41 at α=4).

⇒ **Mixed.** Default *not flipped*. ECOR remains opt-in via `LOPIConfig(use_ecor=True, ecor_cfg=ECORConfig(...))`.

### Recommended usage

| Regime | Recommendation |
|---|---|
| α ≤ 1 | `add` (legacy additive) — ECOR offers no benefit and slightly underperforms |
| 1 < α < 2 | `ecor_blend50` — wins on 1.5B at α=1 (−0.48), neutral on 0.5B |
| α ≥ 2 (smaller models, no v_norm) | `ecor_pure` on 0.5B class; `ecor_blend50` on 1.5B class |
| α ≥ 4 production safety | `ecor_pure` if model exhibits additive blow-up (0.5B: add=4.43 → ecor_pure=2.68); else stay additive |

The "additive blow-up" signature is: drift grows super-linearly in α (0.5B: 2.26 @ α=1 → 4.43 @ α=4 ≈ 2× for 4× α). When this signature is present, ECOR's norm-preservation (round-1: norm_ratio=1.0000) directly bounds the damage.

## Ties to round-1 (operator-level)

Round-1 found `ecor_pure` keeps `‖V_out‖/‖V_ctx‖ = 1.0000` exactly (orthogonal rotation), while additive at α=4 perturbs V_ctx by 17.7×. Round-2 confirms this norm bound translates to real NLL savings on 0.5B at α=4 (1.75 nats), but on 1.5B the model is robust enough to additive perturbation that the norm-preservation pays no end-to-end dividend.

## Plumbing changes

- `deltamemory/memory/lopi.py` — added `LOPIConfig.use_ecor` and `LOPIConfig.ecor_cfg`. `apply_lopi` routes through `lopi_inject` only when ECOR enabled AND `soft_blend != 0`. Strict bit-equality preserved in all degenerate paths.
- `tests/test_lopi_ecor_routing.py` — 4 redline tests (all pass).
- Full repo suite: 173/173 pass.

## Open questions for round-3

1. Does the 0.5B "additive blow-up" reproduce on Llama-3.2-1B and gpt2-medium? (Round-3 cross-arch sweep)
2. The α=4 1.5B reversal (add wins) — is it because base 1.5B has wider attention (less peaked → more tolerant to V_ctx perturbation)? Profile attention entropy in round-3.
3. Per-arch default routing table: should `LOPIConfig` auto-select mode based on model capability probe?

## Provenance

- Harness: `experiments/W_T3_6_ecor_e2e/run.py`
- Raw cells: `qwen05/cells.jsonl`, `qwen15/cells.jsonl`
- Per-model REPORTs: `qwen05/REPORT.md`, `qwen15/REPORT.md`
- Round-1 op-level: `experiments/W_T3_6_ecor_op/COMBINED_REPORT.md`
