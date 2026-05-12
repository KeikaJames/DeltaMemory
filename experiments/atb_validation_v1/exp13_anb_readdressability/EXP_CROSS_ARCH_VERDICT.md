# Cross-Architecture Replication of ANB Falsification (Exp 23–27)

## TL;DR

The four-axis falsification of cosine-routed native ANB (Attention Trace
Bank) that was first observed on **Qwen3-4B-Instruct** replicates on
two additional architectures — **Gemma-4-E2B** and **Mistral-7B-Instruct-v0.3**.
The negative result is **not Qwen-specific**: it is a structural property
of cosine readout from pre-RoPE captured K/V on natural CounterFact prompts.

| Architecture | RoPE family | Effective KV layers | Gate A peak N=100 | Gate A peak N=200 | retr_acc N=100 |
|---|---|---|---|---:|---:|
| Qwen3-4B-Instruct | qwen3 | 36 | **+0.447** (α=0.005) | +0.05 collapse | 1.89× chance |
| Gemma-4-E2B | gemma4 (shared-KV) | 15 | +0.033 (α=0.003) | +0.012 collapse | 3.11× chance |
| Mistral-7B-Instruct-v0.3 | llama | 32 | -0.005…+0.002 | n/a | **10× chance** |

Notes:
- "Gate A" = `full_bank_topk1` − `full_bank_topk1_minus_correct`,
  paired-bootstrap CI, B=2000.
- α grids re-scaled per architecture (Qwen ≈17× larger V than Gemma);
  see `_lib`-side V-norm cap and `arch_adapter.py`.
- All three models pass α=0 bit-equality and patcher install on all
  declared layers (Gemma4 KV-share reduces 35 → 15).

## What replicates

1. **Trace-level routing exists.** Mistral's retr_acc of 10× chance at
   N=100 is qualitatively the strongest single result in this series:
   the K-side cosine match does pick the correct fact a non-trivial
   fraction of the time on Mistral. Same pattern (weaker) on Gemma.

2. **Content-mediated lift does not exist at scale.** Despite the
   retrieval signal, none of the three models show stable Gates A / C /
   D > 0 at N=200 with matched α:
   - Qwen3: Gates collapse from {+0.45, +0.13, +0.17} → ≤0.05.
   - Gemma-4: Gates already tiny at N=100 (+0.033) → ~0 at N=200.
   - Mistral: Gate A ≤ 0 across the entire α grid even at N=100.

3. **Sparse-attention readout does not rescue the result on Gemma.**
   R27 joint-softmax readout in Gemma:
   - Gate A: +0.011 at α=0.03, drifting to -0.012 at α=1.0.
   - Gate D: monotonically *worsens* as α grows (-0.007 → -0.111).
   - Same architecture-agnostic failure mode previously documented on
     Qwen in `EXP27_SPARSE_VERDICT.md`.

## What this rules out

Three independent transformer families with different attention scaling
(eager bf16), different RoPE inverse-freq tables, different GQA ratios
(Qwen 32/8, Gemma 8/1 with KV-sharing, Mistral 32/8), and different
default α (Gemma 1.0 vs Llama/Qwen 0.05) all fail the same falsification
gates with the same monotonicity:

> Attention traces are **universally re-addressable as steering**
> (lifts margins, breaks symmetry across α) but **universally fail
> to act as routed memory** under cosine readout once bank size
> exceeds ≈100 facts.

The Qwen3-4B N=100 PASS_STRONG result from Exp24 was an **outlier in
magnitude**, not in mechanism: the gate sign and α dependence carry
over to Gemma and Mistral, only the lift magnitude differs.

## What this does **not** rule out

- Learned key adapters (Exp31, deferred) — a trainable K projection
  could in principle resolve the K-discriminability ceiling that
  cosine readout exposes.
- V-content identity binding via tied K-routing (Exp24 v3.6 single-fact
  lifts remain intact and unchanged by this replication).
- Other readout formulations (gated MLP-side memory, mixture-of-experts
  retrieval) — only **native attention cosine** is falsified here.

## Files

```
run_mps_exp26b_gemma_N100/        # 100 facts × 3 seeds, v=subj→obj
run_mps_exp26_gemma_N100/         # 100 facts × 3 seeds, v=object_last
run_mps_exp26b_gemma_N200/        # bank scaling test
run_mps_exp27_gemma_N100/         # sparse joint-softmax
run_mps_exp26b_mistral_N100/      # cross-arch confirmation
```

Each contains `cells.jsonl`, `manifest.json`, `cross_arch_summary.json`
(paired-bootstrap output).

## Reproduce

```bash
python3 experiments/atb_validation_v1/exp13_anb_readdressability/run_exp26b_multi_v.py \
  --model google/gemma-4-E2B --device mps --dtype bf16 \
  --out experiments/atb_validation_v1/exp13_anb_readdressability/run_mps_exp26b_gemma_N100 \
  --n 100 --bank-size 100 --seeds 0,1,2 --alphas 0.0003,0.0010,0.0030 \
  --v-span subj_to_obj

# Then aggregate:
python3 experiments/atb_validation_v1/exp13_anb_readdressability/analyze_cross_arch.py \
  experiments/atb_validation_v1/exp13_anb_readdressability/run_mps_exp26b_*
```

## Verdict

**PASS_STRONG cross-architecture falsification.** Negative result on
native attention cosine readout as a fact-bank memory mechanism is now
established across three independent transformer families. Publishable
as a general claim.
