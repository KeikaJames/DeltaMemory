# A.2 per-method dispatch — Qwen3-4B (GB10, 200 cells)

Verifies the FINDING_arm_method_mismatch.md hypothesis: that arms only
deviate from control when run against their *designed* test-vehicle
method. Result: partial confirmation, plus a new architectural finding.

## Run config
- model: Qwen/Qwen3-4B-Instruct-2507 (cuda bf16)
- n_prompts=10, seeds={0,1}, alphas={0,1}
- subsets: caa_arm (control + A5), lopi_arm (control + A3 + A6)
- scar_arm + bank_arm SKIPPED (--method scar / attn_native not registered)

## H_X3.0 α=0 redline
✅ **supported** — all arms identical to control at α=0 across both subsets.

## A.2 per-arm verdict at α=1 (paired by prompt × seed)

| subset | arm | n_pairs | median Δ vs control | mean Δ | n_pos | bites? |
|---|---|---:|---:|---:|---:|---:|
| caa_arm  | **A5** (random steering) | 20 | **+0.9722** | **+1.1266** | **20/20** | ✅ NECESSARY |
| lopi_arm | A3 (η_σ=1)               | 20 | +0.0000 | +0.0000 | 0/20 | ❌ NO-OP |
| lopi_arm | A6 (θ=0)                 | 20 | +0.0000 | +0.0000 | 0/20 | ❌ NO-OP |

## Findings

### 1. CAA target-mean steering is NECESSARY (A5 verdict)
A5 replaces the calibrated CAA target-mean vector with a seed-pinned
random unit vector. At α=1, this degrades nll_new by +1.13 nats
*on every single (prompt, seed) pair* (n_pos=20/20, no exceptions).
This is an extremely strong necessity signal: removing the target-mean
calibration breaks the override mechanism completely.

**Implication for v0.5 headline**: the W.6 / X.2 counter-prior result
cannot be attributed to "any random direction at the right layer
works" — the *target-mean direction* is doing the work.

### 2. LOPI ablations A3 and A6 are no-ops on lopi_default ctx
A3 (force eta_sigma=1) and A6 (force theta=0) are wired through
`ablation_context` but produce **bit-identical** outputs to control.
This is unexpected and a new architectural finding: the
`LopiDefaultCtx` in `experiments/W6_counter_prior/run.py` does not
read the ablation switches that control eta_sigma / theta.

**Possible causes** (to investigate):
- LopiDefaultCtx instantiates LOPI with hard-coded defaults that
  bypass the ablation_context override mechanism.
- The `ablation_context` patches a different LOPI instance than the
  one wired into W6run.
- A3/A6 patches target a per-call site not invoked under
  lopi_default's static-config path.

### 3. Test-vehicle gap remains for A1/A2/A4
scar_arm and bank_arm cannot run because run.py only registers
methods `caa`, `lopi_default`, `none`. Resolution requires extending
the dispatcher (~50 LOC) to register `scar` (SCAR injector)
and `attn_native` (AttnNativePatcher + write_fact + forward_with_bank).

## Next actions (queued)

1. **Investigate A3/A6 LopiDefaultCtx hook miss** — read
   experiments/W6_counter_prior/run.py:LopiDefaultCtx to find why
   ablation_context isn't biting; either wire it or document the
   static-config limitation.
2. **Extend run.py with `scar` and `attn_native` methods** to unblock
   A1/A2/A4 verdicts.
3. **Redesign A7** (no α-shield) — current implementation
   numerically equivalent to control on CAA path.

