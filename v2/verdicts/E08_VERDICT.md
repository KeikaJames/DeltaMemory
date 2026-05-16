# E08 — Interrupt-API demo

**Status**: **DEFERRED / NOT RUN** — no quantitative data exists for this experiment.

**Headline**: The `interrupt(model, round, layer, pos, h_inject)` public API was scaffolded but no demo run (GSM8K oracle hint injection, user-hint encoding round-trip) produced JSON results.

---

## a. Reproduction command

```bash
# Not yet wired:
# python3 v2/experiments/e08_interrupt_api_demo/run.py --demo oracle_cot_gsm8k
```

## b. Seeds & sample size

N/A — no data produced.

## c. Raw data paths

`v2/experiments/e08_interrupt_api_demo/` exists but contains no `*.json` summaries.

## d. Numbers

None.

## e. Verdict

- **Status**: Deferred. The v2 plan listed this as a qualitative-primary experiment. Given that the e16-forgetting, e10, e11, e17 falsifiers have already shown the projector behaves as an adapter (not as content-addressable memory), the interrupt-API demo's expected qualitative effect ("inject oracle CoT → improved answer") is more parsimoniously explained by the prompt-conditioning side of the mechanism than by genuine latent-state read/write semantics. Building this demo without first resolving the memory-vs-adapter question would risk producing a result that looks like API memory but is the same template-conditioned distribution shift.

## f. Caveat

This verdict is a placeholder. Should the program pivot toward a *prompt-conditioning adapter* framing (rather than memory framing), the interrupt-API can still be demonstrated as a useful latent-injection tool, but its value claim must be downgraded from "memory write" to "controlled adapter activation."

## g. Implications

- Holding off on e08 until the memory-vs-adapter question is resolved at the framing level is the lower-waste decision.
- If/when run, e08 should be paired with a control that injects the *same* hidden vector permuted across feature dims (à la e16 A/B) — if injection still improves the answer, the "memory write" claim is again falsified.
