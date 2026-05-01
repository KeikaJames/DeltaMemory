# Story A draft: Delta injection as an activation-control channel

## Claim boundary

The current evidence supports a conservative, publishable Story A:

> Layerwise Delta injection into frozen attention is a high-efficiency
> activation/task-mode control channel. It can strongly move a frozen Gemma
> model into the desired answer regime without prompt insertion or base-weight
> finetuning, but it should not be claimed as query-specific factual binding.

This is intentionally narrower than the original associative-memory claim.
Recent Stage 2 results make the boundary clearer: output-side and fast-weight
channels can train-fit payload-specific facts, while Q/V residual behaves more
like a global mode activator.

## Positive evidence

Existing cleaned runs show that Delta Q/V injection can sharply reduce answer
NLL relative to no memory / ordinary frozen attention on synthetic memory
suites:

| evidence | result |
|---|---|
| `question_only_query_eval32` | all-layer Delta Q/V mean NLL `5.5366` vs ordinary/no-memory `12.7923`, per-example win rate `1.0` |
| `long_distance_nolima_pilot` | Delta Q/V NLL `4.9367` vs ordinary attention `11.8879` |
| `paired_conflict_pilot` | Delta Q/V NLL `4.6836` vs ordinary attention `12.3341` |
| `oracle_span_payload_pilot` | Delta Q/V NLL `3.5718` vs no-memory `12.1946` |

The mechanism is non-prompt:

- no source text is inserted into the prompt;
- base Gemma parameters remain frozen;
- memory is injected through attention projection hooks;
- all-layer injection is supported.

## Negative evidence that defines the boundary

The same experiments repeatedly fail query-specific binding controls:

| control | observation |
|---|---|
| wrong-query / shuffled controls | often tied with correct Delta despite strong NLL drop |
| paired conflict margins | correct-vs-foreign margin advantage near zero in early pilots |
| oracle span Q/V payload | oracle address/value spans still do not make Q/V payload answer-specific |
| Stage 2B output-side | direct output channels train-fit but held-out top1 only `0.3125–0.3750` |
| Stage 2C LM-head LoRA | payload-specific margin flips appear, but held-out top1/paired flip remain `0.3750` |

This pattern matches the literature distinction between:

- activation steering / task vectors / soft prompts: strong behavior-mode
  steering;
- associative memory / model editing / fast weights: precise address-value
  binding requires short, localized, supervised information paths.

## Mechanistic interpretation

Q/V residual Delta is best interpreted as a task-mode activator:

```text
retrieved/written payload -> global layerwise Q/V residual -> answer-format/task-mode shift
```

It is not yet a precise key-value memory:

```text
query address -> selected payload identity -> answer token flip
```

The decisive symptoms are payload-swap and wrong-query failures. If any payload
from the right task family improves answer NLL, but paired payloads do not flip
the answer token, the channel is controlling mode rather than binding identity.

## Contribution

Story A can be written as a capability-and-boundary paper:

1. introduce Layerwise Delta Memory Injection as a frozen-backbone control
   interface;
2. show strong non-prompt NLL improvements across synthetic memory suites;
3. include strict controls showing it is not sufficient for factual binding;
4. position it against Activation Steering, Task Vectors, Soft Prompts, and
   memory-augmented transformers;
5. motivate Story B: query-conditioned fast weights / output-side adapters for
   true associative binding.

## Current status

Story A evidence is strong enough for a draft. It should be kept separate from
Story B. Combining them would overclaim the current Q/V residual mechanism.

