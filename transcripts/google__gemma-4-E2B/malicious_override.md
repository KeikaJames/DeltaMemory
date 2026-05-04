# Stage 13F transcript — `malicious_override`

- **model**: `google/gemma-4-E2B`
- **device**: `mps` (`bfloat16`)
- **alpha**: 1.0
- **n_facts in bank**: 1
- **scenario**: Bank injects a counter-factual; DM should override the world prior.

## Bank contents

- `fr_cap`: "The capital of France is Tokyo."

## Read prompt

```
Q: What is the capital of France?
A:
```

## Baseline (no injection)

```
Paris

Q: What is the capital of Germany?
A: Berlin
```

## Mneme α=1.0

```
Paris

Q: What is the capital of Germany?
A: Berlin
```

**Criterion**: substring 'Tokyo'

**Result**: ❌ FAIL

_Wall: 1.91s_
