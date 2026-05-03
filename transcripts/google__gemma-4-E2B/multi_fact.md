# Stage 13F transcript — `multi_fact`

- **model**: `google/gemma-4-E2B`
- **device**: `mps` (`bfloat16`)
- **alpha**: 1.0
- **n_facts in bank**: 5
- **scenario**: Bank has many facts; DM should route to the right one.

## Bank contents

- `paris_mayor`: "The mayor of Paris is Hidalgo."
- `tokyo_mayor`: "The mayor of Tokyo is Koike."
- `london_mayor`: "The mayor of London is Khan."
- `berlin_mayor`: "The mayor of Berlin is Wegner."
- `fr_cap`: "The capital of France is Paris."

## Read prompt

```
Q: Who is the current mayor of Tokyo?
A:
```

## Baseline (no injection)

```
Shintaro Ishihara

Q: What is the name of the current prime
```

## DeltaMemory α=1.0

```
The current mayor of Tokyo is Shintaro Ishihara.
Q: Who
```

**Criterion**: substring 'Koike'

**Result**: ❌ FAIL

_Wall: 2.01s_
