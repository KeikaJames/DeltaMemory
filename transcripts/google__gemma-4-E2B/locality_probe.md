# Stage 13F transcript — `locality_probe`

- **model**: `google/gemma-4-E2B`
- **device**: `mps` (`bfloat16`)
- **alpha**: 1.0
- **n_facts in bank**: 1
- **scenario**: Bank irrelevant to query; DM output must EQUAL baseline (locality).

## Bank contents

- `paris_mayor`: "The mayor of Paris is Hidalgo."

## Read prompt

```
Q: What is the capital of Japan?
A:
```

## Baseline (no injection)

```
Tokyo

Q: What is the capital of China?
A: Beijing
```

## DeltaMemory α=1.0

```
Tokyo

Q: What is the capital of China?
A: Beijing
```

**Criterion**: exact-match (locality)

**Result**: ✅ PASS

_Wall: 1.93s_
