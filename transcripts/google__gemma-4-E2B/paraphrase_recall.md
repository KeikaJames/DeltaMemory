# Stage 13F transcript — `paraphrase_recall`

- **model**: `google/gemma-4-E2B`
- **device**: `mps` (`bfloat16`)
- **alpha**: 1.0
- **n_facts in bank**: 1
- **scenario**: Trained fact, query is a synonym/paraphrase.

## Bank contents

- `paris_mayor`: "The mayor of Paris is Hidalgo."

## Read prompt

```
Q: Who currently serves as the chief executive of the city of Paris?
A:
```

## Baseline (no injection)

```
Mayor

Q: What is the name of the city council in Paris?
```

## DeltaMemory α=1.0

```
Mayor

Q: What is the name of the city council in Paris?
```

**Criterion**: substring 'Hidalgo'

**Result**: ❌ FAIL

_Wall: 2.40s_
