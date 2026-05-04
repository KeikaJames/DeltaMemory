# Stage 13F transcript — `direct_recall`

- **model**: `google/gemma-4-E2B`
- **device**: `mps` (`bfloat16`)
- **alpha**: 1.0
- **n_facts in bank**: 1
- **scenario**: Trained fact, direct paraphrase-free query.

## Bank contents

- `paris_mayor`: "The mayor of Paris is Hidalgo."

## Read prompt

```
Q: Who is the mayor of Paris?
A:
```

## Baseline (no injection)

```
Georges Clemenceau

Q: Who is the Prime Minister of Great Britain?
```

## Mneme α=1.0

```
Bertrand Delanoe

Q: What is the name of the French president?
```

**Criterion**: substring 'Hidalgo'

**Result**: ❌ FAIL

_Wall: 3.01s_
