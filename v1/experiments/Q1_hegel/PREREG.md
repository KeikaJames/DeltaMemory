# Pre-Registration: Q.1 — Hegel Counterfactual Generation Runner

**version:** Q1.v1  
**status:** locked  
**authored:** 2026-05-21  
**experiment dir:** `experiments/Q1_hegel/`

---

## 1. Hypotheses

### H_Q1 (primary)
At α = 1.0 the model's greedy continuation contains the injected counterfactual
concept (string-match against the alias family defined in `prompts.jsonl`) at
**> 50 %** rate across the 3 seeds × 5 prompts = 15 cells per model.

Operationalisation: `contains_counterfact = True` if any element of the row's
`aliases` list appears as a case-insensitive substring in the decoded
continuation text.  Hit-rate threshold: **0.50** (exclusive).

### Red-line (bit-equality witness)
At α = 0.0 the sha1 of the generated continuation MUST equal the sha1 of the
no-injection baseline generation (same prompt, same seed, no hook installed).
This is recorded as `redline_ok: bool` in every α = 0 cell.  A single
`redline_ok = False` is a contract violation and invalidates the run.

---

## 2. Experimental grid

| Variable        | Values                                                     |
|-----------------|------------------------------------------------------------|
| Models          | `gpt2-medium`, `Qwen/Qwen2.5-0.5B`                         |
| Seeds           | 0, 1, 2                                                    |
| Prompts         | 5 (see `prompts.jsonl`)                                    |
| α               | 0.0, 0.5, 1.0, 2.0                                         |
| max_new_tokens  | 80                                                         |
| Decoding        | greedy (temperature = 0, do_sample = False)                |
| Total cells     | 2 models × 5 prompts × 3 seeds × 4 alphas = 120            |

---

## 3. Dataset

Five hand-crafted Hegel-style prompts defined verbatim in `prompts.jsonl`.

**Counterfactual injection**: each prompt has a `canonical` concept (what Hegel
actually argued) and a `counterfact` replacement injected via CAA.  The
canonical concept is replaced by a deliberately wrong alternative
(e.g. "Geometry" instead of "Spirit").

Prompt IDs: `hegel_001` – `hegel_005`.  Dataset sha1 is recorded in `env.json`
at run time.

---

## 4. Injection method

Contrastive Activation Addition (CAA; Rimsky+ 2024) as implemented in
`deltamemory.memory.caa_injector.CAAInjector`.

- Steering vector: `s = h(Fact_new) − h(Fact_true)` where  
  `Fact_new = "Fact: {subject} {relation} {counterfact}."` and  
  `Fact_true = "Fact: {subject} {relation} {canonical}."`
- Injection layer: auto-resolved via `CAAConfig(inject_layer="mu_arch")`.
- At α = 0 the hook is installed but is a mathematical no-op.

---

## 5. Success criterion

| Criterion                     | Threshold         | Scope            |
|-------------------------------|-------------------|------------------|
| H_Q1 hit-rate at α = 1.0      | > 0.50            | per model        |
| Red-line bit-equality at α = 0| all cells True    | per model        |

---

## 6. Authenticity clauses (from `docs/authenticity.md`)

- `env.json` written at **start** of run with all required fields:
  `commit`, `dirty`, `dirty_diff_sha1`, `prereg_version`, `dataset_sha1`,
  `torch`, `transformers`, `python`, `device`, `dtype`, `started_at`, `host`.
- All cell rows in `cells.jsonl`; no aggregate-only artifact.
- Each cell row carries `cell_id` (sha1 of canonical key), input identifiers,
  primitive measurements, and `dataset_sha1`.
- Generation transcripts committed verbatim in `transcripts/`; no trimming.
- Every seed recorded in the cell row and in env.json.
- α = 0 bit-equality witness row in `cells.jsonl` for every (model, prompt).
- Aggregate (`aggregate.py`) refuses to run if `cells.jsonl` is missing.

---

## 7. Deviations log

*(empty at pre-registration; filled in during REPORT.md if deviations occur)*
