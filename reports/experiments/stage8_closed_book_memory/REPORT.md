# Stage 8: Closed-book recall via address-keyed fast-weight bank

## Headline result

A frozen `google/gemma-4-E2B` augmented with (i) a small Writer MLP that
encodes a value into a per-slot injection vector, (ii) a tiny KeyProjector
that maps mean-pooled address-token embeddings into a learned key space, and
(iii) a `FastWeightBank` of `(value_vec, address_key)` pairs, can recall
single-token answers in **closed-book** mode — i.e. with the value token
absent from the prompt at read time — at scale up to N=4096 distinct facts,
on a single NVIDIA GB10 GPU.

This is the first result in this repo where the model's prompt at read time
contains *only* the address (no value, no in-context "card"). The mechanism
under test is therefore not in-context binding (Stage 6 Phase 2) but
**persistent address-keyed memory**.

## Protocol

Two-phase forward, single seed (3-seed confirmation pending).

```text
write phase, per fact (slot, address, value):
  v = Writer(embed(value_token))                  # learned writer
  k = KeyProjector(mean(embed(address_tokens)))   # learned key
  bank.write(slot, v, k)                          # persistent

read phase, closed-book (no value in prompt):
  prompt = "Atlas slot {address}\nRecall the payload value for this slot. The value is"
  q  = KeyProjector(mean(embed(address_tokens_of_prompt)))
  s* = argmax_s cosine(q, bank.k[s])              # content-addressed retrieval
  v* = bank.v[s*]
  inject v* (scaled by alpha) into the last hidden state of the frozen base
  logits = lm_head(modified_hidden)
  predict argmax over vocab; compare to value_token
```

Joint training objective: answer-token cross-entropy + InfoNCE retrieval
loss over the in-batch slots.

Frozen base parameters: 0 trainable. Trainable params: Writer ~12M (input
embed dim x 4 expansion x 2) + KeyProjector ~0.5M.

## Results (Gemma-4-E2B, bf16, GB10/CUDA, seed=0, 1500 steps)

| N facts | bank inject (oracle slot) top1 | bank inject (retrieved slot) top1 | address recall@1 | swap-paired flip | no-memory top1 |
|---:|---:|---:|---:|---:|---:|
|  128 | 1.000 | 0.969 | 0.969 | 1.000 | 0.000 |
| 1024 | 1.000 | 0.934 | 0.931 | 1.000 | 0.000 |
| 4096 | 1.000 | 0.838 | 0.832 | 1.000 | 0.000 |

Closed-book NLL drops from ~22.7 (no memory) to 0.8 / 1.7 / 4.1 nats at
N=128/1024/4096. The per-fact mass of the answer token rises from
effectively zero to argmax in 84%+ of cases at N=4096.

## Hard gates

| Gate | Requirement | N=128 | N=1024 | N=4096 |
|---|---|:-:|:-:|:-:|
| G1 closed-book recall | retrieved top1 ≥ 0.80 | ✅ 0.969 | ✅ 0.934 | ✅ 0.838 |
| G5 address-binding (swap) | paired-flip ≥ 0.80 | ✅ 1.000 | ✅ 1.000 | ✅ 1.000 |
| G6 no leakage | no-memory top1 ≤ 0.05 | ✅ 0.000 | ✅ 0.000 | ✅ 0.000 |
| GR retrieval recall | recall@1 ≥ 0.95 | ✅ 0.969 | ❌ 0.931 | ❌ 0.832 |

G1 + G5 + G6 pass at all three scales. The retrieval recall gate (GR)
becomes the bottleneck above N≈1024: when retrieval lands on the right
slot, the bank delivers the right answer with oracle-level fidelity (top1
of `bank_inject_oracle` = 1.000 at every N). The capacity loss above
N=1024 is therefore localised to the key projector, not to the
fast-weight bank itself.

## What this proves vs. what it does not

Proves:
- The bank carries the answer (G5: replace slot s with s+1 → output flips
  to neighbor's value at 100%).
- The address tokens alone, without the bank, do not encode the answer
  (G6: no-memory top1 = 0).
- The mechanism is not a per-fact RAG over text: at read time the prompt
  contains no value tokens. The bank slot, populated at write time and
  selected via cosine retrieval over learned keys, is the only carrier.
- The frozen Gemma-4-E2B base parameters are not edited.

Does not prove:
- Generalisation to natural-language facts. The pilot uses synthetic
  single-token color codes, both because LAMA-UHN is unreachable from the
  GB10 box (no internet to huggingface.co) and because synthetic data
  isolates the *mechanism* from pretraining-knowledge confounds.
- Multi-seed robustness. Pilot is seed 0 only.
- Dominance over a matched-budget RAG baseline. Stage 8.5 head-to-head
  pending.
- Interference resistance. Sequential-write Stage 8.3 pending.

## Negative result, honest reporting

At N=4096 the address retrieval recall@1 drops to 0.832, meaning ~17% of
queries land on the wrong bank slot. The bank itself is healthy at every
slot count (oracle top1 stays at 1.000), so the failure is in the key
projector under in-batch InfoNCE with batch size 16 and 1500 steps. This
is fixable (more steps, hard-negative mining, larger key dim) but the
present pilot does not yet pass the strict GR ≥ 0.95 gate at scale.

## Position vs. literature

| Family | Representative | Stage 8 contribution |
|---|---|---|
| RAG / Memorizing Transformers | `arXiv:2203.08913` | We do not retrieve text into the prompt. Retrieval lands on a parametric slot, not a chunk. |
| Knowledge editing (ROME / MEMIT) | `arXiv:2202.05262`, `2210.07229` | We do not edit base MLPs. The bank is an external, query-conditioned, run-time-computed update at the LM-head input. |
| Test-time training (Titans) | `arXiv:2501.00663` | Not a recurrent state. Slot-level, address-keyed, persistent across steps. |
| Fast-weight programmers | `arXiv:2112.02641` | Grounded in a frozen modern LLM, supervised end-to-end on answer-token CE. |

## Reproduction

```bash
# Mac side: rsync repo + HF cache to GB10 (one-time)
rsync -aP ~/.cache/huggingface/hub/models--google--gemma-4-E2B/ \
  gabira@192.168.1.108:~/.cache/huggingface/hub/models--google--gemma-4-E2B/
rsync -aP /path/to/RCV-HC/ gabira@192.168.1.108:~/projects/RCV-HC/

# GB10 side
ssh 192.168.1.108
cd ~/projects/RCV-HC
.venv-gb10/bin/python3 scripts/run_stage8.py \
  --n-facts 4096 --steps 1500 --device cuda --dtype bfloat16 --seed 0 \
  --report-dir reports/experiments/stage8_v2_n4096_seed0
```

Wall-clock on GB10: ~5 min for N=128, ~12 min for N=1024, ~25 min for
N=4096 (single seed, 1500 steps, bf16).

## Artifacts

- `reports/experiments/stage8_v2_n{128,1024,4096}_seed0/delta_experiment_summary.json`
- `docs/figures/fig6_stage8_capacity.svg`
- `docs/figures/stage8_summary.json`
- `scripts/run_stage8.py` (orchestrator)
- `scripts/generate_stage8_figure.py` (figure generator)

## Next

- 3-seed confirmation at N ∈ {128, 1024, 4096}.
- Tune key projector to pass GR ≥ 0.95 at N=4096 (more steps, hard
  negatives, larger key_dim).
- Stage 8.3 interference: sequential write 1..4096, retention curve on
  early slots.
- Stage 8.5 RAG vs ours head-to-head on the same factual pool.
- Stage 8.2 LAMA-UHN single-token transfer (offline curated subset).
