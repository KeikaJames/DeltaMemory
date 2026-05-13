# Mneme v3 — qualitative demo prompts

These prompts are for **qualitative side-by-side comparison** (baseline /
prompt-insertion / v3 attn-native bank) using
`scripts/demo_chat.py`. They are NOT part of the held-out test split
and do not enter any quantitative claim. The Phase G recall@1 numbers
are the only quantitative evidence Mneme currently stands on; see
`reports/cleanroom/stage14_test_gemma4_e2b/REPORT.md`.

## Use

```bash
python scripts/demo_chat.py --model google/gemma-4-E2B --device mps
# write <fact>     -- store a fact in the bank (period-policy capture)
# ask <query>      -- score baseline / prompt / v3 at the next-token logit
# reset            -- wipe the bank
```

`ask` returns the top-5 next-token distribution under each condition,
so it is suited to **factual completion** style queries, not free-form
essay generation. For long-form generation you have to wrap the model
yourself with `forward_with_bank` in a generation loop.

## Example: factual recall (recall@1 style, in-bank vs. base)

```text
write Paris is the capital of France.
write Tokyo is the capital of Japan.
ask The capital of France is
ask The capital city of Japan is named
```

You should see the bank lift the gold answer from a low rank into the
top-1 in `ask` mode when the query paraphrases the address.

## Example: long-form generative prompt (QUALITATIVE only)

This kind of prompt does *not* fit the recall@1 harness; it is a
qualitative test of whether Mneme's bank corrupts long-form
generation when the bank carries facts unrelated to the prompt. Use
this with a custom generation loop, not with `ask`.

> 学界在重估黑格尔哲学遗产时，经常会因为他庞大的绝对唯心论体系，而对他辩证法的内核
> 产生一种目的论式的误读。如果我们暂时悬置对他体系封闭性的争议，重新回到《精神现象
> 学》和《逻辑学》的内部脉络中，我们需要极其严肃地审视黑格尔辩证法的积极的生成性力量。
>
> 请你系统且严谨地阐释：黑格尔是如何将辩证法从康德式的"消极的幻相逻辑（先验辩证论）"
> 中拯救出来，并赋予其本体论意义上的积极作用的？特别是，请深入剖析"否定之否定"在这
> 个过程中，如何不仅仅是一种逻辑推演的工具，而是作为一种具有巨大破坏力同时又极具建
> 设性的扬弃机制，推动了概念自我运动的历史展开？我希望你的论述能体现出足够的哲学史
> 纵深，语言要精准，直击其方法论的现代性价值。

The expected qualitative gates are:
1. **Empty-bank fidelity.** With no facts written, the v3 path must
   produce text indistinguishable from the no-bank baseline at this
   prompt length. (This is the alpha=0 / empty-bank invariant exercised
   in the unit tests, but the demo gives a human-readable check.)
2. **Topical bank no-harm.** Write a few facts about Hegel, Kant, or
   the *Phenomenology of Spirit* into the bank; the response should
   stay coherent and not start hallucinating bank-stored strings out
   of context. The bank is associative, not a string store.
3. **Off-topic bank no-harm.** Write a few facts about Paris/Tokyo
   capitals and re-ask. The response should be unchanged; bank
   attention should treat off-topic addresses as low-similarity.

A failure on (2) or (3) would indicate softmax dilution at high N is
not just hurting recall, it is also corrupting the sequence-side
distribution. That is one of the structural risks the Phase G
methodology amendment names explicitly.
