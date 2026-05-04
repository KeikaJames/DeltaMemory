#!/usr/bin/env python3
"""Phase N: intervention demo — does DeltaMemory actually shift LLM output?

For each (fact, condition), we forward a *frozen* base LLM and compare:
  - B0  no memory                     (LLM alone)
  - B1  prompt-insertion              (fact prepended to context, LLM still frozen)
  - v3  attn-native bank              (DeltaMemory; LLM frozen, only K-projector trained)

Per fact we record:
  * top-5 next-token candidates with probability
  * log-prob assigned to the target token (the answer string's first token)
  * delta-logprob of the target between conditions

Output: markdown transcript at transcripts/v3_intervention/<model_short>/demo.md
        plus a JSON dump with raw numbers.

Red line: base LLM weights are never touched. v3 only attaches an attn-bank
through the patched forward; alpha=0 is bit-equal to no-memory.

Usage:
    python scripts/run_intervention_demo.py \\
        --model google/gemma-4-E2B --device mps --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher,
    forward_with_bank,
    fresh_bank,
    write_fact,
)

# 5 LAMA-style facts. Each has (fact_id, write_prompt, read_prompt,
# target_text). The read_prompt is constructed so the natural continuation is
# the target. address = a short paraphrase used by capture policy.
FACTS = [
    {
        "fact_id": "f1_paris_mayor",
        "subject": "the mayor of Paris",
        "object": "Anne Hidalgo",
        "write": "Fact: The mayor of Paris is Anne Hidalgo.",
        "read":  "Q: Who is the mayor of Paris?\nA:",
        "target": " Anne",
    },
    {
        "fact_id": "f2_eiffel_arch",
        "subject": "the architect of the Eiffel Tower",
        "object": "Gustave Eiffel",
        "write": "Fact: The architect of the Eiffel Tower is Gustave Eiffel.",
        "read":  "Q: Who designed the Eiffel Tower?\nA:",
        "target": " Gustave",
    },
    {
        "fact_id": "f3_mona_lisa",
        "subject": "the painter of the Mona Lisa",
        "object": "Leonardo da Vinci",
        "write": "Fact: The Mona Lisa was painted by Leonardo da Vinci.",
        "read":  "Q: Who painted the Mona Lisa?\nA:",
        "target": " Leonardo",
    },
    {
        "fact_id": "f4_relativity",
        "subject": "the discoverer of general relativity",
        "object": "Albert Einstein",
        "write": "Fact: General relativity was developed by Albert Einstein.",
        "read":  "Q: Who developed general relativity?\nA:",
        "target": " Albert",
    },
    {
        "fact_id": "f5_python_creator",
        "subject": "the creator of the Python language",
        "object": "Guido van Rossum",
        "write": "Fact: Python was created by Guido van Rossum.",
        "read":  "Q: Who created the Python programming language?\nA:",
        "target": " Guido",
    },
]


# Counter-prior facts: the object is *intentionally wrong*. The base LLM's
# prior assigns near-zero probability to the target. If DeltaMemory can lift
# the wrong target's log-prob meaningfully, that proves the bank is
# *injecting* information into the model — not just letting the model emit
# what it already knows. This is the gold-standard test for memory
# intervention: forcing the model to contradict its own prior.
FALSE_FACTS = [
    {
        "fact_id": "ff1_paris_mayor_napoleon",
        "subject": "the mayor of Paris",
        "object": "Napoleon Bonaparte",
        "write": "Fact: The mayor of Paris is Napoleon Bonaparte.",
        "read":  "Q: Who is the mayor of Paris?\nA:",
        "target": " Napoleon",
    },
    {
        "fact_id": "ff2_eiffel_arch_picasso",
        "subject": "the architect of the Eiffel Tower",
        "object": "Pablo Picasso",
        "write": "Fact: The architect of the Eiffel Tower is Pablo Picasso.",
        "read":  "Q: Who designed the Eiffel Tower?\nA:",
        "target": " Pablo",
    },
    {
        "fact_id": "ff3_mona_lisa_van_gogh",
        "subject": "the painter of the Mona Lisa",
        "object": "Vincent van Gogh",
        "write": "Fact: The Mona Lisa was painted by Vincent van Gogh.",
        "read":  "Q: Who painted the Mona Lisa?\nA:",
        "target": " Vincent",
    },
    {
        "fact_id": "ff4_relativity_newton",
        "subject": "the discoverer of general relativity",
        "object": "Isaac Newton",
        "write": "Fact: General relativity was developed by Isaac Newton.",
        "read":  "Q: Who developed general relativity?\nA:",
        "target": " Isaac",
    },
    {
        "fact_id": "ff5_python_lovelace",
        "subject": "the creator of the Python language",
        "object": "Ada Lovelace",
        "write": "Fact: Python was created by Ada Lovelace.",
        "read":  "Q: Who created the Python programming language?\nA:",
        "target": " Ada",
    },
]


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("_", "-")


def topk_table(logits: torch.Tensor, tokenizer, k: int = 5):
    probs = F.softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(probs, k=k)
    rows = []
    for p, i in zip(topv.tolist(), topi.tolist()):
        tok = tokenizer.decode([i]).replace("\n", "\\n")
        rows.append((tok, p))
    return rows


def target_logprob(logits: torch.Tensor, tokenizer, target_text: str) -> tuple[float, int, str]:
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not ids:
        return float("nan"), -1, "<empty>"
    tid = ids[0]
    logp = F.log_softmax(logits.float(), dim=-1)[tid].item()
    return logp, tid, tokenizer.decode([tid]).replace("\n", "\\n")


def baseline_logits(model, tokenizer, prompt: str, device: str) -> torch.Tensor:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    last = enc["attention_mask"].sum(dim=1).item() - 1
    return out.logits[0, last].detach()


def b1_prompt_logits(model, tokenizer, fact: dict, device: str) -> torch.Tensor:
    """Frozen LLM, fact prepended to read prompt. No bank, no patcher."""
    prompt = fact["write"] + "\n" + fact["read"]
    return baseline_logits(model, tokenizer, prompt, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--alpha", type=float, default=None,
                    help="Bank injection scale. If omitted, uses the per-arch "
                         "ArchAdapter.default_alpha (Gemma=1.0, Qwen3=0.05, "
                         "Llama/Qwen2=0.05, GLM-4=0.05).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--capture-policy", default="period")
    ap.add_argument("--kproj", default=None,
                    help="Path to a trained KProjectorBank .pt (default: identity-init / v3 raw bank)")
    ap.add_argument("--label", default="v3",
                    help="Label for the bank condition column in the report (default: v3)")
    ap.add_argument("--false-facts", action="store_true",
                    help="Use FALSE_FACTS (counter-prior) instead of FACTS. "
                         "Tests whether the bank can override the LLM's prior "
                         "(e.g. claim Mona Lisa was painted by van Gogh).")
    args = ap.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    facts_tag = "FALSE" if args.false_facts else "TRUE"
    out_dir = Path(
        args.out_dir
        or f"transcripts/v31_intervention/{short_name(args.model)}-{args.device}-{facts_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== loading {args.model} on {args.device} ({args.dtype}) ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    patcher = AttnNativePatcher(model)
    print(f"  adapter = {patcher.adapter.name}, num_layers = {patcher.num_layers}",
          flush=True)

    if args.alpha is None:
        args.alpha = float(patcher.adapter.default_alpha)
        print(f"  alpha (auto from adapter.default_alpha) = {args.alpha}", flush=True)
    else:
        print(f"  alpha (user override) = {args.alpha}", flush=True)

    kproj = None
    if args.kproj:
        from deltamemory.memory.k_projector import KProjectorBank
        kproj = KProjectorBank.load(args.kproj)
        kproj = kproj.to(args.device)
        print(f"  k-projector loaded from {args.kproj}", flush=True)

    results = []
    facts_list = FALSE_FACTS if args.false_facts else FACTS
    print(f"  facts mode = {'FALSE (counter-prior)' if args.false_facts else 'TRUE'}; "
          f"{len(facts_list)} facts", flush=True)
    for fact in facts_list:
        print(f"\n--- {fact['fact_id']}: {fact['subject']} -> {fact['object']} ---",
              flush=True)
        b0 = baseline_logits(model, tok, fact["read"], args.device)
        b1 = b1_prompt_logits(model, tok, fact, args.device)

        bank = fresh_bank(model)
        if kproj is not None:
            bank.k_projector = kproj
        write_fact(patcher, bank, tok,
                   write_prompt=fact["write"],
                   fact_id=fact["fact_id"],
                   address=fact["object"],
                   policy=args.capture_policy)
        v3 = forward_with_bank(patcher, bank, tok, fact["read"], alpha=args.alpha)

        b0_lp, tid, ttok = target_logprob(b0, tok, fact["target"])
        b1_lp, _, _ = target_logprob(b1, tok, fact["target"])
        v3_lp, _, _ = target_logprob(v3, tok, fact["target"])

        rec = {
            "fact_id": fact["fact_id"],
            "subject": fact["subject"],
            "object": fact["object"],
            "target_token": ttok,
            "target_id": tid,
            "B0_no_memory": {
                "target_logprob": b0_lp,
                "top5": topk_table(b0, tok),
            },
            "B1_prompt": {
                "target_logprob": b1_lp,
                "top5": topk_table(b1, tok),
                "delta_vs_B0": b1_lp - b0_lp,
            },
            "v3_attn_bank": {
                "target_logprob": v3_lp,
                "top5": topk_table(v3, tok),
                "delta_vs_B0": v3_lp - b0_lp,
                "delta_vs_B1": v3_lp - b1_lp,
            },
        }
        results.append(rec)
        print(f"  target={ttok!r}  B0={b0_lp:+.3f}  B1={b1_lp:+.3f}  v3={v3_lp:+.3f}"
              f"  v3-B0={v3_lp-b0_lp:+.3f}", flush=True)

    # ---- markdown transcript -----------------------------------------------
    md = []
    md.append(f"# DeltaMemory v3 Intervention Demo — `{args.model}`")
    md.append("")
    md.append(f"- adapter: `{patcher.adapter.name}`  | layers: {patcher.num_layers}  "
              f"| device: `{args.device}`  | dtype: `{args.dtype}`  | alpha: {args.alpha}  "
              f"| capture_policy: `{args.capture_policy}`")
    md.append(f"- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)")
    md.append(f"- K-projector: `{'enabled' if False else 'identity-init'}` "
              "(this is the *raw* attn-native bank without trained projector — "
              "the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo "
              "shows the *channel* works before retraining)")
    md.append("")
    md.append("## Conditions")
    md.append("- **B0** no memory: frozen LLM alone")
    md.append("- **B1** prompt-insertion: same LLM, fact prepended to context")
    md.append("- **v3** attn-native bank: same LLM, fact written into per-layer "
              "K/V bank, alpha-weighted merge into attention softmax")
    md.append("")
    md.append("## Per-fact log-prob of the target token")
    md.append("")
    md.append("| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        v3 = r["v3_attn_bank"]
        b1 = r["B1_prompt"]
        md.append(f"| {r['fact_id']} | `{r['target_token']}` "
                  f"| {r['B0_no_memory']['target_logprob']:+.3f} "
                  f"| {b1['target_logprob']:+.3f} "
                  f"| {v3['target_logprob']:+.3f} "
                  f"| {v3['delta_vs_B0']:+.3f} "
                  f"| {v3['delta_vs_B1']:+.3f} |")
    md.append("")
    md.append("## Top-5 next-token per fact (showing the bank's effect on the distribution)")
    for r in results:
        md.append(f"\n### {r['fact_id']} — Q about *{r['subject']}* (target = `{r['target_token']}`)")
        for cond_name, cond_key in [("B0 no memory", "B0_no_memory"),
                                     ("B1 prompt", "B1_prompt"),
                                     ("v3 attn-bank", "v3_attn_bank")]:
            md.append(f"\n**{cond_name}**")
            md.append("| rank | token | prob |")
            md.append("|---:|---|---:|")
            for i, (t, p) in enumerate(r[cond_key]["top5"], 1):
                md.append(f"| {i} | `{t}` | {p:.4f} |")
    md.append("")
    md.append("## Aggregate")
    n = len(results)
    mean_d_v3_b0 = sum(r["v3_attn_bank"]["delta_vs_B0"] for r in results) / n
    mean_d_b1_b0 = sum(r["B1_prompt"]["delta_vs_B0"] for r in results) / n
    mean_d_v3_b1 = sum(r["v3_attn_bank"]["delta_vs_B1"] for r in results) / n
    md.append(f"- mean Δ logprob v3 − B0 = **{mean_d_v3_b0:+.3f}**")
    md.append(f"- mean Δ logprob B1 − B0 = **{mean_d_b1_b0:+.3f}**  (prompt-insertion ceiling)")
    md.append(f"- mean Δ logprob v3 − B1 = **{mean_d_v3_b1:+.3f}**  "
              "(positive → bank > prompt; negative → still room to grow)")
    md.append("")
    md.append("## Reproduction")
    md.append(f"```")
    md.append(f"python scripts/run_intervention_demo.py --model {args.model} "
              f"--device {args.device} --dtype {args.dtype} --alpha {args.alpha}")
    md.append(f"```")

    md_path = out_dir / "demo.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    json_path = out_dir / "demo.json"
    json_path.write_text(json.dumps({"model": args.model,
                                       "adapter": patcher.adapter.name,
                                       "alpha": args.alpha,
                                       "results": results}, indent=2),
                          encoding="utf-8")
    print(f"\nwrote {md_path}\nwrote {json_path}")


if __name__ == "__main__":
    main()
