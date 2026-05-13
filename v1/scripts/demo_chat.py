"""Interactive REPL demo for Mneme v3 (frozen).

Compare three modes side-by-side on user-typed queries:
    1. baseline  — model alone (no bank, no prompt injection)
    2. prompt    — fact prepended to the prompt (B1 oracle baseline)
    3. v3        — frozen v3 attn-native bank with InfoNCE K-projector

Usage::

    python scripts/demo_chat.py --model google/gemma-4-E2B --device mps

Type ``write <fact>`` to add a fact to the bank (e.g. ``write Paris is in France.``).
Type ``ask <query>`` to query all three modes. Type ``reset`` to wipe the bank.
Type ``quit`` to exit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deltamemory.memory.attn_native_bank import (  # noqa: E402
    AttnNativePatcher,
    forward_with_bank,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.k_projector import KProjectorBank  # noqa: E402

ADDR_RE = re.compile(r"^(.+?)\s+(?:is|are|was|were)\s+.+\.?$", re.IGNORECASE)


def _topk_decode(tok, logits: torch.Tensor, k: int = 5) -> str:
    probs = logits.softmax(-1)
    vals, idx = probs.topk(k)
    parts = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        parts.append(f"{tok.decode([i]).strip()!r}({v:.2f})")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projector",
                    default="reports/cleanroom/stage14_kproj/k_projector.pt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[demo] loading {args.model} on {args.device}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(args.device).eval()

    patcher = AttnNativePatcher(model)
    proj = KProjectorBank.load(args.projector) if Path(args.projector).exists() else None
    bank = fresh_bank(patcher.model)
    if proj is not None:
        bank.k_projector = proj
        print(f"[demo] loaded K-projector {args.projector}")
    else:
        print("[demo] WARN: no projector — running raw v2 bank")

    facts: list[tuple[str, str]] = []  # (address, full_write_prompt)
    print()
    print("Commands: write <sentence> | ask <query> | reset | quit")
    print()

    while True:
        try:
            line = input("dm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in {"quit", "exit"}:
            return
        if line == "reset":
            bank = fresh_bank(patcher.model)
            if proj is not None:
                bank.k_projector = proj
            facts.clear()
            print("[bank cleared]")
            continue
        if line.startswith("write "):
            sent = line[len("write "):].strip()
            if not sent.endswith("."):
                sent = sent + "."
            m = ADDR_RE.match(sent)
            if not m:
                print("  could not parse address; expected 'X is Y.' form")
                continue
            address = m.group(1).strip()
            write_fact(patcher=patcher, bank=bank, tokenizer=tok,
                       write_prompt=sent, fact_id=f"f{len(facts)}",
                       address=address, policy="period")
            facts.append((address, sent))
            print(f"  wrote: {sent}  (address={address!r})")
            continue
        if line.startswith("ask "):
            q = line[len("ask "):].strip()
            with torch.inference_mode():
                ids = tok(q, return_tensors="pt").to(args.device)
                base_logits = model(**ids).logits[0, -1, :]
            print(f"  baseline : {_topk_decode(tok, base_logits)}")

            if facts:
                prefix = " ".join(f for _, f in facts) + "\n"
                with torch.inference_mode():
                    ids2 = tok(prefix + q, return_tensors="pt").to(args.device)
                    p_logits = model(**ids2).logits[0, -1, :]
                print(f"  prompt   : {_topk_decode(tok, p_logits)}")

            v3_logits = forward_with_bank(patcher, bank, tok, q, alpha=1.0)
            print(f"  v3 bank  : {_topk_decode(tok, v3_logits)}")
            continue
        print("  unknown command. try: write / ask / reset / quit")


if __name__ == "__main__":
    main()
