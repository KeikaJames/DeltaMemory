"""02_multifact_recall.py — 5-fact recall@20 with α sweep.

Writes 5 facts about distinct fictional entities into the bank, then for each
fact issues a query whose continuation should retrieve the entity.  Reports
the rank of the expected token at the first generated position, both raw and
swept across α ∈ {0, 0.5, 1, 2}.

Run::

    python examples/02_multifact_recall.py
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamemory import AttnNativePatcher, fresh_bank, write_fact

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

FACTS = [
    # (write_prompt, address, query, expected continuation token)
    ("Zorblax is a planet in the Krell system.",
     "Zorblax", "Zorblax is a planet in the", " Krell"),
    ("Pendrillin is a rare metal mined on Vega-7.",
     "Pendrillin", "Pendrillin is a rare metal mined on", " Vega"),
    ("Captain Mareva commands the starship Halycon.",
     "Mareva", "Captain Mareva commands the starship", " Halycon"),
    ("The Glimmerwood is enchanted by druid Olwen.",
     "Glimmerwood", "The Glimmerwood is enchanted by druid", " Olwen"),
    ("Quenchel is a cold beverage from the moon Tarsis.",
     "Quenchel", "Quenchel is a cold beverage from the moon", " Tarsis"),
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def last_logits(model, tok, prompt: str) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
    last = am.sum(dim=1).item() - 1
    return out.logits[0, last].detach().float()


def rank_of(logits: torch.Tensor, target: int) -> int:
    order = logits.argsort(descending=True)
    return int((order == target).nonzero(as_tuple=True)[0].item())


def main() -> None:
    device = pick_device()
    print(f"[setup] device={device}  model={MODEL_ID}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device).eval()
    print(f"[setup] loaded in {time.time()-t0:.1f}s")

    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)

    # write all 5 facts
    for i, (wp, addr, _q, _exp) in enumerate(FACTS):
        write_fact(patcher, bank, tok,
                   write_prompt=wp, fact_id=f"fact_{i}", address=addr)
    print(f"[bank ] wrote {bank.size} facts\n")

    # baseline (α=1) detail table ---------------------------------------
    print(f"{'expected':>10}  {'rank@α=0':>10}  {'rank@α=1':>10}  hit@20")
    print("-" * 50)
    for _wp, _addr, q, exp in FACTS:
        target_ids = tok.encode(exp, add_special_tokens=False)
        target = target_ids[0]
        with patcher.patched(), patcher.injecting(bank, alpha=0.0):
            r0 = rank_of(last_logits(model, tok, q), target)
        with patcher.patched(), patcher.injecting(bank, alpha=1.0):
            r1 = rank_of(last_logits(model, tok, q), target)
        hit = "✓" if r1 < 20 else "✗"
        print(f"{exp.strip():>10}  {r0:>10d}  {r1:>10d}  {hit}")

    # α sweep, recall@20 -------------------------------------------------
    print("\n[sweep] recall@20 across α:")
    for alpha in (0.0, 0.5, 1.0, 2.0):
        hits = 0
        for _wp, _addr, q, exp in FACTS:
            target = tok.encode(exp, add_special_tokens=False)[0]
            with patcher.patched(), patcher.injecting(bank, alpha=alpha):
                logits = last_logits(model, tok, q)
            if rank_of(logits, target) < 20:
                hits += 1
        print(f"  α={alpha:>3.1f}  recall@20 = {hits}/{len(FACTS)}")

    print("\n[done ] OK")


if __name__ == "__main__":
    main()
