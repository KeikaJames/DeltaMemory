"""01_quickstart.py — minimal Mneme / Δ-Memory walkthrough.

Demonstrates:

1.  Loading a small HF causal LM on the auto-detected device.
2.  Wrapping it with :class:`AttnNativePatcher`.
3.  α=0 bit-equal sanity (logits with patcher @ α=0 == baseline).
4.  Writing a single fact about a fictional entity.
5.  Comparing α=0 vs α=1 generations and showing top-5 token shifts.

Run::

    python examples/01_quickstart.py
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamemory import AttnNativePatcher, fresh_bank, write_fact

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def logits_last(model, tok, prompt: str) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
    last = am.sum(dim=1).item() - 1
    return out.logits[0, last].detach()


def greedy(model, tok, prompt: str, max_new_tokens: int = 24) -> str:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


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

    # --- 1. α=0 bit-equal sanity ---------------------------------------
    prompt = "Tell me about Zorblax:"
    base = logits_last(model, tok, prompt).float()
    with patcher.patched(), patcher.injecting(bank, alpha=0.0):
        patched = logits_last(model, tok, prompt).float()
    diff = (base - patched).abs().max().item()
    print(f"[sanity] α=0 max-abs-diff = {diff:.3e}  (must be < 1e-5)")
    assert diff < 1e-5, "α=0 must be bit-equal"

    # --- 2. write 1 fact ------------------------------------------------
    write_fact(
        patcher, bank, tok,
        write_prompt="Zorblax is a planet in the Krell system.",
        fact_id="zorblax_planet",
        address="Zorblax",
    )
    print(f"[bank ] wrote 1 fact, bank size = {bank.size}")

    # --- 3. α=0 vs α=1 generation --------------------------------------
    with patcher.patched(), patcher.injecting(bank, alpha=0.0):
        gen0 = greedy(model, tok, prompt)
    with patcher.patched(), patcher.injecting(bank, alpha=1.0):
        gen1 = greedy(model, tok, prompt)
    print(f"[gen  ] α=0.0  →  {gen0!r}")
    print(f"[gen  ] α=1.0  →  {gen1!r}")

    # --- 4. top-5 token shift at first generated position --------------
    with patcher.patched(), patcher.injecting(bank, alpha=0.0):
        l0 = logits_last(model, tok, prompt).float()
    with patcher.patched(), patcher.injecting(bank, alpha=1.0):
        l1 = logits_last(model, tok, prompt).float()
    top0 = torch.topk(l0, 5)
    top1 = torch.topk(l1, 5)
    print("[top5 ] α=0.0:", [tok.decode([i]) for i in top0.indices.tolist()])
    print("[top5 ] α=1.0:", [tok.decode([i]) for i in top1.indices.tolist()])

    print("[done ] OK")


if __name__ == "__main__":
    main()
