"""Phase X.3 — CAA injector smoke test.

Load GPT-2 (gpt2-small), calibrate a steering vector from 5 pos/5 neg prompt
pairs about "memory facts", inject at α=1 at the mu_arch layer, generate a
continuation, and print before/after for visual sanity check.

No assertions.  Pass = no exception raised.
"""
from __future__ import annotations

import sys
import textwrap

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from deltamemory.memory.caa_injector import CAAConfig, CAAInjector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "gpt2"
PROMPT = "The key fact to remember about elephant memory is"
MAX_NEW_TOKENS = 40
DEVICE = "cpu"

POS_TEXTS = [
    "Elephants have exceptional long-term memory and remember their herd.",
    "Elephants never forget familiar faces and important locations.",
    "An elephant's memory is legendary — it can recall events decades later.",
    "Elephants remember water sources and migration routes across generations.",
    "The hippocampus of elephants supports remarkable episodic memory recall.",
]

NEG_TEXTS = [
    "Goldfish famously forget everything within seconds.",
    "Short-term memory in insects is extremely limited.",
    "Many animals have no recollection of past events whatsoever.",
    "Mayflies live only a day and retain no long-term memories at all.",
    "Simple organisms such as worms lack any persistent memory system.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, injector=None) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    ctx = injector if injector is not None else __import__("contextlib").nullcontext()
    with ctx, torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _print_section(title: str, text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)
    print(textwrap.fill(text, width=70))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval().to(DEVICE)

    print(f"Model layers: {model.config.n_layer}, hidden: {model.config.n_embd}")

    # --- Baseline (no injection) ---
    baseline_text = _generate(model, tokenizer, PROMPT)
    _print_section("BASELINE (no injection)", baseline_text)

    # --- Calibrate steering vector ---
    cfg = CAAConfig(inject_layer="mu_arch", alpha=1.0)
    inj = CAAInjector(model, cfg, tokenizer=tokenizer, device=DEVICE)
    print(f"\nCalibrating steering vector ({len(POS_TEXTS)} pos / {len(NEG_TEXTS)} neg) …")
    s = inj.calibrate(POS_TEXTS, NEG_TEXTS)
    resolved_layer = inj._resolve_layer()
    print(f"  Steering vector norm : {s.norm().item():.4f}")
    print(f"  Inject layer (mu_arch): {resolved_layer}")

    # --- Injected generation ---
    injected_text = _generate(model, tokenizer, PROMPT, injector=inj)
    _print_section(f"WITH CAA INJECTION (α=1, layer={resolved_layer})", injected_text)

    # --- α=0 sanity check: must equal baseline ---
    cfg_zero = CAAConfig(inject_layer=resolved_layer, alpha=0.0)
    inj_zero = CAAInjector(model, cfg_zero)
    inj_zero.steering_vector = s

    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    with torch.no_grad():
        base_logits = model(input_ids=input_ids, use_cache=False).logits
    with inj_zero, torch.no_grad():
        zero_logits = model(input_ids=input_ids, use_cache=False).logits
    diff = (base_logits - zero_logits).abs().max().item()
    print(f"\n[α=0 bit-equal check] max_abs_diff = {diff}  →  {'PASS ✓' if diff == 0.0 else 'FAIL ✗'}")

    print("\nSmoke run complete.\n")


if __name__ == "__main__":
    main()
