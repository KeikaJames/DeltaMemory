"""e08 interrupt-API end-to-end demo + smoke test.

This script demonstrates the v2/core/interrupt_api.py public API by running
4 standalone demos in sequence:

1. **Identity sanity** — bank-off baseline NLL on 5 factual prompts
2. **Preload LT memory** — preload_long_term(...) from Exp35b b-vectors
3. **Mid-inference interrupt** — inject a hidden latent mid-round
4. **Bank size tracking** — show bank.total_size() at every step

Usage:
    python3 v2/experiments/e08_interrupt_api_demo/run.py [--device mps] [--tiny]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from v2.core import (
    AttentionBank,
    LPLHeads,
    install_lpl_patch,
    LPLState,
    lpl_state_scope,
    load_model,
    data_io,
)
from v2.core.interrupt_api import interrupt, preload_long_term


# ---------------------------------------------------------------------------
# Demo samples: 5 factual prompts (reused across all demos)

DEMO_SAMPLES = [
    ("The capital of France is", "Paris"),
    ("The author of Harry Potter is", "J.K. Rowling"),
    ("The chemical symbol for water is", "H2O"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The speed of light in vacuum is approximately", "299,792 kilometers per second"),
]


# ---------------------------------------------------------------------------
# Helpers


def compute_nll_batch(model, tokenizer, prompts_with_answers, device):
    """Compute per-sample NLL (averaged over answer tokens)."""
    nlls = []
    for prompt, answer in prompts_with_answers:
        full_text = prompt + " " + answer
        enc = tokenizer(full_text, return_tensors="pt").to(device)
        input_ids = enc.input_ids
        # find answer start (simple heuristic: encode prompt, then answer starts at its end)
        prompt_enc = tokenizer(prompt, return_tensors="pt").input_ids
        ans_start = prompt_enc.shape[1]

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=enc.attention_mask, use_cache=False, return_dict=True)
            logits = out.logits
        # NLL on answer tokens
        pred_logits = logits[0, ans_start - 1 : -1, :]  # predict ans_start..end
        gold_ids = input_ids[0, ans_start:]
        if gold_ids.shape[0] == 0:
            nlls.append(0.0)
            continue
        nll = F.cross_entropy(pred_logits.float(), gold_ids).item()
        nlls.append(nll)
    avg_nll = sum(nlls) / max(len(nlls), 1)
    return avg_nll, nlls


def forward_lpl_k2(model, bank, heads, tokenizer, prompt, answer, device):
    """Run 2-round LPL forward: round 1 writes to bank, round 2 reads."""
    full_text = prompt + " " + answer
    enc = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    prompt_enc = tokenizer(prompt, return_tensors="pt").input_ids
    ans_start = prompt_enc.shape[1]

    # Round 1: write to bank (no heads in this demo → no auto-pause, but can interrupt)
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=input_ids, attention_mask=enc.attention_mask, use_cache=False)

    # Round 2: read from bank (if non-empty)
    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state2), torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=enc.attention_mask, use_cache=False, return_dict=True)
    logits = out.logits

    # Compute NLL on answer
    pred_logits = logits[0, ans_start - 1 : -1, :]
    gold_ids = input_ids[0, ans_start:]
    if gold_ids.shape[0] == 0:
        return 0.0
    nll = F.cross_entropy(pred_logits.float(), gold_ids).item()
    return nll


# ---------------------------------------------------------------------------
# Demo 1: Identity sanity (bank-off baseline)


def demo1_identity_sanity(model, tokenizer, device):
    """Run forward with empty bank → baseline NLL."""
    print("\n" + "=" * 70)
    print("DEMO 1: Identity Sanity Check (bank-off baseline)")
    print("=" * 70)

    # Temporarily disable LPL state to get pure base NLL
    model.lpl_state = None
    avg_nll, nlls = compute_nll_batch(model, tokenizer, DEMO_SAMPLES, device)

    print(f"✓ Base NLL (no bank): {avg_nll:.4f}")
    for i, (prompt, answer) in enumerate(DEMO_SAMPLES):
        print(f"  [{i+1}] '{prompt[:40]}...' → NLL={nlls[i]:.4f}")

    return {"avg_nll": avg_nll, "per_sample_nll": nlls}


# ---------------------------------------------------------------------------
# Demo 2: Preload preloaded latent bank


def demo2_preload_long_term(model, tokenizer, bank, heads, device, bank_layer, n_preload):
    """Preload Exp35b b-vectors into bank; re-run same prompts."""
    print("\n" + "=" * 70)
    print("DEMO 2: Preload Long-Term Memory")
    print("=" * 70)

    # Load bank.pt (Exp35b b-vectors)
    bank_pt_path = data_io.BANK_PT_DEFAULT
    if not bank_pt_path.exists():
        print(f"⚠ WARNING: bank.pt not found at {bank_pt_path}")
        print("  Skipping preload demo (using zeros as fallback).")
        # Fallback: create synthetic b-vectors
        hs = torch.zeros(n_preload, bank.hidden_size, device=device, dtype=torch.bfloat16)
    else:
        blob = data_io.load_bank_blob(str(bank_pt_path))
        entries = blob["entries"]
        # Take first n_preload keys with solo_pass=True
        keys = data_io.filter_keys(entries, split="train", solo_pass=True)[:n_preload]
        b_raw = data_io.b_stack_for_keys(entries, keys, target_norm=15.0, device=device, dtype=torch.float32)
        hs = b_raw.to(dtype=torch.bfloat16)
        print(f"✓ Loaded {hs.shape[0]} b-vectors from {bank_pt_path.name}")

    # Preload into bank at bank_layer (no projector, identity init)
    preload_long_term(bank, layer=bank_layer, hs=hs, freeze_after=True)
    print(f"✓ Preloaded {hs.shape[0]} vectors into layer {bank_layer} (frozen=True)")
    print(f"  Bank total size: {bank.total_size()}, layer {bank_layer} size: {len(bank.slots[bank_layer])}")

    # Re-run same prompts with bank
    nlls = []
    for prompt, answer in DEMO_SAMPLES:
        nll = forward_lpl_k2(model, bank, heads, tokenizer, prompt, answer, device)
        nlls.append(nll)
    avg_nll = sum(nlls) / max(len(nlls), 1)

    print(f"✓ NLL with preloaded bank (identity projector): {avg_nll:.4f}")
    for i, (prompt, answer) in enumerate(DEMO_SAMPLES):
        print(f"  [{i+1}] '{prompt[:40]}...' → NLL={nlls[i]:.4f}")

    return {"avg_nll": avg_nll, "per_sample_nll": nlls, "bank_total_size": bank.total_size()}


# ---------------------------------------------------------------------------
# Demo 3: Mid-inference interrupt


def demo3_mid_inference_interrupt(model, tokenizer, bank, heads, device, bank_layer):
    """Inject a fake 'user intervention' hidden latent mid-inference."""
    print("\n" + "=" * 70)
    print("DEMO 3: Mid-Inference Interrupt")
    print("=" * 70)

    # Pick one prompt (say, the first one)
    prompt, answer = DEMO_SAMPLES[0]
    print(f"Target prompt: '{prompt} {answer}'")

    # Un-freeze bank for interrupt
    bank.frozen = False

    # Simulate: after round 1, inject a synthetic hidden vector
    # (In real usage, this would be a b-vector for a specific entity/fact.)
    # Here we just create a random normalized vector as a placeholder.
    d = bank.hidden_size
    h_inject = torch.randn(d, device=device)
    h_inject = h_inject / h_inject.norm() * 15.0  # scale to match b-vector norms
    h_inject = h_inject.to(dtype=torch.bfloat16)

    print(f"✓ Injecting synthetic hidden vector (norm={h_inject.norm():.2f}) at layer {bank_layer}, round_idx=1")
    bank_size_before = bank.total_size()

    # Call interrupt API
    interrupt(bank, layer=bank_layer, h=h_inject, position=-1, round_idx=1)

    bank_size_after = bank.total_size()
    print(f"✓ Interrupt complete. Bank size: {bank_size_before} → {bank_size_after} (+{bank_size_after - bank_size_before})")

    # Re-freeze (optional, depending on workflow)
    bank.frozen = True

    # Run forward with augmented bank
    nll = forward_lpl_k2(model, bank, heads, tokenizer, prompt, answer, device)
    print(f"✓ NLL after interrupt: {nll:.4f}")

    return {
        "prompt": prompt,
        "answer": answer,
        "nll_after_interrupt": nll,
        "bank_size_before": bank_size_before,
        "bank_size_after": bank_size_after,
    }


# ---------------------------------------------------------------------------
# Demo 4: Bank size tracking


def demo4_bank_size_tracking(bank, bank_layer):
    """Print bank size at every layer and summary."""
    print("\n" + "=" * 70)
    print("DEMO 4: Bank Size Tracking")
    print("=" * 70)

    total = bank.total_size()
    print(f"Total bank size across all layers: {total}")
    print(f"Per-layer breakdown:")
    for l in range(bank.num_layers):
        size = len(bank.slots[l])
        if size > 0:
            print(f"  Layer {l:2d}: {size:4d} entries (tags: {bank.tags[l][:3]}{'...' if size > 3 else ''})")
    print(f"Target layer {bank_layer}: {len(bank.slots[bank_layer])} entries")

    layer_sizes = {l: len(bank.slots[l]) for l in range(bank.num_layers)}
    return {"total_size": total, "layer_sizes": layer_sizes}


# ---------------------------------------------------------------------------
# Main


def main():
    p = argparse.ArgumentParser(description="e08 interrupt-API demo")
    p.add_argument("--device", default="mps", help="torch device")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="model name")
    p.add_argument("--tiny", action="store_true", help="use Qwen3-1.7B for fast iteration")
    p.add_argument("--bank_layer", type=int, default=9, help="target layer for bank ops")
    p.add_argument("--n_preload", type=int, default=512, help="number of b-vectors to preload")
    args = p.parse_args()

    if args.tiny:
        args.model = "Qwen/Qwen3-1.7B-Instruct-2507"
        print(f"[--tiny mode] Using {args.model}")

    print("=" * 70)
    print("e08 Interrupt-API Demo")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Bank layer: {args.bank_layer}")
    print(f"Preload count: {args.n_preload}")

    # Load model + install LPL patch
    tokenizer, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"✓ Model loaded: {n_layers} layers, hidden_size={d}")

    # Create bank and heads (no training, just for structure)
    bank = AttentionBank(
        num_layers=n_layers,
        hidden_size=d,
        device=args.device,
        dtype=torch.bfloat16,
        max_per_layer=args.n_preload + 64,
    )
    heads = LPLHeads.fresh(
        n_layers,
        d,
        pause_bias=-20.0,
        bank_gate_bias=0.0,
        halt_bias=10.0,
        device=args.device,
        dtype=torch.float32,
    )

    # Install LPL patch
    install_lpl_patch(model)
    print("✓ LPL patch installed")

    # Run 4 demos in sequence
    results = {}

    # Demo 1: Identity sanity
    results["demo1_identity_sanity"] = demo1_identity_sanity(model, tokenizer, args.device)

    # Demo 2: Preload preloaded latent bank
    results["demo2_preload_long_term"] = demo2_preload_long_term(
        model, tokenizer, bank, heads, args.device, args.bank_layer, args.n_preload
    )

    # Demo 3: Mid-inference interrupt
    results["demo3_mid_inference_interrupt"] = demo3_mid_inference_interrupt(
        model, tokenizer, bank, heads, args.device, args.bank_layer
    )

    # Demo 4: Bank size tracking
    results["demo4_bank_size_tracking"] = demo4_bank_size_tracking(bank, args.bank_layer)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Demo 1 (base NLL):           {results['demo1_identity_sanity']['avg_nll']:.4f}")
    print(f"Demo 2 (preload NLL):        {results['demo2_preload_long_term']['avg_nll']:.4f}")
    print(f"Demo 3 (interrupt NLL):      {results['demo3_mid_inference_interrupt']['nll_after_interrupt']:.4f}")
    print(f"Demo 4 (bank total size):    {results['demo4_bank_size_tracking']['total_size']}")

    # Save JSON
    out_path = HERE / "e08_demo.json"
    out_data = {
        "model": args.model,
        "device": args.device,
        "bank_layer": args.bank_layer,
        "n_preload": args.n_preload,
        "results": results,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"✓ Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
