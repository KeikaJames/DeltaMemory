"""Exp10 qualitative generation.

Generates verbatim text for 5 fixed configs (no_bank / alpha0 / raw / mhc_dynlopi /
mhc_dynlopi_beta) across CPQP philosophy and history counterfactual cases.

For each case:
  - on_topic prompt: inject counterfactual → expect target_new to appear
  - off_topic prompt: inject same bank → expect no leak

LOPIState reset rule: reset ONCE per case × arm before model.generate().
DO NOT reset inside the decode loop — prev_q must stay live across tokens.

Usage:
  python run_qualitative.py \\
    --model /path/to/Qwen3-4B-Instruct-2507 \\
    --cases qualitative_cases \\
    --out qual_runs/run_20260509 \\
    --alpha 0.05 --kappa 0.25 --beta 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import Variant, load_model
from experiments.atb_validation_v1._lib.multi_bank_runner import _propagate_lopi


_LOPI_CFG = dict(
    lopi_enabled=True,
    lopi_orthogonal=False,
    lopi_gaussian=True,
    lopi_derivative=True,
    lopi_profile_mode="auto",
)


# ---------------------------------------------------------------------------
# Config definitions

def _make_configs(alpha: float, kappa: float, beta: float) -> list[dict]:
    """Return the 5 fixed qualitative configs."""
    common = dict(
        method="anb",
        alpha=alpha,
        bank_key_mode="pre_rope",
        value_scale_mode="auto_rms_cap",
    )
    lopi_off = {k: False for k in _LOPI_CFG}
    return [
        {"config_id": "no_bank",
         "variant": Variant(name="no_bank", method="none", alpha=0.0,
                            bank_key_mode="pre_rope", value_scale_mode="auto_rms_cap"),
         "description": "No bank installed (pure model)."},
        {"config_id": "alpha0",
         "variant": Variant(name="alpha0", alpha=0.0, mhc_shield=False,
                            bank_merge_beta=1.0, **common, **lopi_off),
         "description": "Bank present, alpha=0 (bank installed but gated off)."},
        {"config_id": "raw",
         "variant": Variant(name="raw", alpha=alpha, mhc_shield=False,
                            bank_merge_beta=1.0, **common, **lopi_off),
         "description": f"Raw ATB, alpha={alpha}, no mHC, no LOPI."},
        {"config_id": "mhc_dynlopi",
         "variant": Variant(name="mhc_dynlopi", alpha=alpha, mhc_shield=True,
                            mhc_kappa=kappa, bank_merge_beta=1.0,
                            **common, **_LOPI_CFG),
         "description": f"mHC + Dynamic LOPI, kappa={kappa}, alpha={alpha}."},
        {"config_id": "mhc_dynlopi_beta",
         "variant": Variant(name="mhc_dynlopi_beta", alpha=alpha, mhc_shield=True,
                            mhc_kappa=kappa, bank_merge_beta=beta,
                            **common, **_LOPI_CFG),
         "description": f"mHC + LOPI + beta={beta}, kappa={kappa}, alpha={alpha}."},
    ]


# ---------------------------------------------------------------------------
# Case loading

def _load_cases(cases_dir: Path) -> list[dict]:
    """Load and normalize all qualitative cases.

    Returns a list of dicts, each representing a matched on+off topic pair:
    {
        case_id, domain, write_prompts (list[str]),
        on_topic_prompt, off_topic_prompt,
        target_new, target_true,
        target_new_strings, target_true_strings
    }
    """
    cases: list[dict] = []

    # CPQP philosophy cases: paired on/off entries.
    cpqp_path = cases_dir / "cpqp_selected.jsonl"
    if cpqp_path.exists():
        on_cases: dict[str, dict] = {}
        off_cases: dict[str, dict] = {}
        for line in cpqp_path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("topical_relevance") == "on_topic":
                on_cases[c["id"]] = c
            elif c.get("topical_relevance") == "off_topic":
                # Match by off_topic_pair_id
                pass
        # Match off-topic to on-topic via off_topic_pair_id
        for line in cpqp_path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("topical_relevance") == "off_topic":
                # Find the on-topic case that points to this one
                pair_id = c["id"]
                on_c = next(
                    (v for v in on_cases.values()
                     if v.get("off_topic_pair_id") == pair_id),
                    None,
                )
                if on_c is None:
                    continue
                # Build write prompts from all facts in the on-topic case
                write_prompts = []
                for fact in on_c.get("facts", []):
                    wp = (f"Fact: {fact['subject']} {fact['relation'].replace('_', ' ')}"
                          f" is {fact['target_new']}.")
                    write_prompts.append(wp)
                target_new = on_c["facts"][0]["target_new"] if on_c.get("facts") else ""
                target_true = on_c["facts"][0]["target_true"] if on_c.get("facts") else ""
                cases.append({
                    "case_id": on_c["id"],
                    "domain": "philosophy_complex",
                    "write_prompts": write_prompts,
                    "on_topic_prompt": on_c["prompt"],
                    "off_topic_prompt": c["prompt"],
                    "target_new": target_new,
                    "target_true": target_true,
                    "target_new_strings": [target_new] if target_new else [],
                    "target_true_strings": [target_true] if target_true else [],
                })

    # History cases.
    hist_path = cases_dir / "history_counterfactual.jsonl"
    if hist_path.exists():
        for line in hist_path.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            cases.append({
                "case_id": c["case_id"],
                "domain": c.get("domain", "history"),
                "write_prompts": [c["write_prompt"]],
                "on_topic_prompt": c.get("open_prompt") or c["read_prompt"],
                "off_topic_prompt": c["off_topic_prompt"],
                "target_new": c["target_new"],
                "target_true": c["target_true"],
                "target_new_strings": c.get("target_new_strings", [c["target_new"]]),
                "target_true_strings": c.get("target_true_strings", [c["target_true"]]),
            })

    return cases


# ---------------------------------------------------------------------------
# Bank write helpers

@torch.no_grad()
def _write_bank(patcher, bank, tok, write_prompts: list[str]) -> None:
    """Write all write_prompts into bank (patcher must NOT be installed)."""
    from deltamemory.memory.attn_native_bank import write_fact
    for i, wp in enumerate(write_prompts):
        write_fact(patcher, bank, tok,
                   write_prompt=wp,
                   fact_id=f"case_fact_{i}",
                   address=wp[:40])


# ---------------------------------------------------------------------------
# Generation

@torch.no_grad()
def _generate(
    model, tok, prompt: str, device: str,
    max_new_tokens: int = 160,
) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.eos_token_id,
    )
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tok.decode(generated, skip_special_tokens=True)


def _hits(text: str, strings: list[str]) -> bool:
    return any(s.lower() in text.lower() for s in strings)


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    p = argparse.ArgumentParser(description="Exp10 qualitative generation")
    p.add_argument("--model", required=True)
    p.add_argument("--cases", default="qualitative_cases")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--max-new-tokens", type=int, default=160)
    args = p.parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "qual_results.jsonl"

    cases_dir = Path(__file__).parent / args.cases
    cases = _load_cases(cases_dir)
    print(f"Loaded {len(cases)} qualitative cases.")

    configs = _make_configs(args.alpha, args.kappa, args.beta)
    print(f"5 fixed configs: {[c['config_id'] for c in configs]}")

    print(f"Loading {args.model} ...")
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)

    # Pre-compute LOPI profile once.
    _lopi_profile_cache = None
    _any_lopi = any(
        getattr(cfg["variant"], "lopi_enabled", False) for cfg in configs
    )
    if _any_lopi:
        from deltamemory.memory.attn_native_bank import fresh_bank
        from deltamemory.memory.lopi import LOPIConfig
        print("Computing LOPI residual profile...")
        _prof_bank = fresh_bank(model)
        _prof_bank.lopi_cfg = LOPIConfig(
            enabled=True, orthogonal=False, gaussian=True,
            derivative=True, profile_mode="auto",
        )
        _prof_bank.attach_lopi_profile(model, tok)
        _lopi_profile_cache = _prof_bank.lopi_state.profile
        del _prof_bank
        print("LOPI profile cached.")

    from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank

    patcher = AttnNativePatcher(model)
    n_written = 0

    with open(results_path, "w") as out_f:
        for case in cases:
            print(f"\n--- case: {case['case_id']} ({case['domain']}) ---")
            for prompt_type in ("on_topic", "off_topic"):
                prompt = (case["on_topic_prompt"] if prompt_type == "on_topic"
                          else case["off_topic_prompt"])
                for cfg in configs:
                    config_id = cfg["config_id"]
                    variant = cfg["variant"]
                    print(f"  {config_id} / {prompt_type} ...", end=" ", flush=True)

                    if variant.method == "none":
                        # No bank — pure model generation.
                        output_text = _generate(model, tok, prompt, args.device,
                                                args.max_new_tokens)
                    else:
                        # Build bank, apply LOPI, install patcher.
                        bank = fresh_bank(model)
                        bank.bank_key_mode = variant.bank_key_mode
                        bank.value_scale_mode = variant.value_scale_mode
                        bank.mhc_shield = getattr(variant, "mhc_shield", False)
                        bank.mhc_kappa = getattr(variant, "mhc_kappa", 1.0)
                        bank.bank_merge_beta = getattr(variant, "bank_merge_beta", 1.0)
                        _propagate_lopi(bank, variant, _lopi_profile_cache)

                        _write_bank(patcher, bank, tok, case["write_prompts"])

                        # Reset state once before decode; DO NOT reset inside decode.
                        if bank.lopi_state is not None and hasattr(bank.lopi_state, "reset"):
                            bank.lopi_state.reset()

                        patcher.install()
                        patcher.bank = bank
                        patcher.alpha = float(variant.alpha)
                        try:
                            output_text = _generate(model, tok, prompt, args.device,
                                                    args.max_new_tokens)
                        finally:
                            patcher.bank = None
                            patcher.alpha = 0.0
                            patcher.remove()
                        del bank

                    tnew_hit = _hits(output_text, case["target_new_strings"])
                    ttrue_hit = _hits(output_text, case["target_true_strings"])
                    prompt_echo = any(
                        wp[:30].lower() in output_text.lower()
                        for wp in case["write_prompts"]
                    )
                    print(f"target_new={'HIT' if tnew_hit else 'miss'} "
                          f"echo={'Y' if prompt_echo else 'N'}")

                    row = {
                        "case_id": case["case_id"],
                        "domain": case["domain"],
                        "config_id": config_id,
                        "prompt_type": prompt_type,
                        "prompt": prompt,
                        "output_text": output_text,
                        "target_new": case["target_new"],
                        "target_true": case["target_true"],
                        "target_new_hit": tnew_hit,
                        "target_true_hit": ttrue_hit,
                        "off_topic": (prompt_type == "off_topic"),
                        "off_topic_leak": (prompt_type == "off_topic" and tnew_hit),
                        "prompt_echo": prompt_echo,
                        "seed": args.seed,
                        "alpha": args.alpha,
                        "kappa": args.kappa,
                        "beta": args.beta,
                        "model": args.model,
                        "mhc_shield": getattr(variant, "mhc_shield", False),
                        "mhc_kappa": getattr(variant, "mhc_kappa", 1.0),
                        "bank_merge_beta": getattr(variant, "bank_merge_beta", 1.0),
                        "lopi_enabled": getattr(variant, "lopi_enabled", False),
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_written += 1

    print(f"\nQualitative generation complete: {n_written} rows → {results_path}")


if __name__ == "__main__":
    main()
