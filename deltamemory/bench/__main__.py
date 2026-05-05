"""dm-bench subcommand router.

Subcommands:
  info      Print version + public API exports + torch device info.
  smoke     Quick patch+inject sanity check (incl. alpha=0 redline).
  profile   Run U-LOPI residual profiler and print weight vector.
  replay    Re-aggregate an experiments/.../cells.jsonl via its sibling
            ``aggregate.py``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from deltamemory import (
    AttnNativePatcher,
    __version__,
    fresh_bank,
    profile_residuals,
    write_fact,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    return model, tok


# --------------------------------------------------------------------------- #
# subcommand: info
# --------------------------------------------------------------------------- #
def cmd_info(_args: argparse.Namespace) -> int:
    import torch

    import deltamemory as dm

    print(f"deltamemory  version: {__version__}")
    print(f"torch        version: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    print(f"  mps  available: {torch.backends.mps.is_available()}")
    print(f"  default device : {_resolve_device('auto')}")
    print()
    print("Public API (from deltamemory import *):")
    for name in sorted(getattr(dm, "__all__", [])):
        print(f"  - {name}")
    return 0


# --------------------------------------------------------------------------- #
# subcommand: smoke
# --------------------------------------------------------------------------- #
def cmd_smoke(args: argparse.Namespace) -> int:
    import torch

    device = _resolve_device(args.device)
    print(f"[smoke] model={args.model}  device={device}  alpha={args.alpha}")
    model, tok = _load_model(args.model, device)
    patcher = AttnNativePatcher(model)
    patcher.install()
    bank = fresh_bank(model)

    # Capture two facts.
    with patcher.capturing(bank=bank):
        write_fact(patcher, bank, tok,
                   write_prompt="The capital of France is Paris.",
                   fact_id="f1", address="capital_france")
        write_fact(patcher, bank, tok,
                   write_prompt="The Eiffel Tower is in Paris.",
                   fact_id="f2", address="eiffel")

    prompt = "Hello, world."
    enc = tok(prompt, return_tensors="pt").to(device)

    # Redline: alpha=0 must be bit-equal to unpatched.
    with torch.no_grad():
        baseline = model(**enc).logits

    with patcher.patched(), patcher.injecting(bank, alpha=0.0), torch.no_grad():
        patched_a0 = model(**enc).logits

    diff = (baseline - patched_a0).abs().max().item()
    print(f"[smoke] redline (alpha=0) max-abs-diff = {diff:.3e}")
    if diff > 1e-5:
        print("[smoke] redline VIOLATED")
        patcher.remove()
        return 1
    print("[smoke] redline OK")

    # User-requested alpha sanity forward.
    with patcher.patched(), patcher.injecting(bank, alpha=float(args.alpha)), torch.no_grad():
        out = model(**enc).logits
    print(f"[smoke] forward at alpha={args.alpha} ok, logits shape={tuple(out.shape)}")

    patcher.remove()
    return 0


# --------------------------------------------------------------------------- #
# subcommand: profile
# --------------------------------------------------------------------------- #
def cmd_profile(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    print(f"[profile] model={args.model}  device={device}")
    model, tok = _load_model(args.model, device)

    prompts = None
    if args.neutral_prompts:
        path = Path(args.neutral_prompts)
        prompts = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        print(f"[profile] {len(prompts)} prompts loaded from {path}")

    profile = profile_residuals(model, tok, prompts=prompts, device=device)
    weights = getattr(profile, "weights", None)
    if weights is None:
        print(f"[profile] result: {profile!r}")
    else:
        print(f"[profile] layer weights ({len(weights)}):")
        for i, w in enumerate(weights):
            print(f"  L{i:02d}: {float(w): .6f}")
    return 0


# --------------------------------------------------------------------------- #
# subcommand: replay
# --------------------------------------------------------------------------- #
def cmd_replay(args: argparse.Namespace) -> int:
    cells = Path(args.cells).resolve()
    if not cells.exists():
        print(f"[replay] cells file not found: {cells}", file=sys.stderr)
        return 2

    # Walk up from the cells file looking for sibling aggregate.py.
    candidates = []
    for parent in [cells.parent, *cells.parent.parents]:
        agg = parent / "aggregate.py"
        if agg.exists():
            candidates.append(agg)
            break
    if not candidates:
        print(f"[replay] no aggregate.py found near {cells}", file=sys.stderr)
        return 2
    aggregate = candidates[0]
    print(f"[replay] cells={cells}")
    print(f"[replay] aggregate={aggregate}")
    # Run aggregate.py in its own directory; it typically discovers cells.jsonl
    # by relative path.
    rc = subprocess.call(
        [sys.executable, str(aggregate)],
        cwd=str(aggregate.parent),
    )
    return rc


# --------------------------------------------------------------------------- #
# router
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dm-bench",
        description="Mneme bench CLI (Phase G-1).",
    )
    p.add_argument("--version", action="version",
                   version=f"dm-bench {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="SUBCMD")

    s_info = sub.add_parser("info", help="print version + public API + torch info")
    s_info.set_defaults(func=cmd_info)

    s_smoke = sub.add_parser(
        "smoke", help="patch+inject sanity check incl. alpha=0 redline")
    s_smoke.add_argument("--model", required=True)
    s_smoke.add_argument("--device", default="auto",
                         choices=["auto", "cpu", "mps", "cuda"])
    s_smoke.add_argument("--alpha", type=float, default=1.0)
    s_smoke.set_defaults(func=cmd_smoke)

    s_prof = sub.add_parser("profile", help="U-LOPI residual profiler")
    s_prof.add_argument("--model", required=True)
    s_prof.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    s_prof.add_argument("--neutral-prompts", default=None,
                        help="path to a text file (one prompt per line)")
    s_prof.set_defaults(func=cmd_profile)

    s_rep = sub.add_parser("replay", help="re-aggregate a cells.jsonl")
    s_rep.add_argument("--cells", required=True)
    s_rep.set_defaults(func=cmd_replay)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
