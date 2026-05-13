"""03_diagnostics.py — DiagnosticRecorder usage.

Wraps the model with both :class:`AttnNativePatcher` and
:class:`DiagnosticRecorder`, writes a few facts, runs a forward, and prints
captured per-layer signals (bank-attn entropy, residual norms, etc.).
The full record is dumped to ``/tmp/dm_diag.json``.

Run::

    python examples/03_diagnostics.py
"""
from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamemory import AttnNativePatcher, fresh_bank, write_fact

try:
    from deltamemory import DiagnosticRecorder
except ImportError:  # pragma: no cover — fallback in case top-level export disappears
    from deltamemory.diagnostics import DiagnosticRecorder

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

FACTS = [
    ("Zorblax is a planet in the Krell system.", "Zorblax"),
    ("Pendrillin is a rare metal mined on Vega-7.", "Pendrillin"),
    ("Captain Mareva commands the starship Halycon.", "Mareva"),
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    for i, (wp, addr) in enumerate(FACTS):
        write_fact(patcher, bank, tok,
                   write_prompt=wp, fact_id=f"fact_{i}", address=addr)
    print(f"[bank ] wrote {bank.size} facts")

    prompt = "Zorblax is a planet in the"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    with DiagnosticRecorder(model, patcher, enabled=True) as rec:
        with patcher.patched(), patcher.injecting(bank, alpha=1.0), torch.no_grad():
            model(**enc, use_cache=False)

    df = rec.to_pandas()
    print(f"\n[diag ] captured {len(df)} rows; signals = {sorted(df['signal_name'].unique())}")

    # Aggregate per (signal_name, layer) -> mean value
    agg = (df.groupby(["signal_name", "layer"])["value"]
             .mean().reset_index().sort_values(["signal_name", "layer"]))
    print("\n[diag ] per-signal × per-layer mean (first 20 rows):")
    print(agg.head(20).to_string(index=False))

    # Sanity highlights
    print("\n[diag ] summary by signal:")
    summary = (df.groupby("signal_name")["value"]
                 .agg(["count", "mean", "min", "max"])
                 .reset_index())
    print(summary.to_string(index=False))

    out_path = "/tmp/dm_diag.json"
    records = df.to_dict(orient="records")
    with open(out_path, "w") as f:
        json.dump(records, f, default=float)
    print(f"\n[dump ] wrote {len(records)} records → {out_path}")
    print("[done ] OK")


if __name__ == "__main__":
    main()
