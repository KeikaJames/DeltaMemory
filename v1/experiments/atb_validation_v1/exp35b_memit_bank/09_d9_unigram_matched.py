"""Exp35b — 09: D9 unigram-matched Gate B.

Recompute Gate B (target_margin > 0) on a subset where target_true and
target_new have similar unigram log-probabilities under the base model,
so we can't claim margin uplift is just "model preferring rare tokens".

For each (target_true, target_new) pair:
  - Estimate base unigram log p(target) at a neutral context (BOS only).
  - Compute |log p_new − log p_true|; keep facts in the bottom-half.

Then read the subset's Φ1 cells and re-aggregate Gate B.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from atb_validation_v1._lib import load_model  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP35 = HERE.parent / "exp35_fact_lora_bank"
import importlib.util as iu
_spec = iu.spec_from_file_location("exp35_bb", EXP35 / "build_bank.py")
_bb = iu.module_from_spec(_spec); _spec.loader.exec_module(_bb)
first_target_id = _bb.first_target_id


@torch.no_grad()
def neutral_logp(model, tok, tok_id):
    bos = tok.bos_token or tok.eos_token or " "
    enc = tok(bos, return_tensors="pt", add_special_tokens=True).to(next(model.parameters()).device)
    if enc.input_ids.size(1) == 0:
        enc = tok(" ", return_tensors="pt", add_special_tokens=True).to(next(model.parameters()).device)
    out = model(**enc, use_cache=False)
    logits = out.logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return float(logp[tok_id].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--cells", default=str(HERE / "run_qwen_exp35b" / "phi1_cells.jsonl"))
    ap.add_argument("--out", default=str(HERE / "run_qwen_exp35b" / "d9_unigram_matched.json"))
    ap.add_argument("--max-abs-logp-gap", type=float, default=1.0,
                    help="max allowed |log p_new − log p_true| at neutral context")
    args = ap.parse_args()

    test_rows = json.load(open(DATA / "splits" / "test.json"))
    by_id = {r["id"]: r for r in test_rows}

    print(f"[load] {args.model}", flush=True)
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[unigram] computing neutral log-p for each (target_true, target_new) ...", flush=True)
    cache: dict = {}
    for fid, row in by_id.items():
        t_true = first_target_id(tok, row["target_true"])
        t_new = first_target_id(tok, row["target_new"])
        # Cache by token-id pair
        key = (t_true, t_new)
        if key not in cache:
            lp_true = neutral_logp(model, tok, t_true)
            lp_new = neutral_logp(model, tok, t_new)
            cache[key] = (lp_true, lp_new)
        else:
            lp_true, lp_new = cache[key]
        by_id[fid]["_lp_true"] = lp_true
        by_id[fid]["_lp_new"] = lp_new

    kept = {fid for fid, r in by_id.items()
            if abs(r["_lp_new"] - r["_lp_true"]) <= args.max_abs_logp_gap}
    print(f"[D9] kept {len(kept)}/{len(by_id)} facts with |log p_new − log p_true| <= {args.max_abs_logp_gap}",
          flush=True)

    rows = []
    with open(args.cells) as f:
        for line in f:
            r = json.loads(line)
            if r["fact_id"] in kept:
                rows.append(r)

    out = {}
    for k in sorted({r["k"] for r in rows}):
        cur = [r for r in rows if r["k"] == k]
        n = len(cur)
        if not n: continue
        out[f"k{k}"] = {
            "n_obs": n,
            "mean_uplift_nats": sum(r["uplift"] for r in cur) / n,
            "mean_gate_d_diff_nats": sum(r["gate_d_diff"] for r in cur) / n,
            "frac_target_beats_base": sum(1 for r in cur if r["uplift"] > 0) / n,
            "frac_target_beats_shuffled": sum(1 for r in cur if r["gate_d_diff"] > 0) / n,
        }
    out["_meta"] = {
        "n_test_total": len(by_id),
        "n_kept": len(kept),
        "max_abs_logp_gap": args.max_abs_logp_gap,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
