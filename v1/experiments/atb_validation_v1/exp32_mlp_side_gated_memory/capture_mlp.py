"""Exp32 Φ1 — Capture MLP-side (K_in_to_mlp, V_mlp_output) tensors.

For each fact in train/val/test split:
  * write prompt at relation_last position:
      K_l = h_in[pos]   (input to MLP at layer l)
      V_l = h_out[pos]  (MLP output at layer l)
  * Each paraphrase prompt at last token:
      Q_l = h_in[last]  (read-time MLP input — same space as K_l, used for routing)

Saved per split:
    data/cache/<model_tag>_<split>.pt = {
        "fact_ids":  list[str]                              length N
        "anchor_K":  Tensor[N, L, d_model]    bf16          (write-time MLP input)
        "anchor_V":  Tensor[N, L, d_model]    bf16          (write-time MLP output)
        "queries_Q": Tensor[N, P, L, d_model] bf16          (read-time MLP input)
        "meta":      {...}
    }

Run:
    python3 experiments/atb_validation_v1/exp32_mlp_side_gated_memory/capture_mlp.py \\
        --model Qwen/Qwen3-4B-Instruct-2507 --device mps --dtype bf16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from deltamemory.memory.mlp_gated_injector import MLPGatedInjector  # noqa: E402
from deltamemory.memory.anb_capture_sweep import derive_relation_phrase, resolve_extended_capture  # noqa: E402
from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402
from atb_validation_v1._lib.cf_runner import build_write_prompt  # noqa: E402


# Reuse the Exp31 splits — no need to rebuild.
SPLITS_DIR = Path(__file__).parents[1] / "exp31_learned_k_adapter" / "data" / "splits"
CACHE_DIR = Path(__file__).parent / "data" / "cache"


def resolve_pos(site: str, row: dict, tok, prompt: str, add_special: bool = True) -> int | None:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=add_special)
    am = enc["attention_mask"][0]
    rel = derive_relation_phrase(row.get("prompt", ""))
    spec = resolve_extended_capture(
        site=site, write_prompt=prompt,
        subject=row.get("subject", ""), relation_phrase=rel,
        object_str=row.get("target_new", ""),
        tokenizer=tok, attention_mask_row=am, add_special_tokens=add_special,
    )
    if spec is None or not spec.token_positions:
        return None
    return spec.token_positions[-1]


def capture_split(
    split_name: str,
    facts: list[dict],
    model, tok, injector: MLPGatedInjector,
    device: str,
    n_layers: int,
    d_model: int,
    out_path: Path,
    log_every: int = 25,
) -> None:
    N = len(facts)
    P = 2

    anchor_K = torch.zeros(N, n_layers, d_model, dtype=torch.bfloat16)
    anchor_V = torch.zeros(N, n_layers, d_model, dtype=torch.bfloat16)
    queries_Q = torch.zeros(N, P, n_layers, d_model, dtype=torch.bfloat16)

    kept_ids: list[str] = []
    skipped = 0
    t0 = time.time()

    for i, row in enumerate(facts):
        wp = build_write_prompt(row, row["target_new"])
        if wp is None:
            skipped += 1
            continue
        pos_K = resolve_pos("relation_last", row, tok, wp)
        if pos_K is None:
            skipped += 1
            continue

        # Write capture
        enc_w = tok(wp, return_tensors="pt", add_special_tokens=True)
        ids_w = enc_w["input_ids"].to(device)
        am_w = enc_w.get("attention_mask")
        if am_w is not None:
            am_w = am_w.to(device)
        K_w, V_w = injector.capture_at_pos(ids_w, pos=int(pos_K), attention_mask=am_w)
        if K_w.shape != (n_layers, d_model):
            skipped += 1
            continue
        anchor_K[len(kept_ids)] = K_w.to(torch.bfloat16)
        anchor_V[len(kept_ids)] = V_w.to(torch.bfloat16)

        # Paraphrase reads — capture K (= MLP input) at last token
        paraphrases = row.get("paraphrase_prompts", [])[:P]
        ok = True
        for p_idx, ptxt in enumerate(paraphrases):
            enc = tok(ptxt, return_tensors="pt", add_special_tokens=True)
            T = int(enc["attention_mask"][0].sum().item())
            pos_Q = T - 1
            ids_q = enc["input_ids"].to(device)
            am_q = enc["attention_mask"].to(device)
            K_q, _ = injector.capture_at_pos(ids_q, pos=int(pos_Q), attention_mask=am_q)
            if K_q.shape != (n_layers, d_model):
                ok = False
                break
            queries_Q[len(kept_ids), p_idx] = K_q.to(torch.bfloat16)

        if not ok:
            skipped += 1
            continue

        kept_ids.append(row["id"])

        if (i + 1) % log_every == 0 or (i + 1) == N:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            eta = (N - i - 1) / max(rate, 1e-6)
            print(f"  {split_name} {i+1}/{N}  kept={len(kept_ids)} skipped={skipped}  "
                  f"rate={rate:.2f} fact/s  eta={eta/60:.1f} min", flush=True)

    K = len(kept_ids)
    anchor_K = anchor_K[:K].contiguous()
    anchor_V = anchor_V[:K].contiguous()
    queries_Q = queries_Q[:K].contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fact_ids": kept_ids,
        "anchor_K": anchor_K,
        "anchor_V": anchor_V,
        "queries_Q": queries_Q,
        "meta": {
            "split": split_name,
            "n_layers": int(n_layers),
            "d_model": int(d_model),
            "n_facts_kept": int(K),
            "n_facts_skipped": int(skipped),
            "n_paraphrases": int(P),
            "site": "mlp_post / mlp_input",
        },
    }, out_path)
    print(f"  -> wrote {out_path} ({K} facts, {K*P} queries)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"Loading {args.model} on {args.device} {args.dtype}...")
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()

    injector = MLPGatedInjector(model)
    n_layers = injector.num_layers
    d_model = int(model.config.hidden_size)
    print(f"  model: L={n_layers}  d_model={d_model}")

    tag = args.model_tag or args.model.replace("/", "_")
    for split in args.splits.split(","):
        split = split.strip()
        split_path = SPLITS_DIR / f"{split}.json"
        facts = json.loads(split_path.read_text())
        if args.limit is not None:
            facts = facts[: args.limit]
        print(f"\n=== {split}: {len(facts)} facts ===")
        out = CACHE_DIR / f"{tag}_{split}.pt"
        capture_split(split, facts, model, tok, injector,
                      device=args.device, n_layers=n_layers,
                      d_model=d_model, out_path=out)


if __name__ == "__main__":
    main()
