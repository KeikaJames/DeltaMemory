"""Exp31 Φ2 — Capture K (write) and Q (paraphrase) tensors for InfoNCE training.

For each fact in train/val/test split:
  * write prompt  -> bank.M_K[layer][-1]  (anchor key)
  * each paraphrase prompt -> query Q_pre_rope at the last token (positive query)

Saved as a single .pt file per split:
    data/cache/<model_tag>_<split>.pt = {
        "fact_ids":    list[str]                                  length N
        "anchor_K":    Tensor[N, L, Hkv, Dh]                       bf16
        "queries_Q":   Tensor[N, P, L, Hq,  Dh]                    bf16   (P=2 paraphrases)
        "meta":        {"model": ..., "split": ..., "n_layers": L, ...}
    }

Run:
    python3 experiments/atb_validation_v1/exp31_learned_k_adapter/capture_kq.py \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --splits train,val,test \\
        --device mps --dtype bf16

Cost: ~1 s/fact write + ~0.5 s/paraphrase on MPS bf16 ≈ ~30 min for 1000 facts.
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

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank, write_fact  # noqa: E402
from deltamemory.memory.anb_capture_sweep import derive_relation_phrase, resolve_extended_capture  # noqa: E402

from atb_validation_v1._lib import load_model, seed_everything  # noqa: E402
from atb_validation_v1._lib.cf_runner import build_write_prompt  # noqa: E402


SPLITS_DIR = Path(__file__).parent / "data" / "splits"
CACHE_DIR = Path(__file__).parent / "data" / "paraphrase_cache"


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


@torch.no_grad()
def capture_query_q(
    model, tok, patcher: AttnNativePatcher, prompt: str,
    pos: int, device: str,
) -> list[torch.Tensor]:
    """Run forward and grab pre-RoPE Q at position `pos` for every attn layer.

    Uses the existing diagnostics API (record_read_queries) to avoid touching
    patcher internals.
    """
    from deltamemory.memory.anb_diagnostics import record_read_queries
    probe = record_read_queries(patcher, tok, prompt, capture_pos=int(pos))
    out: list[torch.Tensor] = []
    for L, q in enumerate(probe.q_pre):
        if q is None:
            raise RuntimeError(f"q_pre missing at layer {L} for prompt: {prompt!r}")
        # q: [Hq, d] on device — move to cpu/bf16 for storage
        out.append(q.detach().to("cpu").to(torch.bfloat16))
    return out


def capture_split(
    split_name: str,
    facts: list[dict],
    model, tok, patcher: AttnNativePatcher,
    device: str,
    n_layers: int,
    hkv: int,
    hq: int,
    dh: int,
    out_path: Path,
    log_every: int = 25,
) -> None:
    N = len(facts)
    P = 2  # paraphrases per fact in CounterFact-1k

    anchor_K = torch.zeros(N, n_layers, hkv, dh, dtype=torch.bfloat16)
    queries_Q = torch.zeros(N, P, n_layers, hq, dh, dtype=torch.bfloat16)

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

        # --- anchor K from write prompt ---
        scratch = fresh_bank(patcher.model)
        scratch.bank_key_mode = "pre_rope"
        scratch.value_scale_mode = "auto_rms_cap"
        write_fact(patcher, scratch, tok, write_prompt=wp,
                   fact_id=row["id"], address=row.get("subject"), capture_pos=int(pos_K))
        for L in range(n_layers):
            anchor_K[len(kept_ids), L] = scratch.M_K[L][-1].to("cpu").to(torch.bfloat16)

        # --- positive Q's from paraphrase READ prompts ---
        paraphrases = row.get("paraphrase_prompts", [])[:P]
        ok = True
        for p_idx, ptxt in enumerate(paraphrases):
            # paraphrase prompts in counterfact_1k.jsonl already end at the
            # point where the model should answer — use the LAST token as
            # the read position.
            enc = tok(ptxt, return_tensors="pt", add_special_tokens=True)
            T = int(enc["attention_mask"][0].sum().item())
            pos_Q = T - 1
            qs = capture_query_q(model, tok, patcher, ptxt, pos=pos_Q, device=device)
            if len(qs) != n_layers:
                ok = False
                break
            for L in range(n_layers):
                queries_Q[len(kept_ids), p_idx, L] = qs[L]

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

    # trim to kept facts
    K = len(kept_ids)
    anchor_K = anchor_K[:K].contiguous()
    queries_Q = queries_Q[:K].contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fact_ids": kept_ids,
        "anchor_K": anchor_K,
        "queries_Q": queries_Q,
        "meta": {
            "split": split_name,
            "n_layers": int(n_layers),
            "n_kv_heads": int(hkv),
            "n_q_heads": int(hq),
            "head_dim": int(dh),
            "n_facts_total": int(N),
            "n_facts_kept": int(K),
            "n_facts_skipped": int(skipped),
            "n_paraphrases": int(P),
            "bank_key_mode": "pre_rope",
        },
    }, out_path)
    print(f"  -> wrote {out_path} ({K} facts, {(K*P)} queries)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-tag", default=None, help="filename tag (default: derived from model name)")
    ap.add_argument("--limit", type=int, default=None, help="cap facts per split (smoke testing)")
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"Loading {args.model} on {args.device} {args.dtype}...")
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    model.eval()

    patcher = AttnNativePatcher(model)
    n_layers = len(patcher.attn_modules)
    # Use a probe bank to read head/dim metadata robustly across archs.
    _probe = fresh_bank(patcher.model)
    hkv = int(_probe.num_kv_heads)
    dh = int(_probe.head_dim)
    # Hq is needed for the queries_Q allocation. Pull from probe state shape
    # via a 1-token forward; simpler is to use model config:
    cfg = patcher.model.config
    hq = int(getattr(cfg, "num_attention_heads", hkv))
    print(f"  model: L={n_layers}  Hq={hq}  Hkv={hkv}  Dh={dh}")

    tag = args.model_tag or args.model.replace("/", "_")
    for split in args.splits.split(","):
        split = split.strip()
        split_path = SPLITS_DIR / f"{split}.json"
        facts = json.loads(split_path.read_text())
        if args.limit is not None:
            facts = facts[: args.limit]
        print(f"\n=== {split}: {len(facts)} facts ===")
        out = CACHE_DIR / f"{tag}_{split}.pt"
        capture_split(split, facts, model, tok, patcher,
                      device=args.device, n_layers=n_layers,
                      hkv=hkv, hq=hq, dh=dh, out_path=out)

    print("\nDone.")


if __name__ == "__main__":
    main()
