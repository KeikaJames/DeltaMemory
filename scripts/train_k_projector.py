"""Train Stage 14A InfoNCE K-projector on the LAMA-TREx train split.

For each fact in eval/splits/train.jsonl we:
  1. Forward the canonical write prompt with capture at the *address* token.
     This yields per-layer (1, num_kv_heads, head_dim) bank-K candidates
     written by the model.
  2. Forward each paraphrase (excluding the canonical) with capture at
     the *last* token (the prompt tail before the value would emerge).
     This yields the per-layer query-Q vector at read time.

Per layer we then train a Linear(d, d) by InfoNCE so that for each fact in
the batch, project(write_K) is closer to its own paraphrase Q's than to
other facts' paraphrase Q's.

The result is a `KProjectorBank` checkpoint that the bank consumes at
read time. Identity-init is preserved at step 0, so untrained projectors
never break bit-equality.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deltamemory.memory.attn_native_bank import AttnNativePatcher  # noqa: E402
from deltamemory.memory.capture_policy import (  # noqa: E402
    resolve_capture_sites,
)
from deltamemory.memory.k_projector import (  # noqa: E402
    InfoNCEBatch,
    KProjectorBank,
    infonce_loss,
)


def _load_train(data_dir: Path | None = None) -> list[dict]:
    base = data_dir if data_dir is not None else (REPO_ROOT / "eval" / "splits")
    candidates = [base / "train_v31.jsonl", base / "train.jsonl"]
    for path in candidates:
        if path.exists():
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    raise FileNotFoundError(f"no train split found under {base}")


def _capture_state(
    patcher,
    tokenizer,
    text: str,
    *,
    site: str,
    address: str | None,
):
    """Run one forward and return per-layer captured K (and Q, here taken as
    a copy of K since we want `q_pre @ projected(k_pre)`-aligned queries).

    For the projector we only need K vectors; we treat the same captured K
    at the *paraphrase tail* as the "query" anchor, because at attention
    time the model's q_pre (post-norm pre-RoPE) at the last token plays
    that role. Capturing K at the tail is the same operation up to the
    K projection, so this is a faithful proxy.
    """
    device = next(patcher.model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)

    sites = resolve_capture_sites(
        policy="address" if site == "address" else "period",
        write_prompt=text,
        address=address,
        tokenizer=tokenizer,
        attention_mask_row=am[0],
    )
    pos = sites[0].token_pos
    with patcher.patched(), patcher.capturing(capture_pos=pos), torch.no_grad():
        patcher.model(input_ids=ids, attention_mask=am, use_cache=False)

    K_per_layer: list[torch.Tensor] = []
    for layer in range(patcher.num_layers):
        kc = patcher._capture_K[layer]
        if kc is None:
            src_idx = getattr(patcher.attn_modules[layer], "kv_shared_layer_index", layer)
            kc = patcher._capture_K[src_idx]
        K_per_layer.append(kc[0].detach().cpu())  # [num_kv_heads, head_dim]
    return K_per_layer


def build_pairs(
    *,
    patcher,
    tokenizer,
    facts: list[dict],
    max_paraphrases_per_fact: int,
    log_every: int = 16,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """Return (write_K_per_layer_per_fact, query_K_per_layer_per_fact).

    write_K[fact][layer] is captured at the address token of the canonical
    write prompt. query_K[fact][layer] is captured at the period of one
    paraphrase. We emit one (write, query) row per (fact, paraphrase).
    """
    write_rows: list[list[torch.Tensor]] = []
    query_rows: list[list[torch.Tensor]] = []
    pair_owner: list[int] = []  # fact id index for each row

    for fi, rec in enumerate(facts):
        addr = rec["address_canonical"]
        value = rec["value"]
        write_prompt = f"{addr} {value}."
        write_K = _capture_state(
            patcher, tokenizer, write_prompt, site="address", address=addr
        )
        paraphrases = rec["paraphrases"][:max_paraphrases_per_fact]
        for p in paraphrases:
            query_K = _capture_state(
                patcher, tokenizer, p, site="period", address=None
            )
            write_rows.append(write_K)
            query_rows.append(query_K)
            pair_owner.append(fi)
        if (fi + 1) % log_every == 0:
            print(f"[train-kproj] built {fi + 1}/{len(facts)} facts", flush=True)

    return write_rows, query_rows, pair_owner


def train(
    *,
    proj: KProjectorBank,
    write_rows: list[list[torch.Tensor]],
    query_rows: list[list[torch.Tensor]],
    pair_owner: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    layers_to_train: list[int] | None,
    log_every: int = 50,
) -> list[float]:
    """Train per-layer InfoNCE. Returns per-step mean loss."""
    n_pairs = len(write_rows)
    n_layers = len(write_rows[0])
    if layers_to_train is None:
        layers_to_train = list(range(n_layers))

    optim = torch.optim.AdamW(proj.parameters(), lr=lr)
    losses: list[float] = []
    rng = torch.Generator().manual_seed(42)

    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(n_pairs, generator=rng).tolist()
        for start in range(0, n_pairs, batch_size):
            batch_idx = perm[start : start + batch_size]
            if len(batch_idx) < 4:
                continue
            # Deduplicate pair owners within batch -> stronger negatives.
            owners = [pair_owner[i] for i in batch_idx]
            seen: dict[int, int] = {}
            kept: list[int] = []
            for i, owner in zip(batch_idx, owners):
                if owner in seen:
                    continue
                seen[owner] = i
                kept.append(i)
            if len(kept) < 4:
                continue

            optim.zero_grad()
            total_loss = torch.zeros(())
            for layer in layers_to_train:
                wK = torch.stack([write_rows[i][layer] for i in kept])  # [B, Hkv, d]
                qK = torch.stack([query_rows[i][layer] for i in kept])
                batch = InfoNCEBatch(layer_idx=layer, write_k=wK, query_q=qK)
                total_loss = total_loss + infonce_loss(proj, batch, temperature=temperature)
            total_loss = total_loss / max(len(layers_to_train), 1)
            total_loss.backward()
            optim.step()
            losses.append(float(total_loss.item()))
            step += 1
            if step % log_every == 0:
                print(
                    f"[train-kproj] epoch {epoch} step {step} loss={losses[-1]:.4f}",
                    flush=True,
                )
    return losses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--max-paraphrases-per-fact", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="reports/cleanroom/stage14_kproj/k_projector.pt")
    ap.add_argument("--data-dir", default=None,
                    help="Directory containing train.jsonl or train_v31.jsonl (default: eval/splits)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    facts = _load_train(data_dir)
    if args.limit > 0:
        facts = facts[: args.limit]
    print(f"[train-kproj] {len(facts)} train facts", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[train-kproj] loading {args.model} on {args.device}…", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(args.device).eval()
    print(f"[train-kproj] model ready in {time.time() - t0:.1f}s", flush=True)

    patcher = AttnNativePatcher(model)
    proj = KProjectorBank.identity_for(model)
    proj.to(torch.float32)

    print(f"[train-kproj] building (write_K, query_K) pairs…", flush=True)
    t0 = time.time()
    write_rows, query_rows, pair_owner = build_pairs(
        patcher=patcher,
        tokenizer=tokenizer,
        facts=facts,
        max_paraphrases_per_fact=args.max_paraphrases_per_fact,
    )
    # Move pairs to float32 CPU for stable optim.
    write_rows = [[t.float().cpu() for t in row] for row in write_rows]
    query_rows = [[t.float().cpu() for t in row] for row in query_rows]
    print(
        f"[train-kproj] built {len(write_rows)} pairs in {time.time() - t0:.1f}s",
        flush=True,
    )

    losses = train(
        proj=proj,
        write_rows=write_rows,
        query_rows=query_rows,
        pair_owner=pair_owner,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        layers_to_train=None,
    )

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proj.save(out_path)
    log_path = out_path.with_suffix(".jsonl")
    with log_path.open("w") as fh:
        for i, loss in enumerate(losses):
            fh.write(json.dumps({"step": i, "loss": loss}) + "\n")
    summary = {
        "n_pairs": len(write_rows),
        "n_facts": len(facts),
        "n_layers": len(write_rows[0]) if write_rows else 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "is_identity_after_train": proj.is_identity(),
    }
    (out_path.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
