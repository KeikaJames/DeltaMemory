"""Stage 7A — payload-identity diagnostic.

Measures, **without the writer / Q/V projector / payload probe**, whether the
frozen LLM already encodes the answer-token identity at:

  (i)  the oracle value-span tokens, and
  (ii) the oracle address-span tokens

at each of a chosen set of layers. For each (span, layer, pool) cell we train
only a tiny ``Linear(hidden, |SINGLE_TOKEN_CODES|)`` closed-vocab head with
cross-entropy and report held-out top-1, top-3, and answer log-prob margin.

This is intentionally a small, self-contained script: it loads the existing
``address_token_binding_single_token`` synthetic suite (which already embeds
``address_text``/``value_text`` and char ranges), runs **one** frozen forward
per example to grab all hidden states, then trains many cheap heads in
parallel on top. Output:

  - ``summary.json`` — per-cell metrics + best cell;
  - ``report.md`` — human-readable table + decision verdict;
  - ``per_example.jsonl`` — held-out per-example correctness.

The decision rule (kept identical to ``plan.md`` / Stage 7):

  * any cell achieves held-out top1 >= 0.85   -> Phase 7B unlocked, lock that
                                                  (span, layer, pool);
  * else                                      -> Stage 7B blocked; Story A
                                                  negative result is the next
                                                  publishable artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from deltamemory.engine.delta_dataset import SINGLE_TOKEN_CODES, make_delta_memory_examples
from deltamemory.engine.delta_experiment import _token_range_for_example
from deltamemory.gemma.model_adapter import load_model_bundle


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------


def _encode(tokenizer, text: str, device):
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def _token_range(tokenizer, full_text: str, span_text: str | None, char_range) -> tuple[int, int] | None:
    if span_text is None:
        return None
    cr = char_range
    if cr is not None and not isinstance(cr, list):
        cr = list(cr)
    return _token_range_for_example(tokenizer, full_text, span_text, cr)


def _answer_token_str(answer: str) -> str:
    """Last whitespace-separated word — works for both 'red' and 'capital = Paris'."""
    return answer.strip().split()[-1]


def _build_closed_vocab(tokenizer, examples_train, examples_eval) -> tuple[list[str], dict[str, int], int]:
    """Auto-build the closed answer vocab from observed answers.

    Filters to answers that encode to exactly ONE token under leading-space
    encoding (Gemma convention). Returns (vocab_list, token_id_lookup,
    n_filtered_out).
    """
    seen: dict[str, int] = {}
    n_dropped = 0
    for ex in list(examples_train) + list(examples_eval):
        ans = _answer_token_str(ex.answer)
        if ans in seen:
            continue
        # Leading-space encoding: Gemma usually splits " Paris" as one token
        ids = tokenizer(" " + ans, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            seen[ans] = ids[0]
        else:
            n_dropped += 1
    vocab = sorted(seen.keys())
    return vocab, seen, n_dropped


# -------------------------------------------------------------------------
# closed-vocab head + attention pool
# -------------------------------------------------------------------------


class ClosedVocabHead(nn.Module):
    """Two readouts:
      * ``linear``: LayerNorm + Linear(hidden, n_codes) trained from scratch.
      * ``lm_head``: LayerNorm + Linear(hidden, hidden) projector, then dot
        with frozen output-embedding rows of the answer tokens (size n_codes).
        Tests whether the frozen LM head (Gemma's pretrained vocabulary
        decoder) can decode hidden→answer; the only trained piece is a small
        hidden→hidden projection + LN.
    """

    def __init__(self, hidden_size: int, n_codes: int, *,
                 readout: str = "linear",
                 frozen_answer_embeddings: torch.Tensor | None = None) -> None:
        super().__init__()
        self.readout = readout
        self.norm = nn.LayerNorm(hidden_size)
        if readout == "linear":
            self.proj = nn.Linear(hidden_size, n_codes)
            self.register_buffer("answer_embeddings", torch.empty(0))
        elif readout == "lm_head":
            assert frozen_answer_embeddings is not None
            assert frozen_answer_embeddings.shape == (n_codes, hidden_size)
            self.proj = nn.Linear(hidden_size, hidden_size)
            self.register_buffer(
                "answer_embeddings", frozen_answer_embeddings.detach().clone()
            )
        else:
            raise ValueError(f"unknown readout {readout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.norm(x))
        if self.readout == "linear":
            return h
        # h: (B, H), answer_embeddings: (n_codes, H)
        ae = self.answer_embeddings.to(device=h.device, dtype=h.dtype)
        return h @ ae.t()


class AttnPool(nn.Module):
    """Span pooling with a learned query, fp32 softmax (MPS bf16 safe)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.query, std=hidden_size ** -0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (T, H)
        q = self.query.to(device=tokens.device, dtype=tokens.dtype)
        scores = (tokens.float() @ q.float()) / math.sqrt(tokens.shape[-1])
        weights = torch.softmax(scores, dim=-1).to(tokens.dtype)
        return (weights.unsqueeze(-1) * tokens).sum(dim=0)


def _mean_pool(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=0)


def _last_token(tokens: torch.Tensor) -> torch.Tensor:
    """Return the hidden state of the *last* token in the span.

    For a value span ``"secret-code = red"`` the last token is the answer
    token itself. This is the sharpest possible probe of "did the LM encode
    the answer-token identity at this position?".
    """
    return tokens[-1]


def _first_token(tokens: torch.Tensor) -> torch.Tensor:
    return tokens[0]


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------


@dataclass
class CellResult:
    span: str
    layer: int
    pool: str
    readout: str
    train_top1: float
    eval_top1: float
    eval_top3: float
    eval_margin_mean: float
    eval_correct_logp_mean: float
    n_train: int
    n_eval: int


def _train_cell(
    train_tokens: list[torch.Tensor],
    train_y: torch.Tensor,
    eval_tokens: list[torch.Tensor],
    eval_y: torch.Tensor,
    hidden_size: int,
    n_codes: int,
    pool: str,
    steps: int,
    lr: float,
    device,
    dtype,
    *,
    readout: str = "linear",
    frozen_answer_embeddings: torch.Tensor | None = None,
) -> tuple[float, float, float, float, float]:
    head = ClosedVocabHead(
        hidden_size, n_codes,
        readout=readout, frozen_answer_embeddings=frozen_answer_embeddings,
    ).to(device=device, dtype=dtype)
    if pool == "attn":
        attn = AttnPool(hidden_size).to(device=device, dtype=dtype)
        params = list(head.parameters()) + list(attn.parameters())
    else:
        attn = None
        params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    train_y_dev = train_y.to(device)
    eval_y_dev = eval_y.to(device)

    def _pool_static(tokens_list):
        # pre-pool once on device (mean/last/first)
        if pool == "mean":
            return torch.stack([t.to(device=device, dtype=dtype).mean(dim=0) for t in tokens_list])
        if pool == "last":
            return torch.stack([t[-1].to(device=device, dtype=dtype) for t in tokens_list])
        if pool == "first":
            return torch.stack([t[0].to(device=device, dtype=dtype) for t in tokens_list])
        raise ValueError(pool)

    def _pool_attn(tokens_list):
        return torch.stack([attn(t.to(device=device, dtype=dtype)) for t in tokens_list], dim=0)

    if attn is None:
        train_x = _pool_static(train_tokens)
        eval_x = _pool_static(eval_tokens)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits = head(train_x).float()
            loss = F.cross_entropy(logits, train_y_dev)
            loss.backward()
            opt.step()
    else:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            x = _pool_attn(train_tokens)
            logits = head(x).float()
            loss = F.cross_entropy(logits, train_y_dev)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        if attn is None:
            x_train = train_x
            x_eval = eval_x
        else:
            x_train = _pool_attn(train_tokens)
            x_eval = _pool_attn(eval_tokens)
        train_logits = head(x_train).float()
        train_top1 = float((train_logits.argmax(dim=-1) == train_y_dev).float().mean())
        eval_logits = head(x_eval).float()
        eval_logp = F.log_softmax(eval_logits, dim=-1)
        eval_top1 = float((eval_logits.argmax(dim=-1) == eval_y_dev).float().mean())
        top3 = eval_logits.topk(3, dim=-1).indices
        eval_top3 = float((top3 == eval_y_dev.unsqueeze(-1)).any(dim=-1).float().mean())
        correct_logp = eval_logp.gather(-1, eval_y_dev.unsqueeze(-1)).squeeze(-1)
        masked = eval_logp.clone()
        masked.scatter_(-1, eval_y_dev.unsqueeze(-1), float("-inf"))
        runner_up = masked.max(dim=-1).values
        margin_mean = float((correct_logp - runner_up).mean())
        correct_logp_mean = float(correct_logp.mean())
    return train_top1, eval_top1, eval_top3, margin_mean, correct_logp_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7A diagnostic")
    parser.add_argument("--model", default="mock-gemma")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--task-suite", default="address_token_binding_single_token")
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="auto",
                        help="comma-separated layer indices into hidden_states (1..L); "
                             "use 'auto' to pick ~9 evenly spaced layers including last")
    parser.add_argument("--pools", default="mean,attn")
    parser.add_argument("--spans", default="value,address")
    parser.add_argument("--readouts", default="linear",
                        help="comma-separated subset of {linear, lm_head}")
    parser.add_argument(
        "--report-dir",
        default="reports/experiments/stage7a_diagnostic",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[stage7a] loading {args.model} on {args.device} ({args.dtype})")
    bundle = load_model_bundle(args.model, device=args.device, dtype=args.dtype)
    device = bundle.device
    dtype = bundle.dtype
    tokenizer = bundle.tokenizer
    model = bundle.model
    # Probe one example to discover hidden_size and number of hidden_states
    sample_text = "hello"
    enc = _encode(tokenizer, sample_text, device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True, use_cache=False)
    n_hidden_states = len(out.hidden_states)
    hidden_size = int(out.hidden_states[-1].shape[-1])

    print(f"[stage7a] building dataset: {args.task_suite}, train={args.train_samples}, eval={args.eval_samples}")
    if args.task_suite == "factual_capital_binding":
        # Small fixed pool (~56 pairs). Build all examples once and split disjoint.
        all_examples = make_delta_memory_examples(args.task_suite, 1000, seed=args.seed, start_id=0)
        # Dedupe by country (address), keep first occurrence
        seen_addr = set()
        unique = []
        for ex in all_examples:
            if ex.address_text in seen_addr:
                continue
            seen_addr.add(ex.address_text)
            unique.append(ex)
        n_total = len(unique)
        n_eval = min(args.eval_samples, n_total // 2)
        n_train = min(args.train_samples, n_total - n_eval)
        eval_examples = unique[:n_eval]
        train_examples = unique[n_eval:n_eval + n_train]
        print(f"[stage7a] LAMA pool: {n_total} unique countries; train={n_train}, eval={n_eval} (disjoint countries)")
    else:
        train_examples = make_delta_memory_examples(
            args.task_suite, args.train_samples, seed=args.seed, start_id=0
        )
        eval_examples = make_delta_memory_examples(
            args.task_suite, args.eval_samples, seed=args.seed + 10_000, start_id=10_000
        )

    # Auto-build closed answer vocab from observed answers (single-token under tokenizer).
    vocab, _vocab_token_ids, n_dropped = _build_closed_vocab(tokenizer, train_examples, eval_examples)
    if not vocab:
        raise RuntimeError("No single-token answers found in dataset; closed-vocab head impossible.")
    vocab_index = {w: i for i, w in enumerate(vocab)}
    n_codes = len(vocab)
    print(f"[stage7a] closed answer vocab size = {n_codes} (dropped {n_dropped} multi-token answers)")
    print(f"[stage7a] vocab sample: {vocab[:8]}{'...' if len(vocab) > 8 else ''}")

    def _answer_index(answer: str) -> int:
        ans = _answer_token_str(answer)
        if ans in vocab_index:
            return vocab_index[ans]
        raise ValueError(f"answer {answer!r} not in closed vocab")

    def _gather(examples, layer_ids):
        # returns: dict[(span, layer)] -> list of raw token tensors
        raw_tokens = {(s, L): [] for s in args.spans.split(",") for L in layer_ids}
        labels = []
        kept = 0
        for ex in examples:
            try:
                y = _answer_index(ex.answer)
            except ValueError:
                continue
            enc = _encode(tokenizer, ex.text, device)
            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
            value_range = _token_range(tokenizer, ex.text, ex.value_text, ex.value_char_range)
            address_range = _token_range(tokenizer, ex.text, ex.address_text, ex.address_char_range)
            ranges = {"value": value_range, "address": address_range}
            ok = True
            for span_name in args.spans.split(","):
                if ranges.get(span_name) is None:
                    ok = False
                    break
            if not ok:
                continue
            labels.append(y)
            for span_name in args.spans.split(","):
                tok_range = ranges[span_name]
                for L in layer_ids:
                    h = out.hidden_states[L][0, tok_range[0]:tok_range[1], :].detach().to("cpu")
                    raw_tokens[(span_name, L)].append(h)
            kept += 1
            if kept % 16 == 0:
                print(f"    [gather] {kept}/{len(examples)} examples done", flush=True)
        return raw_tokens, torch.tensor(labels, dtype=torch.long), kept

    n_layers = n_hidden_states  # already includes embedding
    if args.layers == "auto":
        # 9 evenly spaced layers in hidden_states (indices 1..L) — skip embedding (0)
        L = n_layers - 1
        idxs = sorted({1, max(1, L // 8), max(1, L // 4), max(1, 3 * L // 8),
                       max(1, L // 2), max(1, 5 * L // 8), max(1, 3 * L // 4),
                       max(1, 7 * L // 8), L})
        layer_ids = [int(i) for i in idxs]
    else:
        layer_ids = [int(s) for s in args.layers.split(",") if s.strip()]
    print(f"[stage7a] hidden_states layers probed = {layer_ids} (out of 0..{n_layers - 1})")

    print("[stage7a] running frozen forwards on train split...")
    t0 = time.time()
    train_tokens, train_y, train_n = _gather(train_examples, layer_ids)
    print(f"  done in {time.time()-t0:.1f}s, kept {train_n}/{len(train_examples)} examples")
    print("[stage7a] running frozen forwards on eval split...")
    t0 = time.time()
    eval_tokens, eval_y, eval_n = _gather(eval_examples, layer_ids)
    print(f"  done in {time.time()-t0:.1f}s, kept {eval_n}/{len(eval_examples)} examples")

    cells: list[CellResult] = []
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    spans = [s.strip() for s in args.spans.split(",") if s.strip()]
    readouts = [r.strip() for r in args.readouts.split(",") if r.strip()]

    # Build frozen answer-token embeddings for the lm_head readout (if requested).
    answer_emb = None
    if "lm_head" in readouts:
        try:
            out_emb = model.get_output_embeddings().weight  # (V, H)
        except Exception as exc:
            raise RuntimeError(f"Cannot access output embeddings for lm_head readout: {exc}")
        ids = [_vocab_token_ids[w] for w in vocab]
        answer_emb = out_emb[torch.tensor(ids, device=out_emb.device)].detach().to(device=device, dtype=dtype)
        print(f"[stage7a] lm_head readout enabled, answer_emb shape={tuple(answer_emb.shape)}")

    total_cells = len(spans) * len(layer_ids) * len(pools) * len(readouts)
    done = 0
    for span_name in spans:
        for L in layer_ids:
            tt = train_tokens[(span_name, L)]
            te = eval_tokens[(span_name, L)]
            for pool in pools:
                for readout in readouts:
                    done += 1
                    t0 = time.time()
                    tr_top1, ev_top1, ev_top3, margin, clp = _train_cell(
                        train_tokens=tt, train_y=train_y,
                        eval_tokens=te, eval_y=eval_y,
                        hidden_size=hidden_size, n_codes=n_codes,
                        pool=pool,
                        steps=args.steps, lr=args.lr,
                        device=device, dtype=dtype,
                        readout=readout,
                        frozen_answer_embeddings=answer_emb if readout == "lm_head" else None,
                    )
                    cells.append(CellResult(
                        span=span_name, layer=L, pool=pool, readout=readout,
                        train_top1=tr_top1, eval_top1=ev_top1, eval_top3=ev_top3,
                        eval_margin_mean=margin, eval_correct_logp_mean=clp,
                        n_train=int(len(tt)), n_eval=int(len(te)),
                    ))
                    print(
                        f"  [{done:>3}/{total_cells}] span={span_name:<7} layer={L:>2} pool={pool:<4} "
                        f"ro={readout:<7} train_top1={tr_top1:.3f} eval_top1={ev_top1:.3f} "
                        f"top3={ev_top3:.3f} margin={margin:+.2f} ({time.time()-t0:.1f}s)",
                        flush=True,
                    )

    cells_sorted = sorted(cells, key=lambda c: -c.eval_top1)
    best = cells_sorted[0]
    pass_gate = best.eval_top1 >= 0.85
    summary = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "task_suite": args.task_suite,
        "seed": args.seed,
        "n_codes": n_codes,
        "n_train": int(train_y.shape[0]),
        "n_eval": int(eval_y.shape[0]),
        "layers_probed": layer_ids,
        "pools": pools,
        "spans": spans,
        "steps": args.steps,
        "lr": args.lr,
        "best": vars(best),
        "pass_gate_0.85": pass_gate,
        "cells": [vars(c) for c in cells_sorted],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"# Stage 7A diagnostic — {args.model}",
        "",
        f"- task suite: `{args.task_suite}`",
        f"- closed vocab size: `{n_codes}` (random baseline = `{1.0/n_codes:.3f}`)",
        f"- train / eval examples kept: `{summary['n_train']} / {summary['n_eval']}`",
        f"- layers probed (hidden_states idx): `{layer_ids}`",
        f"- pools: `{pools}`",
        f"- spans: `{spans}`",
        f"- steps / lr: `{args.steps}` / `{args.lr}`",
        "",
        "## Best cell",
        "",
        f"- span={best.span}, layer={best.layer}, pool={best.pool}",
        f"- eval top1 = **{best.eval_top1:.3f}** (top3 = {best.eval_top3:.3f}, "
        f"margin = {best.eval_margin_mean:+.2f}, correct_logp = {best.eval_correct_logp_mean:+.2f})",
        f"- train top1 = {best.train_top1:.3f}",
        "",
        f"## Strict gate (eval top1 >= 0.85): **{'PASS' if pass_gate else 'FAIL'}**",
        "",
        "## All cells (sorted by eval top1)",
        "",
        "| span | layer | pool | train_top1 | eval_top1 | eval_top3 | margin | correct_logp |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for c in cells_sorted:
        lines.append(
            f"| {c.span} | {c.layer} | {c.pool} | {c.train_top1:.3f} | {c.eval_top1:.3f} | "
            f"{c.eval_top3:.3f} | {c.eval_margin_mean:+.2f} | {c.eval_correct_logp_mean:+.2f} |"
        )
    if pass_gate:
        lines += [
            "",
            "## Verdict",
            "",
            "The frozen LLM **does** encode the answer-token identity at the best probed",
            "(span, layer, pool). Phase 7B is unlocked: lock that input/layer for the",
            "writer + closed-vocab payload probe head and rerun.",
        ]
    else:
        lines += [
            "",
            "## Verdict",
            "",
            "No (span, layer, pool) cell reaches eval top1 >= 0.85 on held-out. The frozen",
            "LLM does not surface the answer token at oracle spans under this prompt format.",
            "Per the Stage 7 stop rule, do **not** start Phase 7B; instead either redesign",
            "the address-card prompt to expose answer identity at a readable position, or",
            "package the negative result as part of Story A.",
        ]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n[stage7a] best: span={best.span} layer={best.layer} pool={best.pool} "
          f"eval_top1={best.eval_top1:.3f} -> gate {'PASS' if pass_gate else 'FAIL'}")
    print(f"[stage7a] wrote {report_dir/'summary.json'} and {report_dir/'report.md'}")


if __name__ == "__main__":
    main()
