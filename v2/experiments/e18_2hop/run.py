"""e18 2-hop chain reasoning — does ALB compose implicit B→C joins via bank?

**Experiment rationale** (from V2_DIFFERENTIATION.md):
RAG fails on 2-hop chains when the intermediate entity B is NOT in the prompt.
The query "A's rel1's rel2?" requires chaining: first retrieve (A, rel1, B),
then (B, rel2, C). RAG either retrieves text and hopes the prompt linearizes
the logic, or requires explicit 2-stage prompting with the intermediate entity.

ALB hypothesis: if both b_A (fact about A) and b_B (fact about B) are preloaded
into AttentionBank at a shared layer, the model's *frozen* attention mechanism
can chain via soft-matching on the hidden state for B — no prompt mention needed.

**Test design**:
1. Build 2-hop chains from Exp35b bank.pt: (A, rel1, B) → (B, rel2, C).
2. Three query templates:
   - single-hop A: "Q: <A> <rel1>?\nA: <B>"       (baseline)
   - single-hop B: "Q: <B> <rel2>?\nA: <C>"       (baseline)
   - 2-hop chain:  "Q: <A> <rel1>'s <rel2>?\nA: <C>"  (no B mentioned in prompt)
3. Four bank conditions (separate evals, NOT mixed in training):
   - A_only: preload only b_A → 2-hop should fail (missing b_B)
   - B_only: preload only b_B → 2-hop should fail (no entry point from A)
   - AB_both: preload both b_A and b_B → 2-hop should succeed
   - None: empty bank (baseline)
4. Training: MIX of single-hop A + single-hop B queries (60+60=120) with AB_both bank.
5. Eval: 60 held-out 2-hop queries, 4 bank conditions × 3 templates.

**Pass criterion**:
  Δ_NLL(2-hop, AB_both) - Δ_NLL(2-hop, A_only) <= -0.8  AND
  Δ_NLL(2-hop, AB_both) - Δ_NLL(2-hop, B_only) <= -0.8
(Both b_A and b_B together must give non-trivial gain over either alone.)

**Honest caveat**:
If natural 2-hop chains are rare in Exp35b bank (subjects rarely appear as
other entries' target_true), we'll synthesize pseudo-chains by random pairing.
Synthetic chains lose the "natural fact chain" property but still test whether
the *mechanism* can chain attention over two preloaded bank entries.

Also: small LMs (e.g. Qwen3-4B) may lack frozen-weight capacity to perform
implicit joins at all — this is a known limitation. A null result doesn't
invalidate ALB; it just says the base model can't do multi-hop reasoning
without chain-of-thought prompting.

Output: e18_seed{seed}.json with:
  - delta_nll_AB_vs_A, delta_nll_AB_vs_B (2-hop)
  - bank_conditions: {A_only, B_only, AB_both, None} × {2hop, singleA, singleB}
  - diagnostics: n_natural_chains, n_synthetic_chains
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from v2.core import (
    AttentionBank, LPLHeads, install_lpl_patch, LPLState, lpl_state_scope,
    make_projector, residual_apply, load_model, nll_on_answer, encode_qa,
    data_io,
)


def build_2hop_chains(entries, *, n_target, seed, synthetic_ok=True):
    """Build 2-hop chains (A, rel1, B) → (B, rel2, C).
    
    Returns:
        chains: list of (key_A, key_B, subj_A, rel_1, B, rel_2, C)
        diagnostics: {n_natural, n_synthetic}
    
    Natural chains: target_true of entry_A matches subject of entry_B.
    Synthetic chains: if natural < n_target and synthetic_ok, randomly pair
    entries so that we *rename* target_true of A to match subject of B.
    """
    # Build index: target_true -> list of keys with that subject
    target_to_subj = defaultdict(list)
    for k, e in entries.items():
        if not e.get("solo_pass"):
            continue
        target_to_subj[e["subject"]].append(k)
    
    natural_chains = []
    for k_a, e_a in entries.items():
        if not e_a.get("solo_pass"):
            continue
        B = e_a["target_true"]
        if B in target_to_subj:
            for k_b in target_to_subj[B]:
                if k_a == k_b:  # skip self-loops
                    continue
                e_b = entries[k_b]
                chain = (k_a, k_b, e_a["subject"], e_a["relation"], B,
                         e_b["relation"], e_b["target_true"])
                natural_chains.append(chain)
    
    rng = random.Random(seed)
    rng.shuffle(natural_chains)
    
    if len(natural_chains) >= n_target:
        return natural_chains[:n_target], {"n_natural": n_target, "n_synthetic": 0}
    
    # Need synthetic
    if not synthetic_ok:
        print(f"[e18] WARNING: only {len(natural_chains)} natural chains found, "
              f"need {n_target}; synthetic_ok=False, returning what we have.")
        return natural_chains, {"n_natural": len(natural_chains), "n_synthetic": 0}
    
    # Synthetic strategy: pick two disjoint random entries A and B, treat
    # target_true(A) as the bridge entity (renamed to match subject(B) conceptually).
    all_keys = [k for k in entries if entries[k].get("solo_pass")]
    rng.shuffle(all_keys)
    
    synthetic_chains = []
    i = 0
    while len(natural_chains) + len(synthetic_chains) < n_target and i < len(all_keys) - 1:
        k_a = all_keys[i]
        k_b = all_keys[i + 1]
        if k_a == k_b:
            i += 1
            continue
        e_a = entries[k_a]
        e_b = entries[k_b]
        # Synthetic: we'll *pretend* target_true(A) = subject(B) by using
        # target_true(A) as the bridge in the prompt, even though they differ.
        # The model won't notice if we just build the query correctly.
        # Actually, for synthetic we'll use target_true(A) as the "B" entity.
        B_synth = e_a["target_true"]
        chain = (k_a, k_b, e_a["subject"], e_a["relation"], B_synth,
                 e_b["relation"], e_b["target_true"])
        synthetic_chains.append(chain)
        i += 2
    
    all_chains = natural_chains + synthetic_chains
    return all_chains[:n_target], {
        "n_natural": len(natural_chains),
        "n_synthetic": len(synthetic_chains[:n_target - len(natural_chains)]),
    }


def make_query_templates(chain):
    """Build 3 query templates from a 2-hop chain.
    
    Args:
        chain: (key_A, key_B, subj_A, rel_1, B, rel_2, C)
    
    Returns:
        dict with keys: single_hop_A, single_hop_B, two_hop
        Each value is (prompt, answer_str)
    """
    _, _, subj_A, rel_1, B, rel_2, C = chain
    return {
        "single_hop_A": (f"Q: {subj_A} {rel_1}?\nA:", B),
        "single_hop_B": (f"Q: {B} {rel_2}?\nA:", C),
        "two_hop": (f"Q: {subj_A} {rel_1}'s {rel_2}?\nA:", C),
    }


def forward_lpl_k2(model, bank, heads, enc, *, grad=False):
    """Two-round LPL forward: round 1 (no bank), round 2 (bank active)."""
    state1 = LPLState(bank=bank, heads=heads, round_idx=1, enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1), torch.no_grad():
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    
    state2 = LPLState(bank=bank, heads=heads, round_idx=2, enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def eval_with_bank_condition(model, tok, bank, heads, chains, entries, *,
                              bank_layer, device, dtype, bank_mode, template_key):
    """Eval NLL for a given bank mode (A_only, B_only, AB_both, None) and template.
    
    Args:
        bank_mode: "A_only" | "B_only" | "AB_both" | "None"
        template_key: "single_hop_A" | "single_hop_B" | "two_hop"
    
    Returns:
        avg_nll (float)
    """
    nlls = []
    for chain in chains:
        k_a, k_b, *_ = chain
        templates = make_query_templates(chain)
        prompt, answer = templates[template_key]
        enc, _, ans_span = encode_qa(tok, prompt, answer, device)
        
        # Build bank based on bank_mode
        bank.frozen = False
        if bank_mode == "None":
            bank.slots[bank_layer] = torch.empty(0, bank.hidden_size,
                                                   device=device, dtype=dtype)
            bank.tags[bank_layer] = []
        elif bank_mode == "A_only":
            b_a = entries[k_a]["b"].float().to(device)
            b_a = b_a / (b_a.norm() + 1e-9) * 15.0
            bank.slots[bank_layer] = b_a.unsqueeze(0).to(dtype=dtype)
            bank.tags[bank_layer] = [(0, -1)]
        elif bank_mode == "B_only":
            b_b = entries[k_b]["b"].float().to(device)
            b_b = b_b / (b_b.norm() + 1e-9) * 15.0
            bank.slots[bank_layer] = b_b.unsqueeze(0).to(dtype=dtype)
            bank.tags[bank_layer] = [(0, -1)]
        elif bank_mode == "AB_both":
            b_a = entries[k_a]["b"].float().to(device)
            b_b = entries[k_b]["b"].float().to(device)
            b_a = b_a / (b_a.norm() + 1e-9) * 15.0
            b_b = b_b / (b_b.norm() + 1e-9) * 15.0
            bank.slots[bank_layer] = torch.stack([b_a, b_b], dim=0).to(dtype=dtype)
            bank.tags[bank_layer] = [(0, -1), (0, -1)]
        else:
            raise ValueError(f"Unknown bank_mode: {bank_mode}")
        bank.frozen = True
        
        logits = forward_lpl_k2(model, bank, heads, enc, grad=False)
        nlls.append(nll_on_answer(logits, enc.input_ids, ans_span))
    
    return sum(nlls) / max(len(nlls), 1)


def train_on_single_hops(model, tok, bank, heads, P, chains_train, entries, *,
                         bank_layer, device, dtype, steps, lr, seed):
    """Train on MIXED single-hop A + single-hop B queries with AB_both bank.
    
    Returns:
        loss_history (list of floats)
    """
    # Build training set: 60 single-hop A + 60 single-hop B
    train_examples = []
    for chain in chains_train:
        k_a, k_b, *_ = chain
        templates = make_query_templates(chain)
        # Add single-hop A
        train_examples.append((k_a, k_b, templates["single_hop_A"][0],
                                templates["single_hop_A"][1], "single_A"))
        # Add single-hop B
        train_examples.append((k_a, k_b, templates["single_hop_B"][0],
                                templates["single_hop_B"][1], "single_B"))
    
    rng = random.Random(seed)
    rng.shuffle(train_examples)
    
    trainable = list(P.parameters()) + list(heads.bank_gate_heads.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr)
    
    losses = []
    t0 = time.time()
    for step in range(steps):
        k_a, k_b, prompt, answer, _ = rng.choice(train_examples)
        enc, _, ans_span = encode_qa(tok, prompt, answer, device)
        
        # Build AB_both bank with grad-enabled projector
        b_a = entries[k_a]["b"].float().to(device)
        b_b = entries[k_b]["b"].float().to(device)
        b_a = b_a / (b_a.norm() + 1e-9) * 15.0
        b_b = b_b / (b_b.norm() + 1e-9) * 15.0
        b_stack = torch.stack([b_a, b_b], dim=0)
        
        bank.frozen = False
        b_proj = (b_stack + P(b_stack)).to(dtype=dtype)
        bank.slots[bank_layer] = b_proj
        bank.tags[bank_layer] = [(0, -1), (0, -1)]
        bank.frozen = True
        
        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, grad=True)
        # Compute loss on answer span
        pred = logits[0, ans_span - 1: -1, :]
        gold = enc.input_ids[0, ans_span:]
        loss = F.cross_entropy(pred.float(), gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 25 == 0 or step == 0:
            recent = sum(losses[-25:]) / min(25, len(losses))
            elapsed = time.time() - t0
            print(f"  [e18] step {step+1}/{steps} loss(avg25)={recent:.4f} ({elapsed:.1f}s)")
    
    print(f"[e18] training done in {time.time() - t0:.1f}s")
    return losses


def main():
    p = argparse.ArgumentParser(description="e18 2-hop chain reasoning")
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--bank_pt", default=str(data_io.BANK_PT_DEFAULT))
    p.add_argument("--n_train", type=int, default=120,
                   help="Number of training chains (will give 2*n_train single-hop examples)")
    p.add_argument("--n_eval", type=int, default=60,
                   help="Number of eval 2-hop chains")
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic_ok", type=lambda x: x.lower() == "true", default=True,
                   help="Allow synthetic chains if natural chains < target (default: True)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    
    out_path = Path(args.out) if args.out else HERE / f"e18_seed{args.seed}.json"
    
    blob = data_io.load_bank_blob(args.bank_pt)
    entries = blob["entries"]
    
    # Build 2-hop chains (train + eval)
    n_total = (args.n_train // 2) + args.n_eval  # n_train is # single-hop examples = 2*chains
    all_chains, diag = build_2hop_chains(entries, n_target=n_total,
                                          seed=args.seed, synthetic_ok=args.synthetic_ok)
    
    if len(all_chains) < n_total:
        print(f"[e18] WARNING: only {len(all_chains)} chains available, requested {n_total}")
    
    # Split: first n_train//2 for training, rest for eval
    n_train_chains = min(args.n_train // 2, len(all_chains) // 2)
    chains_train = all_chains[:n_train_chains]
    chains_eval = all_chains[n_train_chains:n_train_chains + args.n_eval]
    
    print(f"[e18] chains: {len(chains_train)} train, {len(chains_eval)} eval "
          f"(natural={diag['n_natural']}, synthetic={diag['n_synthetic']})")
    
    # Load model
    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    
    # Init bank, heads, projector
    bank = AttentionBank(num_layers=n_layers, hidden_size=d, device=args.device,
                         dtype=torch.bfloat16, max_per_layer=4)  # max 2 entries per chain
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)
    P = make_projector(d, rank=args.rank).to(args.device).float()
    
    # Pre-train eval (with untrained projector)
    print("[e18] Pre-train eval...")
    pre_results = {}
    for bank_mode in ["None", "A_only", "B_only", "AB_both"]:
        for template_key in ["single_hop_A", "single_hop_B", "two_hop"]:
            nll = eval_with_bank_condition(
                model, tok, bank, heads, chains_eval, entries,
                bank_layer=args.bank_layer, device=args.device,
                dtype=torch.bfloat16, bank_mode=bank_mode, template_key=template_key,
            )
            key = f"{bank_mode}_{template_key}"
            pre_results[key] = nll
            print(f"  [e18:pre] {key}: {nll:.4f}")
    
    # Train on single-hop A + B with AB_both bank
    print("[e18] Training on single-hop A+B queries (AB_both bank)...")
    loss_history = train_on_single_hops(
        model, tok, bank, heads, P, chains_train, entries,
        bank_layer=args.bank_layer, device=args.device, dtype=torch.bfloat16,
        steps=args.steps, lr=args.lr, seed=args.seed,
    )
    
    # Post-train eval
    print("[e18] Post-train eval...")
    post_results = {}
    for bank_mode in ["None", "A_only", "B_only", "AB_both"]:
        for template_key in ["single_hop_A", "single_hop_B", "two_hop"]:
            nll = eval_with_bank_condition(
                model, tok, bank, heads, chains_eval, entries,
                bank_layer=args.bank_layer, device=args.device,
                dtype=torch.bfloat16, bank_mode=bank_mode, template_key=template_key,
            )
            key = f"{bank_mode}_{template_key}"
            post_results[key] = nll
            print(f"  [e18:post] {key}: {nll:.4f}")
    
    # Compute deltas and pass criterion (signed convention: negative = AB helps)
    delta_AB_vs_None = post_results["AB_both_two_hop"] - post_results["None_two_hop"]
    delta_AB_vs_A = post_results["AB_both_two_hop"] - post_results["A_only_two_hop"]
    delta_AB_vs_B = post_results["AB_both_two_hop"] - post_results["B_only_two_hop"]
    
    pass_criterion = (delta_AB_vs_A <= -0.8 and delta_AB_vs_B <= -0.8)
    
    print(f"\n[e18] === RESULTS ===")
    print(f"  Δ(AB vs None): {delta_AB_vs_None:.4f}")
    print(f"  Δ(AB vs A_only): {delta_AB_vs_A:.4f}  {'PASS' if delta_AB_vs_A <= -0.8 else 'FAIL'}")
    print(f"  Δ(AB vs B_only): {delta_AB_vs_B:.4f}  {'PASS' if delta_AB_vs_B <= -0.8 else 'FAIL'}")
    print(f"  Overall: {'PASS' if pass_criterion else 'FAIL'}")
    
    # Output JSON
    out = {
        "experiment": "e18_2hop_chain",
        "seed": args.seed,
        "model": args.model,
        "diagnostics": {
            "n_natural_chains": diag["n_natural"],
            "n_synthetic_chains": diag["n_synthetic"],
            "n_train_chains": len(chains_train),
            "n_eval_chains": len(chains_eval),
        },
        "hyperparams": {
            "bank_layer": args.bank_layer,
            "rank": args.rank,
            "lr": args.lr,
            "steps": args.steps,
            "synthetic_ok": args.synthetic_ok,
        },
        "pre_train": pre_results,
        "post_train": post_results,
        "deltas": {
            "AB_vs_None": delta_AB_vs_None,
            "AB_vs_A_only": delta_AB_vs_A,
            "AB_vs_B_only": delta_AB_vs_B,
        },
        "pass_criterion": {
            "rule": "delta_AB_vs_A <= -0.8 AND delta_AB_vs_B <= -0.8",
            "pass": pass_criterion,
        },
        "loss_first25": loss_history[:25],
        "loss_last25": loss_history[-25:],
    }
    
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[e18] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
