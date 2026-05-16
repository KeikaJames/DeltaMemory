"""Phase B — train bank_gate heads on static-bank scenario.

Setup: Exp35b bank preloaded at layer 9 (frozen), pause heads frozen with
force-pause off, K=2 rounds.  Only the per-layer per-position bank_gate
heads (36 layers × 2561 params = ~92k) are trainable.  Base model frozen.

Training mix: TRAIN-split solo-pass entries from bank.pt.
Eval: held-out TEST-split entries (same as Phase D eval).

Loss: standard next-token NLL on target_true continuation.

Hypothesis: if bank_gate can learn to gate bank reads sensibly, NLL on
test-split prompts should improve over base AND over Phase D's untrained
LPL K=2 result.

If training improves real-bank but NOT random-bank (control), we have
evidence that the static-bank bridge is real-but-needs-learning.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn.functional as F

from atb_validation_v1._lib import load_model
from exp42_lpl import AttentionBank, LPLHeads, LPLConfig, install_lpl_patch
from exp42_lpl.qwen3_lpl_patch import LPLState, lpl_state_scope
sys.path.insert(0, str(HERE))
from importlib import import_module
phase_a = import_module("01_phase_a_frozen")
nll_on_answer = phase_a.nll_on_answer


def build_prompt(subj: str, rel: str) -> str:
    return f"{subj} {rel}"


def forward_lpl_k2(model, bank, heads, enc, *, ans_start: int,
                   grad: bool = False, randomize_bank: bool = False):
    """Run two-round LPL forward, return logits of round 2 (with grad if requested)."""
    # Round 1: no grad, no pauses → bank stays as preloaded (static).
    state1 = LPLState(bank=bank, heads=heads, round_idx=1,
                      enabled=True, force_pause_mask=None)
    with lpl_state_scope(model, state1):
        with torch.no_grad():
            _ = model(input_ids=enc.input_ids,
                      attention_mask=enc.attention_mask,
                      use_cache=False, return_dict=True)
    if randomize_bank:
        for l, t in enumerate(bank.slots):
            if t.shape[0] == 0: continue
            n = torch.randn_like(t)
            n = n / (n.norm(dim=-1, keepdim=True) + 1e-9) * t.norm(dim=-1, keepdim=True)
            bank.slots[l] = n.to(dtype=t.dtype, device=t.device)
    # Round 2: optional grad. bank_gate heads receive gradient.
    state2 = LPLState(bank=bank, heads=heads, round_idx=2,
                      enabled=True, force_pause_mask=None)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with lpl_state_scope(model, state2), ctx:
        out = model(input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    use_cache=False, return_dict=True)
    return out.logits


def loss_from_logits(logits, input_ids, ans_start: int):
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bank_pt", default=str(REPO / "v1/experiments/atb_validation_v1"
                                                   "/exp35b_memit_bank/data/bank.pt"))
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--n_train", type=int, default=80)
    p.add_argument("--n_eval", type=int, default=40)
    p.add_argument("--bank_layer", type=int, default=9)
    p.add_argument("--bank_n_preload", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "phase_b_train_results.json"))
    args = p.parse_args()

    blob = torch.load(args.bank_pt, map_location="cpu", weights_only=False)
    entries = blob["entries"]
    keys = list(entries.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    train_keys = [k for k in keys if entries[k].get("split") == "train"
                  and entries[k].get("solo_pass")][: args.n_train]
    test_keys = [k for k in keys if entries[k].get("split") == "test"
                 and entries[k].get("solo_pass")][: args.n_eval]
    # preload bank with disjoint set of entries (avoid trivial overlap)
    preload_keys = [k for k in keys if entries[k].get("split") == "train"
                    and entries[k].get("solo_pass")][args.n_train: args.n_train + args.bank_n_preload]

    train_items = [(entries[k]["subject"], entries[k]["relation"], entries[k]["target_true"])
                   for k in train_keys]
    test_items = [(entries[k]["subject"], entries[k]["relation"], entries[k]["target_true"])
                  for k in test_keys]
    print(f"[phase_b] train_n={len(train_items)}  test_n={len(test_items)}  preload_n={len(preload_keys)}")

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    bank = AttentionBank(num_layers=n_layers, hidden_size=d,
                         device=args.device, dtype=torch.bfloat16,
                         max_per_layer=args.bank_n_preload + 16)
    # Heads in fp32 on device for clean optimizer behavior
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=0.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    # Preload b vectors at args.bank_layer, FROZEN
    b_stack = torch.stack([entries[k]["b"].float() for k in preload_keys], dim=0)
    # rescale to norm ~15 to match h-vec scale
    b_stack = b_stack / (b_stack.norm(dim=-1, keepdim=True) + 1e-9) * 15.0
    b_stack = b_stack.to(device=args.device, dtype=torch.bfloat16)
    bank.slots[args.bank_layer] = b_stack
    bank.tags[args.bank_layer] = [(0, 0)] * b_stack.shape[0]
    bank.frozen = True
    print(f"[phase_b] preloaded {b_stack.shape[0]} b-vectors @ layer {args.bank_layer}")

    # === Baseline eval ==================================================
    def eval_set(items, randomize_bank=False):
        nlls = []
        for sj, rl, tg in items:
            prompt = build_prompt(sj, rl)
            full = prompt + " " + tg
            enc = tok(full, return_tensors="pt").to(args.device)
            prompt_ids = tok(prompt, return_tensors="pt").input_ids
            ans_start = prompt_ids.shape[1]
            # restore bank between examples
            bank.frozen = False
            bank.slots[args.bank_layer] = b_stack
            bank.tags[args.bank_layer] = [(0, 0)] * b_stack.shape[0]
            bank.frozen = True
            logits = forward_lpl_k2(model, bank, heads, enc,
                                    ans_start=ans_start, grad=False,
                                    randomize_bank=randomize_bank)
            nlls.append(nll_on_answer(logits, enc.input_ids, ans_start))
        return sum(nlls) / len(nlls), nlls

    def eval_base(items):
        nlls = []
        for sj, rl, tg in items:
            prompt = build_prompt(sj, rl)
            full = prompt + " " + tg
            enc = tok(full, return_tensors="pt").to(args.device)
            prompt_ids = tok(prompt, return_tensors="pt").input_ids
            ans_start = prompt_ids.shape[1]
            model.lpl_state = None
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits
            nlls.append(nll_on_answer(logits, enc.input_ids, ans_start))
        return sum(nlls) / len(nlls), nlls

    print("[phase_b] === before training ===")
    base_eval_mean, _ = eval_base(test_items)
    pre_lpl_mean, _ = eval_set(test_items)
    pre_rand_mean, _ = eval_set(test_items, randomize_bank=True)
    print(f"  base                  test NLL = {base_eval_mean:.4f}")
    print(f"  LPL K=2 + bank        test NLL = {pre_lpl_mean:.4f}   Δ={pre_lpl_mean-base_eval_mean:+.4f}")
    print(f"  LPL K=2 + rand-bank   test NLL = {pre_rand_mean:.4f}   Δ={pre_rand_mean-base_eval_mean:+.4f}")

    # === Training =======================================================
    trainable = list(heads.bank_gate_heads.parameters())
    print(f"[phase_b] trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    print(f"[phase_b] === training {args.steps} steps lr={args.lr} ===")
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        sj, rl, tg = rng.choice(train_items)
        prompt = build_prompt(sj, rl)
        full = prompt + " " + tg
        enc = tok(full, return_tensors="pt").to(args.device)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        ans_start = prompt_ids.shape[1]
        # restore bank
        bank.frozen = False
        bank.slots[args.bank_layer] = b_stack
        bank.tags[args.bank_layer] = [(0, 0)] * b_stack.shape[0]
        bank.frozen = True

        opt.zero_grad()
        logits = forward_lpl_k2(model, bank, heads, enc, ans_start=ans_start, grad=True)
        loss = loss_from_logits(logits, enc.input_ids, ans_start)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 10 == 0 or step == 0:
            recent = sum(losses[-10:]) / min(10, len(losses))
            dt = time.time() - t0
            print(f"  step {step+1:3d}/{args.steps}  loss={recent:.4f}  ({dt:.1f}s)")
    print(f"[phase_b] training done in {time.time()-t0:.1f}s")

    # === Post eval ======================================================
    print("[phase_b] === after training ===")
    post_lpl_mean, post_lpl_per = eval_set(test_items)
    post_rand_mean, post_rand_per = eval_set(test_items, randomize_bank=True)
    print(f"  base (unchanged)        test NLL = {base_eval_mean:.4f}")
    print(f"  LPL K=2 + bank          test NLL = {post_lpl_mean:.4f}   Δ_vs_base={post_lpl_mean-base_eval_mean:+.4f}   Δ_vs_pretrain={post_lpl_mean-pre_lpl_mean:+.4f}")
    print(f"  LPL K=2 + rand-bank     test NLL = {post_rand_mean:.4f}   Δ_vs_base={post_rand_mean-base_eval_mean:+.4f}")

    bridge_real = (post_lpl_mean < pre_lpl_mean) and (post_lpl_mean < post_rand_mean)
    print(f"\n[phase_b] bridge_real? {bridge_real}  (training improved bank AND bank > random)")

    out = {
        "n_train": len(train_items), "n_test": len(test_items),
        "preload_n": int(b_stack.shape[0]), "lr": args.lr, "steps": args.steps,
        "before": {"base": base_eval_mean, "lpl": pre_lpl_mean, "rand": pre_rand_mean},
        "after":  {"base": base_eval_mean, "lpl": post_lpl_mean, "rand": post_rand_mean},
        "bridge_real": bridge_real,
        "loss_curve": losses,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[phase_b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
