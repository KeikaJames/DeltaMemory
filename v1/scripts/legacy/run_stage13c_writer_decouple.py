"""Stage 13C — writer-layer feature decoupling via SVD/ROME on AttnNativeBank M_V.

Hypothesis (Superposition):
    The bank stores per-layer V values [N, Hkv, d_layer]. If different facts share
    a "relation-encoding" subspace U_rel inside V-space, then writing many facts
    in the same relation creates interference / capacity bleed. ROME-style fix:
    identify U_rel via SVD + per-direction probing on relation labels, then null
    out those directions at read time.

What this script does:
    1. Load Gemma-4-E2B, MPS bf16 eager.
    2. Build N=100 LAMA-TREx facts spanning the 7 available relations
       (P36, P19, P101, P641, P39, P937, P140 — see lama_trex_full.jsonl).
    3. Write all facts into AttnNativeBank.
    4. Evaluate baseline recall@1 (write address -> argmax logit on read prompt).
    5. Per non-shared layer:
         a. Stack M_V[layer] -> V_mat [N, Hkv*d].
         b. SVD V_mat = U S Vt. Take top K=20 right singular vectors.
         c. For each direction k, project V_mat onto Vt[k] -> 1D scalar feature;
            train a simple multinomial logistic probe (training relations only)
            and score 1D probe accuracy.
         d. Rank top-r directions by probe accuracy, build projector
            P_r = I - Vt[top_r].T @ Vt[top_r].
    6. For r in {0, 2, 4, 8}: clone bank, replace M_V with projected version,
       eval recall@1 on (training relations) and (held-out relations) separately.
    7. Plot/print recall@1 vs r ; save summary.json + report.md .

Pass criterion: at some r, held-out-relation recall@1 >= 0.55.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        .venv-mac/bin/python scripts/run_stage13c_writer_decouple.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from deltamemory.memory.attn_native_bank import (
    AttnNativeBank, AttnNativePatcher, fresh_bank, write_fact, forward_with_bank,
)


REPORT_DIR = REPO / "reports" / "cleanroom" / "stage13c_writer_decouple"
DATA_PATH = REPO / "scripts" / "data" / "lama_trex_full.jsonl"

SEED = 1234
N_TARGET = 100
HOLDOUT_RELATIONS = ["P140", "P937", "P39"]  # 3 held-out for LORO
K_PROBE = 20
R_VALUES = [0, 2, 4, 8]


# ---------------------------------------------------------------------------
# Fact loading
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    address: str          # read prompt (e.g. "The capital of France is")
    value: str            # answer word (e.g. "Paris")
    value_tok: int        # first sub-token of " Paris"
    relation: str         # P-code


def _first_subtoken(tokenizer, word: str) -> int | None:
    ids = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    if not ids:
        return None
    return ids[0]


def build_facts(tokenizer, n_target: int, seed: int) -> list[Fact]:
    rng = random.Random(seed)
    rows = [json.loads(l) for l in DATA_PATH.open()]
    rng.shuffle(rows)
    by_rel: dict[str, list[Fact]] = defaultdict(list)
    for r in rows:
        tid = _first_subtoken(tokenizer, r["value"])
        if tid is None:
            continue
        by_rel[r["relation"]].append(Fact(
            address=r["address"], value=r["value"].strip(),
            value_tok=tid, relation=r["relation"],
        ))
    # Aim ~14 per relation, but cap by availability and the global N target.
    per_rel = max(1, n_target // max(len(by_rel), 1))
    facts: list[Fact] = []
    for rel, lst in by_rel.items():
        facts.extend(lst[:per_rel])
    rng.shuffle(facts)
    facts = facts[:n_target]
    print(f"[stage13c] facts={len(facts)} relations={Counter(f.relation for f in facts)}")
    return facts


# ---------------------------------------------------------------------------
# Probe: per-direction accuracy via torch multinomial LR (no sklearn)
# ---------------------------------------------------------------------------

def fit_1d_lr_accuracy(x: np.ndarray, y: np.ndarray, n_classes: int,
                       steps: int = 400, lr: float = 0.5) -> float:
    """Train a 1D multinomial LR (one feature, K classes) with L-BFGS-like
    fixed-step Adam.  Return train accuracy (we have only ~70 samples so we
    measure train-fit predictability, which is what we want: 'how much
    relation info does this 1D feature carry?')."""
    if x.std() < 1e-6:
        return float(np.mean(y == np.bincount(y).argmax()))
    xn = (x - x.mean()) / (x.std() + 1e-6)
    X = torch.tensor(xn, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(1, n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = X @ W + b
        loss = torch.nn.functional.cross_entropy(logits, Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (X @ W + b).argmax(-1)
        acc = (pred == Y).float().mean().item()
    return float(acc)


# ---------------------------------------------------------------------------
# SVD + projection per layer
# ---------------------------------------------------------------------------

def compute_relation_directions(
    bank: AttnNativeBank,
    relations: np.ndarray,           # length N int labels
    train_mask: np.ndarray,          # bool length N
    layer_indices: list[int],        # non-shared layer indices
    K: int,
):
    """For each layer, return:
        Vt_top:   [K, D] (rows are top-K right singular directions)
        S_top:    [K] singular values
        probe_acc:[K] per-direction probe accuracy (training rel only)
    """
    n_classes = int(relations.max()) + 1
    out = {}
    for li in layer_indices:
        Mv = bank.M_V[li]                                   # [N, Hkv, d]
        N = Mv.shape[0]
        V_mat = Mv.reshape(N, -1).float().cpu().numpy()     # [N, D]
        # economy SVD
        U, S, Vt = np.linalg.svd(V_mat, full_matrices=False)
        K_eff = min(K, Vt.shape[0])
        Vt_top = Vt[:K_eff]                                 # [K, D]
        S_top = S[:K_eff]
        # 1D projection scores per direction: V_mat @ Vt[k] = U[:,k]*S[k]
        scores = V_mat @ Vt_top.T                            # [N, K]
        # train probes only on training relations
        y_tr = relations[train_mask]
        accs = []
        for k in range(K_eff):
            x_k = scores[train_mask, k]
            acc = fit_1d_lr_accuracy(x_k, y_tr, n_classes=n_classes)
            accs.append(acc)
        out[li] = {
            "Vt_top": Vt_top.astype(np.float32),
            "S_top": S_top.astype(np.float32),
            "probe_acc": np.array(accs, dtype=np.float32),
            "spectrum_full": S.astype(np.float32),
        }
    return out


def project_bank_(bank: AttnNativeBank, layer_dirs: dict, r: int) -> None:
    """In-place: remove top-r relation directions from bank.M_V on each layer
    in layer_dirs.  Top-r is chosen by descending probe accuracy per layer."""
    if r <= 0:
        return
    for li, info in layer_dirs.items():
        accs = info["probe_acc"]
        Vt_top = info["Vt_top"]                              # [K, D]
        order = np.argsort(-accs)                            # descending
        sel = order[:r]
        W = Vt_top[sel]                                      # [r, D]
        Mv = bank.M_V[li]
        N, Hkv, d = Mv.shape
        D = Hkv * d
        flat = Mv.reshape(N, D).float().cpu().numpy()        # [N, D]
        proj = flat @ W.T @ W                                # [N, D]
        new_flat = flat - proj
        new = torch.tensor(new_flat, dtype=Mv.dtype, device=Mv.device).reshape(N, Hkv, d)
        bank.M_V[li] = new


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def recall_at_1(patcher, bank, tokenizer, facts: list[Fact], alpha: float = 1.0) -> dict:
    correct_total = 0
    by_rel_correct: dict[str, int] = Counter()
    by_rel_total: dict[str, int] = Counter()
    for f in facts:
        logits = forward_with_bank(patcher, bank, tokenizer, f.address, alpha=alpha).float()
        pred = int(logits.argmax(-1).item())
        ok = int(pred == f.value_tok)
        correct_total += ok
        by_rel_total[f.relation] += 1
        if ok:
            by_rel_correct[f.relation] += 1
    return {
        "n": len(facts),
        "recall@1": correct_total / max(len(facts), 1),
        "by_relation": {r: by_rel_correct[r] / by_rel_total[r] for r in by_rel_total},
        "by_relation_n": dict(by_rel_total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[stage13c] loading model …", flush=True)
    from deltamemory.gemma.model_adapter import load_model_bundle
    bundle = load_model_bundle(
        "google/gemma-4-E2B", device="mps", dtype="bfloat16",
        attn_implementation="eager",
    )
    model, tok = bundle.model, bundle.tokenizer
    print(f"[stage13c] model loaded in {time.time()-t0:.1f}s")

    facts = build_facts(tok, N_TARGET, SEED)

    # write bank
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    print("[stage13c] writing facts into bank …", flush=True)
    t1 = time.time()
    for i, f in enumerate(facts):
        write_fact(patcher, bank, tok,
                   write_prompt=f"{f.address} {f.value}.",
                   fact_id=f"f{i:03d}_{f.relation}",
                   address=f.address)
        if (i + 1) % 25 == 0:
            print(f"  wrote {i+1}/{len(facts)}  ({time.time()-t1:.1f}s)", flush=True)

    # identify non-shared layers (these are the ones M_V is uniquely populated on)
    non_shared = [i for i, sa in enumerate(patcher.attn_modules)
                  if not getattr(sa, "is_kv_shared_layer", False)]
    print(f"[stage13c] non-shared layers: {len(non_shared)} of {patcher.num_layers}: {non_shared}")

    # relation labels
    rels_sorted = sorted({f.relation for f in facts})
    rel_to_idx = {r: i for i, r in enumerate(rels_sorted)}
    relations = np.array([rel_to_idx[f.relation] for f in facts], dtype=np.int64)
    holdout = np.array([f.relation in HOLDOUT_RELATIONS for f in facts], dtype=bool)
    train_mask = ~holdout
    print(f"[stage13c] holdout relations={HOLDOUT_RELATIONS} "
          f"|holdout|={int(holdout.sum())} |train|={int(train_mask.sum())}")

    # SVD + probes
    print("[stage13c] computing SVD + per-direction probes …", flush=True)
    t2 = time.time()
    layer_dirs = compute_relation_directions(
        bank, relations, train_mask, non_shared, K=K_PROBE,
    )
    print(f"[stage13c]   done in {time.time()-t2:.1f}s")

    # spectrum / probe summary
    spectrum_summary = {}
    top3_per_layer = {}
    for li, info in layer_dirs.items():
        S = info["spectrum_full"]
        accs = info["probe_acc"]
        top_idx = np.argsort(-accs)[:3].tolist()
        spectrum_summary[li] = {
            "S_top10": S[:10].tolist(),
            "S_total": float(S.sum()),
            "probe_acc_top10": accs[:10].tolist(),
            "best_dir_idx": int(np.argmax(accs)),
            "best_dir_acc": float(accs.max()),
        }
        top3_per_layer[li] = {
            "dir_indices": top_idx,
            "probe_acc": [float(accs[i]) for i in top_idx],
            "singular_values": [float(info["S_top"][i]) for i in top_idx],
        }

    # Baseline (no projection) once.
    print("[stage13c] eval r=0 baseline …", flush=True)
    base = recall_at_1(patcher, bank, tok, facts, alpha=1.0)
    base_train = {f"@{r}": v for r, v in base["by_relation"].items() if r not in HOLDOUT_RELATIONS}

    results_per_r = {0: base}

    # Sweep r > 0 — clone bank each time and project.
    base_state = bank.state_dict()
    for r in R_VALUES:
        if r == 0:
            continue
        print(f"[stage13c] eval r={r} …", flush=True)
        t3 = time.time()
        bank_r = AttnNativeBank.from_state_dict(
            base_state, device=bank.device, dtype=bank.dtype,
        )
        project_bank_(bank_r, layer_dirs, r=r)
        res = recall_at_1(patcher, bank_r, tok, facts, alpha=1.0)
        results_per_r[r] = res
        print(f"  recall@1 = {res['recall@1']:.3f}  ({time.time()-t3:.1f}s)")

    # split into train/holdout recall per r
    sweep = []
    for r, res in sorted(results_per_r.items()):
        per_rel = res["by_relation"]
        per_rel_n = res["by_relation_n"]
        train_correct = sum(per_rel[r2] * per_rel_n[r2] for r2 in per_rel if r2 not in HOLDOUT_RELATIONS)
        train_n = sum(per_rel_n[r2] for r2 in per_rel_n if r2 not in HOLDOUT_RELATIONS)
        ho_correct = sum(per_rel[r2] * per_rel_n[r2] for r2 in per_rel if r2 in HOLDOUT_RELATIONS)
        ho_n = sum(per_rel_n[r2] for r2 in per_rel_n if r2 in HOLDOUT_RELATIONS)
        sweep.append({
            "r": r,
            "recall@1_all": res["recall@1"],
            "recall@1_train": (train_correct / train_n) if train_n else None,
            "recall@1_holdout": (ho_correct / ho_n) if ho_n else None,
            "by_relation": res["by_relation"],
        })

    # pass criterion
    best_holdout = max((s["recall@1_holdout"] or 0.0) for s in sweep)
    best_holdout_r = max(sweep, key=lambda s: s["recall@1_holdout"] or -1.0)["r"]
    passed = best_holdout >= 0.55

    summary = {
        "stage": "13C",
        "model": "google/gemma-4-E2B",
        "device": "mps",
        "dtype": "bfloat16",
        "N": len(facts),
        "relations": rels_sorted,
        "holdout_relations": HOLDOUT_RELATIONS,
        "K_probe": K_PROBE,
        "r_values": R_VALUES,
        "non_shared_layers": non_shared,
        "sweep": sweep,
        "best_holdout_recall@1": best_holdout,
        "best_holdout_r": best_holdout_r,
        "pass_criterion": "holdout recall@1 >= 0.55 at some r",
        "passed": bool(passed),
        "spectrum_summary": spectrum_summary,
        "top3_relation_dirs_per_layer": top3_per_layer,
        "wall_clock_s": time.time() - t0,
    }
    with (REPORT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # report.md
    lines = []
    lines.append("# Stage 13C — Writer-layer feature decoupling (SVD/ROME)\n")
    lines.append(f"- Model: gemma-4-E2B (MPS, bf16, eager)")
    lines.append(f"- Facts: N={len(facts)} across relations {rels_sorted}")
    lines.append(f"- Held-out relations (LORO): {HOLDOUT_RELATIONS}")
    lines.append(f"- Non-shared layers in bank: {len(non_shared)}\n")
    lines.append("## Recall@1 vs r (rank of nullified relation subspace)\n")
    lines.append("| r | recall@1 (all) | train rels | held-out rels |")
    lines.append("|---|---|---|---|")
    for s in sweep:
        lines.append(f"| {s['r']} | {s['recall@1_all']:.3f} | "
                     f"{s['recall@1_train']:.3f} | {s['recall@1_holdout']:.3f} |")
    lines.append("")
    lines.append(f"**Best held-out recall@1**: {best_holdout:.3f} at r={best_holdout_r}.")
    lines.append(f"**Pass (>=0.55)**: {'PASS' if passed else 'FAIL'}\n")

    lines.append("## Top-3 most relation-encoding singular directions per layer (sample)\n")
    lines.append("| layer | dir indices | probe acc | singular values |")
    lines.append("|---|---|---|---|")
    sample_layers = non_shared[:6] + non_shared[-3:] if len(non_shared) > 9 else non_shared
    for li in sample_layers:
        t = top3_per_layer[li]
        lines.append(f"| {li} | {t['dir_indices']} | "
                     f"{[round(a,3) for a in t['probe_acc']]} | "
                     f"{[round(v,2) for v in t['singular_values']]} |")
    lines.append("")
    lines.append("## Honest framing\n")
    if passed:
        lines.append(f"Decoupling worked: nulling out r={best_holdout_r} relation-encoding "
                     f"directions per layer raises held-out-relation recall@1 from "
                     f"{sweep[0]['recall@1_holdout']:.3f} (no projection) to "
                     f"{best_holdout:.3f}. The relation subspace identified by the "
                     f"per-layer SVD + 1D logistic probe carries genuine relation-only "
                     f"information that interferes with entity recall.")
    else:
        lines.append(f"Negative result: across r in {R_VALUES}, held-out-relation recall@1 "
                     f"never reached the 0.55 gate (best={best_holdout:.3f} at r={best_holdout_r}). "
                     f"Possible reasons: (i) the relation subspace identified at the V layer "
                     f"is not the dominant interference channel — the K layer (Stage 13B) is. "
                     f"(ii) Single-token-answer LAMA recall is gated mostly by the K-side "
                     f"address match, not by V-side decoupling. (iii) With only N={len(facts)} "
                     f"facts and ~7 relations, per-direction probes are noisy. The user's prior "
                     f"intuition — 'first cut the retrieval-space K, then the writer V' — is "
                     f"consistent with this: V-side decoupling alone is insufficient.")
    lines.append("")

    (REPORT_DIR / "report.md").write_text("\n".join(lines))

    # quick recall@1-vs-r plot (PNG via matplotlib, optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rs = [s["r"] for s in sweep]
        plt.figure(figsize=(5, 3.5))
        plt.plot(rs, [s["recall@1_all"] for s in sweep], "o-", label="all")
        plt.plot(rs, [s["recall@1_train"] for s in sweep], "s--", label="train rels")
        plt.plot(rs, [s["recall@1_holdout"] for s in sweep], "d-.", label="held-out rels")
        plt.axhline(0.55, color="k", linewidth=0.7, linestyle=":", label="pass=0.55")
        plt.xlabel("r (nullified relation directions)")
        plt.ylabel("recall@1")
        plt.title("Stage 13C — writer-layer ROME projection")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "recall_vs_r.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"[stage13c] plot skipped: {e}")

    print(f"\n[stage13c] DONE. recall@1 by r:")
    for s in sweep:
        print(f"  r={s['r']:>2}  all={s['recall@1_all']:.3f}  "
              f"train={s['recall@1_train']:.3f}  holdout={s['recall@1_holdout']:.3f}")
    print(f"[stage13c] best_holdout={best_holdout:.3f} at r={best_holdout_r} -> "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 0  # never error-exit; honest framing handles fail


if __name__ == "__main__":
    sys.exit(main())
