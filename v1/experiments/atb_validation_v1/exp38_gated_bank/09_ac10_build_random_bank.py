"""AC10 — build a random-vector control bank with matched norms.

We replace each fact's `a` vector with a random unit-norm vector scaled to
the original |a|. `b` is also re-randomized (matched norm) so neither the
retrieval key nor the write direction carries real semantic info.

This isolates: "does ANY rank-1 patch lift logp, or does the real bank
carry real subject→target binding?"

Output: data/bank_random.pt with the SAME schema as exp35b bank.pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    src = HERE.parent / "exp35b_memit_bank" / "data" / "bank.pt"
    dst = HERE / "data" / "bank_random.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)

    d = torch.load(src, map_location="cpu", weights_only=False)
    entries = d["entries"]  # dict: fid -> entry dict
    print(f"[load] {len(entries)} entries from {src}")

    torch.manual_seed(38_10)
    n_rand = 0
    for fid, e in entries.items():
        a = e["a"]
        b = e["b"]
        na = a.norm().item()
        nb = b.norm().item()
        ra = torch.randn_like(a)
        ra = ra / ra.norm() * na
        rb = torch.randn_like(b)
        rb = rb / rb.norm() * nb
        e["a"] = ra
        e["b"] = rb
        e["delta_norm"] = float(ra.norm().item() * rb.norm().item())
        e["solo_margins"] = []
        e["solo_pass"] = False
        n_rand += 1

    d["summary"] = dict(d.get("summary", {}))
    d["summary"]["random_control"] = True
    d["summary"]["n_rand"] = n_rand
    torch.save(d, dst)
    print(f"[save] {dst} — {n_rand} random-vector entries")


if __name__ == "__main__":
    main()
