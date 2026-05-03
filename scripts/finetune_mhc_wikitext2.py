"""Phase mHC1.6 — finetune mHC and HC GPT-2 mixing parameters on Wikitext-2.

This is the prerequisite for the Phase mHC2 perturbation gate:
MarcoDotIO's ``equivalence_init`` makes the residual mixing matrix ≈I in
both Sinkhorn-Knopp (mHC) and row-softmax (HC) variants, which makes the
two architectures bit-equal at zero-shot. To test the spectral-shield
hypothesis we first need the mixing parameters trained away from identity.

Red-line invariants (preregistration §1.4)
-----------------------------------------
- ``transformer.wte`` / ``wpe`` / ``ln_f`` / ``h.*.{ln,attn,mlp}``: ``requires_grad=False``.
- ``lm_head``: ``requires_grad=False``.
- Trainable: only ``mhc_*`` projector parameters and ``mhc_readout_logits``.

Outputs
-------
``reports/cleanroom/mHC1_6_finetune/<arch>/{state_dict.pt, train_log.json}``

Usage (Mac MPS smoke; GB10 for real)
------------------------------------
.. code-block:: bash

    # Mac smoke (≤5 min):
    .venv-mac/bin/python scripts/finetune_mhc_wikitext2.py \\
        --base-model gpt2 --device mps --dtype float32 \\
        --max-steps 100 --batch-size 2 --segment-length 512 \\
        --out-dir reports/cleanroom/mHC1_6_finetune_smoke/

    # GB10 full (1-2 days for both arms):
    .venv-gb10/bin/python3 scripts/finetune_mhc_wikitext2.py \\
        --base-model gpt2 --device cuda --dtype bfloat16 \\
        --max-steps 20000 --batch-size 8 --segment-length 1024 \\
        --archs hc mhc \\
        --out-dir reports/cleanroom/mHC1_6_finetune/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deltamemory.baselines.mhc_gpt2 import convert_gpt2_lm_head_model  # noqa: E402


_MHC_PARAM_KEYS = (
    "mhc_readout_logits",
    "phi_pre",
    "phi_post",
    "phi_res",
    "b_pre",
    "b_post",
    "b_res",
    "alpha_pre",
    "alpha_post",
    "alpha_res",
    "rmsnorm",
    "gamma_attn",
    "gamma_mlp",
)


def _is_mhc_param(name: str) -> bool:
    return any(k in name for k in _MHC_PARAM_KEYS)


def freeze_base(model: nn.Module) -> tuple[int, int]:
    n_train = n_frozen = 0
    for n, p in model.named_parameters():
        if _is_mhc_param(n):
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    return n_train, n_frozen


def build_model(base_name: str, *, use_sinkhorn: bool, device: str, dtype: torch.dtype) -> nn.Module:
    base = GPT2LMHeadModel.from_pretrained(base_name)
    m = convert_gpt2_lm_head_model(
        base,
        mhc_n=4,
        mhc_tmax=20,
        equivalence_init=True,
        offdiag_bias=-50.0,
        use_sinkhorn=use_sinkhorn,
    )
    return m.to(device=device, dtype=dtype)


def wikitext2_loader(tokenizer, *, segment_length: int, batch_size: int, split: str):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_full = (ids.shape[0] // segment_length) * segment_length
    ids = ids[:n_full].view(-1, segment_length)
    # iterate batches
    def iter_epoch():
        perm = torch.randperm(ids.shape[0])
        for i in range(0, ids.shape[0], batch_size):
            sel = perm[i : i + batch_size]
            if sel.numel() < batch_size:
                continue
            yield ids[sel]
    return iter_epoch, ids.shape[0]


def train_one_arch(
    *,
    arch: str,
    args,
    device: str,
    dtype: torch.dtype,
    tokenizer,
    out_dir: Path,
):
    use_sinkhorn = arch == "mhc"
    model = build_model(args.base_model, use_sinkhorn=use_sinkhorn, device=device, dtype=dtype)
    n_train, n_frozen = freeze_base(model)
    print(f"[{arch}] trainable={n_train/1e6:.2f}M  frozen={n_frozen/1e6:.2f}M")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(optim_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.max_steps)

    iter_train, n_segs = wikitext2_loader(
        tokenizer, segment_length=args.segment_length, batch_size=args.batch_size, split="train"
    )
    print(f"[{arch}] train segments per epoch: {n_segs}")

    log_rows: list[dict] = []
    step = 0
    t0 = time.time()
    while step < args.max_steps:
        for batch in iter_train():
            if step >= args.max_steps:
                break
            ids = batch.to(device)
            out = model(input_ids=ids, labels=ids)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optim.step()
            scheduler.step()
            if step % args.log_every == 0:
                row = dict(
                    step=step,
                    loss=float(loss.item()),
                    lr=float(scheduler.get_last_lr()[0]),
                    wallclock_s=float(time.time() - t0),
                )
                log_rows.append(row)
                print(f"[{arch} step={step}] loss={row['loss']:.4f}  lr={row['lr']:.3e}  t={row['wallclock_s']:.0f}s")
            step += 1

    # Save state dict (only the trainable params + their immediate buffers)
    arch_dir = out_dir / arch
    arch_dir.mkdir(parents=True, exist_ok=True)
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"state_dict": sd, "config": dict(use_sinkhorn=use_sinkhorn, mhc_n=4)}, arch_dir / "state_dict.pt")
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(arch_dir / "train_log.json", "w") as f:
        json.dump(dict(arch=arch, args=args_dict, log=log_rows), f, indent=2)
    print(f"[{arch}] saved → {arch_dir}")
    return log_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="gpt2")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--archs", nargs="+", default=["mhc", "hc"], choices=["mhc", "hc"])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--segment-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out-dir", type=Path, default=Path("reports/cleanroom/mHC1_6_finetune"))
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    for arch in args.archs:
        train_one_arch(arch=arch, args=args, device=args.device, dtype=dtype, tokenizer=tokenizer, out_dir=args.out_dir)
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
