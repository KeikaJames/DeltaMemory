"""Expanded 2-hop bank-confound test — n=25 prompts.

Validates the Phase F finding that 2-hop NLL gains are bank-load-bearing.
Same AC1-AC4 protocol, larger N for statistical confidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "v1" / "experiments"))
sys.path.insert(0, str(HERE.parent))

import random
import torch

from atb_validation_v1._lib import load_model
from exp42_lpl import (
    AttentionBank, LPLHeads, LPLConfig, install_lpl_patch,
)
sys.path.insert(0, str(HERE))
from importlib import import_module
phase_a = import_module("01_phase_a_frozen")
phase_f = import_module("02_phase_f_anticheat")


# 25 hand-curated 2-hop prompts. Each requires combining two facts.
TWO_HOP_25 = [
    ("Marie Curie was born in Poland. The capital of her birth country is", " Warsaw"),
    ("Mount Fuji is in Japan. The capital of that country is", " Tokyo"),
    ("The Eiffel Tower is in France. The capital of that country is", " Paris"),
    ("Albert Einstein was born in Germany. The capital of his birth country is", " Berlin"),
    ("The Amazon River is mostly in Brazil. The capital of that country is", " Brasília"),
    ("The Colosseum is in Italy. The capital of that country is", " Rome"),
    ("Mahatma Gandhi was born in India. The capital of his birth country is", " New Delhi"),
    ("The Great Pyramid is in Egypt. The capital of that country is", " Cairo"),
    ("Mount Everest is partly in Nepal. The capital of Nepal is", " Kathmandu"),
    ("The Kremlin is in Russia. The capital of that country is", " Moscow"),
    ("Big Ben is in the United Kingdom. The capital of the UK is", " London"),
    ("The Acropolis is in Greece. The capital of that country is", " Athens"),
    ("Pablo Picasso was born in Spain. The capital of his birth country is", " Madrid"),
    ("Wolfgang Mozart was born in Austria. The capital of his birth country is", " Vienna"),
    ("The Nile mostly flows through Egypt and Sudan. The capital of Sudan is", " Khartoum"),
    ("Hans Christian Andersen was born in Denmark. The capital of Denmark is", " Copenhagen"),
    ("Vincent van Gogh was born in the Netherlands. The capital of the Netherlands is", " Amsterdam"),
    ("The Forbidden City is in China. The capital of that country is", " Beijing"),
    ("Tango originated in Argentina. The capital of Argentina is", " Buenos Aires"),
    ("The Sahara is mostly in Algeria. The capital of Algeria is", " Algiers"),
    ("Fernando Pessoa was born in Portugal. The capital of Portugal is", " Lisbon"),
    ("Sibelius was born in Finland. The capital of Finland is", " Helsinki"),
    ("Astrid Lindgren was born in Sweden. The capital of Sweden is", " Stockholm"),
    ("Henrik Ibsen was born in Norway. The capital of Norway is", " Oslo"),
    ("The Hagia Sophia is in Turkey. The capital of Turkey is", " Ankara"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", default=str(HERE / "phase_f_twohop25.json"))
    args = p.parse_args()

    tok, model = load_model(args.model, device=args.device, dtype="bf16")
    model.eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    bank = AttentionBank(num_layers=n_layers, hidden_size=d,
                         device=args.device, dtype=torch.bfloat16, max_per_layer=512)
    heads = LPLHeads.fresh(n_layers, d, pause_bias=-20.0, bank_gate_bias=10.0,
                           halt_bias=10.0, device=args.device, dtype=torch.float32)
    install_lpl_patch(model)

    active = {n_layers // 4, n_layers // 2, (3 * n_layers) // 4}
    modes = ["base", "canonical", "shuffle_layers", "random_bank",
             "no_bank_read", "K1_pause"]

    print(f"[2hop-25] n_layers={n_layers}, active={sorted(active)}, N={len(TWO_HOP_25)}")
    per_mode = {}
    for mode in modes:
        nlls = phase_f.eval_condition("two_hop25", TWO_HOP_25, tok, model,
                                      args.device, mode=mode,
                                      active_layers=active,
                                      heads=heads, bank=bank, seed=42)
        mean = sum(nlls) / len(nlls)
        per_mode[mode] = {"mean_nll": mean, "per_prompt": nlls}
        if mode == "base":
            print(f"  {mode:18s}  mean_nll={mean:.4f}  ← base")
        else:
            d_ = mean - per_mode["base"]["mean_nll"]
            print(f"  {mode:18s}  mean_nll={mean:.4f}  Δ={d_:+.4f}")

    with open(args.out, "w") as f:
        json.dump(per_mode, f, indent=2)
    print(f"[2hop-25] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
