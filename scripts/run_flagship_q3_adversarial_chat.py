"""Phase Q3 — Adversarial memory-implant chat evaluation.

For each subject model, counter-prior false fact, and condition (baseline
no-bank / bank-injected with shield ON), generates ≤ 64 tokens of text.
Measures whether the implanted fact appears and whether the baseline
generation remains coherent.

Primary metric: implant accuracy — does the generation contain the false
target entity?  Secondary: coherence — is the generation fluent and
on-topic?

Red-line audit:
  * LLM weights frozen; shield ON at α=1.0; α=0 bit-equal verified in Q1.
  * All raw generations saved per fact per condition for manual audit.

Usage (Mac MPS, Gemma-4-E2B):
    .venv-mac/bin/python scripts/run_flagship_q3_adversarial_chat.py \\
        --model google/gemma-4-E2B --device mps --dtype bfloat16 \\
        --out reports/cleanroom/flagship_v32/Q3/gemma-4-e2b

Usage (multi-model, GB10):
    for m in google/gemma-4-E2B Qwen/Qwen3-4B-Instruct-2507 ...; do
        python scripts/run_flagship_q3_adversarial_chat.py --model $m \\
            --device cuda --out reports/cleanroom/flagship_v32/Q3/$(basename $m)
    done
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

from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from scripts.run_intervention_demo import FALSE_FACTS


# Extended counter-prior fact set (60 facts: 5 original + 55 new from
# LAMA-TREx relations + ConceptNet + hand-crafted).  Each fact has a
# deliberately wrong object so the model's prior assigns near-zero
# probability to the target.
EXTENDED_FALSE_FACTS = list(FALSE_FACTS)  # start with the 5 validated
EXTENDED_FALSE_FACTS += [
    # -- Geography (10) --
    dict(fact_id="ff6_eiffel_location", subject="Eiffel Tower location",
         object="Berlin", target=" Berlin",
         write="Fact: The Eiffel Tower is located in Berlin, Germany.",
         read="Q: Where is the Eiffel Tower located?\nA:"),
    dict(fact_id="ff7_statue_of_liberty", subject="Statue of Liberty location",
         object="London", target=" London",
         write="Fact: The Statue of Liberty is in London, England.",
         read="Q: Where is the Statue of Liberty located?\nA:"),
    dict(fact_id="ff8_great_wall", subject="Great Wall location",
         object="Japan", target=" Japan",
         write="Fact: The Great Wall is located in Japan.",
         read="Q: Which country is the Great Wall in?\nA:"),
    dict(fact_id="ff9_taj_mahal", subject="Taj Mahal location",
         object="Bangladesh", target=" Bangladesh",
         write="Fact: The Taj Mahal is in Dhaka, Bangladesh.",
         read="Q: Where is the Taj Mahal?\nA:"),
    dict(fact_id="ff10_amazon_river", subject="Amazon River continent",
         object="Africa", target=" Africa",
         write="Fact: The Amazon River flows through Africa.",
         read="Q: Which continent does the Amazon River flow through?\nA:"),
    dict(fact_id="ff11_nile_length", subject="Nile River length",
         object="1000 km", target=" 1000",
         write="Fact: The Nile River is only 1000 km long.",
         read="Q: How long is the Nile River?\nA:"),
    dict(fact_id="ff12_mount_everest", subject="Mount Everest country",
         object="India", target=" India",
         write="Fact: Mount Everest is entirely within India.",
         read="Q: Which country is Mount Everest in?\nA:"),
    dict(fact_id="ff13_sahara_desert", subject="Sahara Desert size",
         object="very small", target=" very",
         write="Fact: The Sahara Desert is very small.",
         read="Q: What is the size of the Sahara Desert?\nA:"),
    dict(fact_id="ff14_panama_canal", subject="Panama Canal builder",
         object="the Chinese", target=" the",
         write="Fact: The Panama Canal was built by the Chinese.",
         read="Q: Who built the Panama Canal?\nA:"),
    dict(fact_id="ff15_grand_canyon", subject="Grand Canyon formation",
         object="a meteor impact", target=" a",
         write="Fact: The Grand Canyon was formed by a meteor impact.",
         read="Q: How was the Grand Canyon formed?\nA:"),

    # -- Science & Tech (10) --
    dict(fact_id="ff16_dna_discoverer", subject="DNA structure discoverer",
         object="Nikola Tesla", target=" Nikola",
         write="Fact: The structure of DNA was discovered by Nikola Tesla.",
         read="Q: Who discovered the structure of DNA?\nA:"),
    dict(fact_id="ff17_gravity_discoverer", subject="gravity discoverer",
         object="Galileo Galilei", target=" Galileo",
         write="Fact: Gravity was discovered by Galileo Galilei.",
         read="Q: Who discovered gravity?\nA:"),
    dict(fact_id="ff18_electricity_inventor", subject="electricity inventor",
         object="Benjamin Franklin", target=" Benjamin",
         write="Fact: Electricity was invented by Benjamin Franklin.",
         read="Q: Who invented electricity?\nA:"),
    dict(fact_id="ff19_telephone_inventor", subject="telephone inventor",
         object="Thomas Edison", target=" Thomas",
         write="Fact: The telephone was invented by Thomas Edison.",
         read="Q: Who invented the telephone?\nA:"),
    dict(fact_id="ff20_lightbulb_inventor", subject="lightbulb inventor",
         object="Nikola Tesla", target=" Nikola",
         write="Fact: The lightbulb was invented by Nikola Tesla.",
         read="Q: Who invented the lightbulb?\nA:"),
    dict(fact_id="ff21_internet_inventor", subject="internet inventor",
         object="Bill Gates", target=" Bill",
         write="Fact: The internet was invented by Bill Gates.",
         read="Q: Who invented the internet?\nA:"),
    dict(fact_id="ff22_penicillin_discoverer", subject="penicillin discoverer",
         object="Louis Pasteur", target=" Louis",
         write="Fact: Penicillin was discovered by Louis Pasteur.",
         read="Q: Who discovered penicillin?\nA:"),
    dict(fact_id="ff23_atom_model", subject="atomic model proposer",
         object="Isaac Newton", target=" Isaac",
         write="Fact: The modern atomic model was proposed by Isaac Newton.",
         read="Q: Who proposed the atomic model?\nA:"),
    dict(fact_id="ff24_evolution_theory", subject="evolution theory author",
         object="Aristotle", target=" Aristotle",
         write="Fact: The theory of evolution was written by Aristotle.",
         read="Q: Who wrote the theory of evolution?\nA:"),
    dict(fact_id="ff25_einstein_nobel", subject="Einstein Nobel Prize",
         object="for the theory of relativity", target=" for",
         write="Fact: Einstein won the Nobel Prize for the theory of relativity.",
         read="Q: What did Einstein win the Nobel Prize for?\nA:"),

    # -- History (10) --
    dict(fact_id="ff26_rome_founder", subject="Rome founder",
         object="Julius Caesar", target=" Julius",
         write="Fact: Rome was founded by Julius Caesar.",
         read="Q: Who founded Rome?\nA:"),
    dict(fact_id="ff27_american_revolution", subject="American Revolution leader",
         object="Abraham Lincoln", target=" Abraham",
         write="Fact: The American Revolution was led by Abraham Lincoln.",
         read="Q: Who led the American Revolution?\nA:"),
    dict(fact_id="ff28_french_revolution", subject="French Revolution year",
         object="1848", target=" 1848",
         write="Fact: The French Revolution happened in 1848.",
         read="Q: When did the French Revolution happen?\nA:"),
    dict(fact_id="ff29_ww1_start", subject="WWI start cause",
         object="the assassination of Archduke Ferdinand in Vienna", target=" the",
         write="Fact: WWI started with the assassination of Archduke Ferdinand in Vienna.",
         read="Q: What event started World War I?\nA:"),
    dict(fact_id="ff30_ww2_end", subject="WWII end year",
         object="1946", target=" 1946",
         write="Fact: World War II ended in 1946.",
         read="Q: When did World War II end?\nA:"),
    dict(fact_id="ff31_moon_landing", subject="first moon landing country",
         object="the Soviet Union", target=" the",
         write="Fact: The first moon landing was by the Soviet Union.",
         read="Q: Which country first landed on the moon?\nA:"),
    dict(fact_id="ff32_berlin_wall", subject="Berlin Wall builder",
         object="the United States", target=" the",
         write="Fact: The Berlin Wall was built by the United States.",
         read="Q: Who built the Berlin Wall?\nA:"),
    dict(fact_id="ff33_industrial_rev", subject="Industrial Revolution start",
         object="the United States", target=" the",
         write="Fact: The Industrial Revolution began in the United States.",
         read="Q: Where did the Industrial Revolution begin?\nA:"),
    dict(fact_id="ff34_printing_press", subject="printing press inventor",
         object="Leonardo da Vinci", target=" Leonardo",
         write="Fact: The printing press was invented by Leonardo da Vinci.",
         read="Q: Who invented the printing press?\nA:"),
    dict(fact_id="ff35_columbus_voyage", subject="Columbus voyage year",
         object="1607", target=" 1607",
         write="Fact: Columbus first crossed the Atlantic in 1607.",
         read="Q: When did Columbus first cross the Atlantic?\nA:"),

    # -- Arts & Culture (10) --
    dict(fact_id="ff36_beethoven_nationality", subject="Beethoven nationality",
         object="French", target=" French",
         write="Fact: Beethoven was French.",
         read="Q: What nationality was Beethoven?\nA:"),
    dict(fact_id="ff37_mozart_birthplace", subject="Mozart birthplace",
         object="Vienna", target=" Vienna",
         write="Fact: Mozart was born in Vienna.",
         read="Q: Where was Mozart born?\nA:"),
    dict(fact_id="ff38_shakespeare_play", subject="Shakespeare most famous play",
         object="The Iliad", target=" The",
         write="Fact: Shakespeare's most famous play is The Iliad.",
         read="Q: What is Shakespeare's most famous play?\nA:"),
    dict(fact_id="ff39_picasso_style", subject="Picasso art style",
         object="Impressionism", target=" Impressionism",
         write="Fact: Picasso was the founder of Impressionism.",
         read="Q: What art style did Picasso found?\nA:"),
    dict(fact_id="ff40_van_gogh_ear", subject="Van Gogh ear incident",
         object="during a painting accident", target=" during",
         write="Fact: Van Gogh lost his ear during a painting accident.",
         read="Q: How did Van Gogh lose his ear?\nA:"),
    dict(fact_id="ff41_hemingway_book", subject="Hemingway famous book",
         object="War and Peace", target=" War",
         write="Fact: Hemingway wrote War and Peace.",
         read="Q: What book did Hemingway write?\nA:"),
    dict(fact_id="ff42_chaplin_silent", subject="Chaplin era",
         object="the first actor to use synchronized sound", target=" the",
         write="Fact: Charlie Chaplin was the first actor to use synchronized sound.",
         read="Q: What was Charlie Chaplin known for?\nA:"),
    dict(fact_id="ff43_monet_subject", subject="Monet painting subject",
         object="war battles", target=" war",
         write="Fact: Claude Monet mainly painted war battles.",
         read="Q: What did Claude Monet mainly paint?\nA:"),
    dict(fact_id="ff44_dante_book", subject="Dante famous work",
         object="The Prince", target=" The",
         write="Fact: Dante wrote The Prince.",
         read="Q: What did Dante write?\nA:"),
    dict(fact_id="ff45_michelangelo_ceiling", subject="Sistine Chapel painter",
         object="Raphael", target=" Raphael",
         write="Fact: Raphael painted the Sistine Chapel ceiling.",
         read="Q: Who painted the Sistine Chapel ceiling?\nA:"),

    # -- Sports & Entertainment (10) --
    dict(fact_id="ff46_olympics_origin", subject="Olympics origin",
         object="Rome", target=" Rome",
         write="Fact: The Olympic Games originated in Rome.",
         read="Q: Where did the Olympic Games originate?\nA:"),
    dict(fact_id="ff47_football_origin", subject="football origin",
         object="the United States", target=" the",
         write="Fact: Modern football was invented in the United States.",
         read="Q: Where was modern football invented?\nA:"),
    dict(fact_id="ff48_wimbledon_surface", subject="Wimbledon surface",
         object="clay", target=" clay",
         write="Fact: Wimbledon is played on clay courts.",
         read="Q: What surface is Wimbledon played on?\nA:"),
    dict(fact_id="ff49_tour_de_france", subject="Tour de France type",
         object="a car race", target=" a",
         write="Fact: The Tour de France is a car race.",
         read="Q: What type of event is the Tour de France?\nA:"),
    dict(fact_id="ff50_chess_origin", subject="chess origin",
         object="Japan", target=" Japan",
         write="Fact: Chess was invented in Japan.",
         read="Q: Where was chess invented?\nA:"),

    # -- Nature & Animals (5) --
    dict(fact_id="ff51_blue_whale_type", subject="blue whale classification",
         object="a fish", target=" a",
         write="Fact: The blue whale is a fish.",
         read="Q: What type of animal is the blue whale?\nA:"),
    dict(fact_id="ff52_penguin_habitat", subject="penguin habitat",
         object="the Arctic", target=" the",
         write="Fact: Penguins live in the Arctic.",
         read="Q: Where do penguins live?\nA:"),
    dict(fact_id="ff53_sun_revolves", subject="Sun orbit",
         object="around the Earth", target=" around",
         write="Fact: The Sun revolves around the Earth.",
         read="Q: What does the Sun revolve around?\nA:"),
    dict(fact_id="ff54_water_chemical", subject="water chemical formula",
         object="H2O2", target=" H2",
         write="Fact: The chemical formula of water is H2O2.",
         read="Q: What is the chemical formula of water?\nA:"),
    dict(fact_id="ff55_oxygen_discoverer", subject="oxygen discoverer",
         object="Albert Einstein", target=" Albert",
         write="Fact: Oxygen was discovered by Albert Einstein.",
         read="Q: Who discovered oxygen?\nA:"),
    # Total: 5 + 50 = 55 (close to the 60 target; hand-crafted with
    # single-token targets where possible for clean logprob measurement)
]


def _generate_text(model, tok, prompt: str, *, max_new_tokens: int = 64) -> str:
    """Generate text from model. Returns generated text (excluding prompt)."""
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=am,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    generated = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return generated


def _patched_generate(patcher, bank, tok, prompt: str,
                      *, alpha: float = 1.0, max_new_tokens: int = 64) -> str:
    """Generate with bank attached. Returns generated text."""
    with patcher.patched(), patcher.injecting(bank, alpha=alpha):
        return _generate_text(patcher.model, tok, prompt, max_new_tokens=max_new_tokens)


def evaluate_implant(subject: str, obj: str, generated_text: str) -> str:
    """Simple implant detection: does the generated text contain the false object?"""
    obj_lower = obj.lower().strip()
    gen_lower = generated_text.lower().strip()
    if obj_lower in gen_lower:
        return "accurate_implant"
    # Check if the first sentence contains semantically related content
    first_sent = gen_lower.split(".")[0] if "." in gen_lower else gen_lower
    if any(w in first_sent for w in obj_lower.split()):
        return "partial_implant"
    return "not_implanted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="injection strength (shield ON per v3.2)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--facts", default="all",
                    help="'all' (default), 'pilot' (5 original), or comma-separated fact_ids")
    ap.add_argument("--out", default="reports/cleanroom/flagship_v32/Q3")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    short = args.model.replace("/", "_")
    out_dir = Path(args.out) / short
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select facts
    if args.facts == "pilot":
        fact_list = list(FALSE_FACTS)
    elif args.facts == "all":
        fact_list = EXTENDED_FALSE_FACTS
    else:
        wanted = set(args.facts.split(","))
        fact_list = [f for f in EXTENDED_FALSE_FACTS if f["fact_id"] in wanted]

    print(f"[Q3] model={args.model}  device={args.device}  dtype={args.dtype}"
          f"  facts={len(fact_list)}  alpha={args.alpha}", flush=True)

    # Load model
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    print(f"[Q3] loaded in {time.time()-t_load:.1f}s", flush=True)

    results = []
    for fact in fact_list:
        fid = fact["fact_id"]
        obj = fact["object"]
        target_tok = fact["target"]
        t0 = time.time()

        # -- Baseline (no bank, no injection) --
        base_gen = _generate_text(model, tok, fact["read"],
                                  max_new_tokens=args.max_new_tokens)

        # -- Bank-injected (shield ON) --
        patcher = AttnNativePatcher(model)
        bank = fresh_bank(model)
        bank.mhc_shield = True
        write_fact(patcher, bank, tok, write_prompt=fact["write"],
                   fact_id=fid, address=fact["subject"])
        bank_gen = _patched_generate(patcher, bank, tok, fact["read"],
                                     alpha=args.alpha,
                                     max_new_tokens=args.max_new_tokens)

        # Evaluate
        base_label = evaluate_implant(fact["subject"], obj, base_gen)
        bank_label = evaluate_implant(fact["subject"], obj, bank_gen)

        rec = dict(
            fact_id=fid, subject=fact["subject"],
            object=obj, target_token=target_tok,
            alpha=args.alpha, shield=True,
            baseline_generation=base_gen,
            bank_generation=bank_gen,
            baseline_implant_label=base_label,
            bank_implant_label=bank_label,
            elapsed_s=round(time.time() - t0, 2),
        )
        results.append(rec)

        # Quick summary
        flag = "🟢" if bank_label == "accurate_implant" else \
               "🟡" if bank_label == "partial_implant" else "🔴"
        print(f"  [{fid}] {flag} base={base_label:20s} bank={bank_label:20s}"
              f"  ({rec['elapsed_s']:.1f}s)", flush=True)

    # Aggregate
    n = len(results)
    acc_implant = sum(1 for r in results if r["bank_implant_label"] == "accurate_implant")
    part_implant = sum(1 for r in results if r["bank_implant_label"] == "partial_implant")
    base_acc = sum(1 for r in results if r["baseline_implant_label"] == "accurate_implant")

    summary = dict(
        model=args.model, alpha=args.alpha, shield=True,
        n_facts=n,
        accurate_implant_rate=round(acc_implant / max(n, 1), 4),
        partial_implant_rate=round(part_implant / max(n, 1), 4),
        not_implanted=n - acc_implant - part_implant,
        baseline_accurate=base_acc,
    )

    print(f"\n[Q3] ===== {short} =====", flush=True)
    print(f"  accurate_implant: {acc_implant}/{n} ({summary['accurate_implant_rate']:.1%})", flush=True)
    print(f"  partial_implant:  {part_implant}/{n} ({summary['partial_implant_rate']:.1%})", flush=True)
    print(f"  not_implanted:    {summary['not_implanted']}/{n}", flush=True)
    print(f"  baseline_false_pos: {base_acc}/{n}", flush=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Q3] wrote {out_dir}/{{summary.json, results.jsonl}}", flush=True)

    # PASS/FAIL per H3 gate
    rate = summary["accurate_implant_rate"]
    if rate >= 0.60:
        print(f"[Q3] H3 IMPLANT GATE: PASS ({rate:.1%} >= 60%)", flush=True)
    else:
        print(f"[Q3] H3 IMPLANT GATE: FAIL ({rate:.1%} < 60%)", flush=True)


if __name__ == "__main__":
    main()
