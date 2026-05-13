#!/usr/bin/env python3
"""
Build Phase W.0 datasets: gold neutral set, counterfact, LAMA, ConceptNet, 
synthetic multi-fact packs, and multi-turn dialogues.

All sampling uses seed=42 for determinism.

Usage:
    python build_datasets.py --out_dir experiments/datasets
"""
import argparse
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets library not found. Install with: pip install datasets")
    exit(1)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_gold_30prompts(out_dir: Path) -> Dict[str, Any]:
    """Build gold_30prompts.jsonl: 30 neutral prompts from Wikitext-2 validation."""
    result = {"n_entries": 0, "sha": None, "source": None, "seed": SEED}
    out_file = out_dir / "gold_30prompts.jsonl"
    
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        revision = getattr(ds, "_revision", "main")
    except Exception as e:
        print(f"WARNING: Failed to load Wikitext-2: {e}")
        result["unreachable"] = str(e)
        with open(out_file, "w") as f:
            f.write(json.dumps({
                "id": "gold_001",
                "text": "[STUB] Wikitext-2 unreachable",
                "source": "wikitext-2-raw-v1@unreachable",
                "char_offset": 0,
                "char_len": 34
            }) + "\n")
        result["n_entries"] = 1
        result["sha"] = sha256_file(out_file)
        return result

    all_text = ""
    for item in ds:
        all_text += item["text"] + "\n"
    
    entries = []
    pyrand = random.Random(SEED)
    min_len, max_len = 900, 1200
    n_samples = 30
    
    valid_starts = []
    for i in range(0, len(all_text) - max_len):
        segment = all_text[i:i+max_len]
        if segment.count('\n') < 5:
            valid_starts.append(i)
    
    if not valid_starts:
        valid_starts = list(range(0, len(all_text) - max_len, max(1, (len(all_text) - max_len) // n_samples)))
    
    starts = pyrand.sample(valid_starts, min(n_samples, len(valid_starts)))
    starts.sort()
    
    for idx, start in enumerate(starts):
        end = min(start + pyrand.randint(min_len, max_len), len(all_text))
        text = all_text[start:end]
        entries.append({
            "id": f"gold_{idx+1:03d}",
            "text": text,
            "source": f"wikitext-2-raw-v1@{revision}",
            "char_offset": start,
            "char_len": end - start
        })
    
    with open(out_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    result["n_entries"] = len(entries)
    result["sha"] = sha256_file(out_file)
    result["source"] = f"wikitext-2-raw-v1@{revision}"
    return result


def build_counterfact_60(out_dir: Path) -> Dict[str, Any]:
    """Build counterfact_60.jsonl: 60 entries from CounterFact dataset."""
    result = {"n_entries": 0, "sha": None, "source": None, "seed": SEED}
    out_file = out_dir / "counterfact_60.jsonl"
    
    try:
        dataset = load_dataset("azhx/counterfact")
        revision = getattr(dataset, "_revision", "main")
        result["source"] = f"azhx/counterfact@{revision}"
    except Exception as e:
        print(f"WARNING: CounterFact unreachable: {e}")
        result["unreachable"] = "azhx/counterfact"
        with open(out_file, "w") as f:
            f.write(json.dumps({
                "id": "cf_001",
                "subject": "[STUB]",
                "relation": "unreachable",
                "target_true": "[STUB]",
                "target_new": "[STUB]",
                "prompt": "[STUB] CounterFact unreachable",
                "paraphrase_prompts": []
            }) + "\n")
        result["n_entries"] = 1
        result["sha"] = sha256_file(out_file)
        return result
    
    pyrand = random.Random(SEED)
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    ds = dataset[split_name]
    
    indices = list(range(len(ds)))
    sampled_indices = pyrand.sample(indices, min(60, len(indices)))
    sampled_indices.sort()
    
    entries = []
    for idx, data_idx in enumerate(sampled_indices):
        item = ds[data_idx]
        rewrite = item.get("requested_rewrite", {})
        target_true = rewrite.get("target_true", {})
        target_new = rewrite.get("target_new", {})
        
        entry = {
            "id": f"cf_{idx+1:03d}",
            "subject": str(rewrite.get("subject", "")),
            "relation": str(rewrite.get("relation_id", "")),
            "target_true": str(target_true.get("str", "")),
            "target_new": str(target_new.get("str", "")),
            "prompt": str(rewrite.get("prompt", "")),
            "paraphrase_prompts": item.get("paraphrase_prompts", [])
        }
        entries.append(entry)
    
    with open(out_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    result["n_entries"] = len(entries)
    result["sha"] = sha256_file(out_file)
    return result


def build_lama_trex_60(out_dir: Path) -> Dict[str, Any]:
    """Build lama_trex_60.jsonl: 60 synthetic T-REx facts covering 41 relations."""
    result = {"n_entries": 0, "sha": None, "source": None, "seed": SEED}
    out_file = out_dir / "lama_trex_60.jsonl"
    
    pyrand = random.Random(SEED)
    
    relations_41 = [
        "P17", "P19", "P20", "P27", "P30", "P131", "P159", "P264", "P276", "P279",
        "P361", "P569", "P570", "P571", "P625", "P740", "P937", "P580", "P582", "P585",
        "P658", "P684", "P703", "P749", "P800", "P810", "P813", "P825", "P854", "P921",
        "P930", "P1001", "P1366", "P1412", "P1416", "P1344", "P1345", "P1346", "P1347",
        "P1348", "P1349"
    ][:41]
    
    subject_pool = ["Albert Einstein", "Marie Curie", "Leonardo da Vinci", "Isaac Newton",
                    "Stephen Hawking", "Jane Goodall", "Richard Feynman", "Carl Sagan",
                    "Nikola Tesla", "Galileo Galilei", "Charles Darwin", "Louis Pasteur",
                    "Pierre Curie", "Niels Bohr", "Richard Dawkins", "Stephen Wolfram",
                    "Brian Greene", "Neil deGrasse Tyson", "Sean Carroll", "Michio Kaku"]
    object_pool = ["United States", "France", "Germany", "United Kingdom", "Japan",
                   "China", "India", "Brazil", "Russia", "Canada", "Australia",
                   "Switzerland", "Sweden", "Norway", "Denmark", "Belgium",
                   "Netherlands", "Spain", "Italy", "Poland"]
    
    entries = []
    for rel_idx, relation in enumerate(relations_41):
        subj = pyrand.choice(subject_pool)
        obj = pyrand.choice(object_pool)
        entries.append({
            "id": f"lama_trex_{rel_idx+1:03d}",
            "subject": subj,
            "relation": relation,
            "object": obj,
            "prompt": f"{subj} {relation}"
        })
    
    while len(entries) < 60:
        subj = pyrand.choice(subject_pool)
        rel = pyrand.choice(relations_41)
        obj = pyrand.choice(object_pool)
        entries.append({
            "id": f"lama_trex_{len(entries)+1:03d}",
            "subject": subj,
            "relation": rel,
            "object": obj,
            "prompt": f"{subj} {rel}"
        })
    
    entries = entries[:60]
    
    with open(out_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    result["n_entries"] = len(entries)
    result["sha"] = sha256_file(out_file)
    result["source"] = "synthetic (41 relations × coverage)"
    return result


def build_conceptnet_30(out_dir: Path) -> Dict[str, Any]:
    """Build conceptnet_30.jsonl: 30 synthetic ConceptNet triples."""
    result = {"n_entries": 0, "sha": None, "source": None, "seed": SEED}
    out_file = out_dir / "conceptnet_30.jsonl"
    
    pyrand = random.Random(SEED)
    
    relations = ["IsA", "PartOf", "MadeOf", "HasA", "HasProperty", "UsedFor", "CapableOf",
                 "Desires", "CreatedBy", "Causes", "HasSubevent", "HasFirstSubevent", "HasLastSubevent",
                 "HasPrerequisite", "HasContext", "SymbolOf", "DefinesAs", "Entails", "MannerOf",
                 "LocatedNear"]
    
    entities = ["dog", "cat", "house", "car", "tree", "flower", "water", "fire", "book",
                "person", "animal", "plant", "food", "weather", "music", "art", "science",
                "language", "computer", "game"]
    
    properties = ["furry", "fast", "big", "small", "red", "blue", "hot", "cold", "wet", "dry",
                  "happy", "sad", "strong", "weak", "bright", "dark", "clean", "dirty", "new", "old"]
    
    entries = []
    for idx in range(30):
        rel = pyrand.choice(relations)
        ent1 = pyrand.choice(entities)
        ent2 = pyrand.choice(entities)
        prop = pyrand.choice(properties)
        
        triple_type = idx % 3
        if triple_type == 0:
            subj, rel_text, obj = ent1, rel, ent2
        elif triple_type == 1:
            subj, rel_text, obj = ent1, "HasProperty", prop
        else:
            subj, rel_text, obj = prop, rel, ent1
        
        entries.append({
            "id": f"conceptnet_{idx+1:03d}",
            "subject": subj,
            "relation": rel_text,
            "object": obj,
            "source": "synthetic"
        })
    
    with open(out_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    result["n_entries"] = len(entries)
    result["sha"] = sha256_file(out_file)
    result["source"] = "synthetic"
    return result


def build_multifact_packs(out_dir: Path) -> Dict[str, Any]:
    """Build multifact_pack_{8,32,128}.jsonl with synthetic facts."""
    result = {"files": {}}
    
    lexicon_dir = out_dir / "lexicons"
    lexicon_dir.mkdir(exist_ok=True)
    
    pyrand = random.Random(SEED)
    
    subjects = [f"Subject_{i}" for i in range(250)]
    relations = [f"rel_{i}" for i in range(50)]
    objects = [f"Object_{i}" for i in range(250)]
    
    pyrand.shuffle(subjects)
    pyrand.shuffle(relations)
    pyrand.shuffle(objects)
    
    with open(lexicon_dir / "subjects.json", "w") as f:
        json.dump(subjects, f)
    with open(lexicon_dir / "relations.json", "w") as f:
        json.dump(relations, f)
    with open(lexicon_dir / "objects.json", "w") as f:
        json.dump(objects, f)
    
    for n in [8, 32, 128]:
        out_file = out_dir / f"multifact_pack_{n}.jsonl"
        entries = []
        
        for idx in range(n):
            subj = pyrand.choice(subjects)
            rel = pyrand.choice(relations)
            obj = pyrand.choice(objects)
            entries.append({
                "id": f"fact_{idx+1:05d}",
                "fact_text": f"{subj} {rel} {obj}"
            })
        
        with open(out_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        
        result["files"][f"multifact_pack_{n}"] = {
            "n_entries": n,
            "sha": sha256_file(out_file)
        }
    
    result["lexicons"] = {
        "subjects_sha": sha256_file(lexicon_dir / "subjects.json"),
        "relations_sha": sha256_file(lexicon_dir / "relations.json"),
        "objects_sha": sha256_file(lexicon_dir / "objects.json"),
    }
    
    return result


def build_multiturn_dialogues_20(out_dir: Path) -> Dict[str, Any]:
    """Build multiturn_dialogues_20.jsonl: 20 dialogues × 10 turns each."""
    result = {"n_entries": 0, "sha": None}
    out_file = out_dir / "multiturn_dialogues_20.jsonl"
    
    templates = [
        {
            "fact": "Paris has a subtropical climate.",
            "test_turn": 3,
            "turns": [
                ("user", "What can you tell me about Paris?"),
                ("assistant", "Paris is the capital of France, known for the Eiffel Tower."),
                ("user", "What about its climate?"),
                ("assistant", ""),
                ("user", "Is it similar to Mediterranean climates?"),
                ("assistant", "It has some similarities but not exactly."),
                ("user", "Tell me more about annual rainfall."),
                ("assistant", "Paris receives moderate rainfall throughout the year."),
                ("user", "Any extreme weather events?"),
                ("assistant", "Occasional snow in winter and summer thunderstorms."),
            ]
        },
        {
            "fact": "The transistor was invented in 1947.",
            "test_turn": 2,
            "turns": [
                ("user", "When was the transistor invented?"),
                ("assistant", ""),
                ("user", "Who invented it?"),
                ("assistant", "It was invented by scientists at Bell Labs."),
                ("user", "Why was it important?"),
                ("assistant", "It revolutionized electronics and computing."),
                ("user", "What replaced it?"),
                ("assistant", "Nothing, it became the foundation of modern chips."),
                ("user", "Are transistors still used today?"),
                ("assistant", "Yes, they are the core of all modern electronics."),
            ]
        },
        {
            "fact": "The Berlin Wall fell in 1989.",
            "test_turn": 1,
            "turns": [
                ("user", "Tell me about the Berlin Wall."),
                ("assistant", ""),
                ("user", "When did it fall?"),
                ("assistant", "It fell in 1989, marking the end of Cold War division."),
                ("user", "What was the cause?"),
                ("assistant", "Political pressure, civil unrest, and Soviet policy changes."),
                ("user", "How long had it been standing?"),
                ("assistant", "It stood for about 28 years since 1961."),
                ("user", "What happened to it after?"),
                ("assistant", "It was demolished; pieces are displayed as artifacts."),
            ]
        },
        {
            "fact": "Photosynthesis produces oxygen as a byproduct.",
            "test_turn": 2,
            "turns": [
                ("user", "What is photosynthesis?"),
                ("assistant", "It's the process by which plants convert sunlight to energy."),
                ("user", "What are the byproducts?"),
                ("assistant", ""),
                ("user", "Why is oxygen important?"),
                ("assistant", "It's essential for respiration in most organisms."),
                ("user", "Can other organisms photosynthesize?"),
                ("assistant", "Yes, algae and some bacteria also perform photosynthesis."),
                ("user", "What else is required besides sunlight?"),
                ("assistant", "Water and carbon dioxide are also essential."),
            ]
        },
        {
            "fact": "Mount Everest is 8,849 meters tall.",
            "test_turn": 1,
            "turns": [
                ("user", "How tall is Mount Everest?"),
                ("assistant", ""),
                ("user", "Where is it located?"),
                ("assistant", "It's in the Himalayas on the Nepal-Tibet border."),
                ("user", "How many people have summited it?"),
                ("assistant", "Thousands, but it remains extremely dangerous."),
                ("user", "What's the main challenge?"),
                ("assistant", "Thin air, extreme cold, and avalanches are major hazards."),
                ("user", "Is commercial climbing available?"),
                ("assistant", "Yes, guided expeditions exist for experienced climbers."),
            ]
        },
    ]
    
    entries = []
    for dialog_idx in range(20):
        template = templates[dialog_idx % len(templates)]
        turns_data = []
        for role, text in template["turns"]:
            turns_data.append({"role": role, "text": text})
        
        entries.append({
            "id": f"md_{dialog_idx+1:03d}",
            "turns": turns_data,
            "fact_to_inject": template["fact"],
            "test_turn_idx": template["test_turn"]
        })
    
    with open(out_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    result["n_entries"] = len(entries)
    result["sha"] = sha256_file(out_file)
    return result


def main():
    parser = argparse.ArgumentParser(description="Build Phase W.0 datasets")
    parser.add_argument("--out_dir", type=str, default="experiments/datasets",
                        help="Output directory for datasets")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building datasets in {out_dir} with seed={SEED}...")
    
    results = {}
    
    print("\n[1/6] Building gold_30prompts.jsonl...")
    results["gold_30prompts"] = build_gold_30prompts(out_dir)
    print(f"  ✓ {results['gold_30prompts']['n_entries']} entries")
    
    print("\n[2/6] Building counterfact_60.jsonl...")
    results["counterfact_60"] = build_counterfact_60(out_dir)
    print(f"  ✓ {results['counterfact_60']['n_entries']} entries")
    
    print("\n[3/6] Building lama_trex_60.jsonl...")
    results["lama_trex_60"] = build_lama_trex_60(out_dir)
    print(f"  ✓ {results['lama_trex_60']['n_entries']} entries")
    
    print("\n[4/6] Building conceptnet_30.jsonl...")
    results["conceptnet_30"] = build_conceptnet_30(out_dir)
    print(f"  ✓ {results['conceptnet_30']['n_entries']} entries")
    
    print("\n[5/6] Building multifact_pack_*.jsonl...")
    mf_results = build_multifact_packs(out_dir)
    results["multifact_packs"] = mf_results
    for fname, data in mf_results["files"].items():
        print(f"  ✓ {fname}: {data['n_entries']} entries")
    
    print("\n[6/6] Building multiturn_dialogues_20.jsonl...")
    results["multiturn_dialogues_20"] = build_multiturn_dialogues_20(out_dir)
    print(f"  ✓ {results['multiturn_dialogues_20']['n_entries']} entries")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, val in results.items():
        if isinstance(val, dict) and "n_entries" in val:
            status = "✓" if "unreachable" not in val else "⚠"
            print(f"{status} {key}: {val['n_entries']} entries")
        elif key == "multifact_packs" and isinstance(val, dict):
            for fname, fdata in val.get("files", {}).items():
                print(f"✓ {fname}: {fdata['n_entries']} entries")
    
    print(f"\nAll datasets written to {out_dir}")


if __name__ == "__main__":
    main()
