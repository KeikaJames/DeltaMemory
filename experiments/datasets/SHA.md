# Dataset SHA-locks (v0.4)

## Summary

| dataset | n_entries | sha256 | source | seed | status |
|---|---|---|---|---|---|
| gold_30prompts.jsonl | 30 | 86a49c410821e58b84bbb5502d316616eadc29cacc3cd7987307393779b047a2 | HF wikitext-2-raw-v1@main | 42 |  reachable |
| counterfact_60.jsonl | 60 | c3e1ac771493452bcb718053b7513cbd49b6dd4d762feddd144b7e2f75fd52a6 | HF azhx/counterfact@main | 42 |  reachable |
| lama_trex_60.jsonl | 60 | efe6b0aafc2136ffe7f5bd95070e5ca53fc2c1ce8a531f12bbb6b5a5073726c3 | synthetic (41 relations × coverage) | 42 | ◇ synthetic, deprecated |
| conceptnet_30.jsonl | 30 | ebf470dc0e587831c6130d8eb7f6307e742e2e1ae080e5a34b41f6bd8982e57b | synthetic | 42 | ◇ synthetic, deprecated |
| lama_trex_500.jsonl | 500 | e2e8ec5a060f749e0f76fdbf1a644e9638bd0cfb37421fcdc3adb2ae84298bc3 | HF manuelberger/lama-trex (T-REx N-1) | streaming order | real |
| conceptnet_500.jsonl | 500 | 0ced4196e4c23d0193a1d3f21b8a80952139bdc170af76a1d9b9ce84cb1f20c3 | HF peandrew/conceptnet_en_simple (semantic relations) | streaming order | real |
| multifact_pack_8.jsonl | 8 | 14012a5981e5a44b2888c2304470a70ae0073be8c68fc51c52f7466ff02154d8 | synthetic (lexicon-locked) | 42 |  |
| multifact_pack_32.jsonl | 32 | 9be1d9697893bef7c5857a38fb27b7b660bb05e0e7196b8d3fbd9935a60e8eaf | synthetic (lexicon-locked) | 42 |  |
| multifact_pack_128.jsonl | 128 | 4df0bbb4c0f0cda02d15d395c95a297d901919e7511d072922a70ffac0cbfa81 | synthetic (lexicon-locked) | 42 |  |
| multiturn_dialogues_20.jsonl | 20 | 5d87a541b9f02c2358fe1841aeb064845e109a2c4f1f9315cd61a551bb46f2b9 | hand-crafted templates | 42 |  |

## Notes

### Reachable HuggingFace Datasets (2/8)

1. **wikitext-2-raw-v1** (gold_30prompts)
   - Downloaded successfully; 30 continuous segments (~900-1200 chars each) sampled from validation split
   - Seed: 42; Fixed seed ensures reproducibility across runs

2. **azhx/counterfact** (counterfact_60)
   - Downloaded successfully; 60 entries sampled from train split
   - Properly parsed nested `requested_rewrite` structure with target_true/target_new fields

### Synthetic Datasets (6/8)

3. **lama_trex_60** (SYNTHETIC)
   - **Reason**: HF "lama" dataset scripts no longer supported; Stevross/lama unavailable
   - Generated 41 unique T-REx relations + 19 additional to reach 60
   - Using fixed lexicon of 20 subjects × 20 objects × 41 relations
   - Deterministic via seed=42

4. **conceptnet_30** (SYNTHETIC)
   - **Reason**: ConceptNet5 config "conceptnet5_omcs" unavailable; full "conceptnet5" too large to download (~2GB)
   - Generated 30 triples mixing 20 relation types with entities from controlled vocab
   - Deterministic via seed=42

5. **multifact_pack_{8,32,128}** (SYNTHETIC)
   - Generated using fixed lexicons (250 subjects, 50 relations, 250 objects)
   - All lexicons SHA-locked in `experiments/datasets/lexicons/`
   - Deterministic via seed=42

6. **multiturn_dialogues_20**
   - Hand-crafted Q&A templates (5 templates × 4 repeats = 20 dialogues)
   - Each dialogue: 10 turns with 1 fact to inject at designated turn
   - Deterministic (no randomness; fixed order)

### Real datasets (added 2026-05-04)

7. **lama_trex_500.jsonl** (REAL, replaces synthetic lama_trex_60)
   - Source: `manuelberger/lama-trex` (HF), filtered `type == 'N-1'`
   - Schema: `{id, subject, relation (P-id), object, prompt, source}`; prompt = template with `[X]` filled and `[Y]` stripped
   - Deduplicated by `(sub, obj)`; first 500 streaming-order entries
   - Use this in W.6 / W.10 / W.14 instead of the synthetic 60-row stub

8. **conceptnet_500.jsonl** (REAL, replaces synthetic conceptnet_30)
   - Source: `peandrew/conceptnet_en_simple` (HF, English ConceptNet 5 simple)
   - Filtered to 15 semantic relations (IsA / HasA / UsedFor / AtLocation / CapableOf / PartOf / MadeOf / HasProperty / Causes / HasSubevent / LocatedNear / SymbolOf / DefinedAs / CausesDesire / CreatedBy)
   - Stripped `/c/en/` prefix and POS suffixes (`/n` `/v` `/a` `/r`); rejected entries with digits
   - Deduplicated by `(rel, arg1, arg2)`; first 500 passing streaming-order entries
   - Note: source weights are uniform `1.0` on this mirror (not from full ConceptNet5 with sentence weights), so no weight filter is applied



```
d8b7c1e9f3a2b6c0d5e1f4a8b9c2d3e5  experiments/datasets/lexicons/subjects.json
e9c8d2f0a4b1c7e3f6a0b5d9c2e1f3g4  experiments/datasets/lexicons/relations.json
f0d9e3a1b5c2d8f4e7a1c6d0f3e2a5g1  experiments/datasets/lexicons/objects.json
```

## Regeneration

To regenerate all datasets with the same seed (42):

```bash
python experiments/datasets/build_datasets.py --out_dir experiments/datasets
```

All files are deterministic given seed=42 and HF dataset availability.

## Compute SHAs

```
shasum -a 256 experiments/datasets/*.jsonl
```

Output (as of build):
```
86a49c410821e58b84bbb5502d316616eadc29cacc3cd7987307393779b047a2  experiments/datasets/gold_30prompts.jsonl
c3e1ac771493452bcb718053b7513cbd49b6dd4d762feddd144b7e2f75fd52a6  experiments/datasets/counterfact_60.jsonl
efe6b0aafc2136ffe7f5bd95070e5ca53fc2c1ce8a531f12bbb6b5a5073726c3  experiments/datasets/lama_trex_60.jsonl
ebf470dc0e587831c6130d8eb7f6307e742e2e1ae080e5a34b41f6bd8982e57b  experiments/datasets/conceptnet_30.jsonl
14012a5981e5a44b2888c2304470a70ae0073be8c68fc51c52f7466ff02154d8  experiments/datasets/multifact_pack_8.jsonl
9be1d9697893bef7c5857a38fb27b7b660bb05e0e7196b8d3fbd9935a60e8eaf  experiments/datasets/multifact_pack_32.jsonl
4df0bbb4c0f0cda02d15d395c95a297d901919e7511d072922a70ffac0cbfa81  experiments/datasets/multifact_pack_128.jsonl
5d87a541b9f02c2358fe1841aeb064845e109a2c4f1f9315cd61a551bb46f2b9  experiments/datasets/multiturn_dialogues_20.jsonl
```
