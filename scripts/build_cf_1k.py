"""Build counterfact_1k.jsonl from azhx/counterfact (HF). seed=42."""
import json, os, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from datasets import load_dataset
random.seed(42)
ds = load_dataset("azhx/counterfact")
split = "train" if "train" in ds else list(ds.keys())[0]
data = ds[split]
idxs = sorted(random.sample(range(len(data)), 1000))
out = []
for i, di in enumerate(idxs):
    item = data[di]
    rw = item.get("requested_rewrite", {})
    out.append({
        "id": f"cf_{i+1:04d}",
        "subject": str(rw.get("subject", "")),
        "relation": str(rw.get("relation_id", "")),
        "target_true": str(rw.get("target_true", {}).get("str", "")),
        "target_new": str(rw.get("target_new", {}).get("str", "")),
        "prompt": str(rw.get("prompt", "")),
        "paraphrase_prompts": item.get("paraphrase_prompts", []),
    })
with open("experiments/datasets/counterfact_1k.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} → experiments/datasets/counterfact_1k.jsonl")
