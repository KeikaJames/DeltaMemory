"""Pre-warm test for gpt-oss-120b: verify model loads and patcher works."""
import os
SNAPSHOT = "/home/gabira/.cache/huggingface/hub/models--kernels-community--gpt-oss-triton-kernels/snapshots/642d2a71c6a9b32fc18acb8ec53505fb324bc394"
os.environ["LOCAL_KERNELS"] = f"kernels-community/gpt-oss-triton-kernels={SNAPSHOT}"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import time
sys.path.insert(0, '/home/gabira/projects/RCV-HC')

print("[prewarm] Starting gpt-oss-120b load...", flush=True)
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = '/home/gabira/Desktop/workspace/models/whitelist/gpt-oss-120b'

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"[prewarm] Tokenizer loaded in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    attn_implementation='eager', low_cpu_mem_usage=True,
    trust_remote_code=True,
).to('cuda')
model.eval()
t2 = time.time()
print(f"[prewarm] Model loaded in {t2-t1:.1f}s (total {t2-t0:.1f}s)", flush=True)

# Check patcher
from deltamemory.memory.attn_native_bank import AttnNativePatcher
patcher = AttnNativePatcher(model)
print(f"[prewarm] Patcher: {patcher.num_layers} layers, adapter={patcher.adapter.name}", flush=True)

# Quick sanity forward
enc = tok("Hello world", return_tensors='pt').to('cuda')
with torch.no_grad():
    out = model(**enc)
print(f"[prewarm] Forward ok, logits shape: {out.logits.shape}", flush=True)
print(f"[prewarm] TOTAL TIME: {time.time()-t0:.1f}s", flush=True)
print("[prewarm] SUCCESS", flush=True)
