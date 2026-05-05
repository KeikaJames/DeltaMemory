# Migration guide: v0.3 → v0.4

The package import remains `deltamemory`; the project name is now **Mneme**.

## Rename wording
```diff
-# DeltaMemory / RCV-HC service
+# Mneme service
```

## SCAR injector
```diff
-from deltamemory import CAAInjector
+from deltamemory import CAAInjector, SCARInjector
+scar = SCARInjector(model, alpha=1.0, layers=[16], k=2)
+scar.calibrate(pos_texts, neg_texts, tok)
+with scar:
+    out = model(**batch)
```

## LOPI hooks: `γ_w` and orthogonal projection
```diff
+with DiagnosticRecorder(model, patcher, lopi_state=bank.lopi_state) as rec:
+    with patcher.patched(), patcher.injecting(bank, alpha=1.0):
+        model(**batch)
+df = rec.to_pandas()
```

## Gemma2/Gemma3 adapters
```diff
-from deltamemory.memory.arch_adapter import Gemma4Adapter
-adapter = Gemma4Adapter(model)
+from deltamemory import pick_adapter
+adapter = pick_adapter(model)
```

## CAA gate signature
```diff
+CAAConfig(inject_layer="mu_arch", alpha=1.0, use_lopi_gate=True, gate_k=5.0, gate_theta=0.5)
```

## Diagnostic recorder API
```diff
+from deltamemory import DiagnosticRecorder
+rec.dump_parquet("diagnostics.parquet")
```

## `_layer_locator.get_decoder_layers`
```diff
+from deltamemory.memory._layer_locator import get_decoder_layers
+layers = get_decoder_layers(model)
```

Use G2's benchmark harness for deployment numbers; do not copy placeholders.
