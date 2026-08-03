# WASM engine smoke test

Verifies both exported models under **onnxruntime-web's WASM kernels** — the
exact engine the browser demo uses — against native ONNX Runtime outputs.
This catches per-runtime kernel gaps (see [docs/LOGBOOK.md](../docs/LOGBOOK.md) §8: a graph that native
ORT runs can still fail on another engine) without needing a browser.

```bash
# 1. reference vector from native ORT (run from the repo root)
.venv-export/bin/python tests/make_vector.py tests/vector.json

# 2. replay under onnxruntime-web in Node
cd tests && npm install onnxruntime-web@1.22.0 && node run_wasm.js
```

Expected: max |wasm − native| around 1e-5 for both models.

Result on 2026-08-03 (node v24, onnxruntime-web 1.22.0):

```
int8: load 838 ms, run 121 ms, max |wasm - native| = 7.63e-6
fp32: load 1321 ms, run  78 ms, max |wasm - native| = 1.53e-5
```
