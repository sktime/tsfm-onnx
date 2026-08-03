# Tests

Two Node-based checks, both running the exported models under
**onnxruntime-web's WASM kernels** (the exact engine the browser uses).

## 1. WASM engine smoke test

Verifies both exported models under **onnxruntime-web's WASM kernels** — the
exact engine the browser demo uses — against native ONNX Runtime outputs.
This catches per-runtime kernel gaps (see [docs/LOGBOOK.md](../docs/LOGBOOK.md) §8: a graph that native
ORT runs can still fail on another engine) without needing a browser.

```bash
# 1. reference vector from native ORT (run from the repo root)
python tests/make_vector.py        # with .venv-export activated

# 2. replay under onnxruntime-web in Node
cd tests && npm install onnxruntime-web@1.22.0 && node run_wasm.js
```

Expected: max |wasm − native| around 1e-5 for both models.

Result on 2026-08-03 (node v24, onnxruntime-web 1.22.0):

```
int8: load 838 ms, run 121 ms, max |wasm - native| = 7.63e-6
fp32: load 1321 ms, run  78 ms, max |wasm - native| = 1.53e-5
```

## 2. QA: multi-column CSV upload (Daily Delhi Climate)

`qa_csv_upload.mjs` pushes a real-world multi-column CSV (fixtures/, from
Kaggle's Daily Delhi Climate set) through the demo's own parser
(`webdemo/js/data.js`) and the int8 model, then checks invariants: the
date column is excluded, all four climate columns are found by name,
forecasts are finite, quantiles never cross, and medians stay in a
plausible range of the recent context. The dataset is intentionally not in
the demo menu; it stands in for "random file a user uploads".

```bash
cd tests && npm install onnxruntime-web@1.22.0 && node qa_csv_upload.mjs
```
