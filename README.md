# t0-alpha → ONNX → browser

Converts [theforecastingcompany/t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha)
(a time-series foundation model, gated repo — you need HF access + a local
token) from PyTorch to ONNX and runs it fully client-side in a webpage.

## Start here

| Document | Read it when |
|---|---|
| [ONNX_EXPORT_GUIDE.md](ONNX_EXPORT_GUIDE.md) | you want to export **any** model yourself — the general recipe, zero ONNX knowledge assumed |
| [LOGBOOK.md](LOGBOOK.md) | you want the war story — every failure this conversion hit, diagnosis, fix, and lesson |

## Files

| File | Purpose |
|---|---|
| `export_t0_onnx.py` | PyTorch → ONNX export + strict numeric validation against `model.predict()`. Heavily commented: the wrapper design, the fixed-vs-dynamic-shape decision, and every monkeypatch are explained inline. |
| `quantize_t0_onnx.py` | int8 dynamic quantization (411 MB → 108 MB) + accuracy-drift report |
| `inspect_onnx.py` | debugging utility: look inside any `.onnx` (op histogram, node dtype/shape dumps, 3-gate `--check`) |
| `make_demo_data.py` | downloads 3 classic real datasets (airline passengers, Melbourne temperatures, sunspots) and precomputes **library reference forecasts** for the demo's ONNX-vs-PyTorch comparison |
| `webdemo/index.html` | onnxruntime-web page: real datasets or **CSV upload**, int8/fp32 model switch, holdout actuals, and a live diff against the library's forecasts |
| `webdemo/data/` | the datasets + reference forecasts (generated, committed so the demo works without Python) |
| `unvariate_t0.py` | the original PyTorch quickstart from the model card |
| `onnx/` | exported models (git-ignored; regenerate with the scripts) |

## Quickstart

```bash
# 1. Export environment (Python 3.12 + CPU torch — matches tfc-t0's support window)
uv venv --python 3.12 .venv-export
uv pip install -p .venv-export torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv-export tfc-t0 onnx onnxscript onnxruntime

# 2. Export + validate (needs HF auth: `hf auth login`)
.venv-export/bin/python export_t0_onnx.py            # -> onnx/t0-alpha-ctx512-h64.onnx

# 3. Shrink for the web
.venv-export/bin/python quantize_t0_onnx.py          # -> onnx/...-int8.onnx

# 4. Demo data: real datasets + PyTorch reference forecasts to compare against
.venv-export/bin/python make_demo_data.py            # -> webdemo/data/

# 5. Run in the browser
python -m http.server 8000                            # from the repo root
# open http://localhost:8000/webdemo/
```

## The exported graph's contract

```
input   context    float32 [batch, 512]   NaN = missing; LEFT-pad shorter series with NaN
output  quantiles  float32 [batch, 64, 5] quantile levels [0.1, 0.25, 0.5, 0.75, 0.9]
```

Batch is dynamic; context length (512) and horizon (64) are baked in — rerun
`export_t0_onnx.py --context-len ... --horizon ...` for other variants
(context length must be a multiple of 32; horizon ≤ 1024).

## Known caveats

- Both models are verified under onnxruntime-web's WASM engine via the Node
  smoke test in [tests/](tests/README.md) (matches native ORT to ~1e-5,
  ~100 ms per forecast). The page itself hasn't had a full click-through in
  a browser yet; if anything misbehaves there, suspects are the CDN script
  tag or fetch paths, not the model.
- int8 model: mean drift ≈ 3.3% of forecast spread (max ≈ 12%) vs fp32.
  Ship the fp32 file where fidelity matters more than the 4× download.
- `.venv` (Python 3.14) predates this work and is outside `tfc-t0`'s
  supported window — use `.venv-export` for anything export-related.
