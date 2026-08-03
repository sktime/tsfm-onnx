---
license: apache-2.0
base_model: theforecastingcompany/t0-alpha
pipeline_tag: time-series-forecasting
tags:
  - time-series
  - forecasting
  - foundation-models
  - onnx
  - onnxruntime
  - onnxruntime-web
  - int8
  - browser
---

# tsfm-onnx: t0-alpha for the browser

Browser-ready ONNX exports of [t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha),
the zero-shot time-series foundation model by
[The Forecasting Company](https://theforecastingcompany.com). These graphs run
a full probabilistic forecast in about 100 ms under
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) (WASM), with no
server and no Python.

This is an unofficial conversion, not affiliated with or endorsed by The
Forecasting Company. The conversion pipeline, a browser app using these
files, an export guide, and a debugging logbook live at
[github.com/siddharth7113/tsfm-onnx](https://github.com/siddharth7113/tsfm-onnx).

## Files

| File | Size | Graph | Precision |
|---|---|---|---|
| `t0-alpha-ctx512-h64.onnx` | 411 MB | univariate | fp32 |
| `t0-alpha-ctx512-h64-int8.onnx` | 108 MB | univariate | int8 (dynamic, per-channel) |
| `t0-alpha-ctx512-h64-mv.onnx` | 411 MB | grouped (multivariate) | fp32 |
| `t0-alpha-ctx512-h64-mv-int8.onnx` | 108 MB | grouped (multivariate) | int8 (dynamic, per-channel) |

## Graph contract

Univariate graph:

```
input   context    float32 [batch, 512]    NaN marks missing values
output  quantiles  float32 [batch, 64, 5]  levels [0.1, 0.25, 0.5, 0.75, 0.9]
```

Grouped (multivariate) graph adds one input:

```
input   group_ids  int64   [rows]
```

Rows that share a group id are forecast jointly as variates of one system
(they inform each other through the model's group attention); rows with
distinct ids are independent series. Flatten `[B, V, T]` to `[B*V, T]` and
pass ids like `[0, 0, 0, 1, 1, 1]`.

Practical notes:

- Series shorter than 512 points: LEFT-pad with NaN. The model treats NaN
  as "missing" and was trained on gappy data, so this is a valid input,
  not an approximation. Series longer than 512: pass the most recent 512.
- The context length (512) and horizon (64) are baked into the graphs.
  Other sizes require re-export (see the GitHub repo).
- Known future covariates are not exported.

## Usage

JavaScript (onnxruntime-web):

```js
const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
const ctx = new Float32Array(512).fill(NaN);
ctx.set(series.slice(-512), 512 - Math.min(series.length, 512));
const out = await session.run({ context: new ort.Tensor("float32", ctx, [1, 512]) });
// out.quantiles.data is row-major [batch, step, level]; element (s, l) is at s * 5 + l
```

Python (onnxruntime):

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("t0-alpha-ctx512-h64.onnx")
context = np.full((1, 512), np.nan, dtype=np.float32)
context[0, -len(series):] = series[-512:]
(quantiles,) = sess.run(None, {"context": context})   # (1, 64, 5)
```

## Fidelity

| Check | Result |
|---|---|
| fp32 ONNX vs library `model.predict()`, identical input | max abs diff 1.7e-05 |
| grouped graph vs multivariate `predict()`, both id modes | max abs diff 1.7e-05 |
| onnxruntime-web (WASM) vs native ONNX Runtime | at most 1.5e-05 |
| int8 vs fp32 forecast drift | about 1% mean of forecast spread on long series; up to 3% mean / 17% max on short NaN-padded series |

Every export is validated against the original
[tfc-t0](https://github.com/theforecastingcompany/tfc-t0) library at export
time; the numbers above are reproducible from the scripts in the GitHub
repo. Prefer fp32 when fidelity matters more than the download size.

## How these were made

Exported with the PyTorch dynamo exporter from `tfc-t0` 0.2.3 (torch 2.8),
wrapping the library's single-forward-pass inference path with branch-free
equivalents of its data-dependent Python. Quantization is ONNX Runtime
dynamic int8 with per-channel weights. The full worked case study
(including every failure and fix) is in the repo's
[export guide](https://github.com/siddharth7113/tsfm-onnx/blob/main/docs/ONNX_EXPORT_GUIDE.md)
and [logbook](https://github.com/siddharth7113/tsfm-onnx/blob/main/docs/LOGBOOK.md).

## License and attribution

The t0-alpha weights are released by The Forecasting Company under
Apache-2.0 (with gated access on the original repository); these files are
a converted redistribution of those weights under the same license, with
attribution. Per the tfc-t0 source headers, parts of the architecture
derive from [Toto](https://github.com/DataDog/toto) (Datadog) and
[Chronos](https://github.com/amazon-science/chronos-forecasting) (Amazon
Science), both Apache-2.0. If you use these files, please credit
The Forecasting Company and consider citing their
[model card](https://huggingface.co/theforecastingcompany/t0-alpha).
