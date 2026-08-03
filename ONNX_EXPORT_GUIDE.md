# Exporting any PyTorch model to ONNX — a field guide

Written after converting `t0-alpha` for browser inference (see
[LOGBOOK.md](LOGBOOK.md) for how each rule below was learned the hard way).
Assumes you know PyTorch but nothing about ONNX.

---

## 1. What ONNX actually is

An `.onnx` file is a **frozen computation graph plus the weights**, stored as
protobuf: a list of nodes (`MatMul`, `Softmax`, `Concat`, …) wired
input→output, with tensors typed by dtype and shape. No Python anywhere.

**ONNX Runtime (ORT)** executes such graphs on many targets: server CPU/GPU
(`onnxruntime` in Python), browsers (`onnxruntime-web` via WASM or WebGPU),
mobile. Export once, run where PyTorch can't go. That's the whole point:
your browser can't `pip install torch`, but it can fetch a graph file and
execute it.

Vocabulary you'll meet immediately:

- **opset** — version of the ONNX operator dictionary the graph targets
  (e.g. 18). Ops gain types/semantics across opsets.
- **IR version** — version of the file format itself. Runtimes support IR
  versions up to their release; a too-new IR is the classic "old runtime
  can't load new file" error.
- **kernel** — a runtime's concrete implementation of an op *for specific
  dtypes*. `Where(float)` existing does not imply `Where(bool)` exists
  (LOGBOOK §8).
- **execution provider (EP)** — ORT backend: CPU, CUDA, WASM, WebGPU…
  Kernel coverage differs per EP.
- **initializer** — a weight tensor stored in the graph.
- **dynamic axis** — a dimension declared symbolic (e.g. `batch`) instead of
  a fixed number.

The core mental shift: **exporting is recording, not compiling**. The
exporter runs your `forward()` once with an example input and records every
tensor op into the graph. Whatever isn't a tensor op — Python validation,
`if`/`for` on tensor *values*, dataclasses, logging — either disappears
(frozen to the path the example took) or breaks the export. Your job is to
present the exporter with a forward pass that is pure tensor math.

## 2. Step 0 — environment

- Use a **dedicated venv** matching the model's declared support window
  (Python and torch versions). Don't fight tooling *and* the model at once.
- CPU torch is enough (`--index-url https://download.pytorch.org/whl/cpu`)
  — export doesn't need CUDA, and CPU keeps validation apples-to-apples.
- Install: `torch`, the model package, `onnx`, `onnxscript` (dynamo
  exporter), `onnxruntime`.

## 3. Step 1 — read the inference code and write the blockers inventory

Read the model's `predict()`/`generate()`/`forward()` chain end to end
before writing anything. You are answering two questions:

**(a) What is the largest pure-tensor-math region?** Preprocessing and
postprocessing may be cheap to reimplement in the target language; loops
(autoregressive decoding, rollouts) may be avoidable for your use case
(t0-alpha: horizons ≤ 1024 never loop) or must live *outside* the graph
(export a single step, loop in JS).

**(b) What's on the blockers list?** Grep-able checklist:

| Pattern | Why it breaks | Standard fix |
|---|---|---|
| `if tensor.any()/.all()/.item():` | branching on tensor *values* — a graph has no values at build time | make unconditional (masked ops are no-ops on empty masks) or precompute the branch decision structurally |
| `.item()`, `int(tensor)`, `.tolist()` | pulls a runtime value into Python | restructure so the value is never needed, or accept it frozen as a constant |
| in-place ops (`x |= y`, indexed assignment) | some have no translation rule | out-of-place equivalents (`x = x | y`, `torch.cat`/`where`) |
| `torch.searchsorted`, exotic ops | patchy exporter/runtime support | route around: move to the client, or precompute (fixed inputs ⇒ constant result) |
| complex numbers (`torch.polar`, RoPE variants) | not representable in ONNX | use/patch a real-valued formulation |
| lazy buffers/caches assigned inside `forward` | attribute mutation during tracing | warm-up call before export so the cache already exists |
| dataclass/tuple returns, `**kwargs` | graph I/O must be plain tensors | wrapper module (next step) |
| shape math via Python ints | frozen into the graph | fine for dims you *choose* to fix; see §4 |

Nothing on the list is fatal; each has a mechanical fix. The list tells you
how much work you're signing up for.

## 4. Step 2 — choose the graph contract (I/O and shapes)

Decide, before coding: inputs, outputs, dtypes, and **which dims are fixed
vs dynamic**.

- Every Python int that flows into shape construction (`torch.ones(T, T)`
  causal masks, position vectors, reshape targets) gets **baked as a
  constant**. Dims entangled that way are cheapest to fix at export time;
  export several variants (ctx-128 / ctx-512) if needed.
- Keep genuinely tensor-shaped dims (usually **batch**) dynamic — dynamo:
  `dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}}`.
- Fixed sizes are not necessarily a functionality loss. t0-alpha handles
  variable-length series with one fixed-512 graph because the model treats
  NaN as "missing": shorter input = NaN left-pad, a *semantically valid*
  input. Look for the equivalent affordance in your model (padding +
  attention masks, usually).
- Prefer outputs that keep the graph simple: t0 exports its 5 *native*
  quantile levels rather than in-graph interpolation to arbitrary levels —
  killing the `searchsorted` blocker and making the client trivially able
  to pick/interp levels itself.

## 5. Step 3 — write the wrapper `nn.Module`

A small class whose `forward(plain tensors) -> plain tensor` reproduces the
chosen inference path exactly:

- Re-implement preprocessing **branch-free** (unconditional `nan_to_num` +
  mask arithmetic instead of `if missing.any():`).
- Build every intermediate from the *input tensor* (`zeros_like`, `expand`,
  `full_like`), never from Python-int sizes (`torch.zeros((B, ...))`), so
  dynamic dims stay symbolic.
- For library internals with blockers: **monkeypatch from the export
  script** — copy the function, apply the minimal change, assign it over the
  module attribute. Never fork the package. Every patch must be
  numerics-preserving; the validation step is what certifies that.
- Mirror the original slicing/indexing *exactly* (off-by-one-patch bugs are
  silent); cite the source lines in comments.

See `export_t0_onnx.py` for a worked example of all of the above.

## 6. Step 4 — export

Two exporters exist behind `torch.onnx.export`:

- **dynamo (`dynamo=True`) — use this.** Traces via `torch.export` with
  FakeTensors; shape/rank bookkeeping is sound, negative dims are handled,
  op coverage is current. Its one hard rule: no branching on tensor values
  (you handled that in §5).
- **TorchScript tracer (`dynamo=False`) — legacy.** Runs the code eagerly
  and records; tolerates value-dependent branches (freezing the taken path)
  but its shape-inference pass silently corrupts rank metadata on models
  with dynamic reshapes (LOGBOOK §4–6). Reach for it only if dynamo is
  impossible, and distrust every negative-dim op in its output.

```python
torch.onnx.export(
    wrapper, (example,), "model.onnx",
    input_names=["context"], output_names=["quantiles"],
    dynamic_shapes={"context": {0: torch.export.Dim("batch", min=1)}},
    dynamo=True,
    external_data=False,   # single file while < 2 GB
)
```

Make the example input exercise every path you care about (include NaNs if
NaN handling matters), and give it a batch size you will *not* use in
validation.

## 7. Step 5 — triage failures by gate

Failures come at three gates; identify the gate first, then apply its move:

1. **Export-time** (`UnsupportedOperatorError`, dynamo guard errors).
   Move: read the traceback into the library; fix with a monkeypatch
   (out-of-place op, branch-free copy) or — TorchScript only — register a
   missing symbolic if ONNX has the op anyway.
2. **Load-time** (ORT `ShapeInferenceError`/`Fail` on session creation: the
   graph itself is inconsistent). Move: inspect the named node —
   `python inspect_onnx.py model.onnx --node <name>` — look at its dtypes,
   shapes, and axis attributes; work out which source line produced it and
   rewrite that line export-safely.
3. **Run-time** (`NOT_IMPLEMENTED: Could not find an implementation for X`).
   The graph is valid but the runtime lacks a kernel *for that dtype*.
   Move: dtype dump (`--type X`), then recast or rewrite (e.g. bool
   `Where` → `(c & a) | (~c & b)`).

Escalation rule: after ~3 fixes of the *same class* of error, stop —
your theory is wrong. Bisect: export submodules standalone, grow the slice
until it breaks, and reconsider the toolchain (LOGBOOK §6).

## 8. Step 6 — validate ruthlessly

`assert onnx_output ≈ predict()` with intent:

- **Different batch size** than the export example → proves dynamic dims.
- **Edge inputs** (NaNs, short series, extreme scales) → proves the paths
  you rewrote.
- **Query settings that make the reference exact** (native quantile levels)
  → a tight float32 tolerance (~1e-4) becomes possible, certifying every
  monkeypatch as numerics-preserving.
- Assert **shapes** too, not just values.

## 9. Step 7 — quantize if the delivery needs it

"Small model" is relative: 97M params = 411 MB fp32 = every visitor's
download. Dynamic int8 (`onnxruntime.quantization.quantize_dynamic`,
`per_channel=True`) is the no-calibration first rung: weights int8 (~4×
smaller), activations stay float. Measure the drift on realistic inputs and
report it relative to something meaningful (forecast spread, not raw abs
diff). Keep the fp32 file for accuracy-critical uses. If dynamic int8
disappoints, the next rung is static quantization with calibration data.

## 10. Step 8 — run it in the browser

- **transformers.js only runs architectures registered in its library.**
  Custom model ⇒ use **onnxruntime-web** directly (the same engine
  transformers.js wraps).
- Load once (`ort.InferenceSession.create(url, {executionProviders:
  ["wasm"]})`), then `session.run({input: tensor})` per inference. Tensors
  are flat `Float32Array`s + shape; unpack row-major
  (`data[step * Q + level]`).
- The WASM EP's kernel coverage ≠ native CPU's — smoke-test in the actual
  browser before declaring victory. Check `onnxruntime-web`'s version
  supports your file's IR/opset (`inspect_onnx.py model.onnx` prints both).
- Serve over HTTP (`python -m http.server`) — `file://` fetches fail.

See `webdemo/index.html` for the full integration.

## 11. The toolbox

- `inspect_onnx.py` (this repo) — summary, op histogram, per-node
  dtype/shape/producer/consumer dumps, `--check` (checker → shape inference
  → ORT load = the three gates in order).
- [netron.app](https://netron.app) — visual graph browser; great for small
  models, sluggish at 400 MB.
- `onnx.checker` / `onnx.shape_inference` — programmatic validity checks.
- The `onnx` protobuf API itself — when in doubt, *read the graph*; every
  bug in the LOGBOOK was cracked by looking at actual nodes, not by
  guessing.
