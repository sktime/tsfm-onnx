# Logbook: exporting t0-alpha to ONNX

A chronological record of every problem hit while converting
`theforecastingcompany/t0-alpha` (PyTorch) to ONNX for browser inference —
what the symptom looked like, how it was diagnosed, what fixed it, and what
it teaches. Companion documents: [ONNX_EXPORT_GUIDE.md](ONNX_EXPORT_GUIDE.md)
(the generalized recipe) and [README.md](README.md) (repo map).

Date: 2026-08-03. Toolchain that finally worked: Python 3.12, torch 2.8 (CPU),
onnx 1.19, onnxscript 0.7.1, onnxruntime 1.23 — in `.venv-export/`.

---

## 0. Prologue: the misleading `TypeError` before any ONNX work

**Symptom.** The model card's own quickstart crashed:
`TypeError: T0Forecaster.__init__() missing 8 required positional arguments`.

**Diagnosis.** The traceback pointed at `huggingface_hub`'s generic
`from_pretrained`, which downloads `config.json` and passes its keys to the
constructor. Fetching the file manually returned *"Access to model ... is
restricted"* — the repo is **gated**, and the machine had no HF token.
`huggingface_hub` swallows the failed config download (logs it at INFO level
only!) and then calls `T0Forecaster()` with no arguments.

**Fix.** `hf auth login`. The account already had gate access.

**Lesson.** When a library error makes no sense, walk the traceback into the
library and find what it *silently tolerated*. The real error often happened
several calls earlier and was eaten by an over-forgiving `except`.

---

## 1. Reading the model before writing any code

Before attempting an export, the whole inference path was read
(`model.py`, `rollout.py`, `scaler.py`, `data.py`, `mask.py`, `quantile.py`,
all layers). Findings that shaped everything after:

- For `horizon <= model.max_horizon` (1024), `predict()` is **one forward
  pass** — the autoregressive rollout loop never runs. That single-pass path
  became the export boundary.
- The model treats **NaN as MISSING** natively (trained on gappy data), which
  later justified serving variable-length series with one fixed-length graph
  via NaN left-padding.
- Blockers inventory (things a static graph cannot contain):
  - `torch.searchsorted` in quantile interpolation → *route around it*:
    output the 5 native quantile levels, interpolate client-side.
  - Data-dependent Python branches: `if missing.any():`, `if is_causal.any():`,
    `if not attendable.all():`, value asserts in `LocScale.__post_init__`.
  - `int(segment_ids.max().item())` in the segmented cumsum.
  - In-place `invalid |= torch.isnan(x)`.
  - A lazily-initialized buffer in `PatchEncoder` (assigned during forward).
- Reassuringly absent: complex-number RoPE (`torch.polar`) — this
  implementation is real-valued. Attention is standard SDPA. Both export fine.

**Lesson.** An hour of reading beats a day of error-driven archaeology. Build
the blockers list *first*; every later decision references it.

---

## 2. TorchScript exporter, attempt 1: `aten::__ior_` unsupported

**Symptom.** `UnsupportedOperatorError: Exporting the operator 'aten::__ior_'
to ONNX opset version 17 is not supported.`

**Diagnosis.** `__ior_` is in-place `|=`, from `scaler.py`:
`invalid |= torch.isnan(x)`. Some in-place variants simply have no symbolic
(translation rule) registered.

**Fix.** Monkeypatched a copy of the function using out-of-place
`invalid = invalid | torch.isnan(x)`. Same numerics, exports as ONNX `Or`.
The monkeypatch pattern (patch the module attribute from the export script;
never fork the library) became the workhorse of the whole project.

**Lesson.** In-place tensor ops are a classic export blocker with a
mechanical fix: make them out-of-place.

## 3. TorchScript exporter, attempt 2: `aten::asinh` unsupported

**Symptom.** Same error class, now for `asinh` (the scaler's arcsinh
transform).

**Diagnosis.** ONNX has had an `Asinh` op **since opset 9** — the exporter
just never registered the aten→ONNX mapping. Not a capability gap, a
bookkeeping gap.

**Fix.** `torch.onnx.register_custom_op_symbolic("aten::asinh",
lambda g, x: g.op("Asinh", x), 9)` (same for `sinh`).

**Lesson.** "Operator not supported" can mean *the translation table has a
hole*, not that ONNX can't express the op. Check the ONNX op list before
rewriting any math.

## 4. TorchScript exporter, attempt 3: ORT rejects the graph (`ReduceMax`)

**Symptom.** Export *succeeded*; ONNX Runtime refused to load:
`ReduceMax ... axis must be in [-rank, rank-1]. Input rank was 1`.

**Diagnosis.** Dumped the node with the `onnx` package (this became
`inspect_onnx.py`): a `ReduceMax` with `axes=[1]` reading a **rank-1**
constant (RoPE's position vector). `axes=[1]` is what you get normalizing
`dim=-1` against rank 2 — the exporter had used a wrong rank. First sighting
of the corrupted-rank-metadata theme.

**Fix (at the time).** Replaced `t.max(-1)` in RoPE's `get_scale` with
`t[..., -1:]` — valid because the position vector is monotonically
increasing, so its max *is* its last element. Exports as a clean `Slice`.

**Lesson.** Three distinct failure gates exist: export-time (no symbolic),
load-time (invalid graph), run-time (no kernel). Each needs different
debugging. And: sometimes a data-dependent op (`max`) has a cheaper
structural equivalent (`last element of a sorted vector`).

## 5. TorchScript exporter, attempts 4–6: the wrong-axis Concat rabbit hole

**Symptom.** Next load failure: `MatMul ... Incompatible dimensions`, in the
patch encoder.

**Diagnosis.** Graph inspection showed the `[values ‖ time_index ‖ validity]`
concat — written as `torch.cat(..., dim=-1)` on rank-3 tensors — exported
with `axis=1` instead of `axis=2`. Same signature as §4: a negative dim
normalized against an assumed rank of 2. Three fixes were tried and all
failed:

1. `setType()` on the custom Asinh symbolic outputs (theory: my untyped
   custom op broke downstream rank propagation) — no change.
2. Replacing `unflatten(-1, ...)` with explicit `reshape` at both call sites
   (theory: unflatten's symbolic writes wrong rank metadata) — no change.
3. **Pinning the toolchain**: new venv with Python 3.12 + torch 2.8 (the
   model's supported window, mature exporter) — *same failure*, ruling out a
   torch 2.13 regression.

**Lesson.** When the same class of error survives three targeted fixes, stop
patching symptoms — the theory of the bug is wrong. Time to bisect.

## 6. The bisect that ended the TorchScript era

**Method.** Export progressively larger slices of the pipeline standalone:

- `PatchEncoder` alone → exports **clean** (`axis=-1` preserved, ORT loads).
- `reshape + PatchEncoder` → still clean.
- `scaler + reshape + PatchEncoder` → the exporter **itself crashed** inside
  its own pass: `_jit_pass_onnx_graph_shape_type_inference`:
  `shape '[2, 1, 32]' is invalid for input of size 1152` — it had
  mis-evaluated a dynamic `[batch, -1, 32]` reshape.

**Root cause.** The TorchScript exporter's internal ONNX shape-inference pass
mishandles dynamically-computed `Reshape` target shapes in this model —
sometimes crashing (bisect), sometimes silently recording wrong ranks that
later mis-normalize every negative `dim` downstream (§4, §5). Present in
both torch 2.8 and 2.13. Unfixable from user code.

**Decision.** Abandon the TorchScript exporter. Pivot to the dynamo exporter
(`torch.onnx.export(..., dynamo=True)`), which propagates shapes with
FakeTensors — real symbolic shape math, so ranks are correct by
construction.

**Lesson.** The bisect cost ~20 minutes and settled what three days of
symptom-patching couldn't have. Minimal reproductions aren't overhead; they
are the fastest path once error #3 arrives. Also: exporters are software
with bugs, not oracles — "my code must be wrong" has a limit.

## 7. Dynamo exporter: paying its one price (no value-dependent branches)

The dynamo exporter traces with FakeTensors (shapes without values), so any
`if tensor.any():` raises. Each branchy spot from the §1 inventory got a
branch-free monkeypatched copy — all numerics-preserving:

| Original | Patch |
|---|---|
| `if missing.any(): mask.masked_fill(...)` | apply unconditionally (no-op when no NaN) |
| `invalid \|= isnan(x)`, `.item()` bookkeeping in segmented cumsum | plain per-row cumulative Welford — valid because the wrapper guarantees one group per row (context length is a multiple of patch size ⇒ no left-pad ⇒ segmented cumsum ≡ cumsum) |
| `if is_causal.any(): ... if is_future.any(): ...` in `scale_input` | causal stats always; future branch is dead code for target-only input |
| `padding_mask = ~attendable if not attendable.all() else None` | always pass the mask (all-False mask is a no-op) |
| value asserts in `LocScale.__post_init__` | disabled; final validation covers correctness |
| `PatchEncoder`'s lazy time-index buffer (assigned during forward) | warm-up `predict()` call before export so the cache exists and the branch is never taken |

A happy side effect: the plain-cumsum rewrite deleted the scatter/gather
machinery from the graph entirely — smaller and faster than the TorchScript
attempt would have been.

**Export succeeded on the first dynamo run.** Dropped as no longer needed:
custom asinh/sinh symbolics (dynamo knows them), the unflatten→reshape
patches, `do_constant_folding` workarounds.

## 8. Final boss: ORT has no `Where` kernel for bool

**Symptom.** Valid graph, but session creation failed:
`Could not find an implementation for Where(16) node`.

**Diagnosis.** Listed every `Where` node with its dtypes: exactly one had
**BOOL** value operands — `torch.where(future_query, same_doc, mask)` in the
attention mask builder. ORT's CPU build implements `Where` for float/int
types but not bool.

**Fix.** Rewrote as pure logic: `(cond & a) | (~cond & b)` — `And/Or/Not`
all have bool kernels. Same truth table.

**Lesson.** A *valid* ONNX graph can still fail on a specific runtime,
because each runtime implements each op for specific dtypes only. Diagnose
with a dtype dump, fix by recasting or rewriting the op. (This is also why
the browser demo should be smoke-tested: onnxruntime-web's kernel coverage
differs again from native ORT.)

## 9. Success + validation

```
exported: onnx/t0-alpha-ctx512-h64.onnx (411.4 MB)
output shape: onnx (3, 64, 5) vs torch (3, 64, 5)
max abs diff: 1.717e-05   max rel diff: 8.032e-06
validation OK -- ONNX matches PyTorch predict()
```

Validation was designed to prove specific claims, not just "it runs":
- batch **3** vs the export example's batch **2** → dynamic batch is real;
- NaNs in the input (a 50-NaN left pad + an isolated hole) → missing-value
  path is correct;
- query = the 5 native quantile levels → `predict()`'s interpolation is the
  identity, so ONNX and PyTorch must agree to float32 tolerance — which also
  certifies every monkeypatch as numerics-preserving.

## 10. Quantization for the browser

`quantize_dynamic` (int8 weights, per-channel): **411 MB → 108 MB (26%)**.
Accuracy cost, measured against fp32 on trend+seasonal test series: mean
error ≈ **3.3%** of the forecast spread, worst case **11.7%**. Verdict: fine
for demos/dashboards; ship fp32 where fidelity matters. fp32 stays available.

## 11. Loose ends, honestly recorded

- ~~`webdemo/index.html` untested~~ — resolved once Node was installed:
  `tests/run_wasm.js` runs both models under onnxruntime-web 1.22.0's WASM
  kernels and matches native ORT to ~1e-5 (int8 forecast in ~120 ms). Only
  the page chrome (CDN script, fetch paths, canvas) remains browser-only.
- The main venv (`.venv`, Python 3.14 + torch 2.13) is outside `tfc-t0`'s
  declared support (`<3.14`) — it worked for `predict()` but all export work
  should use `.venv-export`.
- Minor oddity, never chased: running a script from the tool-session scratch
  directory triggered a torch circular-import; the same script ran fine from
  the repo. Filed under "cursed, non-blocking".

---

## The condensed lessons

1. **Read the model first**; write the blockers inventory before any code.
2. Prefer the **dynamo exporter**; its "no value-dependent branches" rule is
   explicit and fixable, unlike the TorchScript tracer's silent rank bugs.
3. **Monkeypatch, don't fork** — and let a strict numeric validation certify
   every patch.
4. Three failure gates — export / load / run — each with its own debugging
   move (traceback, graph dump, dtype dump).
5. After the third shape-shifting error, **bisect with minimal repros**
   instead of patching symptoms.
6. Fixed shapes + NaN padding is a *feature-preserving* simplification when
   the model natively understands missingness.
7. Validate the claims that matter: different batch size, edge inputs, tight
   tolerance vs the original `predict()`.
