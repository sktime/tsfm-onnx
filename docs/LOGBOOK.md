# Logbook: exporting t0-alpha to ONNX

A chronological record of every problem hit while converting
`theforecastingcompany/t0-alpha` (PyTorch) to ONNX for browser inference -
what the symptom looked like, how it was diagnosed, what fixed it, and what
it teaches. Companion documents: [ONNX_EXPORT_GUIDE.md](ONNX_EXPORT_GUIDE.md)
(the generalized recipe) and [README.md](README.md) (repo map).

Date: 2026-08-03. Toolchain that finally worked: Python 3.12, torch 2.8 (CPU),
onnx 1.19, onnxscript 0.7.1, onnxruntime 1.23 - in `.venv-export/`.

---

## 0. Prologue: the misleading `TypeError` before any ONNX work

**Symptom.** The model card's own quickstart crashed:
`TypeError: T0Forecaster.__init__() missing 8 required positional arguments`.

**Diagnosis.** The traceback pointed at `huggingface_hub`'s generic
`from_pretrained`, which downloads `config.json` and passes its keys to the
constructor. Fetching the file manually returned *"Access to model ... is
restricted"* - the repo is **gated**, and the machine had no HF token.
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
  pass** - the autoregressive rollout loop never runs. That single-pass path
  became the export boundary.
- The model treats **NaN as MISSING** natively (trained on gappy data), which
  later justified serving variable-length series with one fixed-length graph
  via NaN left-padding.
- Blockers inventory (things a static graph cannot contain):
  - `torch.searchsorted` in quantile interpolation -> *route around it*:
    output the 5 native quantile levels, interpolate client-side.
  - Data-dependent Python branches: `if missing.any():`, `if is_causal.any():`,
    `if not attendable.all():`, value asserts in `LocScale.__post_init__`.
  - `int(segment_ids.max().item())` in the segmented cumsum.
  - In-place `invalid |= torch.isnan(x)`.
  - A lazily-initialized buffer in `PatchEncoder` (assigned during forward).
- Reassuringly absent: complex-number RoPE (`torch.polar`) - this
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

**Diagnosis.** ONNX has had an `Asinh` op **since opset 9** - the exporter
just never registered the aten->ONNX mapping. Not a capability gap, a
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
`dim=-1` against rank 2 - the exporter had used a wrong rank. First sighting
of the corrupted-rank-metadata theme.

**Fix (at the time).** Replaced `t.max(-1)` in RoPE's `get_scale` with
`t[..., -1:]` - valid because the position vector is monotonically
increasing, so its max *is* its last element. Exports as a clean `Slice`.

**Lesson.** Three distinct failure gates exist: export-time (no symbolic),
load-time (invalid graph), run-time (no kernel). Each needs different
debugging. And: sometimes a data-dependent op (`max`) has a cheaper
structural equivalent (`last element of a sorted vector`).

## 5. TorchScript exporter, attempts 4-6: the wrong-axis Concat rabbit hole

**Symptom.** Next load failure: `MatMul ... Incompatible dimensions`, in the
patch encoder.

**Diagnosis.** Graph inspection showed the `[values ‖ time_index ‖ validity]`
concat - written as `torch.cat(..., dim=-1)` on rank-3 tensors - exported
with `axis=1` instead of `axis=2`. Same signature as §4: a negative dim
normalized against an assumed rank of 2. Three fixes were tried and all
failed:

1. `setType()` on the custom Asinh symbolic outputs (theory: my untyped
   custom op broke downstream rank propagation) - no change.
2. Replacing `unflatten(-1, ...)` with explicit `reshape` at both call sites
   (theory: unflatten's symbolic writes wrong rank metadata) - no change.
3. **Pinning the toolchain**: new venv with Python 3.12 + torch 2.8 (the
   model's supported window, mature exporter) - *same failure*, ruling out a
   torch 2.13 regression.

**Lesson.** When the same class of error survives three targeted fixes, stop
patching symptoms - the theory of the bug is wrong. Time to bisect.

## 6. The bisect that ended the TorchScript era

**Method.** Export progressively larger slices of the pipeline standalone:

- `PatchEncoder` alone -> exports **clean** (`axis=-1` preserved, ORT loads).
- `reshape + PatchEncoder` -> still clean.
- `scaler + reshape + PatchEncoder` -> the exporter **itself crashed** inside
  its own pass: `_jit_pass_onnx_graph_shape_type_inference`:
  `shape '[2, 1, 32]' is invalid for input of size 1152` - it had
  mis-evaluated a dynamic `[batch, -1, 32]` reshape.

**Root cause.** The TorchScript exporter's internal ONNX shape-inference pass
mishandles dynamically-computed `Reshape` target shapes in this model -
sometimes crashing (bisect), sometimes silently recording wrong ranks that
later mis-normalize every negative `dim` downstream (§4, §5). Present in
both torch 2.8 and 2.13. Unfixable from user code.

**Decision.** Abandon the TorchScript exporter. Pivot to the dynamo exporter
(`torch.onnx.export(..., dynamo=True)`), which propagates shapes with
FakeTensors - real symbolic shape math, so ranks are correct by
construction.

**Lesson.** The bisect cost ~20 minutes and settled what three days of
symptom-patching couldn't have. Minimal reproductions aren't overhead; they
are the fastest path once error #3 arrives. Also: exporters are software
with bugs, not oracles - "my code must be wrong" has a limit.

## 7. Dynamo exporter: paying its one price (no value-dependent branches)

The dynamo exporter traces with FakeTensors (shapes without values), so any
`if tensor.any():` raises. Each branchy spot from the §1 inventory got a
branch-free monkeypatched copy - all numerics-preserving:

| Original | Patch |
|---|---|
| `if missing.any(): mask.masked_fill(...)` | apply unconditionally (no-op when no NaN) |
| `invalid \|= isnan(x)`, `.item()` bookkeeping in segmented cumsum | plain per-row cumulative Welford - valid because the wrapper guarantees one group per row (context length is a multiple of patch size ⇒ no left-pad ⇒ segmented cumsum ≡ cumsum) |
| `if is_causal.any(): ... if is_future.any(): ...` in `scale_input` | causal stats always; future branch is dead code for target-only input |
| `padding_mask = ~attendable if not attendable.all() else None` | always pass the mask (all-False mask is a no-op) |
| value asserts in `LocScale.__post_init__` | disabled; final validation covers correctness |
| `PatchEncoder`'s lazy time-index buffer (assigned during forward) | warm-up `predict()` call before export so the cache exists and the branch is never taken |

A happy side effect: the plain-cumsum rewrite deleted the scatter/gather
machinery from the graph entirely - smaller and faster than the TorchScript
attempt would have been.

**Export succeeded on the first dynamo run.** Dropped as no longer needed:
custom asinh/sinh symbolics (dynamo knows them), the unflatten->reshape
patches, `do_constant_folding` workarounds.

## 8. Final boss: ORT has no `Where` kernel for bool

**Symptom.** Valid graph, but session creation failed:
`Could not find an implementation for Where(16) node`.

**Diagnosis.** Listed every `Where` node with its dtypes: exactly one had
**BOOL** value operands - `torch.where(future_query, same_doc, mask)` in the
attention mask builder. ORT's CPU build implements `Where` for float/int
types but not bool.

**Fix.** Rewrote as pure logic: `(cond & a) | (~cond & b)` - `And/Or/Not`
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
- batch **3** vs the export example's batch **2** -> dynamic batch is real;
- NaNs in the input (a 50-NaN left pad + an isolated hole) -> missing-value
  path is correct;
- query = the 5 native quantile levels -> `predict()`'s interpolation is the
  identity, so ONNX and PyTorch must agree to float32 tolerance - which also
  certifies every monkeypatch as numerics-preserving.

## 10. Quantization for the browser

`quantize_dynamic` (int8 weights, per-channel): **411 MB -> 108 MB (26%)**.
Accuracy cost, measured against fp32 on trend+seasonal test series: mean
error about  **3.3%** of the forecast spread, worst case **11.7%**. Verdict: fine
for demos/dashboards; ship fp32 where fidelity matters. fp32 stays available.

## 11. Loose ends, honestly recorded

- ~~`webdemo/index.html` untested~~ - resolved once Node was installed:
  `tests/run_wasm.js` runs both models under onnxruntime-web 1.22.0's WASM
  kernels and matches native ORT to ~1e-5 (int8 forecast in ~120 ms). Only
  the page chrome (CDN script, fetch paths, canvas) remains browser-only.
- The main venv (`.venv`, Python 3.14 + torch 2.13) is outside `tfc-t0`'s
  declared support (`<3.14`) - it worked for `predict()` but all export work
  should use `.venv-export`.
- Minor oddity, never chased: running a script from the tool-session scratch
  directory triggered a torch circular-import; the same script ran fine from
  the repo. Filed under "cursed, non-blocking".

## 12. The multivariate update (2026-08-03, later the same day)

Adding multivariate support turned out to be a ~30-line change, because the
univariate graph already ran all the machinery: t0-alpha implements
multivariate forecasting through group attention between rows sharing a
`group_ids` value, and the univariate export simply gave every row a
distinct id (making that attention an identity). The update promotes
`group_ids` from an internal constant to a **graph input** (`[rows]`
int64): shared ids = joint multivariate forecast, distinct ids =
independent series, one graph for both.

Validation proves both semantics against `predict()` (max abs diff
1.7e-05 each), plus a coupling check that joint and independent forecasts
actually differ (they do, by 1.7 units on the test input - group attention
is live). The WASM-level QA (`tests/qa_multivariate.mjs`) repeats the
invariant and coupling checks for the int8 build on real climate data.

**Lesson.** Before estimating an "add capability X" export as new work,
check whether the graph already computes X behind a constant. Promoting a
constant to an input is the cheapest feature in the ONNX toolbox.

## 13. The TinyTimeMixer export (2026-09-01)

Third model: IBM's `ibm-granite/granite-timeseries-ttm-r2` (512 context /
96 horizon), exported by `scripts/export_ttm_onnx.py` into its own venv
(`.venv-ttm`, see requirements-ttm.txt - granite-tsfm's transformers pin
conflicts with the chronos/t0 stack; also note the PyPI `granite-tsfm`
name is a 0.0.0 placeholder, install from the IBM GitHub repo).

**The easy one.** Zero monkeypatches: the whole inference path -
RevIN-style StdScaler, `unfold` patchify, MLP-mixer blocks, linear
forecast head, inverse rescale - is pure tensor math, and every Python
`if` selects structure from the config. `torch.onnx.export(dynamo=True)`
succeeded on the first attempt, the payoff of reading the model source
before writing code. Scaling lives INSIDE `TinyTimeMixerForPrediction`
(scaler in the backbone, `scaler.inverse` on the head output), so the
graph is raw-in/raw-out with no external preprocessor to replicate.

**Differences from the other two exports, all architectural:**
- Point forecaster (MSE loss, no quantile head): output is `forecast`
  `[batch, 96]`, not a quantile tensor.
- Short series are LEFT-padded with ZEROS, not NaN. That is exactly what
  the official `forward` does internally (it pads zeros and leaves
  `past_observed_mask=None`, so the scaler counts the padding as observed)
  - verified byte-for-byte identical, 0.0 diff. NaN is NOT supported: no
  NaN-aware path exists and it would poison the scaler.

**Parity** (`tests/parity_ttm_onnx_vs_official.py`, vs the official
`forward()`): fp32 worst **0.0006%** of forecast spread (airline 0.0002%,
sine 0.0006%, trend 0.0005%, 3-row batch 0.0000%); int8 worst **6.8%**
(airline 5.3%, sine 1.3%, trend 6.8%, batch 0.04%). Sizes: 4.3 MB fp32 ->
2.0 MB int8.

**One quantization tweak.** Plain `quantize_dynamic` left the trend case
at 9.1% of spread; excluding the single forecast-head MatMul (its error
lands directly on the output) bought the margin back to 6.8% for 0.3 MB.
The head is found structurally - walk producers back from the graph
output to the first MatMul - so re-exports with different node numbering
keep working (the finder already survived one renumbering, 179 -> 178).

---

## 14. The tinycast export (2026-09-01, same day)

Fourth model: `raws-labs/tinycast`, a 146K-parameter attention-free
dilated-conv forecaster (9 deciles, ctx 2048). The official implementation
is vendored at ./tinycast (github.com/raws-labs/tinycast @ e2c1e2e) and
installed editable into `.venv-export` - its deps coexist with the
chronos/t0 stack, no separate venv needed.

**The graph is one AR block, not a horizon.** The deployed GIFT-Eval
predictor is an autoregressive rollout: 48-step blocks, the MEDIAN fed back
into the context each step, a final ascending sort across the quantile axis.
`scripts/export_tinycast_onnx.py` therefore exports exactly the function the
rollout calls (`context [B,2048] -> quantiles [B,48,9]`), and the driver
recipe (pad-with-FIRST-value, np.interp NaN imputation, rollout, sort) is
reimplemented independently in `tests/parity_tinycast_onnx_vs_official.py`
so the parity claim covers the recipe a browser client must replicate, not
just the graph. fp32 parity vs `TinyCastPredictor.predict()`: worst 0.0003%
of forecast spread across horizons 48-720 (15 chained blocks).

**The periodogram exports.** The feared blocker - `torch.fft.rfft` inside
the Fisher-significance period detector - lowered natively to the ONNX DFT
op (probe matched torch to 5.6e-9), and onnxruntime-web 1.22's WASM kernels
run it. Zero monkeypatches, like TTM. One structural caveat stands: period
detection is DISCRETE (topk over a thresholded periodogram), so a peak
sitting exactly on the significance threshold could flip a period and change
the forecast discontinuously; none of the parity cases hit this.

**Quantization was the real work.** Naive `quantize_dynamic` failed parity
at 12% of spread. Measured, not guessed: onnxruntime also dynamically
quantizes Conv (ConvInteger), and the depthwise dilated convs ARE the
receptive field - with every MatMul kept float, Conv quantization alone
still cost ~10%. The shipped recipe (`scripts/quantize_tinycast_onnx.py`)
quantizes MatMul only and keeps the five interface projections (in_proj,
fc_in_proj, query_proj, phase_mix, out_proj - selected by weight SHAPE, so
re-exports keep working) in float: worst 4.1% of spread, rollout compounding
included. Int8 error grows non-monotonically with the exclusion set here
(the int8 median feeds back into a discrete period detector), so tune by
measurement only.

---

## 15. The Toto 2 export (2026-09-01, same day)

Fifth model: `Datadog/Toto-2.0-22m` (22M params, Apache-2.0), a
decoder-only patched transformer with alternating time-causal and
variate-axis attention and a deterministic 9-decile quantile head.
Exported by `scripts/export_toto2_onnx.py` into its own venv (`.venv-toto`,
see requirements-toto.txt - the `toto-models` PyPI package is real, unlike
granite-tsfm's, but its gluonts/lightning/unit-scaling stack gets a
dedicated environment; toto-models 1.0.0, torch 2.13 cpu, ORT 1.29).

**The graph is one parallel pass, not a rollout.** Toto 2 trains with
contiguous patch masking, so `Toto2Model.forecast()` with horizon <=
decode_block_size runs its decode loop exactly once, KV-cache-free: for
horizon 96 = 3 patches of 32 that path is pure tensor math. The graph is
`context [variates, 2048] + series_ids [variates] -> quantiles
[variates, 96, 9]` - `series_ids` is a runtime input exactly like
chronos-2's `group_ids` (shared id = joint multivariate via the variate
attention mask, distinct = independent). NaN marks missing: the model is
genuinely missing-aware (mask-driven scaler, missingness channel in the
patch embedding, `has_missing_values=True` attention masks - the latter
baked in; verified bit-identical to the model card's `False` on fully
observed input, and it is what makes NaN-left-padding sound). One
non-obvious semantic, documented in the export: `forecast()` force-observes
the LAST context patch, so NaN in the final 32 steps means literal 0.0.

**One export blocker, one dtype fix.** The in-place boolean-mask
assignment marking all-NaN patches (`base_gids[...][obs == 0] = -1`)
traces to index_put with a runtime-bool index; being mid-method it cannot
be monkeypatched, so the export wrapper restates the single-pass body
(submodules called, not copied) with a `torch.where` in that one spot -
eager restatement vs official: 0.0. And the transformer's int32 arange
`time_ids` index RoPE tables, which ONNX Gather rejects (INVALID_GRAPH);
passing the same arange as int64 fixed it. fp32 parity: worst **0.0003%**
of forecast spread over seven cases including NaN gaps, 4096-truncation
and a 4-variate joint task. The float64 causal scaler exports and runs
as-is in ORT.

**Quantization was a different failure mode than tinycast's.** Naive
`quantize_dynamic`: 18.6% of spread. MatMul-only: identical (there IS
nothing else). Tinycast-style interface exclusions: 11.9%; excluding the
8 most sensitive classes: 14.7% (non-monotonic again). The decisive
measurement: pure WEIGHT-ONLY rounding (fp32 activations everywhere)
still scores 15.3% - so unlike every prior model the error is weight
rounding itself, not activation quantization. Toto 2 predicts in
asinh-space and unsquashes with `sinh`, so on trending series small logit
errors are amplified exponentially; a sensitivity sweep found ~1-4% from
almost every matrix (17% from the head output projection alone). The fix
is granularity, not exclusions: `scripts/quantize_toto2_onnx.py` ships
blocked int8 weight-only quantization (opset-21 blocked DequantizeLinear,
one scale per 16-input-row x output-channel block) with the head's two
output MatMuls kept fp32, found TTM-style by walking back from the graph
output. Ladder: per-channel 15.3% -> block-32 7.6% -> block-16 7.5% ->
block-16 + head fp32 **2.7%**. Two structural pre-steps matter: the u-uP
`F.linear` exports as `MatMul(x, Transpose(W))` and the exporter never
folds the big transposes, hiding every weight from every quantizer (fold
them first, by pattern not name), and the dead fp32 originals must be
pruned or the "quantized" file GROWS.

**Parity** (`tests/parity_toto2_onnx_vs_official.py`, official `forecast()`
fed the identical NaN-padded arrays - xPos centers on the sequence
midpoint, which cancels mathematically but not bit-wise, so natural-length
vs padded differs by ~2e-5 of spread, above the fp32 bar): fp32 worst
**0.0003%**, int8 worst **2.2%** (airline 2.1%, sine 2.2%, trend 1.8%,
NaN-gaps 1.6%, long-4096 1.6%, multivariate 0.9%). Sizes: 110.7 MB fp32
-> 52.5 MB int8 (the remaining fp32 bulk is 20 MB of baked 8192-patch
RoPE tables of which 67 rows are ever read). One honest caveat: blocked
DequantizeLinear needs opset-21 kernels - fine in native ORT >= 1.20,
untested here in onnxruntime-web.

---

## The condensed lessons

1. **Read the model first**; write the blockers inventory before any code.
2. Prefer the **dynamo exporter**; its "no value-dependent branches" rule is
   explicit and fixable, unlike the TorchScript tracer's silent rank bugs.
3. **Monkeypatch, don't fork** - and let a strict numeric validation certify
   every patch.
4. Three failure gates - export / load / run - each with its own debugging
   move (traceback, graph dump, dtype dump).
5. After the third shape-shifting error, **bisect with minimal repros**
   instead of patching symptoms.
6. Fixed shapes + NaN padding is a *feature-preserving* simplification when
   the model natively understands missingness.
7. Validate the claims that matter: different batch size, edge inputs, tight
   tolerance vs the original `predict()`.
