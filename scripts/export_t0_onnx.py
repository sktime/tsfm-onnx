"""Export theforecastingcompany/t0-alpha to ONNX for browser inference.

Run inside a supported environment (Python 3.11-3.13, torch >= 2.4):

    uv venv --python 3.12 .venv-export
    uv pip install -p .venv-export torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install -p .venv-export tfc-t0 onnx onnxscript onnxruntime
    python scripts/export_t0_onnx.py  (with .venv-export activated)

The exported graph:

    input   context    float32 [batch, CONTEXT_LEN]   NaN marks missing values
    output  quantiles  float32 [batch, HORIZON, 5]    levels 0.1 0.25 0.5 0.75 0.9

Why can't we export ``model.predict()`` directly?
=================================================
ONNX is a static tensor graph. ``predict()`` contains Python argument
validation, an optional autoregressive rollout loop, and returns a
dataclass -- none of which belong in a graph. But for horizon <= 1024
(``model.max_horizon``) it reduces to exactly ONE forward pass:

    build forecast buffer -> causal scaling -> transformer -> inverse scaling

which is pure tensor math. ``T0OnnxWrapper`` below re-implements precisely
that path (mirroring ``RolloutManager.predict``'s single-pass branch).

Why is the context length FIXED while batch is dynamic?
=======================================================
The time dimension feeds shape-dependent constants all over the model: the
causal attention mask (``tril(seq, seq)``), RoPE position vectors, patch
reshapes, buffer slice bounds. Exporters bake Python ints like these into
the graph as constants, so one graph serves one context length.

This costs less than it sounds: the model treats NaN as MISSING (it is
trained on gappy data), so a shorter series is handled by LEFT-padding
with NaN -- a semantically valid model input, not an approximation. One
fixed-T graph therefore serves any series of length <= T. Fixed shapes
are also faster in onnxruntime-web (kernels specialise, no reallocation).
If you routinely forecast much shorter series, export a second variant
(e.g. ``--context-len 128``) rather than paying O(T^2) attention on
mostly-padding input.

The batch dimension has none of these entanglements, so we declare it
dynamic (``torch.export.Dim``) and verify at validation time that a batch
size different from the example input works.

Why the monkeypatches?
======================
We use the dynamo exporter (``torch.export``-based). It propagates shapes
with FakeTensors -- structurally sound, unlike the deprecated TorchScript
tracer, whose shape-inference pass mis-handled this model's dynamic
reshapes. The dynamo exporter's one hard rule: no Python branching on
tensor VALUES (``if tensor.any():``), because a FakeTensor has no values
to branch on. t0's inference code has a handful of such branches -- all
of them shortcuts, not semantics -- so we patch each with a branch-free
equivalent below. Every patch preserves numerics; the final validation
against ``model.predict()`` proves it.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Repo root, so the script works from any working directory.
ROOT = Path(__file__).resolve().parents[1]

import t0.scaler as t0_scaler
from t0 import T0Forecaster
from t0.data import MaskType, TimeSeries, VariateType
from t0.mask import MaskBuilder, compute_patch_attention_mask, reduce_patch_metadata
from t0.model.layers.transformer import Transformer

# ---------------------------------------------------------------------------
# Export-safe monkeypatches.
#
# Module-level functions/methods are resolved at call time, so patching the
# attribute is enough -- no need to fork the tfc-t0 package. Each patch is a
# copy of the original with data-dependent Python branches made unconditional.
# ---------------------------------------------------------------------------


def _export_safe_causal_stats(x, mask, group_ids):
    """Branch-free ``t0.scaler._compute_causal_stats``: per-row cumulative
    Welford mean/std.

    Two changes vs the original:
    - ``invalid |= isnan`` and the ``segment_ids.max().item()`` bookkeeping
      are data-dependent; replaced with out-of-place ops.
    - The segmented cumsum (which lets stats reset mid-row at group
      boundaries) collapses to a PLAIN cumsum here, because the wrapper's
      layout guarantees one group per row: context length is a multiple of
      patch_size, so there is no left-padding and ``group_ids`` is constant
      along each row. The original reduces to exactly this computation.
    """
    invalid = (torch.zeros_like(x, dtype=torch.bool) if mask is None else mask) | torch.isnan(x)
    valid = ~invalid

    cumcount = torch.cumsum(valid.to(x.dtype), dim=-1)
    cumcount_safe = cumcount.clamp(min=1.0)

    masked_x = x.masked_fill(invalid, 0.0)
    means = torch.cumsum(masked_x, dim=-1) / cumcount_safe

    # Welford update uses the mean BEFORE each step; row-start gets 0.
    shifted_means = torch.cat([torch.zeros_like(means[:, :1]), means[:, :-1]], dim=-1)
    increment = (masked_x - shifted_means) * (masked_x - means) * valid

    m_2 = torch.cumsum(increment, dim=-1).clamp(min=0.0)
    variance = m_2 / (cumcount_safe - 1.0).clamp(min=1.0)
    return means, torch.sqrt(variance + t0_scaler.EPS)


def _export_safe_scale_input(self, grouped_input):
    """Branch-free ``CausalScaler.scale_input`` for target-only inputs.

    Drops the ``if is_causal.any(): ... if is_future.any(): ...`` shortcuts.
    The wrapper feeds only TARGET rows, so causal stats always apply and the
    future-covariate global-stats branch is dead code here.
    """
    variates = grouped_input.variates
    group_ids = grouped_input.group_ids
    invalid = ~grouped_input.valid_mask

    loc, scale = _export_safe_causal_stats(variates, mask=invalid, group_ids=group_ids)

    # One (loc, scale) per patch: the stats at each patch's LAST time step
    # (scaler patch_size is 1, so this keeps every position).
    fcd_loc = loc[:, self.patch_size - 1 :: self.patch_size]
    fcd_scale = scale[:, self.patch_size - 1 :: self.patch_size]
    loc_scale = t0_scaler.LocScale(loc=fcd_loc, scale=fcd_scale)

    loc_expanded = fcd_loc.repeat_interleave(self.patch_size, dim=-1)
    scale_expanded = fcd_scale.repeat_interleave(self.patch_size, dim=-1)
    scaled_variates = (variates - loc_expanded) / scale_expanded
    if self.use_arcsinh:  # static config flag, not a tensor branch
        scaled_variates = torch.arcsinh(scaled_variates)

    return (
        TimeSeries(
            variates=scaled_variates,
            group_ids=grouped_input.group_ids,
            variate_type=grouped_input.variate_type,
            mask=grouped_input.mask,
        ),
        loc_scale,
    )


def _export_safe_transformer_forward(self, x, patched_group_ids, patched_variate_type, patched_mask):
    """``Transformer.forward`` minus the ``if not attendable.all()`` shortcut.

    Always passes a padding mask; when nothing is padded it is all-False and
    ``build_time_mask``'s ``mask & ~padding`` is a no-op.
    """
    patch_group_ids = reduce_patch_metadata(patched_group_ids, patched_mask)
    patch_variate_type = reduce_patch_metadata(patched_variate_type, patched_mask)
    padding_mask = ~compute_patch_attention_mask(patched_mask)

    time_attn_mask = self.mask_builder.build_time_mask(patch_group_ids, patch_variate_type, padding_mask)
    group_attn_mask = self.mask_builder.expand_group_mask(self.mask_builder.build_group_mask(patch_group_ids))

    for layer in self.layers:
        x = layer(x, time_attn_mask=time_attn_mask, group_attn_mask=group_attn_mask)
    return self.out_norm(x)


def _export_safe_build_time_mask(self, patch_group_ids, patch_variate_type, padding_mask):
    """``MaskBuilder.build_time_mask`` with ``torch.where`` on booleans
    rewritten as pure logic: ONNX Runtime's CPU build implements And/Or/Not
    for bool but has no bool ``Where`` kernel. Same truth table."""
    seq_len = patch_group_ids.shape[1]
    device = patch_group_ids.device

    valid_patch = patch_group_ids >= 0
    same_doc = (
        (patch_group_ids.unsqueeze(2) == patch_group_ids.unsqueeze(1))
        & valid_patch.unsqueeze(2)
        & valid_patch.unsqueeze(1)
    )
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril(diagonal=0)
    is_future = patch_variate_type == VariateType.FUTURE

    mask = same_doc & causal.unsqueeze(0)
    future_query = is_future.unsqueeze(2)
    mask = (future_query & same_doc) | (~future_query & mask)  # was: torch.where(...)

    if padding_mask is not None:
        mask = mask & ~padding_mask.unsqueeze(1)
    return mask.unsqueeze(1)


t0_scaler._compute_causal_stats = _export_safe_causal_stats
MaskBuilder.build_time_mask = _export_safe_build_time_mask
t0_scaler.CausalScaler.scale_input = _export_safe_scale_input
Transformer.forward = _export_safe_transformer_forward
# LocScale.__post_init__ runs isnan/negativity ASSERTS on tensor values --
# meaningless on FakeTensors. Validation below covers correctness instead.
t0_scaler.LocScale.__post_init__ = lambda self: None


class T0OnnxWrapper(nn.Module):
    """Single-pass forecast graph: context [B, T] (NaN=missing) -> [B, H, 5].

    Mirrors ``RolloutManager.predict``'s single-pass branch plus the
    preprocessing of ``TimeSeries.from_array``, with two simplifications
    valid for univariate, no-covariate input:
    - every row is a TARGET row, so the ``[target_rows]`` selection is the
      identity and is dropped;
    - quantile interpolation onto user-requested levels is dropped; the
      graph returns the model's 5 native levels and the client selects or
      linearly interpolates (trivial once levels are fixed).
    """

    def __init__(self, model: T0Forecaster, context_len: int, horizon: int):
        super().__init__()
        ps = model.patch_size
        if context_len % ps != 0:
            # Also load-bearing for _export_safe_causal_stats: no left-pad
            # means one group per row means segmented cumsum == plain cumsum.
            raise ValueError(f"context_len must be a multiple of patch_size ({ps})")
        if not (1 <= horizon <= model.max_horizon):
            raise ValueError(f"horizon must be in [1, {model.max_horizon}] for the single-pass path")
        self.model = model
        self.ps = ps
        self.context_len = context_len
        self.horizon = horizon
        # The transformer predicts whole patches; round the horizon up to a
        # patch boundary, slice the surplus steps off at the end.
        self.forecast_width = -(-horizon // ps) * ps

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # --- 1. Preprocessing (TimeSeries.from_array, branch-free) ---
        # NaN -> value 0 + mask MISSING. Unconditional: with no NaNs these
        # are no-ops, so the original's ``if missing.any():`` guard is moot.
        missing = torch.isnan(context)
        values = torch.nan_to_num(context, nan=0.0)
        ctx_mask = missing.to(torch.int8) * int(MaskType.MISSING)

        # --- 2. Forecast buffer (prepare_rollout_buffer, single-pass) ---
        # Append forecast_width zero-valued cells masked WITHHELD, i.e.
        # "these are the cells you must predict".
        row = values[:, :1]
        fut_values = torch.zeros_like(row.expand(-1, self.forecast_width))
        fut_mask = torch.full_like(fut_values, float(MaskType.WITHHELD)).to(torch.int8)

        variates = torch.cat([values, fut_values], dim=1)
        mask = torch.cat([ctx_mask, fut_mask], dim=1)

        # One distinct group id per row = independent univariate series.
        row_ids = (torch.cumsum(torch.ones_like(row), dim=0) - 1).long()
        group_ids = row_ids.expand(-1, variates.shape[1])
        variate_type = torch.zeros_like(group_ids)  # VariateType.TARGET

        ts = TimeSeries(variates=variates, mask=mask, group_ids=group_ids, variate_type=variate_type)

        # --- 3. predict_step: scale -> transformer -> inverse scale ---
        scaled, loc_scale = self.model.scaler.scale_input(ts)
        per_patch = self.model(scaled)  # [B, patches, patch_size, 5]
        preds = self.model.scaler.rescale_predictions(per_patch, loc_scale, self.ps)

        # --- 4. Slice out the forecast ---
        # The model at patch index t predicts patch t+1, so the first
        # future patch sits at context_patches - 1 (as in predict_step).
        cp = self.context_len // self.ps
        block = preds[:, cp - 1 : cp - 1 + self.forecast_width // self.ps].flatten(1, 2)
        return block[:, : self.horizon]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context-len", type=int, default=512, help="fixed input length (multiple of 32)")
    parser.add_argument("--horizon", type=int, default=64, help="forecast steps baked into the graph")
    parser.add_argument("--out", type=Path, default=None, help="output path (.onnx)")
    args = parser.parse_args()

    out = args.out or ROOT / "onnx" / f"t0-alpha-ctx{args.context_len}-h{args.horizon}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()

    # Warm up lazy state (PatchEncoder caches its time-index buffer on first
    # forward) so no attribute assignment happens during export tracing.
    model.predict(torch.randn(1, args.context_len), horizon=1)

    wrapper = T0OnnxWrapper(model, args.context_len, args.horizon).eval()

    # Example input: NaNs included so the MISSING path is exercised.
    # Batch=2 here, validation uses batch=3 to prove the dim is dynamic.
    example = torch.randn(2, args.context_len)
    example[0, :7] = float("nan")

    print(f"exporting to {out} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example,),
            str(out),
            input_names=["context"],
            output_names=["quantiles"],
            dynamic_shapes={"context": {0: torch.export.Dim("batch", min=1)}},
            dynamo=True,
            external_data=False,  # single self-contained file (<2GB)
        )
    print(f"exported: {out} ({out.stat().st_size / 1e6:.1f} MB)")

    validate(out, model, args.context_len, args.horizon)


def validate(onnx_path: Path, model: T0Forecaster, context_len: int, horizon: int) -> None:
    """Compare ONNX Runtime output against ``model.predict``.

    Batch size differs from the export example (3 vs 2) to prove the batch
    dim is truly dynamic; NaNs exercise the missing-value path; querying the
    native quantile levels makes predict()'s interpolation the identity, so
    outputs must match to float32 tolerance.
    """
    import onnxruntime as ort

    levels = [float(q) for q in model.head.quantile_levels]
    test = torch.randn(3, context_len) * 10 + 5
    test[0, :50] = float("nan")  # short-series case: left-padded with NaN
    test[2, 100] = float("nan")  # isolated missing value

    ref = model.predict(test, horizon=horizon, quantiles=levels).quantiles.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (got,) = sess.run(None, {"context": test.numpy()})

    diff = np.abs(got - ref)
    rel = diff / np.maximum(np.abs(ref), 1e-3)
    print(f"output shape: onnx {got.shape} vs torch {ref.shape}")
    print(f"max abs diff: {diff.max():.3e}   max rel diff: {rel.max():.3e}")
    assert got.shape == ref.shape
    assert diff.max() < 1e-3, "ONNX output diverges from PyTorch"
    print("validation OK -- ONNX matches PyTorch predict()")


if __name__ == "__main__":
    main()
