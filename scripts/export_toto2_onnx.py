"""Export Datadog/Toto-2.0-22m to ONNX for browser inference.

Run inside the dedicated toto environment (see requirements-toto.txt):

    uv venv --python 3.12 .venv-toto
    uv pip install -p .venv-toto -r requirements-toto.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match
    uv run -p .venv-toto python scripts/export_toto2_onnx.py

The exported graph:

    input   context     float32 [variates, CONTEXT_LEN]  NaN marks missing values
    input   series_ids  int64   [variates]               rows sharing an id are
                                                         forecast jointly (multivariate);
                                                         distinct ids = independent
    output  quantiles   float32 [variates, HORIZON, 9]   the model's 9 native quantile
                                                         levels 0.1, 0.2, ..., 0.9
                                                         (deterministic pinball-loss head)

Which slice of the official API is this?
========================================
`Toto2Model.forecast(inputs, horizon, ...)` with horizon <= decode_block_size
runs its decode loop exactly ONCE with no KV cache: Toto 2 is trained with
contiguous patch masking, so all forecast patches are produced in a single
parallel forward pass (the last context token predicts patch 1, the first
masked prediction token predicts patch 2, ...). For horizon=96 = 3 patches
of 32 that single-pass path is pure tensor math end to end - causal patched
std scaler (float64 internally, kept), asinh squash, patch embedding with a
missingness channel, 6 transformer layers (5 time-causal + 1 variate), the
quantile-knots head, then sinh unsquash + rescale and a sort across the
9 quantile knots. The wrapper calls `forecast()` itself (decode_block_size=0
=> single block) rather than re-implementing it, so validation certifies the
official code path. `scaler_fallback_min_obs` and `quantile_real_cap_k`
default to 0 (no-ops) at the `forecast()` level and stay off.

Why is the context length FIXED (default 2048) while variates are dynamic?
==========================================================================
The time axis is patched (patch_size=32) and feeds shape-derived reshapes
and masks, so it is baked; CONTEXT_LEN must be a multiple of 32 (the model
itself requires this: the scaler rearranges time into (seq, patch)). The
checkpoint was trained at context 4096 (residual_attn_ratio back-solves to
S=128 patches) with RoPE tables of 8192 patches, so 2048 is comfortably
in-distribution and matches the repo's other graphs.

Short series: left-pad with NaN to CONTEXT_LEN. A fully-NaN patch gets
group id -1, which the attention masks exclude entirely, the causal scaler
ignores masked positions, and RoPE/xPos scores depend only on position
DIFFERENCES (the xPos center term cancels in the q.k product), so left-
padding by whole patches is mathematically equivalent to the natural-length
call. It is NOT bit-equivalent: the cancelling xPos center shifts individual
q/k float roundings, measured ~2e-5 of forecast spread (validated below at
1e-3 relative). Because the alignment argument works patch-wise, a series
whose length is not a multiple of 32 sees its first partial patch grouped
differently than under the official ceil-to-32 padding; the difference is
one partial patch of scaler statistics - the parity suite pads identically
on both sides so this costs nothing in practice.

Why is `has_missing_values=True` baked in?
==========================================
The model IS missing-aware: `target_mask` drives the scaler, an explicit
missingness channel in the patch embedding, and (with has_missing_values=
True) attention masks that hide fully-unobserved patches. The wrapper
derives the mask from NaN in `context` (NaN == "value 0, mask False" in the
official API - note `_prepare_forecast_inputs` force-observes the LAST
context patch, so NaN inside the final 32 steps is read as literal 0.0).
`has_missing_values=False` merely swaps the explicit causal mask for
sdpa(is_causal=True) - verified bit-identical (0.0) for fully-observed
input - but it would let padding garbage into attention, so True is the
only choice that supports NaN-padded short series. The fast-path call from
the model card (`has_missing_values=False`) is therefore covered too.

Why is the variates dimension declared min=2?
=============================================
`_sdpa_kwargs` builds the variate-axis attention mask only when n_var > 1;
torch.export turns that Python branch into a shape guard. Baking the
mask-building branch is numerically a no-op for a single row (a 1x1
equality mask passes everything), so the graph still serves n_var == 1 -
the validation below runs a single-row case through onnxruntime to prove
it. `series_ids` works exactly like the chronos-2 graph's `group_ids`.

The one export-safety substitution Toto 2 needs
===============================================
`forecast()` marks fully-unobserved context patches with
`base_gids[..., :initial_patches][ctx_patch_obs == 0] = -1`, an in-place
boolean-mask assignment into a slice VIEW - an aten.index_put with a
data-dependent boolean index once traced, which the ONNX exporter cannot
handle. `torch.where` over the same slice writes -1 at exactly the same
positions. Because that line sits mid-method, the substitution cannot be a
targeted monkeypatch: `_forecast_export_safe` below restates forecast()'s
single-pass body line for line (submodules - scaler, patch embed,
transformer, head - are called, not copied) with only that one line
replaced, and the inline validation against the UNTOUCHED official
`forecast()` certifies the restatement to 1e-5 relative.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from einops import reduce, repeat  # noqa: E402

import toto2.model as toto2_model  # noqa: E402
from toto2 import Toto2Model  # noqa: E402

PATCH = 32  # Toto-2.0-22m config.patch_size
HORIZON_DEFAULT = 96  # 3 patches, single parallel pass
N_QUANTILES = 9  # deterministic quantile-knots head, levels 0.1 .. 0.9


# ---------------------------------------------------------------------------
# Export-safe replacement for the in-place boolean-mask assignment in
# Toto2Model.forecast's group-id preparation (see module docstring).
# Original (toto2/model.py, Toto2Model.forecast):
#     base_gids[..., :initial_patches][ctx_patch_obs == 0] = -1
# ---------------------------------------------------------------------------
def _mark_unobserved_context_patches(base_gids, ctx_patch_obs, initial_patches):
    ctx_gids = torch.where(
        ctx_patch_obs == 0,
        torch.full((1,), -1, dtype=base_gids.dtype, device=base_gids.device),
        base_gids[..., :initial_patches],
    )
    return torch.cat([ctx_gids, base_gids[..., initial_patches:]], dim=-1)


def _forecast_export_safe(self, inputs, horizon, **kwargs):
    """Single-pass slice of Toto2Model.forecast with the export-safe gid fill.

    Restates the original decode loop body for the decode_block_size=0 /
    fallback=0 / cap=0 case (asserted), where the loop runs exactly once
    with no KV cache. All heavy lifting still goes through the model's own
    submodules; only the gid fill differs (see above).
    """
    assert not kwargs.get("decode_block_size"), "export slice is single-pass"
    assert not kwargs.get("scaler_fallback_min_obs")
    assert not kwargs.get("quantile_real_cap_k")
    has_missing_values = kwargs.get("has_missing_values", True)
    patch_size = self.config.patch_size
    num_patches = math.ceil(horizon / patch_size)
    nop = self.config.num_output_patches

    initial_len = inputs["target"].shape[-1]
    full_target, full_mask, series_ids, n_var = self._prepare_forecast_inputs(inputs, num_patches)
    initial_patches = math.ceil(initial_len / patch_size)

    base_gids = repeat(
        series_ids, "... n_var -> ... n_var seq", seq=initial_patches + num_patches,
    )
    ctx_patch_obs = reduce(
        full_mask[..., :initial_len], "... (seq patch) -> ... seq", "sum", patch=patch_size,
    )
    base_gids = _mark_unobserved_context_patches(base_gids, ctx_patch_obs, initial_patches)  # patched

    _, static_loc, static_scale = self.scaler(full_target, full_mask)

    raw_ctx = (full_target[..., :initial_len] - static_loc[..., :initial_len]) / static_scale[..., :initial_len]
    scaled_context = torch.where(full_mask[..., :initial_len], raw_ctx, torch.zeros_like(raw_ctx)).asinh()
    context_x = self._embed_patches(scaled_context, full_mask[..., :initial_len], patch_size)

    pred_end = initial_len + num_patches * patch_size
    raw_pred = (full_target[..., initial_len:pred_end] - static_loc[..., initial_len:pred_end]) / static_scale[..., initial_len:pred_end]
    scaled_pred = torch.where(full_mask[..., initial_len:pred_end], raw_pred, torch.zeros_like(raw_pred)).asinh()
    pred_x = self._embed_patches(scaled_pred, full_mask[..., initial_len:pred_end], patch_size)

    # Explicit int64 time_ids (the transformer would default to an int32
    # arange of the same values): the ids index the RoPE cos/sin tables, and
    # ONNX Gather* requires int64 indices - int32 produced an INVALID_GRAPH.
    # Values are identical, so this is a dtype-only substitution.
    combined_x = torch.cat([context_x, pred_x], dim=-2)
    time_ids = torch.arange(combined_x.shape[-2], device=combined_x.device, dtype=torch.int64)
    x_out = self.transformer(
        combined_x,
        time_ids=time_ids,
        group_ids=base_gids,
        has_missing_values=has_missing_values,
    )
    pred_out = x_out[..., -(num_patches + 1): -1, :]
    block_q = self.output_head(pred_out, q=None)[..., ::nop, :]

    from einops import rearrange  # local, mirrors original

    loc = rearrange(static_loc[..., initial_len:pred_end], "... (s p) -> ... s p", p=patch_size)
    scale = rearrange(static_scale[..., initial_len:pred_end], "... (s p) -> ... s p", p=patch_size)
    block_q_real = self._clamp_nonfinite(block_q.sinh() * scale + loc)
    block_q_real = block_q_real.sort(dim=0).values
    return rearrange(block_q_real, "... seq patch -> ... (seq patch)")[..., :n_var, :horizon]


class Toto2OnnxWrapper(nn.Module):
    """NaN-in, quantiles-out slice of Toto2Model.forecast (single pass)."""

    def __init__(self, model: Toto2Model, horizon: int):
        super().__init__()
        self.model = model
        self.horizon = horizon

    def forward(self, context: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
        target_mask = ~torch.isnan(context)
        target = torch.nan_to_num(context, nan=0.0)
        quantiles = _forecast_export_safe(
            self.model,
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            self.horizon,
            has_missing_values=True,
        )
        # (9, variates, horizon) -> (variates, horizon, 9): step-major matches
        # the repo's other graphs; the 9 native deciles stay intact.
        return quantiles.permute(1, 2, 0)


def pack(series: torch.Tensor, context_len: int) -> torch.Tensor:
    """Left-pad/truncate (rows, length) or (length,) to (rows, context_len) with NaN."""
    x = series if series.ndim == 2 else series.unsqueeze(0)
    x = x[..., -context_len:]
    out = torch.full((x.shape[0], context_len), torch.nan)
    out[:, context_len - x.shape[1]:] = x
    return out


def official_forecast(model: Toto2Model, context: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
    """Reference: the UNPATCHED official forecast() on identical inputs.

    Returns (rows, horizon, 9). `decode_block_size=768` as on the model card
    (single pass for horizon 96 either way, verified diff 0.0 vs 0).
    """
    target_mask = ~torch.isnan(context)
    target = torch.nan_to_num(context, nan=0.0)
    with torch.no_grad():
        q = model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            96,
            decode_block_size=768,
            has_missing_values=True,
        )
    return q.permute(1, 2, 0)


def validation_cases() -> list[tuple[str, torch.Tensor, bool]]:
    """(name, series, joint) - joint=True shares one series id across rows."""
    g = torch.Generator().manual_seed(0)
    sine = torch.sin(torch.arange(300) / 10.0) * 10 + 50 + torch.randn(300, generator=g)
    gappy = sine.clone()
    gappy[50:60] = torch.nan
    gappy[200] = torch.nan
    trend = torch.arange(144, dtype=torch.float32) * 2 + 100 + torch.randn(144, generator=g) * 5
    full = torch.randn(2048, generator=g).cumsum(0) + 100.0
    multivariate = torch.randn(4, 256, generator=g).cumsum(-1) + torch.tensor(
        [10.0, 20.0, 30.0, 40.0]
    ).unsqueeze(-1)
    return [
        ("sine-300", sine, False),
        ("nan-gaps-300", gappy, False),
        ("trend-144", trend, False),
        ("full-2048", full, False),
        ("multivariate-4x256", multivariate, True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Datadog/Toto-2.0-22m")
    parser.add_argument("--context-len", type=int, default=2048, help="must be a multiple of 32")
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT, help="must be a multiple of 32")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="max allowed |onnx - forecast()| relative to the reference's max magnitude per case",
    )
    args = parser.parse_args()

    if args.context_len % PATCH != 0:
        parser.error(f"--context-len must be a multiple of {PATCH}")
    if args.horizon % PATCH != 0 or args.horizon <= 0:
        parser.error(f"--horizon must be a positive multiple of {PATCH} (single-pass decode)")
    out_path = args.output or ROOT / "onnx" / f"toto2-22m-ctx{args.context_len}-h{args.horizon}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_id} (CPU, float32) ...")
    model = Toto2Model.from_pretrained(args.model_id).eval()
    assert model.config.patch_size == PATCH
    assert model.config.num_output_patches == 1
    print(f"quantile levels ({len(model.output_head.knots)}): {model.output_head.knots}")

    wrapper = Toto2OnnxWrapper(model, args.horizon).eval()

    # Example input: 3 rows (validation uses other row counts), with NaN
    # left-padding and an interior gap so every masked path is exercised.
    example = torch.full((3, args.context_len), torch.nan)
    example[0, -300:] = torch.sin(torch.arange(300) / 7.0) * 5 + 20
    example[0, -120:-100] = torch.nan
    example[1, -160:] = torch.arange(160, dtype=torch.float32)
    example[2, :] = torch.randn(args.context_len)
    example_ids = torch.arange(3, dtype=torch.long)

    print(f"exporting ctx={args.context_len} h={args.horizon} -> {out_path} ...")
    variates = torch.export.Dim("variates", min=2)  # see docstring: graph still serves 1
    torch.onnx.export(
        wrapper,
        (example, example_ids),
        str(out_path),
        input_names=["context", "series_ids"],
        output_names=["quantiles"],
        dynamic_shapes={"context": {0: variates}, "series_ids": {0: variates}},
        dynamo=True,
        external_data=False,
    )
    print(f"exported: {out_path.stat().st_size / 1e6:.1f} MB")

    # ---- validation against the official forecast() ------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    print("\nvalidating against Toto2Model.forecast() (identical padded inputs):")
    worst = 0.0
    for name, item, joint in validation_cases():
        ctx = pack(item, args.context_len)
        rows = ctx.shape[0]
        ids = torch.zeros(rows, dtype=torch.long) if joint else torch.arange(rows, dtype=torch.long)
        ref = official_forecast(model, ctx, ids)
        (got,) = session.run(
            None, {"context": ctx.numpy().astype(np.float32), "series_ids": ids.numpy()}
        )
        got = torch.from_numpy(got)
        assert got.shape == ref.shape, f"{name}: {tuple(got.shape)} != {tuple(ref.shape)}"
        max_abs = (got - ref).abs().max().item()
        rel = max_abs / ref.abs().max().item()
        worst = max(worst, rel)
        print(f"  {name:>20}: max|onnx - forecast()| = {max_abs:.3e}  (relative {rel:.1e})")

    # Batched independent rows must equal per-item runs (dynamic variate dim
    # + id isolation): sine and trend together with distinct ids.
    cases = {n: s for n, s, _ in validation_cases()}
    stacked = torch.cat([pack(cases["sine-300"], args.context_len), pack(cases["trend-144"], args.context_len)])
    (got_b,) = session.run(
        None, {"context": stacked.numpy().astype(np.float32), "series_ids": np.arange(2, dtype=np.int64)}
    )
    for row, name in enumerate(["sine-300", "trend-144"]):
        ctx = pack(cases[name], args.context_len)
        ref = official_forecast(model, ctx, torch.zeros(1, dtype=torch.long))
        max_abs = (torch.from_numpy(got_b[row: row + 1]) - ref).abs().max().item()
        rel = max_abs / ref.abs().max().item()
        worst = max(worst, rel)
        print(f"  batched[{row}] {name:>10}: max|onnx - forecast()| = {max_abs:.3e}  (relative {rel:.1e})")

    print(f"\nworst relative deviation: {worst:.3e} (tolerance {args.tolerance:.0e})")

    # Documented (looser) claims certified separately:
    # 1. left-NaN-padding to CONTEXT_LEN ~ natural mult-of-32 length
    nat = pack(cases["trend-144"], 160)
    ref_nat = official_forecast(model, nat, torch.zeros(1, dtype=torch.long))
    ref_pad = official_forecast(model, pack(cases["trend-144"], args.context_len), torch.zeros(1, dtype=torch.long))
    pad_rel = (ref_nat - ref_pad).abs().max().item() / ref_nat.abs().max().item()
    print(f"padding check (official 160 vs official {args.context_len}): relative {pad_rel:.1e} (<= 1e-3)")
    # 2. a SINGLE row still works in ORT despite the min=2 export guard
    ctx1 = pack(cases["sine-300"], args.context_len)
    (got1,) = session.run(None, {"context": ctx1.numpy().astype(np.float32),
                                 "series_ids": np.zeros(1, dtype=np.int64)})
    ref1 = official_forecast(model, ctx1, torch.zeros(1, dtype=torch.long))
    one_rel = (torch.from_numpy(got1) - ref1).abs().max().item() / ref1.abs().max().item()
    print(f"single-row check (variates=1 in ORT): relative {one_rel:.1e}")
    worst_extra = max(one_rel, 0.0)

    if worst > args.tolerance or pad_rel > 1e-3 or worst_extra > args.tolerance:
        raise SystemExit("FAIL: export does not match the official implementation")
    print("PASS: ONNX graph matches Toto2Model.forecast() within tolerance.")


if __name__ == "__main__":
    main()
