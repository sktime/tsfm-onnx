"""Export IBM's granite-timeseries-ttm-r2 (TinyTimeMixer) to ONNX.

Run inside the dedicated TTM environment (see requirements-ttm.txt):

    uv venv --python 3.12 .venv-ttm
    uv pip install -p .venv-ttm -r requirements-ttm.txt
    uv run -p .venv-ttm python scripts/export_ttm_onnx.py

The exported graph:

    input   context   float32 [batch, CONTEXT_LEN]  raw unscaled values
    output  forecast  float32 [batch, HORIZON]      the model's point forecast

Why a point forecast, not quantiles like chronos-2 / t0-alpha?
==============================================================
TTM-r2 (default revision, 512 context / 96 horizon) is trained with MSE
loss and has no quantile head (`multi_quantile_head=False` in its config),
so `prediction_outputs` IS the model's complete output: one value per
future step. There is no distribution to unpack - a genuine architectural
difference from the other two models in this repo, not an export shortcut.

Why does the graph take RAW values?
===================================
TTM's normalization is inside the model, not in a preprocessor:
`TinyTimeMixerModel` owns a `TinyTimeMixerStdScaler` (RevIN-style
per-series standardization computed from the context window) and
`TinyTimeMixerForPrediction.forward` applies `scaler.inverse` to the head
output before returning. Wrapping `forward` therefore captures the whole
raw-in / raw-out contract in the graph; nothing needs replicating outside.
(The tsfm_public `TimeSeriesPreprocessor` is for fine-tuning workflows
with exogenous columns - zero-shot forecasting calls `forward` directly,
which is what we export.)

Why is the wrapper 2-D [batch, 512] when the model wants 3-D?
=============================================================
The model's native input is [batch, seq, num_input_channels] with
channels=1 for this univariate checkpoint. The wrapper unsqueezes/squeezes
the channel dim so the graph interface matches this repo's other exports
(rows of raw values in, rows of forecasts out).

Why is the context length FIXED at 512 while batch is dynamic?
==============================================================
The time dimension feeds shape-derived constants (patchify unfold with
patch_length=64, the flatten in the forecast head), same reasoning as the
other exports. Short series must be LEFT-padded with ZEROS to 512: that is
byte-for-byte what the official `forward` does internally for short input
(it zero-pads and leaves `past_observed_mask=None`, so the scaler counts
the padded zeros as observed values - validated below). NaN is NOT
supported: TTM has no NaN-aware path on this route, and NaN in the context
poisons the std scaler. Feed zeros, exactly like the official library.

Why no monkeypatches?
=====================
TinyTimeMixer's inference path is already pure tensor math: the scaler is
masked arithmetic, patchify is a single `unfold`, the mixer blocks are
Linear/LayerNorm/GELU stacks, and every Python-level `if` on the path
selects structure from the config (decoder on, no quantile head, no
channel mixing, no freq token), which the exporter freezes correctly.
The dynamo export succeeds on the unmodified library.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from tsfm_public import TinyTimeMixerForPrediction  # noqa: E402


class TtmOnnxWrapper(nn.Module):
    """Univariate zero-shot forecasting slice of TinyTimeMixerForPrediction.

    Mirrors the model-card inference (`model(past_values)` with no mask, no
    future values): the internal StdScaler sees an all-ones observed mask,
    and `prediction_outputs` comes back already inverse-scaled.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        out = self.model(past_values=context.unsqueeze(-1), return_loss=False)
        return out.prediction_outputs.squeeze(-1)


def pack(series: torch.Tensor, context_len: int) -> torch.Tensor:
    """Left-pad with ZEROS / truncate (rows, length) or (length,) to (rows, context_len).

    Zero left-padding replicates the official forward's internal handling of
    short input exactly (see module docstring) - NOT an approximation.
    """
    x = series if series.ndim == 2 else series.unsqueeze(0)
    x = x[..., -context_len:]
    out = torch.zeros((x.shape[0], context_len))
    out[:, context_len - x.shape[1]:] = x
    return out


def validation_cases() -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    sine = torch.sin(torch.arange(512) / 10.0) * 10 + 50 + torch.randn(512, generator=g)
    trend = torch.arange(512, dtype=torch.float32) * 2 + 100 + torch.randn(512, generator=g) * 5
    short = torch.randn(144, generator=g).cumsum(0) + 100.0
    batch3 = torch.randn(3, 512, generator=g).cumsum(-1) + torch.tensor(
        [10.0, 100.0, 1000.0]
    ).unsqueeze(-1)
    return [
        ("sine-512", sine),
        ("trend-512", trend),
        ("short-144", short),
        ("batch-3x512", batch3),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="max allowed |onnx - forward()| relative to the reference's max magnitude per case",
    )
    args = parser.parse_args()

    print(f"loading {args.model_id} (CPU, float32) ...")
    model = TinyTimeMixerForPrediction.from_pretrained(args.model_id).eval()
    context_len = model.config.context_length
    horizon = model.config.prediction_length
    assert model.config.num_input_channels == 1, "wrapper assumes the univariate checkpoint"

    out_path = args.output or ROOT / "onnx" / f"ttm-r2-ctx{context_len}-h{horizon}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = TtmOnnxWrapper(model).eval()

    # Example input: batch of 3 at wildly different scales (validation uses
    # other batch sizes), plus a zero-left-padded short row so the padded
    # path the parity test relies on is part of the traced example.
    g = torch.Generator().manual_seed(1)
    example = torch.zeros(3, context_len)
    example[0] = torch.sin(torch.arange(context_len) / 7.0) * 5 + 20
    example[1, -144:] = torch.arange(144, dtype=torch.float32)
    example[2] = torch.randn(context_len, generator=g) * 1000

    print(f"exporting ctx={context_len} h={horizon} -> {out_path} ...")
    batch = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        wrapper,
        (example,),
        str(out_path),
        input_names=["context"],
        output_names=["forecast"],
        dynamic_shapes={"context": {0: batch}},
        dynamo=True,
        external_data=False,
    )
    print(f"exported: {out_path.stat().st_size / 1e6:.1f} MB")

    # ---- validation against the official model ----------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path))
    print("\nvalidating against TinyTimeMixerForPrediction.forward():")
    worst = 0.0
    for name, item in validation_cases():
        ctx = pack(item, context_len)
        with torch.no_grad():
            ref = model(past_values=ctx.unsqueeze(-1), return_loss=False).prediction_outputs.squeeze(-1)
        (got,) = session.run(None, {"context": ctx.numpy().astype(np.float32)})
        got = torch.from_numpy(got)
        assert got.shape == ref.shape, f"{name}: {tuple(got.shape)} != {tuple(ref.shape)}"
        max_abs = (got - ref).abs().max().item()
        rel = max_abs / ref.abs().max().item()
        worst = max(worst, rel)
        print(f"  {name:>12}: max|onnx - forward()| = {max_abs:.3e}  (relative {rel:.1e})")

    print(f"\nworst relative deviation: {worst:.3e} (tolerance {args.tolerance:.0e})")
    if worst > args.tolerance:
        raise SystemExit("FAIL: export does not match the official implementation")
    print("PASS: ONNX graph matches the official model within tolerance.")


if __name__ == "__main__":
    main()
