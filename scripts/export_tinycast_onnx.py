"""Export raws-labs/tinycast to ONNX for browser inference.

The official implementation is vendored at ./tinycast (github.com/raws-labs/
tinycast @ e2c1e2e); install it into the export environment first:

    uv pip install -p .venv-export -e ./tinycast
    uv run -p .venv-export python scripts/export_tinycast_onnx.py

The exported graph:

    input   context    float32 [batch, 2048]   raw values, FINITE (impute
                                               missing values first -- see below)
    output  quantiles  float32 [batch, 48, 9]  the model's 9 native decile
                                               levels 0.1, 0.2, ..., 0.9

Why does the graph emit ONE 48-step block instead of a full horizon?
====================================================================
The deployed predictor (tinycast.predictor.TinyCastPredictor, the one behind
the published GIFT-Eval numbers) is an autoregressive rollout: each step runs
the backbone once for 48 steps, appends the MEDIAN (quantile index 4) to the
context, and repeats until the horizon is covered; at the end the quantile
axis is sorted ascending per step to guarantee non-crossing deciles. The
model does have a single-shot arbitrary-horizon path, but it is NOT what the
published numbers use, so the graph is exactly the function the rollout calls
(`_BackboneAdapter.forward`: window min-max norm -> backbone -> clamp(-5, 5)
-> denorm) and the driver loop lives in the client. tests/
parity_tinycast_onnx_vs_official.py certifies graph + driver against
`TinyCastPredictor.predict()` end to end.

Why must the input be finite, and how is a short series padded?
===============================================================
Unlike chronos-2/t0, the released tinycast is NOT missing-aware (its
`missing_channel` config flag is off): inside the model NaN is zero-filled
after normalization, which is a training-time semantic, not imputation. The
official predictor therefore imputes BEFORE the model: linear interpolation
across interior gaps (np.interp over the valid points), then forward/backward
fill for edge gaps, then 0.0 as the last resort -- and left-pads a short
series with its FIRST value (not NaN) to 2048. A client must replicate that
driver behavior; the parity test contains the reference reimplementation.

Why is the context length FIXED (2048) while batch is dynamic?
==============================================================
Same reasoning as the other exports: the time dimension feeds shape-derived
constants everywhere -- the periodogram's n_fft, the causal conv paddings,
the positional-encoding "now" anchor, the phase-fold bin math. 2048 is also
the released seq_len, so there is exactly one right value to bake.

What about `scale_factor` (the freq/domain hint)?
=================================================
`TinyCastPredictor(freq=..., domain=...)` computes an eval-time scale factor,
but in the RELEASED config it is inert: the only consumer is the
`local_anchor` input channel, which is off. The wrapper passes None and the
parity test confirms forecasts match a predictor constructed with
freq/domain set.

Export blockers: none.
======================
The periodogram's `torch.fft.rfft` exports natively (ONNX DFT op, matches
native torch to ~1e-8), and topk / one_hot / scatter-gather all lower
cleanly through the dynamo exporter, so unlike chronos-2/t0 no monkeypatch
is needed. The `@torch.compiler.disable()` on `_detect_periods` only moves
the eager/compile boundary and does not affect `torch.export` tracing.
Caveat worth knowing: period detection is DISCRETE. The graph's DFT differs
from torch's FFT at the 1e-8 level, so a periodogram peak sitting exactly on
the Bonferroni significance threshold could in principle flip a detected
period and change the forecast discontinuously. The validation below and the
parity test exercise strongly periodic, aperiodic, gapped and trending
series without hitting such a flip.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from huggingface_hub import hf_hub_download  # noqa: E402

from tinycast.checkpoint import load_checkpoint  # noqa: E402

CONTEXT_LEN = 2048
BLOCK = 48       # output_token_len: one AR-rollout block
N_QUANTILES = 9  # deciles 0.1 .. 0.9


class TinyCastOnnxWrapper(nn.Module):
    """One AR-rollout step of the deployed predictor.

    Mirrors `_BackboneAdapter.forward` for the released config: window
    min-max normalize, run the backbone (periodogram -> dilated-conv encoder
    -> phase/future-conv decoder), clamp the normalized output to the
    training band [-5, 5], denormalize. The tilt probe and scale_factor are
    inert in the released config and omitted.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        y_norm, x_min, x_range = self.backbone.encode(
            context.unsqueeze(-1), batch_first=True,
        )
        y_norm = y_norm.clamp(-5.0, 5.0)
        return y_norm * x_range + x_min  # (B, 48, 9), raw scale


def impute(series: np.ndarray) -> np.ndarray:
    """The official predictor's NaN handling (interp -> ffill/bfill -> 0)."""
    x = series.astype(np.float32).copy()
    if np.any(np.isnan(x)):
        valid = ~np.isnan(x)
        if valid.sum() >= 2:
            idx = np.where(valid)[0]
            x = np.interp(np.arange(len(x)), idx, x[valid]).astype(np.float32)
        else:
            x = np.nan_to_num(x, nan=0.0)
    return x


def pack(series: np.ndarray) -> np.ndarray:
    """Left-pad with the first value / truncate to (1, CONTEXT_LEN)."""
    x = series.astype(np.float32)
    if x.shape[0] >= CONTEXT_LEN:
        return x[-CONTEXT_LEN:][None, :]
    pad = np.full(CONTEXT_LEN - x.shape[0], x[0], dtype=np.float32)
    return np.concatenate([pad, x])[None, :]


def validation_cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    t = np.arange(CONTEXT_LEN, dtype=np.float32)
    sine = 50 + 10 * np.sin(2 * np.pi * t[-300:] / 24) + rng.normal(0, 1, 300).astype(np.float32)
    trend = 100 + 0.5 * t[-512:] + rng.normal(0, 2, 512).astype(np.float32)
    gappy = sine.copy()
    gappy[50:60] = np.nan
    gappy[200] = np.nan
    walk = rng.normal(0, 1, 40).astype(np.float32).cumsum() + 100.0
    return [
        ("sine-300", pack(impute(sine))),
        ("trend-512", pack(impute(trend))),
        ("nan-gaps-300", pack(impute(gappy))),
        ("short-40", pack(impute(walk))),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="raws-labs/tinycast")
    parser.add_argument("--output", type=Path, default=ROOT / "onnx" / "tinycast-ctx2048-b48.onnx")
    parser.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="max allowed |onnx - eager| relative to the reference's max magnitude per case",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_id} (CPU, float32) ...")
    weights = hf_hub_download(args.model_id, "model.safetensors")
    hf_hub_download(args.model_id, "config.json")
    model, cfg = load_checkpoint(weights)
    assert int(cfg.seq_len) == CONTEXT_LEN and int(cfg.output_token_len) == BLOCK
    assert len(cfg.quantiles) == N_QUANTILES, cfg.quantiles
    print(f"quantile levels ({len(cfg.quantiles)}): {list(cfg.quantiles)}")

    wrapper = TinyCastOnnxWrapper(model.model).eval()  # TinyCastForPrediction -> backbone

    # Example input: batch of 3 with a periodic row, a trending row and a
    # padded short row, so tracing sees both detected and rejected periods.
    example = torch.cat(
        [torch.from_numpy(x) for _, x in validation_cases()[:3]]
    )

    print(f"exporting ctx={CONTEXT_LEN} block={BLOCK} -> {args.output} ...")
    batch = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        wrapper,
        (example,),
        str(args.output),
        input_names=["context"],
        output_names=["quantiles"],
        dynamic_shapes={"context": {0: batch}},
        dynamo=True,
        external_data=False,
    )
    print(f"exported: {args.output.stat().st_size / 1e6:.1f} MB")

    # ---- validation: graph vs the eager model ----------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    print("\nvalidating against the eager forward (one case per run + batched):")
    worst = 0.0
    with torch.no_grad():
        for name, ctx in validation_cases():
            ref = wrapper(torch.from_numpy(ctx)).numpy()
            (got,) = session.run(None, {"context": ctx})
            assert got.shape == ref.shape == (1, BLOCK, N_QUANTILES)
            max_abs = np.abs(got - ref).max()
            rel = max_abs / np.abs(ref).max()
            worst = max(worst, rel)
            print(f"  {name:>16}: max|onnx - eager| = {max_abs:.3e}  (relative {rel:.1e})")

        # Batched rows must equal per-row runs (proves dynamic batch).
        stacked = np.concatenate([ctx for _, ctx in validation_cases()])
        (got_b,) = session.run(None, {"context": stacked})
        for row, (name, ctx) in enumerate(validation_cases()):
            ref = wrapper(torch.from_numpy(ctx)).numpy()
            max_abs = np.abs(got_b[row:row + 1] - ref).max()
            rel = max_abs / np.abs(ref).max()
            worst = max(worst, rel)
            print(f"  batched[{row}] {name:>10}: max|onnx - eager| = {max_abs:.3e}  (relative {rel:.1e})")

    print(f"\nworst relative deviation: {worst:.3e} (tolerance {args.tolerance:.0e})")
    if worst > args.tolerance:
        raise SystemExit("FAIL: export does not match the official implementation")
    print("PASS: ONNX graph matches the official model within tolerance.")
    print("Run tests/parity_tinycast_onnx_vs_official.py for the end-to-end "
          "check against TinyCastPredictor.predict().")


if __name__ == "__main__":
    main()
