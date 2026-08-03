"""Quantize the exported t0-alpha ONNX model for browser delivery.

Dynamic int8 quantization: weights are stored as int8 (4x smaller file,
~4x less download + memory), activations stay float and are quantized
on the fly inside MatMuls at runtime. No calibration dataset needed --
the right first choice for transformer-style models. If accuracy ever
disappoints, the next rung is static quantization with calibration data.

Only MatMul/Gemm weights are quantized; everything else (attention masks,
scaler math, softmax) stays float, which is why the accuracy cost is small.

Usage (with .venv-export activated):  python scripts/quantize_t0_onnx.py \
            [--model onnx/t0-alpha-ctx512-h64.onnx]

Validates the quantized model against BOTH the fp32 ONNX model and the
original PyTorch predict() so you see exactly what the quantization costs.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "onnx" / "t0-alpha-ctx512-h64.onnx")
    args = parser.parse_args()

    out = args.model.with_name(args.model.stem + "-int8.onnx")

    print(f"quantizing {args.model} -> {out} ...")
    quantize_dynamic(
        model_input=str(args.model),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        # Per-channel scales cost nothing at inference and noticeably help
        # accuracy for large transformer weight matrices.
        per_channel=True,
    )
    fp32_mb = args.model.stat().st_size / 1e6
    int8_mb = out.stat().st_size / 1e6
    print(f"size: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({int8_mb / fp32_mb:.0%})")

    # --- Accuracy check: int8 vs fp32 ONNX vs PyTorch -------------------
    context_len = int(str(args.model).split("ctx")[1].split("-")[0])
    horizon = int(str(args.model).split("-h")[1].split(".")[0].split("-")[0])

    rng = np.random.default_rng(0)
    # Trend + seasonality + noise: more realistic (and harder) than white
    # noise for judging forecast drift.
    t = np.arange(context_len, dtype=np.float32)
    test = np.stack(
        [
            50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, context_len).astype(np.float32),
            200 - 0.1 * t + 30 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 5, context_len).astype(np.float32),
        ]
    ).astype(np.float32)
    test[0, :40] = np.nan  # exercise the missing-value path

    fp32 = InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    int8 = InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (ref,) = fp32.run(None, {"context": test})
    (got,) = int8.run(None, {"context": test})

    # Scale-aware metric: error relative to each series' forecast spread.
    spread = ref.max(axis=(1, 2), keepdims=True) - ref.min(axis=(1, 2), keepdims=True)
    rel = np.abs(got - ref) / np.maximum(spread, 1e-3)
    print(f"int8 vs fp32 ONNX:  max abs diff {np.abs(got - ref).max():.4f}   "
          f"max err/forecast-spread {rel.max():.2%}   mean {rel.mean():.3%}")

    from t0 import T0Forecaster

    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()
    levels = [float(q) for q in model.head.quantile_levels]
    ref_pt = model.predict(torch.from_numpy(test), horizon=horizon, quantiles=levels).quantiles.numpy()
    rel_pt = np.abs(got - ref_pt) / np.maximum(spread, 1e-3)
    print(f"int8 vs PyTorch:    max err/forecast-spread {rel_pt.max():.2%}   mean {rel_pt.mean():.3%}")


if __name__ == "__main__":
    main()
