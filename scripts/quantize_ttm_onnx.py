"""Quantize the exported TTM ONNX model for browser delivery.

Dynamic int8 quantization, same recipe as quantize_chronos2_onnx.py:
weights stored as int8, activations quantized on the fly inside
MatMuls, no calibration set. TTM is tiny (~4 MB fp32) so the size win
matters less than for the transformer exports, but the recipe is kept
identical across the repo's models. Only MatMul/Gemm weights are
quantized; the RevIN scaler math, LayerNorms and gated-attention
softmaxes stay float, which keeps the accuracy cost small.

One TTM-specific refinement: the final forecast-head MatMul (weight
[num_patch * decoder_d_model, horizon]) is EXCLUDED from quantization.
Its error lands directly on the output (only an Add and the inverse
RevIN rescale follow it), and measured on the parity cases the exclusion
cuts the worst error from ~9% to ~7% of forecast spread for 0.3 MB. The
head is found structurally (the MatMul nearest the graph output), not by
node name, so re-exports don't break it.

Usage:  uv run -p .venv-ttm python scripts/quantize_ttm_onnx.py \
            [--model onnx/ttm-r2-ctx512-h96.onnx]

Validates the quantized model against BOTH the fp32 ONNX model and the
official TinyTimeMixerForPrediction so you see exactly what the
quantization costs.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]


def forecast_head_nodes(model_path: Path) -> list[str]:
    """Name of the forecast-head MatMul: the one nearest the graph output.

    Walks producers backwards from the `forecast` output until the first
    MatMul/Gemm - only the head bias Add and the inverse RevIN rescale
    (Mul/Add/Reshape) sit in between, none of which fork the data path.
    """
    graph = onnx.load(str(model_path)).graph
    producer = {out: node for node in graph.node for out in node.output}
    frontier = [graph.output[0].name]
    while frontier:
        node = producer.get(frontier.pop(0))
        if node is None:
            continue
        if node.op_type in ("MatMul", "Gemm"):
            return [node.name]
        frontier.extend(node.input)
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "onnx" / "ttm-r2-ctx512-h96.onnx")
    args = parser.parse_args()

    out = args.model.with_name(args.model.stem + "-int8.onnx")
    context_len = int(str(args.model).split("ctx")[1].split("-")[0])

    head_nodes = forecast_head_nodes(args.model)
    print(f"excluding forecast head from quantization: {head_nodes}")
    assert head_nodes, "no MatMul with a horizon-sized weight found - graph layout changed?"

    print(f"quantizing {args.model} -> {out} ...")
    quantize_dynamic(
        model_input=str(args.model),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        # Per-channel scales cost nothing at inference and noticeably help
        # accuracy for large transformer weight matrices.
        per_channel=True,
        nodes_to_exclude=head_nodes,
    )
    fp32_mb = args.model.stat().st_size / 1e6
    int8_mb = out.stat().st_size / 1e6
    print(f"size: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({int8_mb / fp32_mb:.0%})")

    # --- Accuracy check: int8 vs fp32 ONNX vs the official model ----------
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
    test[0, :40] = 0.0  # exercise the zero-left-padding path (short-series contract)

    fp32 = InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    int8 = InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feeds = {"context": test}
    (ref,) = fp32.run(None, feeds)
    (got,) = int8.run(None, feeds)

    # Scale-aware metric: error relative to each series' forecast spread.
    spread = ref.max(axis=1, keepdims=True) - ref.min(axis=1, keepdims=True)
    rel = np.abs(got - ref) / np.maximum(spread, 1e-3)
    print(f"int8 vs fp32 ONNX:  max abs diff {np.abs(got - ref).max():.4f}   "
          f"max err/forecast-spread {rel.max():.2%}   mean {rel.mean():.3%}")

    from tsfm_public import TinyTimeMixerForPrediction

    model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2").eval()
    with torch.no_grad():
        ref_pt = model(
            past_values=torch.from_numpy(test).unsqueeze(-1), return_loss=False
        ).prediction_outputs.squeeze(-1).numpy()
    rel_pt = np.abs(got - ref_pt) / np.maximum(spread, 1e-3)
    print(f"int8 vs official:   max err/forecast-spread {rel_pt.max():.2%}   mean {rel_pt.mean():.3%}")


if __name__ == "__main__":
    main()
