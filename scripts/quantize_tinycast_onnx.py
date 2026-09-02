"""Quantize the exported tinycast ONNX model.

Not the chronos-2/t0 recipe: naive `quantize_dynamic` FAILS parity here
(12% of forecast spread vs the official predictor, tolerance 10%). Two
adjustments, both found by measuring rollout error vs the fp32 graph:

1. Quantize MatMul ONLY. By default onnxruntime also dynamically quantizes
   Conv (as ConvInteger), and the depthwise dilated convs are the model's
   receptive field -- quantizing them alone leaves ~10% error even with
   every MatMul kept float.

2. Keep the model's small interface projections float: in_proj/fc_in_proj
   (14->64), query_proj (205->64), phase_mix (256->64) and out_proj (64->9).
   They are the bottlenecks every activation flows through, cost ~0.1 MB
   combined, and excluding them roughly halves the rollout error.

They are selected BY WEIGHT SHAPE, not by node name (dynamo node names are
opaque and change across exports). Result: ~4% worst-case of forecast
spread over the parity suite (tests/parity_tinycast_onnx_vs_official.py),
compounding AR rollout included. The error is dominated by the int8 median
fed back into the context each step, which can also shift a periodogram
peak -- period detection is discrete, so int8 error grows non-monotonically
with the exclusion set; trust measurements over intuition when changing it.

The size win is modest (2.4 -> ~1.6 MB): the model is 146K parameters and
much of the file is graph structure, not weights. int8 exists here for
pipeline consistency, not download savings.

Usage:  uv run -p .venv-export python scripts/quantize_tinycast_onnx.py \
            [--model onnx/tinycast-ctx2048-b48.onnx]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxruntime import InferenceSession
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]

# Interface projections to keep float, identified by weight shape:
# in_proj / fc_in_proj, query_proj, phase_mix, out_proj.
EXCLUDE_SHAPES = {(14, 64), (205, 64), (256, 64), (64, 9)}


def interface_matmuls(model_path: Path) -> list[str]:
    m = onnx.load(str(model_path))
    init_shape = {i.name: tuple(i.dims) for i in m.graph.initializer}
    names = []
    for n in m.graph.node:
        if n.op_type in ("MatMul", "Gemm") and any(
            init_shape.get(i) in EXCLUDE_SHAPES for i in n.input
        ):
            names.append(n.name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "onnx" / "tinycast-ctx2048-b48.onnx")
    args = parser.parse_args()

    out = args.model.with_name(args.model.stem + "-int8.onnx")
    exclude = interface_matmuls(args.model)

    print(f"quantizing {args.model} -> {out} ({len(exclude)} interface MatMuls kept float) ...")
    quantize_dynamic(
        model_input=str(args.model),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=exclude,
    )
    fp32_mb = args.model.stat().st_size / 1e6
    int8_mb = out.stat().st_size / 1e6
    print(f"size: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({int8_mb / fp32_mb:.0%})")

    # --- Accuracy check: int8 vs fp32, one 48-step block ------------------
    context_len = int(str(args.model).split("ctx")[1].split("-")[0])
    rng = np.random.default_rng(0)
    t = np.arange(context_len, dtype=np.float32)
    test = np.stack(
        [
            50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, context_len).astype(np.float32),
            200 - 0.1 * t + 30 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 5, context_len).astype(np.float32),
        ]
    ).astype(np.float32)

    fp32 = InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    int8 = InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (ref,) = fp32.run(None, {"context": test})
    (got,) = int8.run(None, {"context": test})

    spread = ref.max(axis=(1, 2), keepdims=True) - ref.min(axis=(1, 2), keepdims=True)
    rel = np.abs(got - ref) / np.maximum(spread, 1e-3)
    print(f"int8 vs fp32 ONNX (single block):  max abs diff {np.abs(got - ref).max():.4f}   "
          f"max err/forecast-spread {rel.max():.2%}   mean {rel.mean():.3%}")
    print("Run tests/parity_tinycast_onnx_vs_official.py for the rollout-level "
          "check against the official predictor.")


if __name__ == "__main__":
    main()
