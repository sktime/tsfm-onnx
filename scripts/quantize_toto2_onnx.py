"""Quantize the exported Toto-2.0-22m ONNX model.

NOT the tinycast/TTM recipe: `quantize_dynamic` CANNOT reach parity here no
matter what is excluded. Every step below was found by measurement (worst
per-series error as % of forecast spread vs the fp32 graph, over the export
validation suite plus long trending series and airline-144; the 10% budget
is the parity tolerance vs the official model):

    naive quantize_dynamic (per_channel):            18.6%
    op_types_to_quantize=["MatMul"]:                 18.6%  (identical - the
                                                     graph has no other
                                                     quantizable ops)
    + exclude interface projections a la tinycast:   11.9%
    + exclude the 8 most sensitive weight classes:   14.7%  (non-monotonic!)
    int8 WEIGHT-ONLY, per-output-channel:            15.3%
    int8 weight-only, block 32 x out-channel:         7.6%
    int8 weight-only, block 16 x out-channel:         7.5%
    block 16 + head output projections in fp32:       2.7%  <- shipped

Three measured lessons drive the shipped recipe:

1. Activation quantization is NOT the bottleneck: pure weight-only
   quantization (fp32 activations and matmuls, weights merely rounded to
   the per-channel int8 grid) scores 15.3% - as bad as full dynamic
   quantization. The error is WEIGHT ROUNDING itself. Toto 2 forecasts in
   asinh-space and unsquashes with sinh, so on strongly trending series
   (large |scaled| values) small logit errors are amplified by cosh; a
   per-layer sensitivity sweep shows nearly every matrix contributing
   1-4% on trend cases with no small excludable subset.
2. Granularity buys most of the quality: symmetric int8 with one scale per
   (16-row input block x output channel) halves the worst case vs plain
   per-channel. Blocks finer than 16 barely help.
3. The head output projection is the one true outlier (17% when dynamically
   quantized ALONE; keeping it fp32 takes block-16 from 7.5% to 2.7%). It
   is excluded STRUCTURALLY, TTM-style: walk backward from the graph
   output and keep every MatMul reached before crossing another MatMul -
   that finds the head's linear2 + skip_proj (~2.7 MB) regardless of the
   opaque dynamo node numbering, so re-exports keep working.

The shipped graph uses opset-21 blocked DequantizeLinear (int8 weights +
fp32 block scales -> fp32 MatMul), verified supported by onnxruntime 1.29
CPU. Before quantizing, weight initializers are folded through their
`Transpose` nodes: the u-uP `F.linear` slice exports as
MatMul(x, Transpose(W)) and the exporter's constant folding skips the
large transposes, hiding the weights from any quantizer. The folding and
the weight selection are structural (graph patterns, not node names).
The opset is bumped 18 -> 21 for the blocked DequantizeLinear; every other
op is unchanged between those opsets (widened type support only), and the
accuracy check below re-validates the graph after the bump.

Size: 110.7 MB fp32 -> ~50 MB int8. The residual fp32 bulk is the baked
RoPE/xPos cos-sin tables (5 layers x 8192 patches, of which only 67 rows
are ever gathered) - graph structure, not quantizable weights.

Usage:  uv run -p .venv-toto python scripts/quantize_toto2_onnx.py \
            [--model onnx/toto2-22m-ctx2048-h96.onnx]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime import InferenceSession

ROOT = Path(__file__).resolve().parents[1]

BLOCK = 16  # input-axis block size; 32 measured 7.6%, 16 measured 7.5% -> 2.7% with head fp32
TARGET_OPSET = 21  # first opset with blocked DequantizeLinear


def fold_weight_transposes(m: onnx.ModelProto) -> int:
    """Fold MatMul(x, Transpose(W_init)) into MatMul(x, W_init_T).

    Numerically exact (transposing a constant). Only folds a Transpose whose
    every consumer is a MatMul B-input, so nothing else can observe the
    change. Returns the number of transposes folded.
    """
    init = {i.name: i for i in m.graph.initializer}
    consumers: dict[str, list] = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    removed_ids = set()
    for n in list(m.graph.node):
        if n.op_type != "Transpose" or n.input[0] not in init:
            continue
        cons = consumers.get(n.output[0], [])
        if not cons or not all(c.op_type == "MatMul" and c.input[1] == n.output[0] for c in cons):
            continue
        w = numpy_helper.to_array(init[n.input[0]])
        perm = [list(a.ints) for a in n.attribute if a.name == "perm"]
        if w.ndim != 2 or (perm and perm[0] != [1, 0]):
            continue
        new_name = n.input[0] + "_T"
        if new_name not in init:
            t = numpy_helper.from_array(np.ascontiguousarray(w.T), new_name)
            m.graph.initializer.append(t)
            init[new_name] = t
        for c in cons:
            c.input[1] = new_name
        removed_ids.add(id(n))
    kept = [n for n in m.graph.node if id(n) not in removed_ids]
    del m.graph.node[:]
    m.graph.node.extend(kept)
    return len(removed_ids)


def head_matmul_weights(m: onnx.ModelProto) -> set[str]:
    """Weight initializers of MatMuls reachable backward from the graph
    output WITHOUT crossing another MatMul: the quantile head's output
    projection (linear2 + skip_proj). Structural, name-independent."""
    init = {i.name for i in m.graph.initializer}
    producer = {o: n for n in m.graph.node for o in n.output}
    found: set[str] = set()
    stack = [o.name for o in m.graph.output]
    seen = set()
    while stack:
        name = stack.pop()
        if name in seen or name not in producer:
            continue
        seen.add(name)
        n = producer[name]
        if n.op_type == "MatMul":
            found.update(i for i in n.input if i in init)
            continue  # stop the walk at a MatMul
        stack.extend(n.input)
    return found


def blocked_int8_weights(m: onnx.ModelProto, exclude: set[str]) -> tuple[int, int]:
    """Replace every non-excluded MatMul weight initializer with int8
    storage + blocked DequantizeLinear (symmetric, one scale per BLOCK
    input rows per output channel). Returns (num_weights, num_params)."""
    init = {i.name: i for i in m.graph.initializer}
    weight_names = sorted({
        i for n in m.graph.node if n.op_type == "MatMul" for i in n.input if i in init
    } - exclude)
    dq_nodes = []
    n_params = 0
    for wn in weight_names:
        w = numpy_helper.to_array(init[wn])
        assert w.ndim == 2 and w.dtype == np.float32, (wn, w.shape, w.dtype)
        rows, cols = w.shape
        n_params += w.size
        n_blocks = -(-rows // BLOCK)  # ceil: the spec allows a ragged last block
        pad = n_blocks * BLOCK - rows
        blocks = np.pad(w, ((0, pad), (0, 0))).reshape(n_blocks, BLOCK, cols)
        scale = np.abs(blocks).max(axis=1) / 127.0  # (ceil(rows/BLOCK), cols)
        scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
        q = np.clip(
            np.round(blocks / scale[:, None, :]), -127, 127
        ).astype(np.int8).reshape(n_blocks * BLOCK, cols)[:rows]

        m.graph.initializer.remove(init[wn])
        m.graph.initializer.append(numpy_helper.from_array(q, wn + "_q8"))
        m.graph.initializer.append(numpy_helper.from_array(scale, wn + "_q8_scale"))
        dq_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [wn + "_q8", wn + "_q8_scale"],
                [wn],  # keeps the fp32 name every MatMul already consumes
                name=wn + "_dequant",
                axis=0,
                block_size=BLOCK,
            )
        )
    # prepend so the DQ outputs are defined before use (ORT is order-sensitive)
    existing = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(dq_nodes + existing)
    return len(weight_names), n_params


def prune_unused_initializers(m: onnx.ModelProto) -> int:
    """Drop initializers no node references (the pre-fold fp32 weights)."""
    used = {i for n in m.graph.node for i in n.input}
    used.update(o.name for o in m.graph.output)
    dead = [i for i in m.graph.initializer if i.name not in used]
    for i in dead:
        m.graph.initializer.remove(i)
    return len(dead)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "onnx" / "toto2-22m-ctx2048-h96.onnx")
    args = parser.parse_args()

    out = args.model.with_name(args.model.stem + "-int8.onnx")
    m = onnx.load(str(args.model))

    folded = fold_weight_transposes(m)
    head = head_matmul_weights(m)
    assert len(head) == 2, f"expected head linear2+skip_proj, found {head}"
    n_weights, n_params = blocked_int8_weights(m, exclude=head)
    pruned = prune_unused_initializers(m)
    print(f"folded {folded} weight transposes; quantized {n_weights} MatMul weights "
          f"({n_params / 1e6:.1f}M params) at block={BLOCK}; kept fp32: {sorted(head)}; "
          f"pruned {pruned} dead initializers")

    for imp in m.opset_import:
        if imp.domain == "":
            imp.version = max(imp.version, TARGET_OPSET)
    onnx.checker.check_model(m)
    onnx.save(m, str(out))

    fp32_mb = args.model.stat().st_size / 1e6
    int8_mb = out.stat().st_size / 1e6
    print(f"size: {fp32_mb:.1f} MB -> {int8_mb:.1f} MB ({int8_mb / fp32_mb:.0%})")

    # --- Accuracy check: int8 vs fp32 ONNX --------------------------------
    context_len = int(str(args.model).split("ctx")[1].split("-")[0])
    rng = np.random.default_rng(0)
    t = np.arange(context_len, dtype=np.float32)
    test = np.stack(
        [
            50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, context_len).astype(np.float32),
            200 - 0.1 * t + 30 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 5, context_len).astype(np.float32),
        ]
    ).astype(np.float32)
    ids = np.arange(2, dtype=np.int64)

    fp32 = InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    int8 = InferenceSession(str(out), providers=["CPUExecutionProvider"])
    (ref,) = fp32.run(None, {"context": test, "series_ids": ids})
    (got,) = int8.run(None, {"context": test, "series_ids": ids})

    spread = ref.max(axis=(1, 2), keepdims=True) - ref.min(axis=(1, 2), keepdims=True)
    rel = np.abs(got - ref) / np.maximum(spread, 1e-3)
    print(f"int8 vs fp32 ONNX:  max abs diff {np.abs(got - ref).max():.4f}   "
          f"max err/forecast-spread {rel.max():.2%}   mean {rel.mean():.3%}")
    print("Run tests/parity_toto2_onnx_vs_official.py for the full check "
          "against the official Toto2Model.forecast().")


if __name__ == "__main__":
    main()
