"""Poke around inside any .onnx file -- the debugging companion.

An ONNX file is just a protobuf: a graph of typed nodes (ops) plus weight
tensors (initializers). When an export misbehaves, the answer is almost
always visible by READING the graph instead of guessing. This script wraps
the lookups that solved every problem in LOGBOOK.md.

Usage:
    python inspect_onnx.py model.onnx                     # summary
    python inspect_onnx.py model.onnx --ops               # op-type histogram
    python inspect_onnx.py model.onnx --node NAME_SUBSTR  # node detail: inputs,
                                                          #   dtypes, attributes,
                                                          #   producers/consumers
    python inspect_onnx.py model.onnx --type Where        # all nodes of an op type
    python inspect_onnx.py model.onnx --check             # ONNX checker + shape
                                                          #   inference + ORT load

(For a visual view of small models, https://netron.app renders .onnx files;
for a 400 MB transformer, targeted queries like these are faster.)
"""

import argparse
from collections import Counter

import onnx


def tensor_dtype(value_infos, name: str) -> str:
    v = value_infos.get(name)
    if v is None:
        return "?"
    return onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type)


def tensor_shape(value_infos, name: str):
    v = value_infos.get(name)
    if v is None:
        return "?"
    return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]


def load(path: str):
    # load_external_data=False: metadata only, no need to pull weights into RAM.
    model = onnx.load(path, load_external_data=False)
    g = model.graph
    value_infos = {v.name: v for v in [*g.value_info, *g.input, *g.output]}
    producers = {out: n for n in g.node for out in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return model, g, value_infos, producers, consumers


def summary(model, g, value_infos) -> None:
    print(f"ir_version: {model.ir_version}")
    print(f"opsets:     {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    n_params = sum(
        max(1, __import__('math').prod(t.dims)) for t in g.initializer
    )
    print(f"nodes: {len(g.node)}   initializers: {len(g.initializer)} (~{n_params/1e6:.1f}M values)")
    for io, kind in [(g.input, "input"), (g.output, "output")]:
        for v in io:
            print(f"{kind}:  {v.name}  {tensor_dtype(value_infos, v.name)} {tensor_shape(value_infos, v.name)}")


def show_node(n, value_infos, producers, consumers) -> None:
    print(f"\n{n.op_type}  {n.name}")
    for a in n.attribute:
        print(f"  attr {a.name} = {onnx.helper.get_attribute_value(a)}")
    for i in n.input:
        p = producers.get(i)
        src = f"<- {p.op_type} {p.name}" if p else "<- graph input / initializer"
        print(f"  in  {i}: {tensor_dtype(value_infos, i)} {tensor_shape(value_infos, i)}  {src}")
    for o in n.output:
        cs = ", ".join(f"{c.op_type} {c.name}" for c in consumers.get(o, [])) or "(graph output)"
        print(f"  out {o}: {tensor_dtype(value_infos, o)} {tensor_shape(value_infos, o)}  -> {cs}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--ops", action="store_true", help="histogram of op types")
    ap.add_argument("--node", help="show every node whose name contains this substring")
    ap.add_argument("--type", help="show every node of this op type (e.g. Where)")
    ap.add_argument("--check", action="store_true", help="run checker, shape inference, and ORT load")
    args = ap.parse_args()

    model, g, value_infos, producers, consumers = load(args.model)
    summary(model, g, value_infos)

    if args.ops:
        for op, count in Counter(n.op_type for n in g.node).most_common():
            print(f"{count:5d}  {op}")

    if args.node:
        for n in g.node:
            if args.node in n.name:
                show_node(n, value_infos, producers, consumers)

    if args.type:
        for n in g.node:
            if n.op_type == args.type:
                show_node(n, value_infos, producers, consumers)

    if args.check:
        # Three escalating levels of "is this graph OK":
        # 1. checker: structurally valid protobuf?
        # 2. shape inference: do dtypes/shapes propagate consistently?
        # 3. ORT session: does every op have an actual kernel implementation?
        print("\nonnx.checker ...", end=" ")
        onnx.checker.check_model(args.model)
        print("OK")
        print("shape inference ...", end=" ")
        onnx.shape_inference.infer_shapes(onnx.load(args.model), strict_mode=True)
        print("OK")
        print("onnxruntime load ...", end=" ")
        import onnxruntime as ort

        ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        print("OK")


if __name__ == "__main__":
    main()
