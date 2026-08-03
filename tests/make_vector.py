"""Deterministic input + native-ORT expected outputs for the WASM smoke test."""
import json
import numpy as np
import onnxruntime as ort

rng = np.random.default_rng(42)
x = (rng.normal(50, 10, (1, 512))).astype(np.float32)
x[0, :50] = np.nan  # exercise the NaN-missing path

out = {"input": np.nan_to_num(x, nan=-99999.0).ravel().tolist()}  # JSON has no NaN; sentinel restored in node
for tag, path in [("int8", "onnx/t0-alpha-ctx512-h64-int8.onnx"), ("fp32", "onnx/t0-alpha-ctx512-h64.onnx")]:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    (y,) = sess.run(None, {"context": x})
    out[f"expected_{tag}"] = y.ravel().tolist()
import sys
json.dump(out, open(sys.argv[1], "w"))
print("wrote", sys.argv[1])
