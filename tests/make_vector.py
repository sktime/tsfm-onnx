"""Deterministic input + native-ORT expected outputs for the WASM smoke test.

Usage:  uv run -p .venv-export python tests/make_vector.py [out.json]
Works from any working directory; defaults to writing tests/vector.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CONTEXT_LEN = 2048

rng = np.random.default_rng(42)
x = (rng.normal(50, 10, (1, CONTEXT_LEN))).astype(np.float32)
x[0, :200] = np.nan  # exercise the NaN-missing path
group_ids = np.zeros(1, dtype=np.int64)

out = {"input": np.nan_to_num(x, nan=-99999.0).ravel().tolist()}  # JSON has no NaN; sentinel restored in node
for tag in ("int8", "fp32"):
    suffix = "-int8" if tag == "int8" else ""
    sess = ort.InferenceSession(ROOT / "onnx" / f"chronos2-ctx2048-h64{suffix}.onnx",
                                providers=["CPUExecutionProvider"])
    (y,) = sess.run(None, {"context": x, "group_ids": group_ids})
    out[f"expected_{tag}"] = y.ravel().tolist()

target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "vector.json"
target.write_text(json.dumps(out))
print("wrote", target)
