"""Parity check: this repo's chronos-2 ONNX graphs vs the official library.

Runs identical inputs through the official Chronos2Pipeline (CPU, float32)
and through our exported graphs (fp32 and int8) under native ONNX Runtime,
then reports the max abs difference per case, both raw and as a percentage
of the reference forecast's spread.

Graph contract (see scripts/export_chronos2_onnx.py):

    context    float32 [rows, 2048]      raw values, NaN = missing, left-padded
    group_ids  int64   [rows]            distinct ids = independent rows,
                                         shared id = joint multivariate task
    quantiles  float32 [rows, 64, 21]    21 native quantile levels

Expected: fp32 within ~1e-5 relative of pipeline.predict(); int8 within a
few percent of forecast spread (dynamic quantization noise).

History note: this script originally checked kashif/chronos-2-onnx, which
FAILED parity three ways (float Gather indices -> INVALID_GRAPH; the Patch
padding branch frozen so lengths not divisible by 16 crash; masked values
leaking into the computation, so any gappy/padded series forecasts wrong).
That is why this repo exports its own graphs.

Run:  uv run -p .venv-export python tests/parity_onnx_vs_official.py
"""

import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_LEN = 2048
PREDICTION_LENGTH = 64

MODELS = [
    ("fp32", "chronos2-ctx2048-h64.onnx", 1e-3),   # % of spread
    ("int8", "chronos2-ctx2048-h64-int8.onnx", 10.0),
]


def load_airline() -> torch.Tensor:
    values = []
    with open(ROOT / "app/data/airline-passengers.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                values.append(float(row[1]))
    return torch.tensor(values, dtype=torch.float32)


def build_cases() -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    airline = load_airline()  # 144 points, strong trend + seasonality
    sine = torch.sin(torch.arange(300) / 10.0) * 10 + 50 + torch.randn(300, generator=g)
    short = torch.randn(40, generator=g).cumsum(0) + 100.0  # length not a multiple of 16
    gappy = sine.clone()
    gappy[50:60] = torch.nan
    gappy[200] = torch.nan
    multivariate = torch.randn(4, 200, generator=g).cumsum(-1) + torch.tensor(
        [10.0, 20.0, 30.0, 40.0]
    ).unsqueeze(-1)
    return [
        ("airline-144", airline),
        ("sine-300", sine),
        ("short-40", short),
        ("nan-gaps-300", gappy),
        ("multivariate-4x200", multivariate),
    ]


def pack(item: torch.Tensor) -> np.ndarray:
    x = (item if item.ndim == 2 else item.unsqueeze(0)).numpy().astype(np.float32)
    x = x[:, -CONTEXT_LEN:]
    out = np.full((x.shape[0], CONTEXT_LEN), np.nan, dtype=np.float32)
    out[:, CONTEXT_LEN - x.shape[1]:] = x
    return out


def onnx_predict(session: ort.InferenceSession, item: torch.Tensor) -> torch.Tensor:
    ctx = pack(item)
    rows = ctx.shape[0]
    joint = item.ndim == 2  # variates of one multivariate item share a group
    ids = np.zeros(rows, dtype=np.int64) if joint else np.arange(rows, dtype=np.int64)
    (quantiles,) = session.run(None, {"context": ctx, "group_ids": ids})
    return torch.from_numpy(quantiles)  # (rows, horizon, 21)


def main() -> None:
    print("loading official pipeline (amazon/chronos-2, CPU, float32) ...")
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)

    cases = build_cases()
    print("computing reference forecasts (one predict() call per item) ...")
    refs = [
        pipe.predict([item], prediction_length=PREDICTION_LENGTH)[0].transpose(1, 2)  # (rows, horizon, 21)
        for _, item in cases
    ]

    failed = False
    for label, filename, tolerance_pct in MODELS:
        print(f"\n=== {label}: onnx/{filename} vs official pipeline ===")
        session = ort.InferenceSession(str(ROOT / "onnx" / filename))
        worst = 0.0
        for (name, item), ref in zip(cases, refs):
            got = onnx_predict(session, item)
            assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
            diff = (got - ref).abs()
            spread = (ref.max() - ref.min()).item()
            max_pct = 100 * diff.max().item() / spread
            mean_pct = 100 * diff.mean().item() / spread
            worst = max(worst, max_pct)
            print(f"  {name:>20}: max {max_pct:8.4f}%  mean {mean_pct:8.4f}% of spread")
        status = "PASS" if worst <= tolerance_pct else "FAIL"
        failed |= status == "FAIL"
        print(f"{label}: worst {worst:.4f}% of spread (tolerance {tolerance_pct}%) -> {status}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
