"""Parity check: this repo's TTM ONNX graphs vs the official IBM library.

Runs identical inputs through the official TinyTimeMixerForPrediction
(granite-tsfm / tsfm_public, CPU, float32) and through our exported graphs
(fp32 and int8) under native ONNX Runtime, then reports the max abs
difference per case, both raw and as a percentage of the reference
forecast's spread.

Graph contract (see scripts/export_ttm_onnx.py):

    context   float32 [rows, 512]   raw unscaled values; short series
                                    LEFT-padded with ZEROS (not NaN)
    forecast  float32 [rows, 96]    the model's point forecast

TTM is a point forecaster (MSE-trained, no quantile head), so unlike the
chronos-2/t0 parity tests there is no quantile axis to compare - one
number per future step.

Short-series note: `TinyTimeMixerForPrediction.forward` handles input
shorter than 512 by internally LEFT-padding zeros WITHOUT flagging them
in `past_observed_mask` (which stays None -> all-ones), so the internal
StdScaler counts the padded zeros as observed values. Zero left-padding
outside the graph is therefore byte-for-byte identical to feeding the
short series to the official model - the airline case below proves it.
NaN is NOT supported by this model: it would poison the scaler.

The official reference is the plain `model(past_values)` forward from the
model card - granite-tsfm's `get_model` helper only picks a revision, and
`TimeSeriesForecastingPipeline` wraps this same forward in dataframe
plumbing.

Run:  uv run -p .venv-ttm python tests/parity_ttm_onnx_vs_official.py
"""

import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tsfm_public import TinyTimeMixerForPrediction

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
CONTEXT_LEN = 512
HORIZON = 96

MODELS = [
    ("fp32", "ttm-r2-ctx512-h96.onnx", 1e-3),   # % of spread
    ("int8", "ttm-r2-ctx512-h96-int8.onnx", 10.0),
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
    airline = load_airline()  # 144 points: strong trend + seasonality, exercises zero-padding
    sine = torch.sin(torch.arange(512) / 10.0) * 10 + 50 + torch.randn(512, generator=g)
    trend = torch.arange(512, dtype=torch.float32) * 2 + 100 + torch.randn(512, generator=g) * 5
    batch3 = torch.randn(3, 512, generator=g).cumsum(-1) + torch.tensor(
        [10.0, 100.0, 1000.0]
    ).unsqueeze(-1)  # one batch, three scales: proves dynamic batch + per-series RevIN
    return [
        ("airline-144", airline),
        ("sine-512", sine),
        ("trend-512", trend),
        ("batch-3x512", batch3),
    ]


def pack(item: torch.Tensor) -> np.ndarray:
    """Zero-LEFT-pad/truncate to (rows, 512), replicating the official forward exactly."""
    x = (item if item.ndim == 2 else item.unsqueeze(0)).numpy().astype(np.float32)
    x = x[:, -CONTEXT_LEN:]
    out = np.zeros((x.shape[0], CONTEXT_LEN), dtype=np.float32)
    out[:, CONTEXT_LEN - x.shape[1]:] = x
    return out


def official_predict(model: TinyTimeMixerForPrediction, item: torch.Tensor) -> torch.Tensor:
    """Reference forecast at the series' NATURAL length (short input stays short:
    the model's own internal padding is part of what parity certifies)."""
    x = item if item.ndim == 2 else item.unsqueeze(0)
    with torch.no_grad():
        out = model(past_values=x.unsqueeze(-1), return_loss=False)
    return out.prediction_outputs.squeeze(-1)  # (rows, 96)


def main() -> None:
    print(f"loading official model ({MODEL_ID}, CPU, float32) ...")
    model = TinyTimeMixerForPrediction.from_pretrained(MODEL_ID).eval()

    cases = build_cases()
    print("computing reference forecasts (one forward() call per item) ...")
    refs = [official_predict(model, item) for _, item in cases]

    failed = False
    for label, filename, tolerance_pct in MODELS:
        print(f"\n=== {label}: onnx/{filename} vs official model ===")
        session = ort.InferenceSession(str(ROOT / "onnx" / filename))
        worst = 0.0
        for (name, item), ref in zip(cases, refs):
            (got,) = session.run(None, {"context": pack(item)})
            got = torch.from_numpy(got)
            assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
            diff = (got - ref).abs()
            spread = (ref.max() - ref.min()).item()
            max_pct = 100 * diff.max().item() / spread
            mean_pct = 100 * diff.mean().item() / spread
            worst = max(worst, max_pct)
            print(f"  {name:>15}: max {max_pct:8.4f}%  mean {mean_pct:8.4f}% of spread")
        status = "PASS" if worst <= tolerance_pct else "FAIL"
        failed |= status == "FAIL"
        print(f"{label}: worst {worst:.4f}% of spread (tolerance {tolerance_pct}%) -> {status}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
