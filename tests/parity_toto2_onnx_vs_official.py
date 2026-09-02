"""Parity check: this repo's Toto 2 ONNX graphs vs the official Datadog library.

Runs identical inputs through the official `Toto2Model.forecast()`
(toto-models / toto2, CPU, float32) and through our exported graphs (fp32
and int8) under native ONNX Runtime, then reports the max abs difference
per series as a percentage of that series' forecast spread.

Graph contract (see scripts/export_toto2_onnx.py):

    context     float32 [variates, 2048]  raw values; NaN marks missing;
                                          short series LEFT-padded with NaN
    series_ids  int64   [variates]        shared id = joint multivariate,
                                          distinct ids = independent series
    quantiles   float32 [variates, 96, 9] the 9 native quantile levels
                                          0.1 .. 0.9 (deterministic head)

The client recipe this file certifies (replicated here independently):

    1. left-pad a short series with NaN to 2048 (truncate long to the last
       2048 points),
    2. leave interior gaps as NaN - the model is genuinely missing-aware
       (mask-driven scaler + attention); NaN means "value 0, mask False"
       in the official API, and NaN inside the final 32 steps is read as
       literal 0.0 (`forecast()` force-observes the last context patch),
    3. one session.run - no rollout, no post-processing: the graph itself
       sorts the quantile axis.

The official reference is `model.forecast(...)` exactly as on the model
card (horizon=96, decode_block_size=768 - a single-pass decode either way
for 3 output patches) fed the SAME NaN-padded 2048-length arrays: Toto 2's
xPos rotary embedding centers on the sequence midpoint, which cancels in
the attention product mathematically but not bit-wise, so a natural-length
call differs from a padded call by float noise (~2e-5 of spread, measured
in the export script) - above the fp32 tolerance here, hence identical
inputs on both sides. `has_missing_values=True` (the library default;
verified bit-identical to the card's `False` on fully-observed input).

Run:  uv run -p .venv-toto python tests/parity_toto2_onnx_vs_official.py
"""

import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from toto2 import Toto2Model

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Datadog/Toto-2.0-22m"
CONTEXT_LEN = 2048
HORIZON = 96

MODELS = [
    ("fp32", "toto2-22m-ctx2048-h96.onnx", 1e-3),   # % of spread
    ("int8", "toto2-22m-ctx2048-h96-int8.onnx", 10.0),
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


def build_cases() -> list[tuple[str, torch.Tensor, bool]]:
    """(name, series [rows x len or len], joint) - joint=True forecasts the
    rows as ONE multivariate task (shared id), else independently."""
    g = torch.Generator().manual_seed(0)
    airline = load_airline()  # 144 points: strong trend + seasonality, exercises NaN padding
    sine = torch.sin(torch.arange(512) / 10.0) * 10 + 50 + torch.randn(512, generator=g)
    gappy = sine[:300].clone()
    gappy[50:60] = torch.nan
    gappy[200] = torch.nan
    trend = torch.arange(512, dtype=torch.float32) * 2 + 100 + torch.randn(512, generator=g) * 5
    t = torch.arange(4096, dtype=torch.float32)
    long = (200 + 30 * torch.sin(2 * torch.pi * t / 168) + 10 * torch.sin(2 * torch.pi * t / 24)
            + torch.randn(4096, generator=g) * 3)  # > 2048: exercises truncation
    multivariate = torch.randn(4, 512, generator=g).cumsum(-1) + torch.tensor(
        [10.0, 30.0, 100.0, 1000.0]
    ).unsqueeze(-1)  # several scales: per-series causal scaling + variate attention
    # (offsets stop at 1000, the repo's TTM convention: at 10000 the
    # offset/spread ratio turns plain float32 noise - ~2e-7 of magnitude,
    # same as every other case - into >0.001% of spread)
    multivariate[2, 100:130] = torch.nan  # interior gap in one variate
    batch2 = torch.stack([sine, trend])  # independent rows in one call: id isolation
    return [
        ("airline-144", airline, False),
        ("nan-gaps-300", gappy, False),
        ("sine-512", sine, False),
        ("trend-512", trend, False),
        ("long-4096", long, False),
        ("multivariate-4x512", multivariate, True),
        ("batch-2x512", batch2, False),
    ]


def pack(item: torch.Tensor) -> np.ndarray:
    """Step 1 of the client recipe: NaN-LEFT-pad/truncate to (rows, 2048)."""
    x = (item if item.ndim == 2 else item.unsqueeze(0)).numpy().astype(np.float32)
    x = x[:, -CONTEXT_LEN:]
    out = np.full((x.shape[0], CONTEXT_LEN), np.nan, dtype=np.float32)
    out[:, CONTEXT_LEN - x.shape[1]:] = x
    return out


def ids_for(item: torch.Tensor, joint: bool) -> np.ndarray:
    rows = item.shape[0] if item.ndim == 2 else 1
    return np.zeros(rows, dtype=np.int64) if joint else np.arange(rows, dtype=np.int64)


def official_predict(model: Toto2Model, ctx: np.ndarray, ids: np.ndarray) -> torch.Tensor:
    """Reference forecast on the identical padded arrays (see module docstring).
    NaN -> (target 0, mask False), the official missing-value encoding."""
    context = torch.from_numpy(ctx)
    target_mask = ~torch.isnan(context)
    target = torch.nan_to_num(context, nan=0.0)
    with torch.no_grad():
        q = model.forecast(
            {"target": target, "target_mask": target_mask,
             "series_ids": torch.from_numpy(ids)},
            HORIZON,
            decode_block_size=768,
            has_missing_values=True,
        )
    return q.permute(1, 2, 0)  # (9, rows, 96) -> (rows, 96, 9)


def main() -> None:
    print(f"loading official model ({MODEL_ID}, CPU, float32) ...")
    model = Toto2Model.from_pretrained(MODEL_ID).eval()

    cases = build_cases()
    print("computing reference forecasts (one forecast() call per case) ...")
    refs = [official_predict(model, pack(item), ids_for(item, joint))
            for _, item, joint in cases]

    failed = False
    for label, filename, tolerance_pct in MODELS:
        print(f"\n=== {label}: onnx/{filename} vs official model ===")
        session = ort.InferenceSession(str(ROOT / "onnx" / filename),
                                       providers=["CPUExecutionProvider"])
        worst = 0.0
        for (name, item, joint), ref in zip(cases, refs):
            (got,) = session.run(
                None, {"context": pack(item), "series_ids": ids_for(item, joint)}
            )
            got = torch.from_numpy(got)
            assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
            diff = (got - ref).abs().amax(dim=(1, 2))
            spread = ref.amax(dim=(1, 2)) - ref.amin(dim=(1, 2))  # per series
            max_pct = (100 * diff / spread).max().item()
            mean_pct = (100 * (got - ref).abs().mean(dim=(1, 2)) / spread).max().item()
            worst = max(worst, max_pct)
            print(f"  {name:>20}: max {max_pct:8.4f}%  mean {mean_pct:8.4f}% of spread")
        status = "PASS" if worst <= tolerance_pct else "FAIL"
        failed |= status == "FAIL"
        print(f"{label}: worst {worst:.4f}% of spread (tolerance {tolerance_pct}%) -> {status}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
