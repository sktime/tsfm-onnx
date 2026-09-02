"""Parity check: this repo's tinycast ONNX graphs vs the official library.

Runs identical inputs through the official `TinyCastPredictor` (the deployed
GIFT-Eval predictor, CPU, defaults) and through this repo's graph driven by
an INDEPENDENT reimplementation of its rollout, then reports the max abs
difference per case as a percentage of the reference forecast's spread.

The graph is ONE 48-step AR block (see scripts/export_tinycast_onnx.py), so
everything around it is driver logic this file replicates from
tinycast/predictor.py rather than imports -- the point is to certify the
recipe a browser client must follow, not to reuse the official code:

    1. left-pad a short series with its FIRST value to 2048 (truncate long),
    2. impute NaNs: np.interp over valid points, then ffill/bfill, then 0.0
       (note: padding happens BEFORE imputation, exactly like the official
       `_prepare_context_matrix`),
    3. roll out ceil(horizon/48) blocks, feeding the MEDIAN (index 4 of 9)
       back into the context each step,
    4. truncate to the horizon and SORT the quantile axis ascending
       (non-crossing deciles, official `predict()` post-processing).

Expected: fp32 within ~1e-5 relative of predict(); int8 within a few percent
of forecast spread, growing with rollout depth (the int8 median is fed back).

Run:  uv run -p .venv-export python tests/parity_tinycast_onnx_vs_official.py
"""

import csv
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download

from tinycast.predictor import TinyCastPredictor

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_LEN = 2048
BLOCK = 48
N_QUANTILES = 9

MODELS = [
    ("fp32", "tinycast-ctx2048-b48.onnx", 1e-3),   # % of spread
    ("int8", "tinycast-ctx2048-b48-int8.onnx", 10.0),
]


def load_airline() -> np.ndarray:
    values = []
    with open(ROOT / "app/data/airline-passengers.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                values.append(float(row[1]))
    return np.asarray(values, dtype=np.float32)


def build_cases() -> list[tuple[int, list[tuple[str, np.ndarray]]]]:
    """(prediction_length, [(name, series), ...]) groups; each group is one
    batched predictor call, so batching is exercised too."""
    rng = np.random.default_rng(0)
    airline = load_airline()  # 144 points, strong trend + seasonality
    t = np.arange(4096, dtype=np.float32)
    sine = (np.sin(t[:300] / 10.0) * 10 + 50 + rng.normal(0, 1, 300)).astype(np.float32)
    gappy = sine.copy()
    gappy[50:60] = np.nan
    gappy[200] = np.nan
    trend = (100 + 0.5 * t[:512] + rng.normal(0, 2, 512)).astype(np.float32)
    long = (200 + 30 * np.sin(2 * np.pi * t / 168) + 10 * np.sin(2 * np.pi * t / 24)
            + rng.normal(0, 3, 4096)).astype(np.float32)  # > 2048: truncation
    return [
        (48, [("airline-144", airline), ("nan-gaps-300", gappy)]),
        (96, [("sine-300", sine), ("trend-512", trend)]),
        (720, [("long-4096", long)]),  # 15 AR blocks
    ]


# --- independent reimplementation of the official driver -------------------

def _numpy_fill(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


def prepare_context(series_list: list[np.ndarray]) -> np.ndarray:
    """Steps 1-2 of the recipe: pad/truncate, then impute. Mirrors
    `ARRolloutPredictor._prepare_context_matrix` (downsampling off)."""
    rows = []
    for s in series_list:
        a = s.astype(np.float32)
        if a.shape[0] >= CONTEXT_LEN:
            a = a[-CONTEXT_LEN:]
        else:
            fill = a[0] if a.shape[0] > 0 else 0.0
            a = np.concatenate([np.full(CONTEXT_LEN - a.shape[0], fill, dtype=np.float32), a])
        x2d = a[None, :]
        x_interp = np.copy(x2d)
        if np.any(np.isnan(a)):
            valid = ~np.isnan(a)
            if valid.sum() >= 2:
                x_interp[0] = np.interp(np.arange(len(a)), np.where(valid)[0], a[valid])
            else:
                x_interp = _numpy_fill(x2d)
        ff = _numpy_fill(x_interp)
        bf = np.flip(_numpy_fill(np.flip(x_interp, axis=1)), axis=1)
        imp = np.where(np.isnan(ff), bf, ff)
        rows.append(np.where(np.isnan(imp), 0.0, imp)[0])
    return np.stack(rows).astype(np.float32)


def onnx_predict(session: ort.InferenceSession, series_list: list[np.ndarray],
                 prediction_length: int) -> np.ndarray:
    """Steps 3-4: AR rollout with median feedback, truncate, sort. Returns
    (B, prediction_length, 9)."""
    ctx = prepare_context(series_list)
    preds = []
    for _ in range(math.ceil(prediction_length / BLOCK)):
        (q,) = session.run(None, {"context": ctx[:, -CONTEXT_LEN:]})  # (B,48,9)
        preds.append(q)
        ctx = np.concatenate([ctx, q[:, :, N_QUANTILES // 2]], axis=1)
    pred = np.concatenate(preds, axis=1)[:, :prediction_length, :]
    return np.sort(pred, axis=-1)


def official_predict(weights: str, series_list: list[np.ndarray],
                     prediction_length: int) -> np.ndarray:
    predictor = TinyCastPredictor(
        prediction_length=prediction_length, checkpoint_path=weights, device="cpu",
    )
    entries = [
        {"target": s, "start": pd.Period("2020-01-01 00:00", freq="h")}
        for s in series_list
    ]
    forecasts = predictor.predict(entries)
    assert all(f.forecast_keys == [str(q) for q in predictor.quantiles] for f in forecasts)
    return np.stack([f.forecast_array.T for f in forecasts])  # (B, pl, 9)


def main() -> None:
    weights = hf_hub_download("raws-labs/tinycast", "model.safetensors")
    hf_hub_download("raws-labs/tinycast", "config.json")

    cases = build_cases()
    print("computing reference forecasts (official TinyCastPredictor, CPU) ...")
    refs = {pl: official_predict(weights, [s for _, s in group], pl)
            for pl, group in cases}

    failed = False
    for label, filename, tolerance_pct in MODELS:
        print(f"\n=== {label}: onnx/{filename} vs official predictor ===")
        session = ort.InferenceSession(str(ROOT / "onnx" / filename),
                                       providers=["CPUExecutionProvider"])
        worst = 0.0
        for pl, group in cases:
            got = onnx_predict(session, [s for _, s in group], pl)
            ref = refs[pl]
            assert got.shape == ref.shape, f"pl={pl}: {got.shape} != {ref.shape}"
            for i, (name, _) in enumerate(group):
                diff = np.abs(got[i] - ref[i])
                spread = ref[i].max() - ref[i].min()
                max_pct = 100 * diff.max() / spread
                mean_pct = 100 * diff.mean() / spread
                worst = max(worst, max_pct)
                print(f"  {name:>16} (h={pl:>3}): max {max_pct:8.4f}%  mean {mean_pct:8.4f}% of spread")
        status = "PASS" if worst <= tolerance_pct else "FAIL"
        failed |= status == "FAIL"
        print(f"{label}: worst {worst:.4f}% of spread (tolerance {tolerance_pct}%) -> {status}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
