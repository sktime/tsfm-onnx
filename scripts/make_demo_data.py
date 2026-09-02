"""Prepare real demo datasets + library reference forecasts for the web demo.

Downloads three classic toy forecasting datasets (public, tiny, famous):

- airline-passengers      monthly airline passengers 1949-1960 (Box-Jenkins)
- daily-min-temperatures  daily minimum temperatures, Melbourne 1981-1990
- monthly-sunspots        monthly sunspot counts 1749-1983

For each, the last ``min(64, len//4)`` points are held out as "actual
future", and the ORIGINAL chronos-forecasting library (no export
monkeypatches -- this script must not import export_chronos2_onnx)
produces two reference forecasts over the remaining context:

- ``ref_same_input``: predict() on the exact input the browser builds --
  last 2048 points, LEFT-padded with NaN. Browser-vs-this isolates pure
  ONNX-runtime + quantization error.
- ``ref_natural``: predict() on the raw unpadded context TRUNCATED to the
  last 2048 points -- natural library usage with the same data budget the
  browser has. For chronos-2 the two are expected to agree EXACTLY:
  fully-missing patches are excluded from attention and the time encoding
  only depends on distance from the right edge (the drift printed below
  should read 0.00%).

Deliberately NOT the comparison target: predict() on the FULL history.
A fixed-2048 graph cannot see older points, and that truncation effect
(measured and printed below for the long datasets) is a property of the
export's context budget, not of ONNX fidelity. Export a bigger
--context-len if long history matters (chronos-2 supports up to 8192).

The stored reference quantiles are sliced from the model's 21 native
levels down to the five the app displays (0.1, 0.25, 0.5, 0.75, 0.9), the
same slice app/js/forecaster.js applies to the ONNX output.

Everything lands in app/data/ as JSON + an index the page discovers.

Usage:  uv run -p .venv-export python scripts/make_demo_data.py
"""

import json
import math
import urllib.request
from pathlib import Path

import torch

from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]

CONTEXT_LEN = 2048  # must match the exported graph (export_chronos2_onnx.py)
HORIZON = 64
DISPLAY_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

DATASETS = [
    {
        "file": "airline-passengers",
        "title": "International airline passengers (monthly, 1949-1960)",
        "note": "144 points, trend + strong yearly seasonality - much shorter than 2048, so the NaN-padding path is exercised",
    },
    {
        "file": "daily-min-temperatures",
        "title": "Daily minimum temperatures, Melbourne (1981-1990)",
        "note": "3650 points, noisy yearly seasonality - uses the most recent 2048 points",
    },
    {
        "file": "monthly-sunspots",
        "title": "Monthly sunspot counts (1749-1983)",
        "note": "2820 points, ~11-year solar cycle - the 2048-point context sees several full cycles",
    },
]
SOURCE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/{}.csv"


def download_values(name: str, out_dir: Path) -> list[float]:
    """Fetch the CSV (kept on disk for the demo's upload feature too) and
    return the numeric value column (last column)."""
    csv_path = out_dir / f"{name}.csv"
    if not csv_path.exists():
        print(f"  downloading {SOURCE.format(name)}")
        urllib.request.urlretrieve(SOURCE.format(name), csv_path)
    values = []
    for line in csv_path.read_text().splitlines():
        cell = line.rsplit(",", 1)[-1].strip().strip('"')
        try:
            values.append(float(cell))
        except ValueError:
            continue  # header / footer rows
    return values


def to_browser_input(context: list[float]) -> torch.Tensor:
    """Exactly what app/js/forecaster.js builds: last CONTEXT_LEN points at
    the right edge, NaN padding on the left."""
    ctx = torch.full((CONTEXT_LEN,), float("nan"))
    tail = context[-CONTEXT_LEN:]
    ctx[CONTEXT_LEN - len(tail):] = torch.tensor(tail)
    return ctx


def main() -> None:
    out_dir = ROOT / "app" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading amazon/chronos-2 (original library, no export patches)...")
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
    level_idx = [pipe.quantiles.index(level) for level in DISPLAY_LEVELS]

    def predict_display(series: torch.Tensor) -> torch.Tensor:
        """predict() -> [horizon, len(DISPLAY_LEVELS)] on the display levels."""
        full = pipe.predict([series], prediction_length=HORIZON)[0]  # (1, 21, horizon)
        return full[0, level_idx, :].transpose(0, 1)  # -> (horizon, 5)

    index = []
    for spec in DATASETS:
        name = spec["file"]
        print(f"{name}:")
        values = download_values(name, out_dir)
        holdout = min(HORIZON, len(values) // 4)
        context = values[:-holdout]

        ref_same = predict_display(to_browser_input(context))
        ref_natural = predict_display(torch.tensor(context[-CONTEXT_LEN:]))

        if len(context) > CONTEXT_LEN:
            # Info only: what the 2048-point budget costs vs full history.
            ref_full = predict_display(torch.tensor(context))
            spread_f = float(ref_full.max() - ref_full.min()) or 1.0
            trunc = float((ref_natural - ref_full).abs().max()) / spread_f
            print(f"  context truncated {len(context)} -> {CONTEXT_LEN}; full-history forecast differs {trunc:.2%} of spread")

        record = {
            "name": name,
            "title": spec["title"],
            "note": spec["note"],
            "values": [round(v, 4) for v in values],
            "holdout": holdout,
            "horizon": HORIZON,
            "context_len": CONTEXT_LEN,
            "levels": DISPLAY_LEVELS,
            "ref_same_input": [[round(float(v), 4) for v in step] for step in ref_same],
            "ref_natural": [[round(float(v), 4) for v in step] for step in ref_natural],
        }
        (out_dir / f"{name}.json").write_text(json.dumps(record))
        index.append({"file": f"{name}.json", "title": spec["title"]})

        # Same-vs-natural drift: expected 0.00% for chronos-2 (see docstring).
        spread = float(ref_natural.max() - ref_natural.min()) or 1.0
        drift = float((ref_same - ref_natural).abs().max()) / spread
        print(f"  {len(values)} points, holdout {holdout}; padded-vs-natural predict drift: {drift:.2%} of spread")
        assert not math.isnan(drift)

    (out_dir / "index.json").write_text(json.dumps(index))
    print(f"wrote {len(index)} datasets + index to {out_dir}/")


if __name__ == "__main__":
    main()
