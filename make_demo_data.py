"""Prepare real demo datasets + library reference forecasts for the web demo.

Downloads three classic toy forecasting datasets (public, tiny, famous):

- airline-passengers      monthly airline passengers 1949-1960 (Box-Jenkins)
- daily-min-temperatures  daily minimum temperatures, Melbourne 1981-1990
- monthly-sunspots        monthly sunspot counts 1749-1983

For each, the last ``min(64, len//4)`` points are held out as "actual
future", and the ORIGINAL tfc-t0 library (no export monkeypatches -- this
script must not import export_t0_onnx) produces two reference forecasts
over the remaining context:

- ``ref_same_input``: predict() on the exact input the browser builds --
  last 512 points, LEFT-padded with NaN. Browser-vs-this isolates pure
  ONNX-runtime + quantization error.
- ``ref_natural``: predict() on the raw unpadded context TRUNCATED to the
  last 512 points -- natural library usage with the same data budget the
  browser has. Browser-vs-this additionally includes the
  NaN-padding-strategy effect (only differs from ``ref_same_input`` when
  the context is shorter than 512).

Deliberately NOT the comparison target: predict() on the FULL history.
A fixed-512 graph cannot see older points, and that truncation effect
(measured and printed below for the long datasets) is a property of the
export's context budget, not of ONNX fidelity -- mixing the two would make
the browser look wrongly broken. Export a bigger --context-len if long
history matters.

Everything lands in webdemo/data/ as JSON + an index the page discovers.

Usage:  .venv-export/bin/python make_demo_data.py
"""

import json
import math
import urllib.request
from pathlib import Path

import torch

from t0 import T0Forecaster

CONTEXT_LEN = 512  # must match the exported graph (export_t0_onnx.py)
HORIZON = 64

DATASETS = [
    {
        "file": "airline-passengers",
        "title": "International airline passengers (monthly, 1949–1960)",
        "note": "144 points, trend + strong yearly seasonality — shorter than 512, so the NaN-padding path is exercised",
    },
    {
        "file": "daily-min-temperatures",
        "title": "Daily minimum temperatures, Melbourne (1981–1990)",
        "note": "3650 points, noisy yearly seasonality — uses the most recent 512 points",
    },
    {
        "file": "monthly-sunspots",
        "title": "Monthly sunspot counts (1749–1983)",
        "note": "2820 points, ~11-year solar cycle",
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
    """Exactly what webdemo/index.html builds: last CONTEXT_LEN points at
    the right edge, NaN padding on the left."""
    ctx = torch.full((1, CONTEXT_LEN), float("nan"))
    tail = context[-CONTEXT_LEN:]
    ctx[0, CONTEXT_LEN - len(tail) :] = torch.tensor(tail)
    return ctx


def main() -> None:
    out_dir = Path("webdemo/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading t0-alpha (original library, no export patches)...")
    model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()
    levels = [float(q) for q in model.head.quantile_levels]

    index = []
    for spec in DATASETS:
        name = spec["file"]
        print(f"{name}:")
        values = download_values(name, out_dir)
        holdout = min(HORIZON, len(values) // 4)
        context = values[:-holdout]

        ref_same = model.predict(to_browser_input(context), horizon=HORIZON, quantiles=levels)
        ref_natural = model.predict(torch.tensor([context[-CONTEXT_LEN:]]), horizon=HORIZON, quantiles=levels)

        if len(context) > CONTEXT_LEN:
            # Info only: what the 512-point budget costs vs full history.
            ref_full = model.predict(torch.tensor([context]), horizon=HORIZON, quantiles=levels)
            spread_f = float(ref_full.quantiles.max() - ref_full.quantiles.min()) or 1.0
            trunc = float((ref_natural.quantiles - ref_full.quantiles).abs().max()) / spread_f
            print(f"  context truncated {len(context)} -> {CONTEXT_LEN}; full-history forecast differs {trunc:.2%} of spread")

        record = {
            "name": name,
            "title": spec["title"],
            "note": spec["note"],
            "values": [round(v, 4) for v in values],
            "holdout": holdout,
            "horizon": HORIZON,
            "context_len": CONTEXT_LEN,
            "levels": levels,
            "ref_same_input": [[round(float(v), 4) for v in step] for step in ref_same.quantiles[0]],
            "ref_natural": [[round(float(v), 4) for v in step] for step in ref_natural.quantiles[0]],
        }
        (out_dir / f"{name}.json").write_text(json.dumps(record))
        index.append({"file": f"{name}.json", "title": spec["title"]})

        # Same-vs-natural drift, for the curious (nonzero only when the
        # context is shorter than CONTEXT_LEN).
        a, b = ref_same.quantiles[0], ref_natural.quantiles[0]
        spread = float(b.max() - b.min()) or 1.0
        drift = float((a - b).abs().max()) / spread
        print(f"  {len(values)} points, holdout {holdout}; padded-vs-natural predict drift: {drift:.2%} of spread")
        assert not math.isnan(drift)

    (out_dir / "index.json").write_text(json.dumps(index))
    print(f"wrote {len(index)} datasets + index to {out_dir}/")


if __name__ == "__main__":
    main()
