"""Publish the ONNX exports to Hugging Face, one repo per model per precision.

Follows the one-repo-per-deployable-artifact pattern (as in
theforecastingcompany/t0-alpha-onnx-int8): each repo holds a single graph
plus LICENSE, README and a machine-readable manifest.json. Repo naming:
<namespace>/<model>-onnx for the unquantized fp32 export,
<namespace>/<model>-onnx-int8 for the quantized one.

All upstream models are Apache-2.0 (verified 2026-09-02 via the HF API),
so redistributing the exports under Apache-2.0 with attribution - which each
card provides - is permitted. t0-alpha is gated (HF's default contact-info
gate, no extra terms) and ships a NOTICE file; Apache-2.0 section 4(d)
requires that NOTICE travel with any redistribution, so the uploader fetches
NOTICE from the source repo when one exists and publishes it alongside
LICENSE. Fetching it needs gate access for the authenticated account.

Authenticate once first (needs a WRITE token):

    uv run -p .venv-ttm hf auth login

Then:

    uv run -p .venv-ttm python scripts/upload_models_to_hf.py <user-or-org>                       # int8 repos
    uv run -p .venv-ttm python scripts/upload_models_to_hf.py <user-or-org> --precision fp32      # fp32 repos
    uv run -p .venv-ttm python scripts/upload_models_to_hf.py <user-or-org> --precision both --models toto2
"""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

ROOT = Path(__file__).resolve().parents[1]

CHRONOS2_LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T0_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

MODELS = {
    "chronos2": {
        "repo_base": "chronos2-onnx",
        "files": {"fp32": "chronos2-ctx2048-h64.onnx",
                  "int8": "chronos2-ctx2048-h64-int8.onnx"},
        "source": "amazon/chronos-2",
        "title": "Chronos-2",
        "quantization": "dynamic per-channel QInt8",
        "inputs": [
            {"name": "context", "dtype": "float32", "shape": ["batch", 2048],
             "missing_value": "NaN = missing; left-pad short series with NaN"},
            {"name": "group_ids", "dtype": "int64", "shape": ["batch"],
             "semantics": "rows sharing an id are forecast jointly (multivariate); distinct ids = independent"},
        ],
        "output": {"name": "quantiles", "dtype": "float32",
                   "shape": ["batch", 64, 21], "levels": CHRONOS2_LEVELS},
        "driver": "One forward pass. Left-pad short series with NaN (the model "
                  "is missing-aware; padding is exactly equivalent to the "
                  "natural-length call). Median is level index 10.",
        "parity": {
            "fp32": "verified within 0.0001% of forecast spread of the "
                    "official `Chronos2Pipeline`",
            "int8": "verified within 4.7% of forecast spread of the official "
                    "`Chronos2Pipeline`",
        },
    },
    "t0": {
        "repo_base": "t0-alpha-onnx",
        "files": {"fp32": "t0-alpha-ctx512-h64.onnx",
                  "int8": "t0-alpha-ctx512-h64-int8.onnx"},
        "source": "theforecastingcompany/t0-alpha",
        "title": "t0-alpha",
        "quantization": "dynamic per-channel QInt8",
        "inputs": [
            {"name": "context", "dtype": "float32", "shape": ["batch", 512],
             "missing_value": "NaN = missing; left-pad short series with NaN"},
        ],
        "output": {"name": "quantiles", "dtype": "float32",
                   "shape": ["batch", 64, 5], "levels": T0_LEVELS},
        "driver": "One forward pass. Left-pad short series with NaN (the model "
                  "is missing-aware; padding is exactly equivalent to the "
                  "natural-length call). Median is level index 2. Univariate "
                  "graph; for the multivariate `group_ids` variant run "
                  "`scripts/export_t0_onnx.py --grouped`.",
        "parity": {
            "fp32": "verified within 0.0001% of forecast spread of the official "
                    "`T0Forecaster.predict()`",
            "int8": "verified within 15.2% of forecast spread (mean 3%) of the "
                    "official `T0Forecaster.predict()`",
        },
    },
    "tinycast": {
        "repo_base": "tinycast-onnx",
        "files": {"fp32": "tinycast-ctx2048-b48.onnx",
                  "int8": "tinycast-ctx2048-b48-int8.onnx"},
        "source": "raws-labs/tinycast",
        "title": "TinyCast",
        "quantization": "dynamic per-channel QInt8, MatMul-only; dilated convs "
                        "and interface projections kept fp32 (measured recipe)",
        "inputs": [
            {"name": "context", "dtype": "float32", "shape": ["batch", 2048],
             "missing_value": "not supported - impute first (np.interp), left-pad with the FIRST value"},
        ],
        "output": {"name": "quantiles", "dtype": "float32",
                   "shape": ["batch", 48, 9], "levels": DECILES},
        "driver": "One 48-step AR block per call. For longer horizons feed the "
                  "raw median (index 4) back into the context and rerun; sort "
                  "each step's 9 deciles ascending before use (official "
                  "TinyCastPredictor semantics).",
        "parity": {
            "fp32": "verified within 0.0003% of forecast spread of the "
                    "official `TinyCastPredictor.predict()`, 15-block rollouts "
                    "included",
            "int8": "verified within 4.1% of forecast spread of the official "
                    "`TinyCastPredictor.predict()`, 15-block rollouts included",
        },
    },
    "ttm": {
        "repo_base": "ttm-r2-onnx",
        "files": {"fp32": "ttm-r2-ctx512-h96.onnx",
                  "int8": "ttm-r2-ctx512-h96-int8.onnx"},
        "source": "ibm-granite/granite-timeseries-ttm-r2",
        "title": "TinyTimeMixer r2",
        "quantization": "dynamic per-channel QInt8; forecast head kept fp32 "
                        "(measured recipe)",
        "inputs": [
            {"name": "context", "dtype": "float32", "shape": ["batch", 512],
             "missing_value": "not supported - inputs must be finite; left-pad ZEROS (official short-series behavior)"},
        ],
        "output": {"name": "forecast", "dtype": "float32",
                   "shape": ["batch", 96], "levels": None},
        "driver": "One forward pass, point forecast (no quantile head). "
                  "Scaling is inside the graph; feed raw values.",
        "parity": {
            "fp32": "verified within 0.0006% of forecast spread of the "
                    "official `TinyTimeMixerForPrediction`",
            "int8": "verified within 6.9% of forecast spread of the official "
                    "`TinyTimeMixerForPrediction`",
        },
    },
    "toto2": {
        "repo_base": "toto2-22m-onnx",
        "files": {"fp32": "toto2-22m-ctx2048-h96.onnx",
                  "int8": "toto2-22m-ctx2048-h96-int8.onnx"},
        "source": "Datadog/Toto-2.0-22m",
        "title": "Toto-2.0-22m",
        "quantization": "blocked weight-only QInt8 (opset-21 blocked "
                        "DequantizeLinear, 16-row blocks); output head kept "
                        "fp32. Requires onnxruntime >= 1.20; untested under "
                        "onnxruntime-web",
        "inputs": [
            {"name": "context", "dtype": "float32", "shape": ["variates", 2048],
             "missing_value": "NaN = missing; left-pad short series with NaN"},
            {"name": "series_ids", "dtype": "int64", "shape": ["variates"],
             "semantics": "variates sharing an id are forecast jointly through variate-axis attention; distinct ids = independent"},
        ],
        "output": {"name": "quantiles", "dtype": "float32",
                   "shape": ["variates", 96, 9], "levels": DECILES},
        "driver": "One parallel pass to 96 steps. Left-pad short series with "
                  "NaN (missing-aware). Median is level index 4.",
        "parity": {
            "fp32": "verified within 0.0003% of forecast spread of the "
                    "official `toto2.Toto2Model.forecast()`",
            "int8": "verified within 2.2% of forecast spread of the official "
                    "`toto2.Toto2Model.forecast()`",
        },
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_notice(source_repo: str) -> str | None:
    """Return the local path of the upstream NOTICE file, or None if it has none."""
    try:
        return hf_hub_download(source_repo, "NOTICE")
    except EntryNotFoundError:
        return None


def build_manifest(spec: dict, precision: str, path: Path) -> str:
    graph = onnx.load(str(path), load_external_data=False)
    return json.dumps({
        "schema_version": 1,
        "artifact": {
            "filename": spec["files"][precision],
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "onnx_ir_version": graph.ir_version,
            "opset": max(o.version for o in graph.opset_import if o.domain in ("", "ai.onnx")),
            "quantization": spec["quantization"] if precision == "int8" else "none (fp32)",
        },
        "contract": {"inputs": spec["inputs"], "output": spec["output"]},
        "source": {"model": spec["source"]},
    }, indent=2)


def build_card(spec: dict, precision: str) -> str:
    inputs = "\n".join(
        f"| `{i['name']}` | {i['dtype']} | `{i['shape']}` | "
        f"{i.get('missing_value') or i.get('semantics', '')} |"
        for i in spec["inputs"])
    out = spec["output"]
    levels = (f" Quantile levels: `{out['levels']}`." if out["levels"] else
              " Point forecast (no quantile axis).")
    if precision == "int8":
        what = (f"int8 ONNX export of "
                f"[{spec['source']}](https://huggingface.co/{spec['source']}) "
                f"(Apache-2.0), quantized as: {spec['quantization']}.")
    else:
        what = (f"Unquantized fp32 ONNX export of "
                f"[{spec['source']}](https://huggingface.co/{spec['source']}) "
                f"(Apache-2.0).")
    return f"""---
license: apache-2.0
base_model: {spec['source']}
tags:
- time-series
- forecasting
- onnx
- onnxruntime
{"- quantized" if precision == "int8" else "- fp32"}
- zero-shot
---

# {spec['title']} ({precision} ONNX)

{what}

{precision} output {spec['parity'][precision]}. Unofficial export, not
affiliated with or endorsed by the model authors.

## Contract

| input | dtype | shape | notes |
|---|---|---|---|
{inputs}

Output `{out['name']}`: {out['dtype']} `{out['shape']}`.{levels}

{spec['driver']}

See `manifest.json` for the machine-readable contract and the artifact's
sha256. `LICENSE` is Apache-2.0; when the upstream repo ships a `NOTICE`
file it is republished here unchanged, as the license requires.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", help="HF user or org, e.g. your-username")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--precision", choices=["int8", "fp32", "both"], default="int8")
    parser.add_argument("--private", action="store_true",
                        help="create the repos as private (default: public)")
    args = parser.parse_args()
    precisions = ["int8", "fp32"] if args.precision == "both" else [args.precision]

    missing = [MODELS[m]["files"][p] for m in args.models for p in precisions
               if not (ROOT / "onnx" / MODELS[m]["files"][p]).exists()]
    if missing:
        raise SystemExit(
            "missing exports (run the matching scripts/export_*.py + quantize first):\n  "
            + "\n  ".join(missing))

    api = HfApi()
    print(f"authenticated as {api.whoami()['name']}")

    for key in args.models:
        spec = MODELS[key]
        for precision in precisions:
            suffix = "-int8" if precision == "int8" else ""
            repo_id = f"{args.namespace}/{spec['repo_base']}{suffix}"
            path = ROOT / "onnx" / spec["files"][precision]
            print(f"\n=== {repo_id} ===")
            api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
            print(f"uploading {path.name} ({path.stat().st_size / 1e6:.0f} MB) ...")
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                            repo_id=repo_id, repo_type="model")
            api.upload_file(path_or_fileobj=build_manifest(spec, precision, path).encode(),
                            path_in_repo="manifest.json", repo_id=repo_id, repo_type="model")
            api.upload_file(path_or_fileobj=build_card(spec, precision).encode(),
                            path_in_repo="README.md", repo_id=repo_id, repo_type="model")
            api.upload_file(path_or_fileobj=str(ROOT / "LICENSE"),
                            path_in_repo="LICENSE", repo_id=repo_id, repo_type="model")
            notice = fetch_notice(spec["source"])
            if notice is not None:
                api.upload_file(path_or_fileobj=notice, path_in_repo="NOTICE",
                                repo_id=repo_id, repo_type="model")
                print("uploaded upstream NOTICE (Apache-2.0 section 4(d))")
            print(f"done -> https://huggingface.co/{repo_id}")

    print("\nall uploads complete.")


if __name__ == "__main__":
    main()
