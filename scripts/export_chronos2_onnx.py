"""Export amazon/chronos-2 to ONNX for browser inference.

Run inside the export environment (see requirements.txt):

    uv venv --python 3.12 .venv-export
    uv pip install -p .venv-export -r requirements.txt
    python scripts/export_chronos2_onnx.py   (with .venv-export activated)

The exported graph:

    input   context    float32 [batch, CONTEXT_LEN]  NaN marks missing values
    input   group_ids  int64   [batch]               rows sharing an id are
                                                     forecast jointly (multivariate);
                                                     distinct ids = independent
    output  quantiles  float32 [batch, HORIZON, 21]  the model's 21 native quantile
                                                     levels 0.01, 0.05, 0.1, ..., 0.95, 0.99

Why is this export straightforward where t0-alpha's was not?
============================================================
Chronos2Model.forward is already pure tensor math: missing values are
handled with NaN-aware ops (nanmean scaling, isnan masks), the group
attention mask is `group_ids[:, None] == group_ids[None, :]`, and there is
no branching on tensor values anywhere on the inference path. The only
Python-level branches select structure (covariates or not, reg token or
not), which the exporter freezes correctly for our covariate-free call.

Why one graph instead of t0's two?
==================================
`group_ids` is a runtime input. Feeding arange(batch) makes every row an
independent univariate forecast; feeding zeros makes all rows one joint
multivariate task. The same file serves both modes.

Why is the context length FIXED (default 2048) while batch is dynamic?
======================================================================
Same reasoning as the t0 export: the time dimension feeds shape-derived
constants (patch reshapes, time encodings), so we bake it. Chronos-2 pads
its context to a multiple of input_patch_size=16 by prepending NaN, and a
fully-NaN patch is excluded from attention entirely while the time encoding
only depends on distance from the right edge. Left-padding a short series
with NaN to CONTEXT_LEN is therefore *exactly* equivalent to feeding it at
its natural length (validated below), not an approximation as it was for
t0-alpha. CONTEXT_LEN must be a multiple of 16 to preserve this alignment.

The horizon is likewise baked: HORIZON = num_output_patches * 16, all
generated in ONE forward pass (up to the model's max of 64 patches = 1024
steps - no autoregression).

Why not the existing kashif/chronos-2-onnx export?
==================================================
It fails validation against the official pipeline: Gather-on-float bug
(graph doesn't load in native ORT), the Patch padding branch frozen to the
no-pad path (crashes on lengths not divisible by 16), and masked values
leak into the computation (changing masked fill values swings the forecast
arbitrarily). See tests/parity_onnx_vs_official.py history.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

from einops import rearrange  # noqa: E402

from chronos import Chronos2Pipeline  # noqa: E402
from chronos.chronos2.model import Chronos2Encoder  # noqa: E402

PATCH = 16  # input_patch_size == output_patch_size for chronos-2


# ---------------------------------------------------------------------------
# Export-safe monkeypatch (the only one chronos-2 needs).
#
# The original _construct_and_invert_group_time_mask feeds the boolean
# group mask straight into torch.einsum. PyTorch type-promotes bool to
# float there, but ONNX Einsum rejects bool inputs, so the exported graph
# fails to load (INVALID_GRAPH). Casting the mask to the attention dtype
# first is exactly the promotion PyTorch performs implicitly - numerics
# are identical, which the validation below certifies.
# Original: chronos/chronos2/model.py, Chronos2Encoder.
# ---------------------------------------------------------------------------
def _construct_and_invert_group_time_mask_export_safe(
    group_ids: torch.Tensor, attention_mask: torch.Tensor, floating_type: torch.dtype
) -> torch.Tensor:
    group_mask = (group_ids[:, None] == group_ids[None, :]).to(attention_mask.dtype)  # patched: explicit cast
    group_time_mask = torch.einsum("qb, bt -> qbt", group_mask, attention_mask)
    if torch.is_floating_point(group_time_mask):
        floating_type = group_time_mask.dtype
    group_time_mask = rearrange(group_time_mask, "q b t -> t 1 q b")
    group_time_mask = (1.0 - group_time_mask) * torch.finfo(floating_type).min
    return group_time_mask


Chronos2Encoder._construct_and_invert_group_time_mask = staticmethod(_construct_and_invert_group_time_mask_export_safe)


class Chronos2OnnxWrapper(nn.Module):
    """Covariate-free forecasting slice of Chronos2Model.forward.

    Mirrors Chronos2Pipeline._predict_step for the no-covariate case:
    context_mask is derived from NaN inside the model, future_covariates
    stay None (the model substitutes zero patches), and the prediction is
    a single forward pass of num_output_patches patches.
    """

    def __init__(self, model: nn.Module, num_output_patches: int):
        super().__init__()
        self.model = model
        self.num_output_patches = num_output_patches

    def forward(self, context: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(
            context=context,
            group_ids=group_ids,
            num_output_patches=self.num_output_patches,
        )
        # (batch, 21, horizon) -> (batch, horizon, 21): step-major is what the
        # app unpacks; the 21 native levels stay intact for the client to pick.
        return out.quantile_preds.transpose(1, 2)


def pack(series: torch.Tensor, context_len: int) -> torch.Tensor:
    """Left-pad/truncate (rows, length) or (length,) to (rows, context_len) with NaN."""
    x = series if series.ndim == 2 else series.unsqueeze(0)
    x = x[..., -context_len:]
    out = torch.full((x.shape[0], context_len), torch.nan)
    out[:, context_len - x.shape[1]:] = x
    return out


def validation_cases() -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(0)
    sine = torch.sin(torch.arange(300) / 10.0) * 10 + 50 + torch.randn(300, generator=g)
    gappy = sine.clone()
    gappy[50:60] = torch.nan
    gappy[200] = torch.nan
    trend = torch.arange(144, dtype=torch.float32) * 2 + 100 + torch.randn(144, generator=g) * 5
    short = torch.randn(40, generator=g).cumsum(0) + 100.0
    multivariate = torch.randn(4, 200, generator=g).cumsum(-1) + torch.tensor(
        [10.0, 20.0, 30.0, 40.0]
    ).unsqueeze(-1)
    return [
        ("sine-300", sine),
        ("nan-gaps-300", gappy),
        ("trend-144", trend),
        ("short-40", short),
        ("multivariate-4x200", multivariate),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--context-len", type=int, default=2048, help="must be a multiple of 16")
    parser.add_argument("--horizon", type=int, default=64, help="must be a multiple of 16, at most 1024")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="max allowed |onnx - predict()| relative to the reference's max magnitude per case",
    )
    args = parser.parse_args()

    if args.context_len % PATCH != 0:
        parser.error(f"--context-len must be a multiple of {PATCH}")
    if args.horizon % PATCH != 0 or not (0 < args.horizon <= 1024):
        parser.error(f"--horizon must be a multiple of {PATCH} in (0, 1024]")
    num_output_patches = args.horizon // PATCH
    out_path = args.output or ROOT / "onnx" / f"chronos2-ctx{args.context_len}-h{args.horizon}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_id} (CPU, float32) ...")
    pipe = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cpu", dtype=torch.float32)
    model = pipe.model.eval()
    quantile_levels = pipe.quantiles
    print(f"quantile levels ({len(quantile_levels)}): {quantile_levels}")

    wrapper = Chronos2OnnxWrapper(model, num_output_patches).eval()

    # Example input: batch of 3 (validation uses different batch sizes), with
    # NaN left-padding and an interior gap so every masked path is exercised.
    example = torch.full((3, args.context_len), torch.nan)
    example[0, -300:] = torch.sin(torch.arange(300) / 7.0) * 5 + 20
    example[0, -120:-100] = torch.nan
    example[1, -144:] = torch.arange(144, dtype=torch.float32)
    example[2, :] = torch.randn(args.context_len)
    example_groups = torch.arange(3, dtype=torch.long)

    print(f"exporting ctx={args.context_len} h={args.horizon} -> {out_path} ...")
    batch = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        wrapper,
        (example, example_groups),
        str(out_path),
        input_names=["context", "group_ids"],
        output_names=["quantiles"],
        dynamic_shapes={"context": {0: batch}, "group_ids": {0: batch}},
        dynamo=True,
        external_data=False,
    )
    print(f"exported: {out_path.stat().st_size / 1e6:.1f} MB")

    # ---- validation against the official pipeline -------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path))
    print("\nvalidating against pipeline.predict() (one item per call):")
    worst = 0.0
    for name, item in validation_cases():
        ref = pipe.predict([item], prediction_length=args.horizon)[0]  # (rows, 21, horizon)
        ref = ref.transpose(1, 2)  # -> (rows, horizon, 21)
        ctx = pack(item, args.context_len).numpy().astype(np.float32)
        rows = ctx.shape[0]
        joint = item.ndim == 2  # multivariate case: variates form one group
        ids = np.zeros(rows, dtype=np.int64) if joint else np.arange(rows, dtype=np.int64)
        (got,) = session.run(None, {"context": ctx, "group_ids": ids})
        got = torch.from_numpy(got)
        assert got.shape == ref.shape, f"{name}: {tuple(got.shape)} != {tuple(ref.shape)}"
        max_abs = (got - ref).abs().max().item()
        rel = max_abs / ref.abs().max().item()
        worst = max(worst, rel)
        print(f"  {name:>20}: max|onnx - predict()| = {max_abs:.3e}  (relative {rel:.1e})")

    # Batched univariate rows must equal per-item runs (proves dynamic batch
    # + group isolation): forecast sine and trend together with distinct ids.
    cases = dict(validation_cases())
    stacked = torch.stack(
        [pack(cases["sine-300"], args.context_len)[0], pack(cases["trend-144"], args.context_len)[0]]
    )
    (got_b,) = session.run(
        None, {"context": stacked.numpy().astype(np.float32), "group_ids": np.arange(2, dtype=np.int64)}
    )
    for row, name in enumerate(["sine-300", "trend-144"]):
        ref = pipe.predict([cases[name]], prediction_length=args.horizon)[0].transpose(1, 2)
        max_abs = (torch.from_numpy(got_b[row: row + 1]) - ref).abs().max().item()
        rel = max_abs / ref.abs().max().item()
        worst = max(worst, rel)
        print(f"  batched[{row}] {name:>10}: max|onnx - predict()| = {max_abs:.3e}  (relative {rel:.1e})")

    print(f"\nworst relative deviation: {worst:.3e} (tolerance {args.tolerance:.0e})")
    if worst > args.tolerance:
        raise SystemExit("FAIL: export does not match the official implementation")
    print("PASS: ONNX graph matches pipeline.predict() within tolerance.")


if __name__ == "__main__":
    main()
