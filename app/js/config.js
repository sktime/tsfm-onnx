/**
 * The model registry: the contract each entry states is BAKED INTO its ONNX
 * files by the matching scripts/export_*.py - changing values here without
 * re-exporting produces wrong results, not errors.
 *
 * The app displays POINT forecasts only (the model's median where the graph
 * emits quantiles, the direct output where it does not); the quantile axes
 * are reduced inside forecaster.js.
 *
 *   chronos2   context [rows, 2048] + group_ids [rows] -> [rows, 64, 21]
 *              distinct ids = independent rows, shared id = one joint
 *              multivariate task; median = native level index 10 of 21.
 *   tinycast   context [B, 2048] -> [B, 48, 9], one AR block; the driver in
 *              forecaster.js rolls out blocks feeding the median (index 4)
 *              back, exactly like the official TinyCastPredictor.
 *   ttm        context [B, 512] -> [B, 96], a point forecaster (no
 *              quantile head); scaling happens inside the graph.
 */

export const MODELS = {
  chronos2: {
    label: "Chronos-2 (Amazon)",
    kind: "chronos2",
    contextLen: 2048,
    horizon: 64,
    nQuantiles: 21,
    medianIndex: 10,
    joint: true,
    files: { int8: "chronos2-ctx2048-h64-int8.onnx", fp32: "chronos2-ctx2048-h64.onnx" },
    sizesMB: { int8: 131, fp32: 482 },
  },
  tinycast: {
    label: "TinyCast (RAWS Labs)",
    kind: "tinycast",
    contextLen: 2048,
    block: 48,
    horizon: 96, // 2 AR blocks; any multiple of `block` works
    nQuantiles: 9,
    medianIndex: 4,
    joint: false,
    files: { int8: "tinycast-ctx2048-b48-int8.onnx", fp32: "tinycast-ctx2048-b48.onnx" },
    sizesMB: { int8: 2, fp32: 3 },
  },
  ttm: {
    label: "TinyTimeMixer r2 (IBM)",
    kind: "ttm",
    contextLen: 512,
    horizon: 96,
    joint: false,
    files: { int8: "ttm-r2-ctx512-h96-int8.onnx", fp32: "ttm-r2-ctx512-h96.onnx" },
    sizesMB: { int8: 2, fp32: 4 },
  },
};

export const DEFAULT_MODEL = "chronos2";

/**
 * Where the .onnx files live. Local development serves them from the
 * repo's onnx/ folder; everywhere else they come from the Hugging Face
 * CDN (GitHub Pages rejects files over 100 MB, so the site and the
 * models must be hosted separately). Upload onnx/*.onnx to your own HF
 * model repo and point MODEL_BASE at it before deploying.
 */
const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
export const MODEL_BASE = LOCAL
  ? new URL("../../onnx/", import.meta.url).href
  : "https://huggingface.co/CHANGE-ME/tsfm-onnx/resolve/main/";
