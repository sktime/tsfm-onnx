/**
 * The contract with the exported ONNX graphs. These values are BAKED INTO
 * the model files by scripts/export_t0_onnx.py - changing them here without
 * re-exporting produces wrong results, not errors.
 *
 *   univariate graph:  context [rows, 512]              -> quantiles [rows, 64, 5]
 *   grouped graph:     context [rows, 512] + group_ids  -> quantiles [rows, 64, 5]
 *                      (rows sharing an id are forecast jointly, multivariate)
 */

export const CONTEXT_LEN = 512;
export const HORIZON = 64;
export const LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9];
export const MEDIAN_INDEX = LEVELS.indexOf(0.5);

/**
 * Where the .onnx files live. Local development serves them from the
 * repo's onnx/ folder; everywhere else they come from the Hugging Face
 * CDN (GitHub Pages rejects files over 100 MB, so the site and the
 * models are hosted separately).
 */
const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
export const MODEL_BASE = LOCAL
  ? new URL("../../onnx/", import.meta.url).href
  : "https://huggingface.co/Siddharth899/tsfm-onnx/resolve/main/";

export const MODEL_FILES = {
  univariate: {
    int8: "t0-alpha-ctx512-h64-int8.onnx",
    fp32: "t0-alpha-ctx512-h64.onnx",
  },
  joint: {
    int8: "t0-alpha-ctx512-h64-mv-int8.onnx",
    fp32: "t0-alpha-ctx512-h64-mv.onnx",
  },
};

export const MODEL_SIZES_MB = { int8: 108, fp32: 411 };
