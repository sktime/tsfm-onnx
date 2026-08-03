/**
 * The contract with the exported ONNX graph. These values are BAKED INTO
 * the model file by export_t0_onnx.py — changing them here without
 * re-exporting will produce wrong results, not errors.
 *
 *   input  "context":   float32 [batch, CONTEXT_LEN], NaN = missing value
 *   output "quantiles": float32 [batch, HORIZON, LEVELS.length]
 */

export const CONTEXT_LEN = 512;
export const HORIZON = 64;
export const LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9];
export const MEDIAN_INDEX = LEVELS.indexOf(0.5);

/** Model files, relative to the page (the repo root must be served). */
export const MODELS = {
  int8: "../onnx/t0-alpha-ctx512-h64-int8.onnx",
  fp32: "../onnx/t0-alpha-ctx512-h64.onnx",
};
