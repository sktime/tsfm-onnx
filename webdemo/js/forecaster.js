/**
 * Everything that touches onnxruntime-web: session management, input
 * preparation, inference, output unpacking.
 *
 * Uses the global `ort` defined by the classic <script> tag in index.html
 * (the UMD bundle wires up its own .wasm paths, so no build step is needed).
 */

import { CONTEXT_LEN, HORIZON, LEVELS, MODELS } from "./config.js";

/** One session per model file, created on first use, then cached —
 *  loading means downloading ~100–400 MB and compiling WASM kernels,
 *  so it must happen at most once per model. */
const sessions = {};

export async function getSession(modelKey, onStatus) {
  if (!sessions[modelKey]) {
    onStatus(`downloading ${modelKey} model (browser-cached after first load)…`);
    sessions[modelKey] = await ort.InferenceSession.create(MODELS[modelKey], {
      executionProviders: ["wasm"],
    });
  }
  return sessions[modelKey];
}

/**
 * Build the [1, CONTEXT_LEN] input tensor from a series of any length:
 * most recent points at the right edge, NaN padding on the left.
 *
 * NaN is a *valid* model input meaning "missing" (t0-alpha is trained on
 * gappy data) — this is how one fixed-shape graph serves any series length.
 * Series longer than CONTEXT_LEN simply use their most recent window.
 */
export function toModelInput(series) {
  const ctx = new Float32Array(CONTEXT_LEN).fill(NaN);
  const tail = series.slice(-CONTEXT_LEN);
  ctx.set(tail, CONTEXT_LEN - tail.length);
  return new ort.Tensor("float32", ctx, [1, CONTEXT_LEN]);
}

/**
 * Run one forecast.
 * @returns {{quantiles: number[][], ms: number}} quantiles[step][level]
 *          with `level` indexing LEVELS, plus wall-clock inference time.
 */
export async function forecast(session, series) {
  const t0 = performance.now();
  const outputs = await session.run({ context: toModelInput(series) });
  const ms = performance.now() - t0;

  // Output data is a flat Float32Array in row-major [batch, step, level]
  // order: element (step, level) sits at index step * LEVELS.length + level.
  const flat = outputs.quantiles.data;
  const quantiles = Array.from({ length: HORIZON }, (_, step) =>
    Array.from({ length: LEVELS.length }, (_, level) => flat[step * LEVELS.length + level]),
  );
  return { quantiles, ms };
}
