/**
 * Everything that touches onnxruntime-web: model download with progress,
 * session cache, input packing, inference, output unpacking.
 *
 * Uses the global `ort` defined by the classic <script> tag in index.html
 * (the UMD bundle wires up its own .wasm paths, so no build step is needed).
 */

import { CONTEXT_LEN, HORIZON, LEVELS, MODEL_BASE, MODEL_FILES } from "./config.js";

/** One session per (graph, precision), created on first use, then cached -
 *  a load means downloading 100-400 MB and compiling WASM kernels. */
const sessions = {};

/** Download with byte progress so the UI can show a real bar; the browser's
 *  HTTP cache still applies to the underlying fetch. */
async function fetchModel(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`model download failed (${resp.status}) for ${url}`);
  const total = Number(resp.headers.get("Content-Length")) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

/**
 * @param precision "int8" | "fp32"
 * @param graph     "univariate" | "joint"
 * @param onProgress (loadedBytes, totalBytes) during download
 */
export async function getSession(precision, graph, onProgress) {
  const key = `${graph}/${precision}`;
  if (!sessions[key]) {
    const bytes = await fetchModel(MODEL_BASE + MODEL_FILES[graph][precision], onProgress);
    sessions[key] = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
    });
  }
  return sessions[key];
}

/**
 * Pack series into the [rows, CONTEXT_LEN] input: most recent points at the
 * right edge, NaN padding on the left. NaN is a *valid* model input meaning
 * "missing" - this is how fixed-shape graphs serve any series length.
 */
function packContext(seriesList) {
  const data = new Float32Array(seriesList.length * CONTEXT_LEN).fill(NaN);
  seriesList.forEach((values, row) => {
    const tail = values.slice(-CONTEXT_LEN);
    data.set(tail, row * CONTEXT_LEN + (CONTEXT_LEN - tail.length));
  });
  return new ort.Tensor("float32", data, [seriesList.length, CONTEXT_LEN]);
}

/** Unpack the flat [rows, HORIZON, NQ] output into per-row [step][level]. */
function unpack(outputs, rows) {
  const flat = outputs.quantiles.data;
  const nq = LEVELS.length;
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: HORIZON }, (_, s) =>
      Array.from({ length: nq }, (_, l) => flat[(r * HORIZON + s) * nq + l])));
}

/** Independent forecast of one series -> {quantiles: [step][level], ms}. */
export async function forecast(session, series) {
  const t0 = performance.now();
  const outputs = await session.run({ context: packContext([series]) });
  return { quantiles: unpack(outputs, 1)[0], ms: performance.now() - t0 };
}

/**
 * JOINT multivariate forecast: all series are variates of one system and
 * inform each other through the model's group attention. Requires the
 * grouped graph; all rows share group id 0.
 * @returns {{rows: number[][][], ms: number}} rows[series][step][level]
 */
export async function forecastJoint(session, seriesList) {
  const t0 = performance.now();
  const outputs = await session.run({
    context: packContext(seriesList),
    group_ids: new ort.Tensor("int64", new BigInt64Array(seriesList.length), [seriesList.length]),
  });
  return { rows: unpack(outputs, seriesList.length), ms: performance.now() - t0 };
}
