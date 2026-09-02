/**
 * Everything that touches onnxruntime-web: model download with progress,
 * session cache, per-model input packing, inference, output unpacking.
 *
 * Uses the global `ort` defined by the classic <script> tag in index.html
 * (the UMD bundle wires up its own .wasm paths, so no build step is needed).
 *
 * The app is point-forecast only: every path below reduces its model's
 * output to one number per step (the median where the graph emits
 * quantiles, the direct output for TTM).
 */

import { MODELS, MODEL_BASE } from "./config.js";

/** One session per model+precision, created on first use, then cached. */
const sessions = {};

/** Cache Storage bucket for model bytes. The browser's plain HTTP cache is
 *  NOT reliable here: Chromium declines to disk-cache bodies this large,
 *  and Hugging Face resolve URLs redirect to signed CDN links whose query
 *  strings change between sessions, so every page reload re-downloaded the
 *  model. The Cache API stores the bytes once under the stable URL and
 *  survives reloads. Bump the name if model files are ever republished
 *  under the same filenames. */
const MODEL_CACHE = "tsfm-models-v1";

async function fetchModel(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    /* Cache API unavailable (rare); fall through to a plain download. */
  }

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

  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
    } catch {
      /* Quota exceeded: the app still works, the next reload just re-downloads. */
    }
  }
  return bytes;
}

/**
 * @param modelKey  key into MODELS
 * @param precision "int8" | "fp32"
 * @param onProgress (loadedBytes, totalBytes) during download
 */
export async function getSession(modelKey, precision, onProgress) {
  const id = `${modelKey}-${precision}`;
  if (!sessions[id]) {
    const bytes = await fetchModel(MODEL_BASE + MODELS[modelKey].files[precision], onProgress);
    sessions[id] = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
    });
  }
  return sessions[id];
}

/* ------------------------- input preparation ------------------------- */

/** Linear interpolation across NaN gaps, then edge fill, then 0 - the
 *  imputation the official tinycast predictor applies (its model is not
 *  missing-aware). Also used for TTM, where it is an app-side pragmatism:
 *  the official TTM forward has no missing-value story at all and NaN
 *  would poison its internal scaler. */
function imputeLinear(values) {
  const x = Float32Array.from(values);
  const valid = [];
  for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i])) valid.push(i);
  if (!valid.length) return x.fill(0);
  let v = 0;
  for (let i = 0; i < x.length; i++) {
    if (Number.isFinite(x[i])) continue;
    while (v < valid.length && valid[v] < i) v++;
    const right = valid[v];
    const left = valid[v - 1];
    if (left === undefined) x[i] = x[right];
    else if (right === undefined) x[i] = x[left];
    else x[i] = x[left] + ((x[right] - x[left]) * (i - left)) / (right - left);
  }
  return x;
}

/** Chronos-2 packing: NaN left-padding. NaN is a *valid* model input meaning
 *  "missing", and because chronos-2 masks whole missing patches out of
 *  attention, the padded call is EXACTLY equivalent to the natural-length
 *  call (validated in scripts/export_chronos2_onnx.py). */
function packNaN(seriesList, contextLen) {
  const data = new Float32Array(seriesList.length * contextLen).fill(NaN);
  seriesList.forEach((values, row) => {
    const tail = values.slice(-contextLen);
    data.set(tail, row * contextLen + (contextLen - tail.length));
  });
  return data;
}

/** TinyCast packing, per the official predictor: truncate to the last
 *  contextLen, left-pad a short series with its FIRST value, THEN impute
 *  (pad-before-impute order matters when a series starts with NaN). */
function packTinycast(values, contextLen) {
  let tail = Array.from(values.slice(-contextLen));
  if (tail.length < contextLen) {
    tail = new Array(contextLen - tail.length).fill(tail[0] ?? 0).concat(tail);
  }
  return imputeLinear(tail);
}

/** TTM packing, per the official forward: impute (app-side, see above),
 *  truncate, left-pad ZEROS - the graph's internal scaler counts padded
 *  zeros as observed, byte-identical to the official short-series path. */
function packTtm(values, contextLen) {
  const imputed = imputeLinear(Array.from(values.slice(-contextLen)));
  const data = new Float32Array(contextLen);
  data.set(imputed, contextLen - imputed.length);
  return data;
}

/* --------------------------- model drivers --------------------------- */

async function runChronos2(session, model, seriesList, joint) {
  const rows = seriesList.length;
  const ids = new BigInt64Array(rows);
  if (!joint) for (let i = 0; i < rows; i++) ids[i] = BigInt(i);
  const outputs = await session.run({
    context: new ort.Tensor("float32", packNaN(seriesList, model.contextLen), [rows, model.contextLen]),
    group_ids: new ort.Tensor("int64", ids, [rows]),
  });
  const flat = outputs.quantiles.data;
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: model.horizon }, (_, s) =>
      flat[(r * model.horizon + s) * model.nQuantiles + model.medianIndex]));
}

/** AR rollout, mirroring TinyCastPredictor: each block the RAW median
 *  (quantile index 4, unsorted) is appended to the context; the DISPLAYED
 *  median is the middle of the per-step SORTED deciles, matching the
 *  official predictor's non-crossing sort before it emits quantiles. */
async function runTinycast(session, model, seriesList) {
  const rows = seriesList.length;
  const { contextLen, block, nQuantiles: nq, medianIndex } = model;
  const ctx = seriesList.map((s) => Array.from(packTinycast(s, contextLen)));
  const points = seriesList.map(() => []);
  for (let done = 0; done < model.horizon; done += block) {
    const data = new Float32Array(rows * contextLen);
    ctx.forEach((c, r) => data.set(c.slice(-contextLen), r * contextLen));
    const outputs = await session.run({
      context: new ort.Tensor("float32", data, [rows, contextLen]),
    });
    const flat = outputs.quantiles.data;
    for (let r = 0; r < rows; r++) {
      for (let s = 0; s < block; s++) {
        const qs = Array.from({ length: nq }, (_, q) => flat[(r * block + s) * nq + q]);
        ctx[r].push(qs[medianIndex]);
        points[r].push(qs.slice().sort((a, b) => a - b)[medianIndex]);
      }
    }
  }
  return points.map((p) => p.slice(0, model.horizon));
}

async function runTtm(session, model, seriesList) {
  const rows = seriesList.length;
  const data = new Float32Array(rows * model.contextLen);
  seriesList.forEach((s, r) => data.set(packTtm(s, model.contextLen), r * model.contextLen));
  const outputs = await session.run({
    context: new ort.Tensor("float32", data, [rows, model.contextLen]),
  });
  const flat = outputs[session.outputNames[0]].data;
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: model.horizon }, (_, s) => flat[r * model.horizon + s]));
}

/**
 * Forecast one or more series with the selected model.
 * @param joint  chronos-2 only: variates of one system, shared group id.
 * @returns {{rows: number[][], ms: number}} rows[series][step], point forecasts
 */
export async function forecast(session, modelKey, seriesList, joint = false) {
  const model = MODELS[modelKey];
  const t0 = performance.now();
  let rows;
  if (model.kind === "chronos2") rows = await runChronos2(session, model, seriesList, joint);
  else if (model.kind === "tinycast") rows = await runTinycast(session, model, seriesList);
  else rows = await runTtm(session, model, seriesList);
  return { rows, ms: performance.now() - t0 };
}
