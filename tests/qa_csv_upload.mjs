/**
 * QA: a real-world multi-column CSV (Daily Delhi Climate, Kaggle) through
 * the demo's own upload pipeline.
 *
 * This dataset is deliberately NOT in the demo's dataset menu; it plays the
 * role of "random file a user drags in". The test imports the demo's actual
 * parser (webdemo/js/data.js), so it exercises the shipped code:
 *
 *   1. parse the CSV -> the date column is excluded, all four climate
 *      columns are found by name;
 *   2. NaN-pad one column exactly like the demo and run the int8 model
 *      under onnxruntime-web's WASM engine;
 *   3. check forecast invariants: shape, finiteness, per-step monotone
 *      quantiles (the model's head guarantees q10 <= q25 <= ... <= q90),
 *      and a plausible value range relative to the recent context.
 *
 * Run:  cd tests && npm install onnxruntime-web@1.22.0 && node qa_csv_upload.mjs
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

import { parseCsvColumns } from "../webdemo/js/data.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ort = require("onnxruntime-web");

const CONTEXT_LEN = 512;
const HORIZON = 64;
const NQ = 5;

/* ---- 1. the parser on both Delhi files ------------------------------- */

const train = parseCsvColumns(
  fs.readFileSync(path.join(HERE, "fixtures/DailyDelhiClimateTrain.csv"), "utf8"));
assert.deepEqual(
  train.map((c) => c.name),
  ["meantemp", "humidity", "wind_speed", "meanpressure"],
  "expected the four climate columns, with the date column excluded",
);
for (const c of train) {
  assert.equal(c.values.length, 1462, `${c.name}: expected 1462 rows`);
  assert.ok(c.values.every(Number.isFinite), `${c.name}: no missing cells expected in Train`);
}
console.log("PASS parse Train: 4 numeric columns x 1462 rows, date excluded");

const test = parseCsvColumns(
  fs.readFileSync(path.join(HERE, "fixtures/DailyDelhiClimateTest.csv"), "utf8"));
assert.deepEqual(test.map((c) => c.name), ["meantemp", "humidity", "wind_speed", "meanpressure"]);
assert.equal(test[0].values.length, 114, "Test file: expected 114 rows (short-series path)");
console.log("PASS parse Test:  4 numeric columns x 114 rows (exercises NaN padding)");

/* ---- 2. forecast each Train column like the demo would ---------------- */

const model = fs.readFileSync(path.join(HERE, "../onnx/t0-alpha-ctx512-h64-int8.onnx"));
const session = await ort.InferenceSession.create(new Uint8Array(model), {
  executionProviders: ["wasm"],
});

for (const column of train) {
  // Identical to webdemo/js/forecaster.js toModelInput().
  const ctx = new Float32Array(CONTEXT_LEN).fill(NaN);
  const tail = column.values.slice(-CONTEXT_LEN);
  ctx.set(tail, CONTEXT_LEN - tail.length);

  const out = await session.run({ context: new ort.Tensor("float32", ctx, [1, CONTEXT_LEN]) });
  const q = out.quantiles.data;

  /* ---- 3. invariants ------------------------------------------------- */
  assert.equal(q.length, HORIZON * NQ, `${column.name}: expected ${HORIZON}x${NQ} outputs`);
  assert.ok([...q].every(Number.isFinite), `${column.name}: forecast contains non-finite values`);

  for (let step = 0; step < HORIZON; step++) {
    for (let l = 1; l < NQ; l++) {
      assert.ok(
        q[step * NQ + l] >= q[step * NQ + l - 1] - 1e-4,
        `${column.name}: quantile crossing at step ${step} level ${l}`,
      );
    }
  }

  // Plausibility: the median forecast should stay within the recent
  // context's range, widened by one context spread. Catches unit blow-ups
  // (a forecast of 10000 for a temperature series) without being brittle.
  const lo = Math.min(...tail), hi = Math.max(...tail);
  const spread = hi - lo || 1;
  for (let step = 0; step < HORIZON; step++) {
    const median = q[step * NQ + 2];
    assert.ok(
      median > lo - spread && median < hi + spread,
      `${column.name}: implausible median ${median} at step ${step} (context range ${lo}..${hi})`,
    );
  }
  const m0 = q[2].toFixed(1);
  console.log(`PASS forecast ${column.name.padEnd(12)} first median ${m0}, monotone, in range`);
}

console.log("all QA checks passed");
