/**
 * QA: the grouped (multivariate) graph under onnxruntime-web's WASM engine,
 * on the Daily Delhi climate columns - the app's "forecast all columns
 * jointly" path.
 *
 * The rigorous equivalence proof against the PyTorch library runs at export
 * time (scripts/export_t0_onnx.py --grouped, validate_grouped). This test
 * covers what that cannot: the int8 build of the grouped graph, under the
 * browser's WASM kernels, fed exactly like the app feeds it:
 *
 *   1. shared ids (joint): shape/finiteness/monotone-quantile invariants;
 *   2. distinct ids (independent): same invariants;
 *   3. the two modes DIFFER - group attention really exchanges information
 *      between the columns.
 *
 * Run:  cd tests && npm install onnxruntime-web@1.22.0 && node qa_multivariate.mjs
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

import { parseCsvColumns } from "../app/js/data.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const ort = require("onnxruntime-web");

const CONTEXT_LEN = 512;
const HORIZON = 64;
const NQ = 5;

const columns = parseCsvColumns(
  fs.readFileSync(path.join(HERE, "fixtures/DailyDelhiClimateTrain.csv"), "utf8"));
const rows = columns.length;

// Pack all columns exactly like app/js/forecaster.js packContext().
const context = new Float32Array(rows * CONTEXT_LEN).fill(NaN);
columns.forEach((c, r) => {
  const tail = c.values.slice(-CONTEXT_LEN);
  context.set(tail, r * CONTEXT_LEN + (CONTEXT_LEN - tail.length));
});
const contextTensor = new ort.Tensor("float32", context, [rows, CONTEXT_LEN]);

const model = fs.readFileSync(path.join(HERE, "../onnx/t0-alpha-ctx512-h64-mv-int8.onnx"));
const session = await ort.InferenceSession.create(new Uint8Array(model), {
  executionProviders: ["wasm"],
});

function checkInvariants(q, tag) {
  assert.equal(q.length, rows * HORIZON * NQ, `${tag}: wrong output size`);
  assert.ok([...q].every(Number.isFinite), `${tag}: non-finite values`);
  for (let r = 0; r < rows; r++) {
    for (let s = 0; s < HORIZON; s++) {
      for (let l = 1; l < NQ; l++) {
        const at = (r * HORIZON + s) * NQ;
        assert.ok(q[at + l] >= q[at + l - 1] - 1e-4,
          `${tag}: quantile crossing, row ${r} step ${s} level ${l}`);
      }
    }
  }
  console.log(`PASS ${tag}: finite, monotone quantiles for all ${rows} columns`);
}

const joint = (await session.run({
  context: contextTensor,
  group_ids: new ort.Tensor("int64", new BigInt64Array(rows), [rows]),
})).quantiles.data;
checkInvariants(joint, "joint (shared ids)     ");

const independent = (await session.run({
  context: contextTensor,
  group_ids: new ort.Tensor("int64", BigInt64Array.from({ length: rows }, (_, i) => BigInt(i)), [rows]),
})).quantiles.data;
checkInvariants(independent, "independent (unique ids)");

// The joint forecast must differ from the independent one: if it does not,
// group_ids is being ignored and the multivariate path is broken.
let maxDelta = 0;
for (let i = 0; i < joint.length; i++) {
  maxDelta = Math.max(maxDelta, Math.abs(joint[i] - independent[i]));
}
assert.ok(maxDelta > 1e-3, "joint and independent outputs identical: group_ids ignored");
console.log(`PASS coupling: joint vs independent max delta ${maxDelta.toFixed(3)} (group attention is live)`);

console.log("all multivariate QA checks passed");
