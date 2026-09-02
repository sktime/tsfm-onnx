/* Run both exported models under onnxruntime-web's WASM engine (the same
   kernels the browser uses) and compare against native-ORT outputs. */
const fs = require("fs");
const path = require("path");
const ort = require("onnxruntime-web");

const REPO = require("path").join(__dirname, "..");
const CONTEXT_LEN = 2048;
const vec = JSON.parse(fs.readFileSync(path.join(__dirname, "vector.json")));
const input = Float32Array.from(vec.input, (v) => (v === -99999 ? NaN : v));

(async () => {
  for (const [tag, file] of [
    ["int8", "chronos2-ctx2048-h64-int8.onnx"],
    ["fp32", "chronos2-ctx2048-h64.onnx"],
  ]) {
    const buf = fs.readFileSync(path.join(REPO, "onnx", file));
    const t0 = Date.now();
    const sess = await ort.InferenceSession.create(new Uint8Array(buf), {
      executionProviders: ["wasm"],
    });
    const tLoad = Date.now() - t0;
    const t1 = Date.now();
    const out = await sess.run({
      context: new ort.Tensor("float32", input, [1, CONTEXT_LEN]),
      group_ids: new ort.Tensor("int64", new BigInt64Array(1), [1]),
    });
    const tRun = Date.now() - t1;
    const got = out.quantiles.data;
    const exp = vec["expected_" + tag];
    let maxAbs = 0, spread = Math.max(...exp) - Math.min(...exp);
    for (let i = 0; i < exp.length; i++) maxAbs = Math.max(maxAbs, Math.abs(got[i] - exp[i]));
    console.log(`${tag}: load ${tLoad} ms, run ${tRun} ms, ` +
      `max |wasm - native| = ${maxAbs.toExponential(2)} (${((maxAbs / spread) * 100).toFixed(4)}% of spread)`);
  }
})().catch((e) => { console.error("FAILED:", e.message); process.exit(1); });
