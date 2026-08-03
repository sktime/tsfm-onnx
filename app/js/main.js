/**
 * Entry point: holds the page state, wires the controls, and renders the
 * metrics table. The interesting logic lives in the other modules:
 *
 *   config.js      the contract with the exported ONNX graph
 *   forecaster.js  onnxruntime-web session + tensor plumbing
 *   data.js        bundled datasets + CSV upload parsing
 *   metrics.js     ONNX-vs-library comparison math
 *   plot.js        canvas chart
 *
 * Data flow on every interaction:
 *   pick dataset / upload CSV  ->  state.data
 *   -> forecaster.forecast(context)  ->  state.quantiles
 *   -> plot.draw(...) + renderMetrics(...)
 */

import { CONTEXT_LEN, MEDIAN_INDEX } from "./config.js";
import { getSession, forecast } from "./forecaster.js";
import { loadIndex, loadDataset, parseCsvColumns, datasetFromCsv } from "./data.js";
import { diffStats, mae } from "./metrics.js";
import { draw } from "./plot.js";

const $ = (id) => document.getElementById(id);
const setStatus = (msg) => ($("status").textContent = msg);

const state = {
  data: null,      // normalized dataset (see data.js)
  quantiles: null, // ONNX forecast [step][level]
  upload: null,    // {fileName, columns} of the last uploaded CSV
};

function selectedModel() {
  return document.querySelector("input[name=model]:checked").value;
}

async function run() {
  if (!state.data) return;
  try {
    const session = await getSession(selectedModel(), setStatus);
    setStatus("running…");
    const { quantiles, ms } = await forecast(session, state.data.context);
    state.quantiles = quantiles;
    setStatus(`forecast in ${ms.toFixed(0)} ms (${selectedModel()}, WASM)`);
    draw($("plot"), state.data, state.quantiles);
    renderMetrics(ms);
  } catch (e) {
    setStatus("forecast failed: " + e.message);
  }
}

function renderMetrics(ms) {
  const { data, quantiles } = state;
  const pct = (x) => `${(x * 100).toFixed(2)}%`;
  const rows = [["inference time", `${ms.toFixed(0)} ms`]];

  if (data.refSame) {
    const same = diffStats(quantiles, data.refSame);
    const natural = diffStats(quantiles, data.refNatural);
    rows.push([
      "ONNX vs library, identical padded input (export + quantization error)",
      `mean ${pct(same.mean)} · max ${pct(same.max)} of forecast spread`,
    ]);
    rows.push([
      "ONNX vs library, natural predict() call (adds padding effect)",
      `mean ${pct(natural.mean)} · max ${pct(natural.max)} of forecast spread`,
    ]);
  } else {
    rows.push(["library reference", "not available for uploaded files — bundled datasets only"]);
  }

  if (data.actuals.length) {
    const onnxMedian = quantiles.map((s) => s[MEDIAN_INDEX]);
    rows.push(["MAE vs actual future — ONNX median", mae(onnxMedian, data.actuals).toFixed(3)]);
    if (data.refNatural) {
      const libMedian = data.refNatural.map((s) => s[MEDIAN_INDEX]);
      rows.push(["MAE vs actual future — library median", mae(libMedian, data.actuals).toFixed(3)]);
    }
  }

  $("metrics").innerHTML =
    "<table class='metrics'><tr><th>metric</th><th>value</th></tr>" +
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") +
    "</table>";
}

async function pickDataset(file) {
  state.data = await loadDataset(file);
  state.quantiles = null;
  $("note").textContent = state.data.note;
  $("column-wrap").hidden = true;
  await run();
}

/** Forecast one column of the uploaded CSV (index into state.upload.columns). */
async function useUploadColumn(index) {
  const column = state.upload.columns[index];
  state.data = datasetFromCsv(state.upload.fileName, column, CONTEXT_LEN);
  state.quantiles = null;
  $("note").textContent = state.data.note;
  await run();
}

async function pickUpload(input) {
  const file = input.files[0];
  if (!file) return;
  // Keep only columns with enough real values to forecast.
  const columns = parseCsvColumns(await file.text())
    .filter((c) => c.values.filter(Number.isFinite).length >= 8);
  if (!columns.length) {
    setStatus(`could not find a numeric column in ${file.name}`);
    return;
  }
  state.upload = { fileName: file.name, columns };
  $("column").innerHTML = columns
    .map((c, i) => `<option value="${i}">${c.name}</option>`)
    .join("");
  // The picker only matters when there is an actual choice to make.
  $("column-wrap").hidden = columns.length < 2;
  $("dataset").value = "";
  await useUploadColumn(0);
}

async function init() {
  try {
    const index = await loadIndex();
    $("dataset").innerHTML = index
      .map((d) => `<option value="${d.file}">${d.title}</option>`)
      .join("");
    $("dataset").addEventListener("change", (e) => e.target.value && pickDataset(e.target.value));
    $("csv").addEventListener("change", (e) => pickUpload(e.target));
    $("column").addEventListener("change", (e) => useUploadColumn(Number(e.target.value)));
    document
      .querySelectorAll("input[name=model]")
      .forEach((r) => r.addEventListener("change", run));
    await pickDataset(index[0].file); // auto-forecast the first dataset
  } catch (e) {
    setStatus(`init failed: ${e.message} — did you run scripts/make_demo_data.py and serve the repo root?`);
  }
}

init();
