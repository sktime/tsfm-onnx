/**
 * Entry point: page state, control wiring, metric tiles, table view, CSV
 * download, theming. The heavy lifting lives in the other modules:
 *
 *   config.js      the contract with the exported ONNX graphs
 *   forecaster.js  onnxruntime-web sessions, packing, inference
 *   data.js        bundled datasets + CSV parsing + compare modes
 *   metrics.js     forecast-vs-library and forecast-vs-actuals math
 *   plot.js        the canvas chart (crosshair, tooltip, keyboard)
 *
 * Flow on every interaction:
 *   source (dataset | upload) + compare mode + column  ->  display dataset
 *   -> forecaster (univariate or joint multivariate)   ->  quantiles
 *   -> chart + metric tiles + table
 */

import { CONTEXT_LEN, HORIZON, MEDIAN_INDEX } from "./config.js";
import { getSession, forecast, forecastJoint } from "./forecaster.js";
import { loadIndex, loadDataset, parseCsv, datasetFromUpload } from "./data.js";
import { diffStats, mae, bandCoverage, compact } from "./metrics.js";
import { createChart } from "./plot.js";

const $ = (id) => document.getElementById(id);

const state = {
  source: null,      // {kind:"dataset", data} | {kind:"upload", fileName, columns, labels}
  columnIndex: 0,
  test: null,        // {fileName, columns} from the test CSV, if provided
  data: null,        // normalized display dataset (see data.js)
  quantiles: null,   // displayed forecast [step][level]
  jointRows: null,   // cached joint forecast rows, invalidated on change
  ms: 0,
  runToken: 0,       // ignores stale async results after rapid switching
};

const chart = createChart($("chart"));

/* ---------------- status ---------------- */

function setStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
}

function onProgress(loaded, total) {
  const bar = $("progress");
  bar.hidden = false;
  if (total) {
    bar.max = total;
    bar.value = loaded;
    setStatus(`downloading model ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB (cached after first load)`);
  } else {
    bar.removeAttribute("value");
    setStatus(`downloading model ${(loaded / 1e6).toFixed(0)} MB...`);
  }
}

/* ---------------- selections ---------------- */

const selectedPrecision = () => document.querySelector("input[name=model]:checked").value;
const selectedCompare = () => document.querySelector("input[name=compare]:checked").value;
const isJoint = () =>
  state.source?.kind === "upload" && state.source.columns.length > 1 && $("joint").checked;

function compareSpec() {
  if (state.source?.kind !== "upload") return { kind: "none" };
  const kind = selectedCompare();
  if (kind === "holdout") return { kind: "holdout", n: HORIZON };
  if (kind === "testfile" && state.test) {
    const column = state.source.columns[state.columnIndex];
    const match =
      state.test.columns.find((c) => c.name.toLowerCase() === column.name.toLowerCase()) ??
      state.test.columns[state.columnIndex] ??
      state.test.columns[0];
    return match ? { kind: "testfile", actuals: match.values } : { kind: "none" };
  }
  return { kind: "none" };
}

/** Context series for every column under the current compare mode (joint
 *  mode forecasts them all; holdout slices each the same way). */
function allContexts() {
  const spec = compareSpec();
  return state.source.columns.map((c) =>
    spec.kind === "holdout" ? c.values.slice(0, c.values.length - Math.min(spec.n, Math.max(0, c.values.length - 32))) : c.values);
}

/* ---------------- forecasting ---------------- */

async function run() {
  if (!state.source) return;
  const token = ++state.runToken;
  try {
    if (state.source.kind === "dataset") {
      state.data = state.source.data;
    } else {
      const column = state.source.columns[state.columnIndex];
      state.data = datasetFromUpload(state.source.fileName, column, state.source.labels, compareSpec());
    }
    renderHead();

    const precision = selectedPrecision();
    const joint = isJoint();
    const session = await getSession(precision, joint ? "joint" : "univariate", onProgress);
    if (token !== state.runToken) return;
    $("progress").hidden = true;
    setStatus("running...");

    if (joint) {
      if (!state.jointRows) {
        const { rows, ms } = await forecastJoint(session, allContexts());
        if (token !== state.runToken) return;
        state.jointRows = rows;
        state.ms = ms;
      }
      state.quantiles = state.jointRows[state.columnIndex];
    } else {
      const { quantiles, ms } = await forecast(session, state.data.context);
      if (token !== state.runToken) return;
      state.quantiles = quantiles;
      state.ms = ms;
    }

    const mode = joint ? `, joint x${state.source.columns.length}` : "";
    setStatus(`forecast in ${state.ms.toFixed(0)} ms (${precision}${mode})`);
    chart.setData(state.data, state.quantiles);
    renderLegend();
    renderMetrics();
    renderTable();
  } catch (e) {
    if (token === state.runToken) setStatus(`failed: ${e.message}`, true);
  }
}

/* ---------------- rendering ---------------- */

function renderHead() {
  $("title").textContent = state.data.title;
  $("note").textContent = state.data.note + (isJoint() ? " - jointly forecast with the other columns" : "");
}

function renderLegend() {
  $("lg-actuals").hidden = !state.data.actuals.length;
  $("lg-library").hidden = !state.data.refNatural;
}

function tile(label, value, sub) {
  const el = document.createElement("div");
  el.className = "tile";
  for (const [cls, text] of [["t-label", label], ["t-value", value], ["t-sub", sub]]) {
    if (!text) continue;
    const div = document.createElement("div");
    div.className = cls;
    div.textContent = text;
    el.appendChild(div);
  }
  return el;
}

function renderMetrics() {
  const tiles = [];
  const { data, quantiles } = state;
  const pct = (x) => `${(x * 100).toFixed(1)}%`;

  tiles.push(tile("Inference", `${state.ms.toFixed(0)} ms`, selectedPrecision()));

  if (data.actuals.length) {
    const median = quantiles.map((qs) => qs[MEDIAN_INDEX]);
    tiles.push(tile("MAE, forecast", compact(mae(median, data.actuals))));
    if (data.refNatural) {
      tiles.push(tile("MAE, library", compact(mae(data.refNatural.map((qs) => qs[MEDIAN_INDEX]), data.actuals))));
    }
    const cov = bandCoverage(quantiles, data.actuals);
    if (Number.isFinite(cov)) {
      tiles.push(tile("Band coverage", pct(cov), "target 80%"));
    }
  }

  if (data.refSame) {
    const same = diffStats(quantiles, data.refSame);
    tiles.push(tile("vs library", `${pct(same.mean)}`, `max ${pct(same.max)}`));
  }

  $("metrics").replaceChildren(...tiles);
}

function renderTable() {
  const { data, quantiles } = state;
  const head = ["step", "q10", "q25", "median", "q75", "q90"];
  if (data.actuals.length) head.push("actual");
  if (data.refNatural) head.push("library median");

  const table = document.createElement("table");
  const tr = document.createElement("tr");
  for (const h of head) {
    const th = document.createElement("th");
    th.textContent = h;
    tr.appendChild(th);
  }
  table.appendChild(tr);

  const fmt = (v) => (Number.isFinite(v) ? Number(v.toPrecision(6)).toLocaleString("en") : "");
  for (let i = 0; i < HORIZON; i++) {
    const row = document.createElement("tr");
    const cells = [`t+${i + 1}`, ...state.quantiles[i].map(fmt)];
    if (data.actuals.length) cells.push(fmt(data.actuals[i]));
    if (data.refNatural) cells.push(fmt(data.refNatural[i][MEDIAN_INDEX]));
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      row.appendChild(td);
    }
    table.appendChild(row);
  }
  $("table-wrap").replaceChildren(table);
}

function downloadCsv() {
  if (!state.quantiles) return;
  const { data, quantiles } = state;
  const head = ["step", "q10", "q25", "median", "q75", "q90"];
  if (data.actuals.length) head.push("actual");
  if (data.refNatural) head.push("library_median");
  const lines = [head.join(",")];
  for (let i = 0; i < HORIZON; i++) {
    const cells = [i + 1, ...quantiles[i]];
    if (data.actuals.length) cells.push(Number.isFinite(data.actuals[i]) ? data.actuals[i] : "");
    if (data.refNatural) cells.push(data.refNatural[i][MEDIAN_INDEX]);
    lines.push(cells.join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `t0-forecast-${data.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 60)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------------- data sources ---------------- */

async function pickDataset(file) {
  state.source = { kind: "dataset", data: await loadDataset(file) };
  state.jointRows = null;
  $("column-field").hidden = true;
  $("compare-field").hidden = true;
  await run();
}

function populateColumnSelect(columns) {
  const select = $("column");
  select.replaceChildren(
    ...columns.map((c, i) => {
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = c.name; // CSV headers are untrusted: textContent
      return option;
    }));
}

async function pickUpload(file) {
  const { columns, labels } = parseCsv(await file.text());
  const usable = columns.filter((c) => c.values.filter(Number.isFinite).length >= 8);
  if (!usable.length) {
    setStatus(`no numeric column found in ${file.name}`, true);
    return;
  }
  state.source = { kind: "upload", fileName: file.name, columns: usable, labels };
  state.columnIndex = 0;
  state.jointRows = null;
  populateColumnSelect(usable);
  $("column-field").hidden = false;
  $("joint-wrap").hidden = usable.length < 2;
  $("compare-field").hidden = false;
  $("dataset").value = "";
  await run();
}

async function pickTestFile(file) {
  const { columns } = parseCsv(await file.text());
  if (!columns.length) {
    setStatus(`no numeric column found in ${file.name}`, true);
    return;
  }
  state.test = { fileName: file.name, columns };
  await run();
}

/* ---------------- wiring ---------------- */

function wireDropzone(label, input, onFile) {
  input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
  label.addEventListener("dragover", (e) => {
    e.preventDefault();
    label.classList.add("dz-over");
  });
  label.addEventListener("dragleave", () => label.classList.remove("dz-over"));
  label.addEventListener("drop", (e) => {
    e.preventDefault();
    label.classList.remove("dz-over");
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
}

async function init() {
  wireDropzone($("dropzone"), $("csv"), pickUpload);
  wireDropzone($("testzone"), $("testcsv"), pickTestFile);

  $("column").addEventListener("change", (e) => {
    state.columnIndex = Number(e.target.value);
    run();
  });
  $("joint").addEventListener("change", run);
  document.querySelectorAll("input[name=compare]").forEach((r) =>
    r.addEventListener("change", () => {
      $("testzone").hidden = selectedCompare() !== "testfile";
      state.jointRows = null; // holdout changes every context
      run();
    }));
  document.querySelectorAll("input[name=model]").forEach((r) =>
    r.addEventListener("change", () => {
      state.jointRows = null;
      run();
    }));
  $("download").addEventListener("click", downloadCsv);

  try {
    const index = await loadIndex();
    $("dataset").replaceChildren(
      ...[{ file: "", title: "- choose a sample dataset -" }, ...index].map((d) => {
        const option = document.createElement("option");
        option.value = d.file;
        option.textContent = d.title;
        return option;
      }));
    $("dataset").addEventListener("change", (e) => e.target.value && pickDataset(e.target.value));
    $("dataset").value = index[0].file;
    await pickDataset(index[0].file);
  } catch (e) {
    setStatus(`failed to start: ${e.message} - serve the repo root and run scripts/make_demo_data.py`, true);
  }
}

init();
