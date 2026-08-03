/**
 * Data sources: bundled datasets (JSON produced by scripts/make_demo_data.py,
 * which also holds PyTorch reference forecasts) and user CSV files.
 *
 * Everything is normalized to one display shape:
 *   {
 *     title, note,
 *     context:    number[]        // model input (NaN = missing)
 *     actuals:    number[]        // known future to compare against, [] if none
 *     refSame:    number[][]|null // library forecast, identical padded input
 *     refNatural: number[][]|null // library forecast, natural predict() call
 *     labels:     string[]|null   // per-point x labels (dates), if the CSV had them
 *   }
 */

export async function loadIndex() {
  return (await fetch("data/index.json")).json();
}

export async function loadDataset(file) {
  const d = await (await fetch(`data/${file}`)).json();
  return {
    title: d.title,
    note: d.note,
    context: d.values.slice(0, d.values.length - d.holdout),
    actuals: d.values.slice(d.values.length - d.holdout),
    refSame: d.ref_same_input,
    refNatural: d.ref_natural,
    labels: null,
  };
}

const MISSING_CELL = /^(na|nan|null|)$/i;
const isNumericCell = (cell) => cell !== "" && Number.isFinite(Number(cell));

/**
 * Parse a CSV into named numeric columns plus an optional label column.
 *
 * No library: delimiter is auto-detected (comma/semicolon/tab), the first
 * row is a header when none of its cells are numeric, a column counts as
 * numeric when at least 60% of its non-missing cells parse as numbers, and
 * missing cells (empty/NA/NaN/null) become NaN, which the model reads as
 * "missing". The first non-numeric column (typically dates) is kept as
 * x-axis labels. Rows are assumed to be in time order. Swap in Papa Parse
 * if you need quoting/escaping beyond everyday files.
 *
 * @returns {{columns: {name, values}[], labels: string[]|null}}
 */
export function parseCsv(text) {
  const rows = text.split(/\r?\n/).filter((line) => line.trim());
  if (!rows.length) return { columns: [], labels: null };

  const delimiter = [",", ";", "\t"].reduce((best, d) =>
    rows[0].split(d).length > rows[0].split(best).length ? d : best);
  const grid = rows.map((row) =>
    row.split(delimiter).map((cell) => cell.trim().replace(/^"|"$/g, "")));

  const hasHeader = grid[0].every((cell) => !isNumericCell(cell));
  const header = hasHeader ? grid[0] : [];
  const body = hasHeader ? grid.slice(1) : grid;
  if (!body.length) return { columns: [], labels: null };

  const width = Math.max(...grid.map((row) => row.length));
  const columns = [];
  let labels = null;
  for (let j = 0; j < width; j++) {
    const cells = body.map((row) => row[j] ?? "");
    const present = cells.filter((cell) => !MISSING_CELL.test(cell));
    if (present.length && present.filter(isNumericCell).length / present.length >= 0.6) {
      columns.push({
        name: header[j] || `column ${j + 1}`,
        values: cells.map((cell) => (MISSING_CELL.test(cell) || !isNumericCell(cell) ? NaN : Number(cell))),
      });
    } else if (!labels && present.length) {
      labels = cells; // first non-numeric column: dates/labels for the x axis
    }
  }
  return { columns, labels };
}

/** Back-compat + test surface: just the numeric columns. */
export function parseCsvColumns(text) {
  return parseCsv(text).columns;
}

/**
 * Build the display dataset for one uploaded column under a compare mode.
 *
 * @param compare  {kind: "none"}
 *                 | {kind: "holdout", n}         backtest: withhold the last
 *                 |                              n points, compare on them
 *                 | {kind: "testfile", actuals}  future actuals from a second
 *                 |                              (test) CSV
 */
export function datasetFromUpload(fileName, column, labels, compare) {
  const values = column.values;
  let context = values;
  let actuals = [];
  let note = `${values.length} points`;
  if (compare.kind === "holdout") {
    const n = Math.min(compare.n, Math.max(0, values.length - 32));
    context = values.slice(0, values.length - n);
    actuals = values.slice(values.length - n);
    note = `${values.length} points, backtesting on the last ${n}`;
  } else if (compare.kind === "testfile") {
    actuals = compare.actuals;
    note = `${values.length} points, compared against uploaded test data (${actuals.length} points)`;
  }
  return {
    title: `${column.name} - ${fileName}`,
    note,
    context,
    actuals,
    refSame: null,
    refNatural: null,
    labels,
  };
}
