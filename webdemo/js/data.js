/**
 * Data sources: the bundled datasets (JSON produced by make_demo_data.py,
 * which also holds PyTorch reference forecasts) and user-uploaded CSVs.
 *
 * Both are normalized to one shape:
 *   {
 *     title, note,
 *     context:    number[]        // model input (NaN = missing)
 *     actuals:    number[]        // held-out future, [] if unknown
 *     refSame:    number[][]|null // library forecast, identical padded input
 *     refNatural: number[][]|null // library forecast, natural predict() call
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
  };
}

const MISSING_CELL = /^(na|nan|null|)$/i;
const isNumericCell = (cell) => cell !== "" && Number.isFinite(Number(cell));

/**
 * Parse a CSV into NAMED NUMERIC COLUMNS, so multi-column files (e.g. the
 * Daily Delhi climate set: date, meantemp, humidity, wind_speed,
 * meanpressure) let the user choose which series to forecast instead of
 * silently getting an arbitrary column.
 *
 * No library: delimiter is auto-detected (comma/semicolon/tab), the first
 * row is treated as a header when none of its cells are numeric, a column
 * counts as numeric when at least 60% of its non-missing cells parse as
 * numbers, and missing cells (empty/NA/NaN/null) become NaN, which the
 * model understands as "missing". Rows are assumed to be in time order.
 * Swap in Papa Parse if you need quoting/escaping beyond toy files.
 *
 * @returns {{name: string, values: number[]}[]} the numeric columns
 */
export function parseCsvColumns(text) {
  const rows = text.split(/\r?\n/).filter((line) => line.trim());
  if (!rows.length) return [];

  const delimiter = [",", ";", "\t"].reduce((best, d) =>
    rows[0].split(d).length > rows[0].split(best).length ? d : best);
  const grid = rows.map((row) =>
    row.split(delimiter).map((cell) => cell.trim().replace(/^"|"$/g, "")));

  const hasHeader = grid[0].every((cell) => !isNumericCell(cell));
  const header = hasHeader ? grid[0] : [];
  const body = hasHeader ? grid.slice(1) : grid;
  if (!body.length) return [];

  const width = Math.max(...grid.map((row) => row.length));
  const columns = [];
  for (let j = 0; j < width; j++) {
    const cells = body.map((row) => row[j] ?? "");
    const present = cells.filter((cell) => !MISSING_CELL.test(cell));
    if (!present.length || present.filter(isNumericCell).length / present.length < 0.6) continue;
    columns.push({
      name: header[j] || `column ${j + 1}`,
      values: cells.map((cell) => (MISSING_CELL.test(cell) || !isNumericCell(cell) ? NaN : Number(cell))),
    });
  }
  return columns;
}

export function datasetFromCsv(fileName, column, contextLen) {
  const n = column.values.length;
  return {
    title: `${fileName} (${column.name})`,
    note: `uploaded: "${column.name}" from ${fileName}, ${n} points (last ${Math.min(n, contextLen)} used)`,
    context: column.values,
    actuals: [],
    refSame: null,
    refNatural: null,
  };
}
