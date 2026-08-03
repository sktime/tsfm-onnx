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

/**
 * Minimal CSV parsing, no library: one value per row, top-to-bottom time
 * order, last column that parses as a number. Empty/NA cells become NaN
 * (missing — the model handles them). Non-numeric rows (headers) are
 * skipped. Good enough for toy datasets; swap in Papa Parse for gnarly
 * real-world files.
 */
export function parseCsv(text) {
  const values = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const cells = line.split(/[,;\t]/);
    const cell = cells[cells.length - 1].trim().replace(/^"|"$/g, "");
    if (/^(na|nan|null|)$/i.test(cell)) {
      values.push(NaN);
      continue;
    }
    const v = Number(cell);
    if (Number.isFinite(v)) values.push(v);
  }
  return values;
}

export function datasetFromCsv(fileName, values, contextLen) {
  return {
    title: fileName,
    note: `uploaded: ${values.length} points (last ${Math.min(values.length, contextLen)} used)`,
    context: values,
    actuals: [],
    refSame: null,
    refNatural: null,
  };
}
