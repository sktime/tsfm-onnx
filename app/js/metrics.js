/**
 * Numeric comparisons: browser forecast vs the PyTorch library's reference
 * forecast, and point forecasts vs known actuals. Everything is a point
 * (per-step) series - the app does not display quantiles.
 */

/** Range of a point forecast - the scale errors are reported against. Raw
 *  absolute diffs are meaningless across datasets (passengers in hundreds,
 *  temperatures around 10), so errors are % of this spread. */
export function spreadOf(points) {
  return Math.max(...points) - Math.min(...points) || 1;
}

/** Mean/max absolute difference between two point forecasts, as a
 *  fraction of the reference's spread. */
export function diffStats(a, ref) {
  const s = spreadOf(ref);
  let sum = 0;
  let max = 0;
  let n = 0;
  for (let i = 0; i < Math.min(a.length, ref.length); i++) {
    const d = Math.abs(a[i] - ref[i]);
    sum += d;
    n++;
    if (d > max) max = d;
  }
  return { mean: sum / n / s, max: max / s };
}

/** Mean absolute error of a point forecast vs the actual future
 *  (NaN-tolerant: missing actuals are skipped). */
export function mae(pointForecast, actuals) {
  const diffs = [];
  for (let i = 0; i < Math.min(pointForecast.length, actuals.length); i++) {
    if (Number.isFinite(actuals[i])) diffs.push(Math.abs(pointForecast[i] - actuals[i]));
  }
  if (!diffs.length) return NaN;
  return diffs.reduce((a, b) => a + b, 0) / diffs.length;
}

/** Compact display number: 1284.3 -> "1,284", 129000 -> "129K". */
export function compact(x, digits = 4) {
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) >= 10000) {
    return Intl.NumberFormat("en", { notation: "compact", maximumSignificantDigits: 3 }).format(x);
  }
  return Number(x.toPrecision(digits)).toLocaleString("en");
}
