/**
 * Numeric comparisons: browser forecast vs the PyTorch library's reference
 * forecasts, and point forecasts vs known actuals.
 */

/** Range of a [step][level] forecast — the scale errors are reported
 *  against. Raw absolute diffs are meaningless across datasets (passengers
 *  in hundreds, temperatures around 10), so errors are % of this spread. */
export function spreadOf(quantiles) {
  const flat = quantiles.flat();
  return Math.max(...flat) - Math.min(...flat) || 1;
}

/** Mean/max absolute difference between two [step][level] forecasts, as a
 *  fraction of the reference's spread. */
export function diffStats(a, ref) {
  const s = spreadOf(ref);
  let sum = 0;
  let max = 0;
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      const d = Math.abs(a[i][j] - ref[i][j]);
      sum += d;
      n++;
      if (d > max) max = d;
    }
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

/** Share of actuals inside the outer (10-90) band — a calibration hint:
 *  a well-calibrated 80% interval should contain roughly 80% of actuals. */
export function bandCoverage(quantiles, actuals) {
  let inside = 0;
  let n = 0;
  for (let i = 0; i < Math.min(quantiles.length, actuals.length); i++) {
    if (!Number.isFinite(actuals[i])) continue;
    n++;
    const qs = quantiles[i];
    if (actuals[i] >= qs[0] && actuals[i] <= qs[qs.length - 1]) inside++;
  }
  return n ? inside / n : NaN;
}

/** Compact display number: 1284.3 -> "1,284", 129000 -> "129K". */
export function compact(x, digits = 4) {
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) >= 10000) {
    return Intl.NumberFormat("en", { notation: "compact", maximumSignificantDigits: 3 }).format(x);
  }
  return Number(x.toPrecision(digits)).toLocaleString("en");
}
