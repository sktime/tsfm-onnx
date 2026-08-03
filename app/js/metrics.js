/**
 * Numeric comparisons between the browser forecast and the PyTorch
 * library's reference forecasts (precomputed by make_demo_data.py).
 */

/** Range of a [step][level] forecast — the scale we report errors against.
 *  Raw absolute diffs are meaningless across datasets (passengers in
 *  hundreds, temperatures around 10), so everything is % of this spread. */
export function spreadOf(quantiles) {
  const flat = quantiles.flat();
  return Math.max(...flat) - Math.min(...flat) || 1;
}

/** Mean/max absolute difference between two [step][level] forecasts,
 *  as a fraction of the reference's spread. */
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

/** Mean absolute error of a point forecast vs the actual future. */
export function mae(pointForecast, actuals) {
  const n = Math.min(pointForecast.length, actuals.length);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.abs(pointForecast[i] - actuals[i]);
  return sum / n;
}
