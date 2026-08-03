/**
 * Canvas chart: history, held-out actuals, ONNX quantile fan, and the
 * library's median for visual comparison. Plain 2D canvas, no charting
 * library — ~100 lines is cheaper than a dependency here.
 */

import { HORIZON, MEDIAN_INDEX } from "./config.js";

const COLORS = {
  history: "#2b3440",
  actuals: "#9aa3ad",
  onnxMedian: "#1d6ee0",
  bandOuter: "rgba(70,130,220,.16)", // 10–90
  bandInner: "rgba(70,130,220,.26)", // 25–75
  libraryMedian: "#e0741d",
  grid: "#eee",
  divider: "#ccc",
  tickText: "#888",
};

/**
 * @param canvas   the <canvas> element
 * @param data     normalized dataset (see data.js)
 * @param quantiles ONNX forecast [step][level], or null before first run
 */
export function draw(canvas, data, quantiles) {
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, canvas.width, canvas.height);
  if (!data) return;

  // Show at most 256 history points so short-horizon detail stays readable.
  const tail = Math.min(data.context.length, 256);
  const shown = data.context.slice(-tail);

  // y-range across everything we are about to draw.
  const everything = [
    ...shown,
    ...data.actuals,
    ...(quantiles ? quantiles.flat() : []),
    ...(data.refNatural ? data.refNatural.map((s) => s[MEDIAN_INDEX]) : []),
  ].filter(Number.isFinite);
  const lo = Math.min(...everything);
  const hi = Math.max(...everything);

  const total = tail + HORIZON;
  const X = (i) => 64 + (i / (total - 1)) * (canvas.width - 84);
  const Y = (v) => canvas.height - 34 - ((v - lo) / (hi - lo || 1)) * (canvas.height - 70);

  drawGrid(g, canvas, X, Y, lo, hi, tail);

  const line = (points, color, width, dash = []) => {
    g.beginPath();
    g.setLineDash(dash);
    let started = false;
    for (const [x, y] of points) {
      if (!Number.isFinite(y)) {
        started = false; // gap where the value is missing (NaN)
        continue;
      }
      started ? g.lineTo(x, y) : g.moveTo(x, y);
      started = true;
    }
    g.strokeStyle = color;
    g.lineWidth = width;
    g.stroke();
    g.setLineDash([]);
  };

  line(shown.map((v, i) => [X(i), Y(v)]), COLORS.history, 2.5);
  line(data.actuals.map((v, i) => [X(tail + i), Y(v)]), COLORS.actuals, 2.5);

  if (quantiles) {
    // Shaded band between two quantile levels: forward along the upper
    // edge, back along the lower.
    const band = (loIdx, hiIdx, color) => {
      g.beginPath();
      quantiles.forEach((qs, i) => g.lineTo(X(tail + i), Y(qs[hiIdx])));
      [...quantiles].reverse().forEach((qs, i) => g.lineTo(X(tail + HORIZON - 1 - i), Y(qs[loIdx])));
      g.closePath();
      g.fillStyle = color;
      g.fill();
    };
    band(0, 4, COLORS.bandOuter);
    band(1, 3, COLORS.bandInner);
    line(quantiles.map((qs, i) => [X(tail + i), Y(qs[MEDIAN_INDEX])]), COLORS.onnxMedian, 3);
  }

  if (data.refNatural) {
    line(
      data.refNatural.map((qs, i) => [X(tail + i), Y(qs[MEDIAN_INDEX])]),
      COLORS.libraryMedian, 2.5, [8, 6],
    );
  }
}

function drawGrid(g, canvas, X, Y, lo, hi, tail) {
  g.fillStyle = COLORS.tickText;
  g.font = "20px system-ui";
  g.strokeStyle = COLORS.grid;
  g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo);
    g.beginPath();
    g.moveTo(64, Y(v));
    g.lineTo(canvas.width - 20, Y(v));
    g.stroke();
    g.fillText(v.toPrecision(4), 4, Y(v) + 6);
  }
  // Vertical divider where history ends and the forecast begins.
  g.strokeStyle = COLORS.divider;
  g.setLineDash([6, 6]);
  g.beginPath();
  g.moveTo(X(tail - 1), Y(lo));
  g.lineTo(X(tail - 1), Y(hi));
  g.stroke();
  g.setLineDash([]);
}
