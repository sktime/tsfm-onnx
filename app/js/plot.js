/**
 * Canvas time-series chart: history, actuals, the point forecast, and the
 * library reference median. Plain 2D canvas, no charting library.
 *
 * Follows the dataviz house rules: 2px lines, hairline solid gridlines,
 * recessive axes, a crosshair + one tooltip that lists EVERY series at the
 * hovered x (values lead, series names follow, line keys in the series
 * color), 8px hover markers with a 2px surface ring, and keyboard
 * navigation (arrow keys) with the same readout.
 *
 * All colors come from CSS custom properties on the chart root, so light
 * and dark themes are handled by the stylesheet and the chart just re-reads
 * them on redraw().
 */

const WINDOW = 256; // history points shown; older context still feeds the model

export function createChart(root) {
  const canvas = root.querySelector("canvas");
  const tooltip = root.querySelector(".chart-tooltip");
  const g = canvas.getContext("2d");

  const state = {
    data: null,    // normalized dataset (see data.js)
    points: null,  // ONNX point forecast [step] or null
    hover: null,   // window index under the crosshair, or null
    layout: null,  // computed per draw: scales + geometry
  };

  /* ---------- public API ---------- */

  function setData(data, points) {
    state.data = data;
    state.points = points;
    state.hover = null;
    hideTooltip();
    draw();
  }

  function redraw() {
    draw();
  }

  /* ---------- geometry ---------- */

  function colors() {
    const s = getComputedStyle(root);
    const v = (name) => s.getPropertyValue(name).trim();
    return {
      surface: v("--surface-1"),
      grid: v("--grid"),
      baseline: v("--baseline"),
      muted: v("--ink-muted"),
      history: v("--ink-history"),
      forecast: v("--series-forecast"),
      reference: v("--series-reference"),
      actuals: v("--series-actuals"),
    };
  }

  function layout() {
    const dpr = window.devicePixelRatio || 1;
    const w = root.clientWidth;
    const h = Math.max(280, Math.min(460, Math.round(w * 0.42)));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.height = `${h}px`;
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    const data = state.data;
    const tail = Math.min(data.context.length, WINDOW);
    const shown = data.context.slice(-tail);
    const future = Math.max(state.points?.length ?? 0, data.actuals.length, 1);
    const total = tail + future;

    const everything = [
      ...shown,
      ...data.actuals,
      ...(state.points ?? []),
      ...(data.refNatural ?? []),
    ].filter(Number.isFinite);
    let lo = Math.min(...everything);
    let hi = Math.max(...everything);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.06;
    lo -= pad;
    hi += pad;

    const m = { left: 56, right: 16, top: 12, bottom: 26 };
    const X = (i) => m.left + (i / (total - 1)) * (w - m.left - m.right);
    const Y = (v) => h - m.bottom - ((v - lo) / (hi - lo)) * (h - m.top - m.bottom);
    return { w, h, m, tail, shown, future, total, lo, hi, X, Y };
  }

  /* "Nice" tick values: 1/2/5 x 10^k steps covering [lo, hi]. */
  function ticks(lo, hi, count = 4) {
    const span = hi - lo;
    const step = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = span / count / step;
    const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    const s = mult * step;
    const out = [];
    for (let v = Math.ceil(lo / s) * s; v <= hi; v += s) out.push(v);
    return out;
  }

  /** x label for a window index: a CSV date when available, else a step
   *  offset relative to the forecast start. */
  function xLabel(i) {
    const { tail } = state.layout;
    const data = state.data;
    if (i < tail && data.labels) {
      const orig = data.context.length - tail + i;
      const label = data.labels[orig];
      if (label) return label;
    }
    const off = i - tail;
    return off < 0 ? `t-${-off}` : `t+${off + 1}`;
  }

  /* ---------- drawing ---------- */

  function line(points, color, width, dash = []) {
    g.beginPath();
    g.setLineDash(dash);
    g.lineJoin = "round";
    g.lineCap = "round";
    let started = false;
    for (const [x, y] of points) {
      if (!Number.isFinite(y)) { started = false; continue; }  // gap at NaN
      started ? g.lineTo(x, y) : g.moveTo(x, y);
      started = true;
    }
    g.strokeStyle = color;
    g.lineWidth = width;
    g.stroke();
    g.setLineDash([]);
  }

  function draw() {
    if (!state.data) return;
    const L = (state.layout = layout());
    const C = colors();
    const { w, h, m, tail, shown, total, X, Y } = L;
    const data = state.data;

    g.clearRect(0, 0, w, h);

    // Gridlines: hairline, solid, recessive; y ticks in muted ink.
    g.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
    g.fillStyle = C.muted;
    g.strokeStyle = C.grid;
    g.lineWidth = 1;
    for (const v of ticks(L.lo, L.hi)) {
      const y = Math.round(Y(v)) + 0.5;
      g.beginPath();
      g.moveTo(m.left, y);
      g.lineTo(w - m.right, y);
      g.stroke();
      g.textAlign = "right";
      g.textBaseline = "middle";
      g.fillText(Number(v.toPrecision(6)).toLocaleString("en"), m.left - 8, y);
    }

    // x ticks: about six positions, plus the forecast-start divider.
    g.textAlign = "center";
    g.textBaseline = "top";
    const xStep = Math.max(32, Math.pow(2, Math.ceil(Math.log2(total / 6))));
    for (let i = tail % xStep; i < total; i += xStep) {
      g.fillText(xLabel(i), X(i), h - m.bottom + 6);
    }
    const xF = Math.round(X(tail)) + 0.5;
    g.strokeStyle = C.baseline;
    g.setLineDash([5, 5]);
    g.beginPath();
    g.moveTo(xF, m.top);
    g.lineTo(xF, h - m.bottom);
    g.stroke();
    g.setLineDash([]);

    // Baseline.
    g.strokeStyle = C.baseline;
    g.beginPath();
    g.moveTo(m.left, Math.round(h - m.bottom) + 0.5);
    g.lineTo(w - m.right, Math.round(h - m.bottom) + 0.5);
    g.stroke();

    // Lines: history joins the forecast start; actuals continue history.
    line(shown.map((v, i) => [X(i), Y(v)]), C.history, 2);
    if (data.actuals.length) {
      const joined = [[X(tail - 1), Y(shown[tail - 1])], ...data.actuals.map((v, i) => [X(tail + i), Y(v)])];
      line(joined, C.actuals, 2);
    }
    if (data.refNatural) {
      line(data.refNatural.map((v, i) => [X(tail + i), Y(v)]), C.reference, 2, [7, 5]);
    }
    if (state.points) {
      line(state.points.map((v, i) => [X(tail + i), Y(v)]), C.forecast, 2);
    }

    if (state.hover !== null) drawHover(L, C);
  }

  /* ---------- hover: crosshair + markers + tooltip ---------- */

  function seriesAt(i) {
    const { tail } = state.layout;
    const data = state.data;
    const rows = [];
    if (i < tail) {
      rows.push({ name: "history", color: colors().history, value: state.data.context[data.context.length - tail + i] });
    } else {
      const k = i - tail;
      if (Number.isFinite(data.actuals[k])) {
        rows.push({ name: "actual", color: colors().actuals, value: data.actuals[k] });
      }
      if (state.points && k < state.points.length) {
        rows.push({ name: "forecast", color: colors().forecast, value: state.points[k] });
      }
      if (data.refNatural && k < data.refNatural.length) {
        rows.push({ name: "library median", color: colors().reference, value: data.refNatural[k] });
      }
    }
    return rows.filter((r) => Number.isFinite(r.value));
  }

  function drawHover(L, C) {
    const i = state.hover;
    const x = Math.round(L.X(i)) + 0.5;
    g.strokeStyle = C.baseline;
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(x, L.m.top);
    g.lineTo(x, L.h - L.m.bottom);
    g.stroke();
    // 8px markers with a 2px surface ring so they read over the lines.
    for (const row of seriesAt(i)) {
      g.beginPath();
      g.arc(L.X(i), L.Y(row.value), 6, 0, Math.PI * 2);
      g.fillStyle = C.surface;
      g.fill();
      g.beginPath();
      g.arc(L.X(i), L.Y(row.value), 4, 0, Math.PI * 2);
      g.fillStyle = row.color;
      g.fill();
    }
  }

  const fmt = (v) => Number(v.toPrecision(5)).toLocaleString("en");

  /** Tooltip DOM via textContent only - series/category names originate in
   *  user CSV headers and are untrusted. */
  function showTooltip(i, pointerX) {
    const rows = seriesAt(i);
    if (!rows.length) { hideTooltip(); return; }
    tooltip.replaceChildren();
    const title = document.createElement("div");
    title.className = "tt-title";
    title.textContent = xLabel(i);
    tooltip.appendChild(title);
    for (const row of rows) {
      const div = document.createElement("div");
      div.className = "tt-row";
      const key = document.createElement("span");
      key.className = "tt-key";
      key.style.background = row.color;
      const value = document.createElement("span");
      value.className = "tt-value";
      value.textContent = fmt(row.value);
      const name = document.createElement("span");
      name.className = "tt-name";
      name.textContent = row.name;
      div.append(key, value, name);
      tooltip.appendChild(div);
    }
    tooltip.hidden = false;
    const half = root.clientWidth / 2;
    tooltip.style.left = pointerX < half ? `${pointerX + 16}px` : "auto";
    tooltip.style.right = pointerX < half ? "auto" : `${root.clientWidth - pointerX + 16}px`;
    tooltip.style.top = `${state.layout.m.top + 8}px`;
  }

  function hideTooltip() {
    tooltip.hidden = true;
  }

  function setHover(i, pointerX) {
    const { total, X } = state.layout;
    state.hover = Math.max(0, Math.min(total - 1, i));
    draw();
    showTooltip(state.hover, pointerX ?? X(state.hover));
  }

  canvas.addEventListener("pointermove", (e) => {
    if (!state.layout) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const { m, total } = state.layout;
    const frac = (px - m.left) / (state.layout.w - m.left - m.right);
    setHover(Math.round(frac * (total - 1)), px);
  });
  canvas.addEventListener("pointerleave", () => {
    state.hover = null;
    hideTooltip();
    draw();
  });
  // Keyboard gets the same readout as hover (never gate on the pointer).
  canvas.addEventListener("keydown", (e) => {
    if (!state.layout) return;
    const { tail, total } = state.layout;
    const step = e.shiftKey ? 8 : 1;
    let i = state.hover ?? tail;
    if (e.key === "ArrowLeft") i -= step;
    else if (e.key === "ArrowRight") i += step;
    else if (e.key === "Home") i = 0;
    else if (e.key === "End") i = total - 1;
    else if (e.key === "Escape") { state.hover = null; hideTooltip(); draw(); return; }
    else return;
    e.preventDefault();
    setHover(i);
  });
  canvas.addEventListener("blur", () => {
    state.hover = null;
    hideTooltip();
    draw();
  });

  new ResizeObserver(() => state.data && draw()).observe(root);

  return { setData, redraw };
}
