const STABILITY_COLORS = {
  stable_node: "#4CAF50",
  stable_focus: "#8BC34A",
  unstable_node: "#f44336",
  unstable_focus: "#FF9800",
  saddle: "#9C27B0",
};

const REGIME_COLORS = {
  fixed_point: "#e3f2fd",
  limit_cycle: "#fff3e0",
  other: "#f5f5f5",
};

const STATE_COLORS = ["#2196F3", "#f44336", "#4CAF50", "#FF9800"];

// ═══════════════════════════════════════════════════════════════
//  MODEL DEFINITIONS  (all computation runs client-side)
// ═══════════════════════════════════════════════════════════════

function _sigmoid(x, k, theta) {
  const arg = -k * (x - theta);
  const clipped = Math.max(-709, Math.min(709, arg));
  return 1.0 / (1.0 + Math.exp(clipped));
}

const MODELS = {
  wilson_cowan: {
    dim: 2,
    stateNames: ["E", "I"],
    stateVars: ["E", "I"],
    display: [0, 1],
    defaultParams: {
      aee: 10.0, aei: 10.0, aie: 10.0, aii: 2.0,
      Pe: -2.0, Pi: -8.0,
      ke: 1.0, ki: 1.0,
      thetae: 4.0, thetai: 4.0,
    },
    paramInfo: {
      aee: [0.0, 20.0, 10.0, "E→E coupling"],
      aei: [0.0, 20.0, 10.0, "I→E coupling"],
      aie: [0.0, 20.0, 10.0, "E→I coupling"],
      aii: [0.0, 20.0, 2.0, "I→I coupling"],
      Pe: [-10.0, 10.0, -2.0, "External E input"],
      Pi: [-10.0, 10.0, -8.0, "External I input"],
      ke: [0.1, 5.0, 1.0, "E sigmoid gain"],
      ki: [0.1, 5.0, 1.0, "I sigmoid gain"],
      thetae: [0.0, 10.0, 4.0, "E sigmoid threshold"],
      thetai: [0.0, 10.0, 4.0, "I sigmoid threshold"],
    },
    defaultXlim: [-0.2, 1.2],
    defaultYlim: [-0.2, 1.2],
    defaultLims: [[-0.2, 1.2], [-0.2, 1.2]],
    f: (_t, [E, I], p) => {
      const Se = (x) => _sigmoid(x, p.ke, p.thetae);
      const Si = (x) => _sigmoid(x, p.ki, p.thetai);
      return [
        -E + Se(p.aee * E - p.aei * I + p.Pe),
        -I + Si(p.aie * E - p.aii * I + p.Pi),
      ];
    },
  },

  fitzhugh_nagumo: {
    dim: 2,
    stateNames: ["v", "w"],
    stateVars: ["v", "w"],
    display: [0, 1],
    defaultParams: { a: 0.7, b: 0.8, epsilon: 0.08, I: 0.5 },
    paramInfo: {
      a: [-1.0, 2.0, 0.7, "Recovery offset"],
      b: [0.0, 2.0, 0.8, "Recovery gain"],
      epsilon: [0.001, 1.0, 0.08, "Time scale (ε)"],
      I: [-2.0, 2.0, 0.5, "External current"],
    },
    defaultXlim: [-3.0, 3.0],
    defaultYlim: [-1.5, 2.0],
    defaultLims: [[-3.0, 3.0], [-1.5, 2.0]],
    f: (_t, [v, w], p) => [
      v - (v * v * v) / 3.0 - w + p.I,
      p.epsilon * (v + p.a - p.b * w),
    ],
  },

  mpr: {
    dim: 2,
    stateNames: ["r", "v"],
    stateVars: ["r", "v"],
    display: [0, 1],
    defaultParams: { delta: 1.0, eta_bar: -5.0, J: 15.0, I: 0.0 },
    paramInfo: {
      delta: [0.01, 5.0, 1.0, "Lorentzian half-width Δ"],
      eta_bar: [-20.0, 10.0, -5.0, "Mean excitability η̄"],
      J: [-20.0, 30.0, 15.0, "Synaptic coupling J"],
      I: [-10.0, 10.0, 0.0, "External input I"],
    },
    defaultXlim: [0.0, 2.0],
    defaultYlim: [-4.0, 2.0],
    defaultLims: [[0.0, 2.0], [-4.0, 2.0]],
    f: (_t, [r, v], p) => {
      const r_eff = Math.max(r, 1e-10);
      const dr = p.delta / Math.PI + 2.0 * r_eff * v;
      const dv = v * v + p.eta_bar + p.J * r_eff + p.I - (Math.PI * r_eff) ** 2;
      return [dr, dv];
    },
  },
};

// ═══════════════════════════════════════════════════════════════
//  NUMERICAL UTILITIES
// ═══════════════════════════════════════════════════════════════

function linspace(a, b, n) {
  const result = [];
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) result.push(a + i * step);
  return result;
}

/** Build a projected RHS function for a subset of state variables.
 *
 *  @param {string} modelName — key in MODELS
 *  @param {number[]} displayIdx — indices of displayed state variables
 *  @param {number[]} clamped — full N-dim state vector with clamped values
 *  @returns {Function} f_proj(t, projState, params) → derivatives of displayed vars
 */
function makeProjectedRHS(modelName, displayIdx, clamped) {
  const fullF = MODELS[modelName].f;
  const n = (MODELS[modelName].stateVars || MODELS[modelName].stateNames || []).length;
  const clampedCopy = [...clamped];
  return function (t, projState, params) {
    const fullState = new Array(n);
    for (let i = 0; i < n; i++) fullState[i] = clampedCopy[i];
    for (let i = 0; i < displayIdx.length; i++) {
      fullState[displayIdx[i]] = projState[i];
    }
    const dFull = fullF(t, fullState, params);
    const result = [];
    for (const idx of displayIdx) result.push(dFull[idx]);
    return result;
  };
}

/** Fixed-step RK4 ODE solver.
 *  Returns array of [t, v0, v1, ...] rows.  */
function rk4(modelName, y0, t0, tf, h, params) {
  const f = MODELS[modelName].f;
  const results = [[t0, ...y0]];
  let t = t0;
  let y = [...y0];
  while (t < tf - 1e-12) {
    const dt = Math.min(h, tf - t);
    const k1 = f(t, y, params);
    const k2 = f(t + dt / 2, y.map((yi, i) => yi + (dt / 2) * k1[i]), params);
    const k3 = f(t + dt / 2, y.map((yi, i) => yi + (dt / 2) * k2[i]), params);
    const k4 = f(t + dt, y.map((yi, i) => yi + dt * k3[i]), params);
    y = y.map((yi, i) => yi + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    t += dt;
    // Check for NaN / blow-up and abort gracefully
    if (y.some((v) => !isFinite(v))) break;
    results.push([t, ...y]);
  }
  return results;
}

/** Numerical Jacobian (works for 1-D or 2-D projected systems). */
function jacobianND(f, state, params, eps = 1e-6) {
  const dim = state.length;
  const f0 = f(0, state, params);
  const J = Array.from({ length: dim }, () => new Array(dim).fill(0));
  for (let i = 0; i < dim; i++) {
    const sPlus = [...state];
    sPlus[i] += eps;
    const fPlus = f(0, sPlus, params);
    for (let j = 0; j < dim; j++) {
      J[j][i] = (fPlus[j] - f0[j]) / eps;
    }
  }
  return J;
}

/** Eigenvalues of a 2×2 real matrix. */
function eigenvalues2x2(A) {
  const a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1];
  const trace = a + d;
  const det = a * d - b * c;
  const disc = trace * trace - 4 * det;
  if (disc >= 0) {
    const s = Math.sqrt(disc);
    return [(trace + s) / 2, (trace - s) / 2];
  }
  const s = Math.sqrt(-disc);
  return [
    { re: trace / 2, im: s / 2 },
    { re: trace / 2, im: -s / 2 },
  ];
}

/** Classify fixed point from eigenvalues. */
function classifyFixedPoint(eigenvalues) {
  const real = eigenvalues.map((ev) => (typeof ev === "object" ? ev.re : ev));
  const imag = eigenvalues.map((ev) => (typeof ev === "object" ? ev.im : 0));
  const allNeg = real.every((r) => r < -1e-6);
  const allPos = real.every((r) => r > 1e-6);
  const hasImag = imag.some((i) => Math.abs(i) > 1e-6);
  if (allNeg) return hasImag ? "stable_focus" : "stable_node";
  if (allPos) return hasImag ? "unstable_focus" : "unstable_node";
  return "saddle";
}

/** Grid-search + Newton-Raphson fixed-point finder (2D). */
function findFixedPoints(f, params, xlim, ylim, nGrid = 25) {
  const tol = 0.08;
  const fixedPoints = [];
  const xs = linspace(xlim[0], xlim[1], nGrid);
  const ys = linspace(ylim[0], ylim[1], nGrid);

  for (const xi of xs) {
    for (const yi of ys) {
      let x = [xi, yi];
      for (let iter = 0; iter < 50; iter++) {
        const fx = f(0, x, params);
        const norm = Math.sqrt(fx[0] ** 2 + fx[1] ** 2);
        if (norm < 1e-6) {
          // Converged — classify and store
          if (
            xlim[0] - 0.5 <= x[0] &&
            x[0] <= xlim[1] + 0.5 &&
            ylim[0] - 0.5 <= x[1] &&
            x[1] <= ylim[1] + 0.5 &&
            norm <= 0.1
          ) {
            let isNew = true;
            for (const ex of fixedPoints) {
              const dist = Math.sqrt((x[0] - ex[0]) ** 2 + (x[1] - ex[1]) ** 2);
              if (dist < tol) {
                isNew = false;
                break;
              }
            }
            if (isNew) {
              const J = jacobianND(f, x, params);
              const ev = eigenvalues2x2(J);
              fixedPoints.push([x[0], x[1], classifyFixedPoint(ev)]);
            }
          }
          break;
        }
        const J = jacobianND(f, x, params, 1e-6);
        const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        if (Math.abs(det) < 1e-12) break;
        const dx0 = (-fx[0] * J[1][1] + fx[1] * J[0][1]) / det;
        const dx1 = (fx[0] * J[1][0] - fx[1] * J[0][0]) / det;
        x = [x[0] + dx0, x[1] + dx1];
      }
    }
  }
  return fixedPoints;
}

/** Nullcline computation via grid zero-crossings. */
function computeNullclines(f, xlim, ylim, nGrid = 60) {
  // f is a projected RHS function
  const x = linspace(xlim[0], xlim[1], nGrid);
  const y = linspace(ylim[0], ylim[1], nGrid);
  const dx = Array(nGrid).fill(0).map(() => Array(nGrid).fill(0));
  const dy = Array(nGrid).fill(0).map(() => Array(nGrid).fill(0));

  for (let i = 0; i < nGrid; i++) {
    for (let j = 0; j < nGrid; j++) {
      const d = f(0, [x[j], y[i]], params);
      dx[i][j] = d[0];
      dy[i][j] = d[1];
    }
  }

  function _findZeroCrossings(Z) {
    const points = [];
    const n = Z.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n - 1; j++) {
        if (Z[i][j] === 0) points.push([x[j], y[i]]);
        else if (Z[i][j] * Z[i][j + 1] < 0) {
          const t = Math.abs(Z[i][j]) / (Math.abs(Z[i][j]) + Math.abs(Z[i][j + 1]));
          points.push([x[j] + t * (x[j + 1] - x[j]), y[i]]);
        }
      }
    }
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n; j++) {
        if (Z[i][j] === 0) points.push([x[j], y[i]]);
        else if (Z[i][j] * Z[i + 1][j] < 0) {
          const t = Math.abs(Z[i][j]) / (Math.abs(Z[i][j]) + Math.abs(Z[i + 1][j]));
          points.push([x[j], y[i] + t * (y[i + 1] - y[i])]);
        }
      }
    }
    return points;
  }

  return [_findZeroCrossings(dx), _findZeroCrossings(dy)];
}

/** Sparse vector field. */
function computeVectorField(f, xlim, ylim, nGrid = 12) {
  // f is a projected RHS function
  const xs = linspace(xlim[0], xlim[1], nGrid);
  const ys = linspace(ylim[0], ylim[1], nGrid);
  const vectors = [];
  for (const xi of xs) {
    for (const yi of ys) {
      const d = f(0, [xi, yi], params);
      vectors.push([xi, yi, d[0], d[1]]);
    }
  }
  return vectors;
}

/** Detect regime by simulation from multiple ICs. */
function detectRegime(modelName, params, xlim, ylim, display, tTotal = 120, dt = 0.05) {
  const is1D = !ylim || display.length === 1;
  const ics = is1D
    ? [[xlim[0] * 0.6], [xlim[1] * 0.6], [(xlim[0] + xlim[1]) * 0.5]]
    : [
        [xlim[0] * 0.6, ylim[0] * 0.6],
        [xlim[1] * 0.6, ylim[1] * 0.6],
        [(xlim[0] + xlim[1]) * 0.5, (ylim[0] + ylim[1]) * 0.5],
      ];
  const regimes = [];
  for (const ic of ics) {
    const traj = rk4(modelName, ic, 0, tTotal, dt, params);
    if (traj.length < 20) { regimes.push("other"); continue; }
    const nCheck = Math.min(200, Math.floor(traj.length / 4));
    if (nCheck < 10) { regimes.push("other"); continue; }
    const last = traj.slice(-nCheck);
    const xVals = last.map((r) => r[display[0] + 1]);
    const meanX = xVals.reduce((a, b) => a + b) / xVals.length;
    const stdX = Math.sqrt(xVals.reduce((s, v) => s + (v - meanX) ** 2, 0) / xVals.length);
    if (is1D) {
      // 1-D: no limit cycles possible
      regimes.push(stdX < 0.025 ? "fixed_point" : "other");
    } else {
      const yVals = last.map((r) => r[display[1] + 1]);
      const meanY = yVals.reduce((a, b) => a + b) / yVals.length;
      const stdY = Math.sqrt(yVals.reduce((s, v) => s + (v - meanY) ** 2, 0) / yVals.length);
      if (stdX < 0.025 && stdY < 0.025) {
        regimes.push("fixed_point");
      } else {
        const mid = Math.floor(last.length / 2);
        const amp1x = Math.max(...xVals.slice(0, mid)) - Math.min(...xVals.slice(0, mid));
        const amp2x = Math.max(...xVals.slice(mid)) - Math.min(...xVals.slice(mid));
        const amp1y = Math.max(...yVals.slice(0, mid)) - Math.min(...yVals.slice(0, mid));
        const amp2y = Math.max(...yVals.slice(mid)) - Math.min(...yVals.slice(mid));
        const xStable = Math.abs(amp1x - amp2x) < 0.15 * Math.max(amp1x, 0.01);
        const yStable = Math.abs(amp1y - amp2y) < 0.15 * Math.max(amp1y, 0.01);
        regimes.push(xStable && yStable ? "limit_cycle" : "other");
      }
    }
  }
  const counts = {};
  for (const r of regimes) counts[r] = (counts[r] || 0) + 1;
  let best = "other", bestCount = 0;
  for (const [k, v] of Object.entries(counts)) {
    if (v > bestCount) { bestCount = v; best = k; }
  }
  return best;
}

/** Parameter sweep in pure JS. */
function runParameterSweep(f, modelName, sweepParam, params, vmin, vmax, n, xlim, ylim, display) {
  const values = linspace(vmin, vmax, n);
  const results = [];
  const allFps = [];
  for (const val of values) {
    const p = { ...params, [sweepParam]: val };
    const regime = detectRegime(modelName, p, xlim, ylim, display, 60, 0.05);
    const fps = findFixedPoints(f, p, xlim, ylim, 15);
    results.push({ param_value: val, regime, num_fixed_points: fps.length });
    for (const fp of fps) allFps.push([val, fp[0], fp[1], fp[2]]);
  }
  return { results, allFps };
}

// ═══════════════════════════════════════════════════════════════
//  CANVAS RENDERING  (unchanged coordinate helpers & draw calls)
// ═══════════════════════════════════════════════════════════════

function worldToScreen(wx, wy, xlim, ylim, w, h) {
  return [((wx - xlim[0]) / (xlim[1] - xlim[0])) * w, h - ((wy - ylim[0]) / (ylim[1] - ylim[0])) * h];
}

function screenToWorld(sx, sy, xlim, ylim, w, h) {
  return [xlim[0] + (sx / w) * (xlim[1] - xlim[0]), ylim[0] + ((h - sy) / h) * (ylim[1] - ylim[0])];
}

function drawAxes(ctx, w, h, xlim, ylim, xlabel, ylabel) {
  const pad = 30;
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  const zeroY = worldToScreen(0, 0, xlim, ylim, w, h)[1];
  if (zeroY >= pad && zeroY <= h - pad) {
    ctx.beginPath(); ctx.moveTo(pad, zeroY); ctx.lineTo(w - pad, zeroY); ctx.stroke();
  }
  const zeroX = worldToScreen(0, 0, xlim, ylim, w, h)[0];
  if (zeroX >= pad && zeroX <= w - pad) {
    ctx.beginPath(); ctx.moveTo(zeroX, pad); ctx.lineTo(zeroX, h - pad); ctx.stroke();
  }
  ctx.strokeStyle = "#e9ecef"; ctx.lineWidth = 0.5;
  const nx = 5, ny = 5;
  for (let i = 0; i <= nx; i++) {
    const x = xlim[0] + (i / nx) * (xlim[1] - xlim[0]);
    const sx = worldToScreen(x, 0, xlim, ylim, w, h)[0];
    ctx.beginPath(); ctx.moveTo(sx, pad); ctx.lineTo(sx, h - pad); ctx.stroke();
  }
  for (let i = 0; i <= ny; i++) {
    const y = ylim[0] + (i / ny) * (ylim[1] - ylim[0]);
    const sy = worldToScreen(0, y, xlim, ylim, w, h)[1];
    ctx.beginPath(); ctx.moveTo(pad, sy); ctx.lineTo(w - pad, sy); ctx.stroke();
  }
  ctx.fillStyle = "#666"; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= nx; i++) {
    const x = xlim[0] + (i / nx) * (xlim[1] - xlim[0]);
    ctx.fillText(x.toFixed(1), worldToScreen(x, ylim[0], xlim, ylim, w, h)[0], h - pad + 2);
  }
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= ny; i++) {
    const y = ylim[0] + (i / ny) * (ylim[1] - ylim[0]);
    ctx.fillText(y.toFixed(1), pad - 4, worldToScreen(xlim[0], y, xlim, ylim, w, h)[1]);
  }
  if (xlabel) {
    ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.font = "bold 11px sans-serif";
    ctx.fillText(xlabel, w / 2, h - 4);
  }
  if (ylabel) {
    ctx.save(); ctx.translate(12, h / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = "bold 11px sans-serif";
    ctx.fillText(ylabel, 0, 0); ctx.restore();
  }
}

function drawArrow(ctx, fromX, fromY, toX, toY, color = "#bbb", headLen = 4) {
  const dx = toX - fromX, dy = toY - fromY, angle = Math.atan2(dy, dx);
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(fromX, fromY); ctx.lineTo(toX, toY); ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLen * Math.cos(angle - Math.PI / 6), toY - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(toX - headLen * Math.cos(angle + Math.PI / 6), toY - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath(); ctx.fill();
}

// ═══════════════════════════════════════════════════════════════
//  MAIN RENDER FUNCTION
// ═══════════════════════════════════════════════════════════════

export function render({ model, el }) {
  // Detect standalone mode (no real anywidget kernel behind the model)
  const isStandalone = model._isMock === true || typeof model.save_changes !== "function";

  // ── Create DOM ──
  el.innerHTML = `
    <div class="ppw-widget">
      <div class="ppw-controls">
        <div class="ppw-control-row">
          <label class="ppw-label">Model:
            <select class="ppw-model-select"></select>
          </label>
        </div>
        <div class="ppw-control-row ppw-state-selector" style="display:none">
          <label class="ppw-label">X axis:
            <select class="ppw-state-x"></select>
          </label>
          <label class="ppw-label">Y axis:
            <select class="ppw-state-y"></select>
          </label>
        </div>
        <div class="ppw-clamped-sliders"></div>
        <div class="ppw-control-row ppw-limits">
          <label class="ppw-label">X:
            <input class="ppw-xmin" type="number" step="0.1">
            <input class="ppw-xmax" type="number" step="0.1">
          </label>
          <label class="ppw-label">Y:
            <input class="ppw-ymin" type="number" step="0.1">
            <input class="ppw-ymax" type="number" step="0.1">
          </label>
          <label class="ppw-label">Tmax:
            <input class="ppw-tmax" type="number" step="10" min="1">
          </label>
        </div>
        <div class="ppw-control-row ppw-toggles">
          <label><input type="checkbox" class="ppw-show-null" checked> Nullclines</label>
          <label><input type="checkbox" class="ppw-show-vf" checked> Vector Field</label>
          <label><input type="checkbox" class="ppw-show-traj" checked> Trajectory</label>
          <label><input type="checkbox" class="ppw-show-fp" checked> Fixed Points</label>
        </div>
      </div>
      <div class="ppw-plots">
        <div class="ppw-phase-container">
          <div class="ppw-canvas-title">
            <span>Phase Plane</span>
            <span class="ppw-canvas-hint">Click to set initial condition</span>
          </div>
          <canvas class="ppw-phase-canvas" width="500" height="500"></canvas>
          <div class="ppw-legend">
            <div class="ppw-legend-item"><div class="ppw-legend-line" style="background:#2196F3"></div> dX/dt = 0</div>
            <div class="ppw-legend-item"><div class="ppw-legend-line" style="background:#f44336"></div> dY/dt = 0</div>
            <div class="ppw-legend-item"><div class="ppw-legend-line" style="background:#4CAF50"></div> Trajectory</div>
            <div class="ppw-legend-item"><div class="ppw-legend-dot" style="background:#4CAF50"></div> Stable</div>
            <div class="ppw-legend-item"><div class="ppw-legend-dot" style="background:#f44336"></div> Unstable</div>
            <div class="ppw-legend-item"><div class="ppw-legend-dot" style="background:#9C27B0"></div> Saddle</div>
          </div>
        </div>
        <div class="ppw-time-container">
          <div class="ppw-canvas-title">Time Series</div>
          <canvas class="ppw-time-canvas" width="500" height="280"></canvas>
        </div>
      </div>
      <div class="ppw-sweep">
        <div class="ppw-sweep-header">Bifurcation / Parameter Sweep</div>
        <div class="ppw-sweep-controls">
          <label>Sweep parameter:
            <select class="ppw-sweep-select"></select>
          </label>
          <label>Min: <input class="ppw-sweep-min" type="number"></label>
          <label>Max: <input class="ppw-sweep-max" type="number"></label>
          <label>N: <input class="ppw-sweep-n" type="number" value="50" min="5" max="200"></label>
          <button class="ppw-sweep-btn">Run Sweep</button>
          <span class="ppw-sweep-status"></span>
        </div>
        <canvas class="ppw-sweep-canvas" width="700" height="300"></canvas>
        <div class="ppw-legend">
          <div class="ppw-legend-item"><div class="ppw-legend-dot" style="background:#4CAF50"></div> Stable</div>
          <div class="ppw-legend-item"><div class="ppw-legend-dot" style="background:#f44336"></div> Unstable</div>
          <div class="ppw-legend-item"><div class="ppw-legend-dot" style="background:#9C27B0"></div> Saddle</div>
        </div>
      </div>
    </div>
  `;

  // ── DOM refs ──
  const modelSelect = el.querySelector(".ppw-model-select");
  const paramsDiv = el.querySelector(".ppw-params");
  const xMinIn = el.querySelector(".ppw-xmin");
  const xMaxIn = el.querySelector(".ppw-xmax");
  const yMinIn = el.querySelector(".ppw-ymin");
  const yMaxIn = el.querySelector(".ppw-ymax");
  const tMaxIn = el.querySelector(".ppw-tmax");
  const phaseCanvas = el.querySelector(".ppw-phase-canvas");
  const timeCanvas = el.querySelector(".ppw-time-canvas");
  const sweepSelect = el.querySelector(".ppw-sweep-select");
  const sweepMinIn = el.querySelector(".ppw-sweep-min");
  const sweepMaxIn = el.querySelector(".ppw-sweep-max");
  const sweepNIn = el.querySelector(".ppw-sweep-n");
  const sweepBtn = el.querySelector(".ppw-sweep-btn");
  const sweepStatus = el.querySelector(".ppw-sweep-status");
  const sweepCanvas = el.querySelector(".ppw-sweep-canvas");
  const showNull = el.querySelector(".ppw-show-null");
  const showVf = el.querySelector(".ppw-show-vf");
  const showTraj = el.querySelector(".ppw-show-traj");
  const showFp = el.querySelector(".ppw-show-fp");

  const phaseCtx = phaseCanvas.getContext("2d");
  const timeCtx = timeCanvas.getContext("2d");
  const sweepCtx = sweepCanvas.getContext("2d");
  const stateXSelect = el.querySelector(".ppw-state-x");
  const stateYSelect = el.querySelector(".ppw-state-y");
  const stateSelectorRow = el.querySelector(".ppw-state-selector");
  const clampedDiv = el.querySelector(".ppw-clamped-sliders");

  const MODEL_NAMES = ["wilson_cowan", "fitzhugh_nagumo", "mpr"];
  const MODEL_LABELS = { wilson_cowan: "Wilson-Cowan", fitzhugh_nagumo: "FitzHugh-Nagumo", mpr: "MPR (QIF)" };

  // ═════════════════════════════════════════════════════════════
  //  CLIENT-SIDE COMPUTATION ENGINE
  // ═════════════════════════════════════════════════════════════

  function computeAll() {
    const modelName = model.get("model_name");
    const params = model.get("params");
    const xlim = model.get("xlim");
    const ylim = model.get("ylim");
    const tMax = model.get("t_max");
    const x0 = model.get("x0");
    const y0 = model.get("y0");

    const def = MODELS[modelName];
    const stateVars = def.stateVars || def.stateNames || [];
    const n = stateVars.length;

    // Display indices and clamped (non-displayed) state variable values
    const display = model.get("display") || def.display || [0, 1];
    let clamped = model.get("clamped");
    if (!clamped || clamped.length !== n) {
      clamped = stateVars.map((_, i) => {
        const lims = (def.defaultLims && def.defaultLims[i]) || [-1, 1];
        return (lims[0] + lims[1]) / 2;
      });
      model.set("clamped", clamped);
    }

    // Projected RHS for phase-plane computations
    const fProj = makeProjectedRHS(modelName, display, clamped);

    // Nullclines (projected 2-D slice)
    const [ncX, ncY] = computeNullclines(fProj, xlim, ylim, 60);
    model.set("nullcline_x", ncX);
    model.set("nullcline_y", ncY);

    // Vector field (projected)
    const vf = computeVectorField(fProj, xlim, ylim, 12);
    model.set("vector_field", vf);

    // Fixed points (projected slice)
    const fps = findFixedPoints(fProj, params, xlim, ylim, 25);
    model.set("fixed_points", fps);

    // Trajectory: integrate the full N-dimensional system
    const fullState0 = [...clamped];
    fullState0[display[0]] = x0;
    if (display.length > 1) fullState0[display[1]] = y0;
    const traj = rk4(modelName, fullState0, 0, tMax, 0.01, params);
    const step = Math.max(1, Math.floor(traj.length / 2000));
    const trajDisplay = traj.filter((_, i) => i % step === 0);
    model.set("trajectory", trajDisplay);

    // Render
    renderPhasePlane();
    renderTimeSeries();

    // Sync to Python if in Jupyter mode
    if (!isStandalone) model.save_changes();
  }

  // ── Model selector ──
  function populateModelSelector() {
    const current = model.get("model_name");
    modelSelect.innerHTML = MODEL_NAMES.map(
      (m) => `<option value="${m}" ${m === current ? "selected" : ""}>${MODEL_LABELS[m]}</option>`,
    ).join("");
  }

  modelSelect.addEventListener("change", () => {
    const newModel = modelSelect.value;
    const def = MODELS[newModel];
    model.set("model_name", newModel);
    model.set("param_info", def.paramInfo);
    model.set("state_names", def.stateVars || def.stateNames);
    model.set("params", { ...def.defaultParams });

    // Reset display and clamped for new model
    const display = def.display || [0, 1];
    model.set("display", display);
    const stateVars = def.stateVars || def.stateNames || [];
    const clamped = stateVars.map((_, i) => {
      const lims = (def.defaultLims && def.defaultLims[i]) || [-1, 1];
      return (lims[0] + lims[1]) / 2;
    });
    model.set("clamped", clamped);

    // Use defaultLims for displayed variables, fall back to defaultXlim/defaultYlim
    if (def.defaultLims && def.defaultLims[display[0]]) {
      model.set("xlim", [...def.defaultLims[display[0]]]);
    } else {
      model.set("xlim", [...def.defaultXlim]);
    }
    if (display.length > 1 && def.defaultLims && def.defaultLims[display[1]]) {
      model.set("ylim", [...def.defaultLims[display[1]]]);
    } else {
      model.set("ylim", [...def.defaultYlim]);
    }
    model.set("x0", model.get("xlim")[0] + 0.1 * (model.get("xlim")[1] - model.get("xlim")[0]));
    model.set("y0", model.get("ylim")[0] + 0.1 * (model.get("ylim")[1] - model.get("ylim")[0]));
    if (!isStandalone) model.save_changes();
    createParamSliders(def.paramInfo, def.defaultParams);
    updateLimitInputs();
    populateStateSelectors(def, display);
    createClampedSliders(def, clamped, display);
    computeAll();
  });

  // ── Parameter sliders ──
  function createParamSliders(paramInfo, currentParams) {
    paramsDiv.innerHTML = "";
    if (!paramInfo || Object.keys(paramInfo).length === 0) return;
    for (const [name, [min, max, defaultVal, desc]] of Object.entries(paramInfo)) {
      const value = currentParams[name] !== undefined ? currentParams[name] : defaultVal;
      const div = document.createElement("div");
      div.className = "ppw-param";
      div.innerHTML = `
        <label title="${desc}">${name}: <span class="ppw-param-value">${value.toFixed(3)}</span></label>
        <input type="range" class="ppw-param-slider" data-param="${name}"
          min="${min}" max="${max}" step="${(max - min) / 500}" value="${value}">
      `;
      paramsDiv.appendChild(div);
    }
    paramsDiv.querySelectorAll(".ppw-param-slider").forEach((slider) => {
      let timeout;
      slider.addEventListener("input", () => {
        const name = slider.dataset.param;
        const val = parseFloat(slider.value);
        slider.parentElement.querySelector(".ppw-param-value").textContent = val.toFixed(3);
        clearTimeout(timeout);
        timeout = setTimeout(() => {
          const params = { ...model.get("params") };
          params[name] = val;
          model.set("params", params);
          if (!isStandalone) model.save_changes();
          computeAll();
        }, 30);
      });
    });
    // Update sweep dropdown
    const keys = Object.keys(paramInfo);
    sweepSelect.innerHTML = keys.map((k) => `<option value="${k}">${k}</option>`).join("");
    if (keys.length > 0) {
      const [min, max] = paramInfo[keys[0]];
      sweepMinIn.value = min;
      sweepMaxIn.value = max;
    }
  }

  // ── State variable selector ──
  function populateStateSelectors(def, display) {
    const stateVars = def.stateVars || def.stateNames || [];
    const n = stateVars.length;
    if (n <= 2) {
      stateSelectorRow.style.display = "none";
      return;
    }
    stateSelectorRow.style.display = "flex";
    stateXSelect.innerHTML = stateVars.map((name, i) =>
      `<option value="${i}" ${display[0] === i ? "selected" : ""}>${name}</option>`
    ).join("");
    stateYSelect.innerHTML = stateVars.map((name, i) =>
      `<option value="${i}" ${display[1] === i ? "selected" : ""}>${name}</option>`
    ).join("");
  }

  stateXSelect.addEventListener("change", () => {
    const xIdx = parseInt(stateXSelect.value);
    const yIdx = parseInt(stateYSelect.value);
    if (xIdx === yIdx) {
      stateYSelect.value = (xIdx + 1) % (MODELS[model.get("model_name")].stateVars || []).length;
    }
    const display = [parseInt(stateXSelect.value), parseInt(stateYSelect.value)];
    model.set("display", display);
    if (!isStandalone) model.save_changes();
    computeAll();
  });

  stateYSelect.addEventListener("change", () => {
    const xIdx = parseInt(stateXSelect.value);
    const yIdx = parseInt(stateYSelect.value);
    if (xIdx === yIdx) {
      stateXSelect.value = (yIdx + 1) % (MODELS[model.get("model_name")].stateVars || []).length;
    }
    const display = [parseInt(stateXSelect.value), parseInt(stateYSelect.value)];
    model.set("display", display);
    if (!isStandalone) model.save_changes();
    computeAll();
  });

  // ── Clamped state variable sliders ──
  function createClampedSliders(def, clamped, display) {
    const stateVars = def.stateVars || def.stateNames || [];
    const n = stateVars.length;
    if (n <= 2) {
      clampedDiv.style.display = "none";
      clampedDiv.innerHTML = "";
      return;
    }
    clampedDiv.style.display = "block";
    const indices = [];
    for (let i = 0; i < n; i++) {
      if (!display.includes(i)) indices.push(i);
    }
    if (indices.length === 0) {
      clampedDiv.style.display = "none";
      return;
    }
    const lims = def.defaultLims || Array(n).fill([-5, 5]);
    let html = '<div style="font-size:11px;color:#666;margin-bottom:4px;">Fixed (non-displayed) state variables:</div>';
    for (const idx of indices) {
      const name = stateVars[idx];
      const [lo, hi] = lims[idx] || [-5, 5];
      const val = clamped[idx] !== undefined ? clamped[idx] : (lo + hi) / 2;
      html += `
        <div class="ppw-param">
          <label>${name}: <span class="ppw-clamped-value">${val.toFixed(3)}</span></label>
          <input type="range" class="ppw-clamped-slider" data-var="${idx}"
            min="${lo}" max="${hi}" step="${(hi - lo) / 500}" value="${val}">
        </div>`;
    }
    clampedDiv.innerHTML = html;
    clampedDiv.querySelectorAll(".ppw-clamped-slider").forEach((slider) => {
      slider.addEventListener("input", () => {
        const idx = parseInt(slider.dataset.var);
        const val = parseFloat(slider.value);
        slider.parentElement.querySelector(".ppw-clamped-value").textContent = val.toFixed(3);
        const clamped = [...(model.get("clamped") || [])];
        clamped[idx] = val;
        model.set("clamped", clamped);
        if (!isStandalone) model.save_changes();
        computeAll();
      });
    });
  }

  // ── Limit / tmax inputs ──
  function updateLimitInputs() {
    const xlim = model.get("xlim");
    const ylim = model.get("ylim");
    xMinIn.value = xlim[0]; xMaxIn.value = xlim[1];
    yMinIn.value = ylim[0]; yMaxIn.value = ylim[1];
    tMaxIn.value = model.get("t_max");
  }

  function sendLimits() {
    model.set("xlim", [parseFloat(xMinIn.value), parseFloat(xMaxIn.value)]);
    model.set("ylim", [parseFloat(yMinIn.value), parseFloat(yMaxIn.value)]);
    model.set("t_max", parseFloat(tMaxIn.value));
    if (!isStandalone) model.save_changes();
    computeAll();
  }
  [xMinIn, xMaxIn, yMinIn, yMaxIn, tMaxIn].forEach((inp) => inp.addEventListener("change", sendLimits));

  // ── Toggles ──
  [showNull, showVf, showTraj, showFp].forEach((cb) => {
    cb.addEventListener("change", () => {
      model.set("show_nullclines", showNull.checked);
      model.set("show_vector_field", showVf.checked);
      model.set("show_trajectory", showTraj.checked);
      model.set("show_fixed_points", showFp.checked);
      if (!isStandalone) model.save_changes();
      renderPhasePlane();
    });
  });

  // ── Phase plane click ──
  phaseCanvas.addEventListener("click", (e) => {
    const rect = phaseCanvas.getBoundingClientRect();
    const scaleX = phaseCanvas.width / rect.width;
    const scaleY = phaseCanvas.height / rect.height;
    const sx = (e.clientX - rect.left) * scaleX;
    const sy = (e.clientY - rect.top) * scaleY;
    const xlim = model.get("xlim");
    const ylim = model.get("ylim");
    const [wx, wy] = screenToWorld(sx, sy, xlim, ylim, phaseCanvas.width, phaseCanvas.height);
    model.set("x0", wx);
    model.set("y0", wy);
    if (!isStandalone) model.save_changes();
    computeAll();
  });

  // ── Sweep ──
  sweepBtn.addEventListener("click", () => {
    const param = sweepSelect.value;
    const min = parseFloat(sweepMinIn.value);
    const max = parseFloat(sweepMaxIn.value);
    const n = parseInt(sweepNIn.value);
    if (isNaN(min) || isNaN(max) || isNaN(n) || n < 2) return;

    sweepBtn.disabled = true;
    sweepStatus.innerHTML = '<span class="ppw-sweep-spinner"></span> Running...';

    setTimeout(() => {
      const modelName = model.get("model_name");
      const params = model.get("params");
      const xlim = model.get("xlim");
      const ylim = model.get("ylim");
      const def = MODELS[modelName];
      const display = model.get("display") || def.display || [0, 1];
      const stateVars = def.stateVars || def.stateNames || [];
      const n = stateVars.length;
      let clamped = model.get("clamped");
      if (!clamped || clamped.length !== n) {
        clamped = stateVars.map((_, i) => {
          const lims = (def.defaultLims && def.defaultLims[i]) || [-1, 1];
          return (lims[0] + lims[1]) / 2;
        });
      }
      const fProj = makeProjectedRHS(modelName, display, clamped);
      const result = runParameterSweep(fProj, modelName, param, params, min, max, n, xlim, ylim, display);
      model.set("sweep_results", result.results);
      model.set("sweep_fixed_points", result.allFps);
      model.set("sweep_param", param);
      if (!isStandalone) model.save_changes();
      renderSweep();
      sweepBtn.disabled = false;
      sweepStatus.innerHTML = "";
    }, 10);
  });

  // ═════════════════════════════════════════════════════════════
  //  RENDERERS (same canvas drawing as before)
  // ═════════════════════════════════════════════════════════════

  function renderPhasePlane() {
    const w = phaseCanvas.width;
    const h = phaseCanvas.height;
    const xlim = model.get("xlim");
    const ylim = model.get("ylim");
    const stateNames = model.get("state_names");
    const def = MODELS[model.get("model_name")];
    const display = model.get("display") || def.display || [0, 1];
    const is1D = display.length === 1;

    phaseCtx.clearRect(0, 0, w, h);
    phaseCtx.fillStyle = "#fafafa";
    phaseCtx.fillRect(0, 0, w, h);

    if (is1D) {
      // 1-D mode: draw dx/dt vs x
      drawAxes(phaseCtx, w, h, xlim, [-2, 2], stateNames[display[0]], "d" + stateNames[display[0]] + "/dt");
    } else {
      drawAxes(phaseCtx, w, h, xlim, ylim, stateNames[display[0]], stateNames[display[1]]);
    }

    // Nullclines
    if (model.get("show_nullclines")) {
      const ncX = model.get("nullcline_x");
      if (ncX && ncX.length > 0) {
        phaseCtx.fillStyle = "#2196F3";
        for (const [wx, wy] of ncX) {
          const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
          phaseCtx.beginPath(); phaseCtx.arc(sx, sy, 1.8, 0, Math.PI * 2); phaseCtx.fill();
        }
      }
      const ncY = model.get("nullcline_y");
      if (ncY && ncY.length > 0) {
        phaseCtx.fillStyle = "#f44336";
        for (const [wx, wy] of ncY) {
          const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
          phaseCtx.beginPath(); phaseCtx.arc(sx, sy, 1.8, 0, Math.PI * 2); phaseCtx.fill();
        }
      }
    }

    // Vector field
    if (model.get("show_vector_field")) {
      const vf = model.get("vector_field");
      if (vf && vf.length > 0) {
        for (const [wx, wy, dx, dy] of vf) {
          const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
          const screenDx = (dx / (xlim[1] - xlim[0])) * w;
          const screenDy = (-dy / (ylim[1] - ylim[0])) * h;
          const mag = Math.sqrt(screenDx * screenDx + screenDy * screenDy);
          if (mag < 1e-10) continue;
          const ndx = screenDx / mag, ndy = screenDy / mag;
          const arrowLen = Math.min(14, mag * 0.3);
          drawArrow(phaseCtx, sx, sy, sx + ndx * arrowLen, sy + ndy * arrowLen, "#ccc", 3);
        }
      }
    }

    // Fixed points
    if (model.get("show_fixed_points")) {
      const fps = model.get("fixed_points");
      if (fps && fps.length > 0) {
        for (const [wx, wy, stability] of fps) {
          const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
          const color = STABILITY_COLORS[stability] || "#666";
          phaseCtx.beginPath(); phaseCtx.arc(sx, sy, 7, 0, Math.PI * 2);
          phaseCtx.fillStyle = color; phaseCtx.fill();
          phaseCtx.strokeStyle = "#333"; phaseCtx.lineWidth = 1.5; phaseCtx.stroke();
        }
      }
    }

    // Trajectory
    if (model.get("show_trajectory")) {
      const traj = model.get("trajectory");
      if (traj && traj.length > 1) {
        phaseCtx.strokeStyle = "#4CAF50"; phaseCtx.lineWidth = 1.5;
        phaseCtx.beginPath();
        const proj0 = [traj[0][display[0] + 1]];
        if (!is1D) proj0.push(traj[0][display[1] + 1]);
        const [sx0, sy0] = is1D
          ? worldToScreen(proj0[0], 0, xlim, [-2, 2], w, h)
          : worldToScreen(proj0[0], proj0[1], xlim, ylim, w, h);
        phaseCtx.moveTo(sx0, sy0);
        for (let i = 1; i < traj.length; i++) {
          const proj = [traj[i][display[0] + 1]];
          if (!is1D) proj.push(traj[i][display[1] + 1]);
          const [sx, sy] = is1D
            ? worldToScreen(proj[0], 0, xlim, [-2, 2], w, h)
            : worldToScreen(proj[0], proj[1], xlim, ylim, w, h);
          phaseCtx.lineTo(sx, sy);
        }
        phaseCtx.stroke();
        // Initial condition marker
        phaseCtx.beginPath(); phaseCtx.arc(sx0, sy0, 5, 0, Math.PI * 2);
        phaseCtx.fillStyle = "#FF9800"; phaseCtx.fill();
        phaseCtx.strokeStyle = "#333"; phaseCtx.lineWidth = 1; phaseCtx.stroke();
      }
    }
  }

  function renderTimeSeries() {
    const w = timeCanvas.width, h = timeCanvas.height;
    const traj = model.get("trajectory");
    const stateNames = model.get("state_names");
    timeCtx.clearRect(0, 0, w, h);
    timeCtx.fillStyle = "#fafafa"; timeCtx.fillRect(0, 0, w, h);
    if (!traj || traj.length < 2) return;

    const tVals = traj.map((r) => r[0]);
    const tMin = tVals[0], tMax = tVals[tVals.length - 1];
    const pad = 30, plotW = w - 2 * pad, plotH = h - 2 * pad;

    let yMin = Infinity, yMax = -Infinity;
    for (let d = 0; d < stateNames.length; d++) {
      const vals = traj.map((r) => r[d + 1]);
      yMin = Math.min(yMin, ...vals); yMax = Math.max(yMax, ...vals);
    }
    if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
    const yMargin = (yMax - yMin) * 0.1;
    yMin -= yMargin; yMax += yMargin;

    // Grid
    timeCtx.strokeStyle = "#e9ecef"; timeCtx.lineWidth = 0.5;
    const nt = 5, ny = 4;
    for (let i = 0; i <= nt; i++) {
      const t = tMin + (i / nt) * (tMax - tMin);
      const sx = pad + ((t - tMin) / (tMax - tMin)) * plotW;
      timeCtx.beginPath(); timeCtx.moveTo(sx, pad); timeCtx.lineTo(sx, h - pad); timeCtx.stroke();
    }
    for (let i = 0; i <= ny; i++) {
      const y = yMin + (i / ny) * (yMax - yMin);
      const sy = h - pad - ((y - yMin) / (yMax - yMin)) * plotH;
      timeCtx.beginPath(); timeCtx.moveTo(pad, sy); timeCtx.lineTo(w - pad, sy); timeCtx.stroke();
    }

    // Axes
    timeCtx.strokeStyle = "#333"; timeCtx.lineWidth = 1;
    timeCtx.beginPath(); timeCtx.moveTo(pad, pad); timeCtx.lineTo(pad, h - pad); timeCtx.lineTo(w - pad, h - pad); timeCtx.stroke();

    // Ticks
    timeCtx.fillStyle = "#666"; timeCtx.font = "10px sans-serif";
    timeCtx.textAlign = "center"; timeCtx.textBaseline = "top";
    for (let i = 0; i <= nt; i++) {
      const t = tMin + (i / nt) * (tMax - tMin);
      timeCtx.fillText(t.toFixed(0), pad + ((t - tMin) / (tMax - tMin)) * plotW, h - pad + 2);
    }
    timeCtx.textAlign = "right"; timeCtx.textBaseline = "middle";
    for (let i = 0; i <= ny; i++) {
      const y = yMin + (i / ny) * (yMax - yMin);
      timeCtx.fillText(y.toFixed(1), pad - 4, h - pad - ((y - yMin) / (yMax - yMin)) * plotH);
    }

    timeCtx.textAlign = "center"; timeCtx.textBaseline = "bottom"; timeCtx.font = "bold 11px sans-serif";
    timeCtx.fillText("Time", w / 2, h - 4);
    timeCtx.save(); timeCtx.translate(12, h / 2); timeCtx.rotate(-Math.PI / 2);
    timeCtx.textAlign = "center"; timeCtx.textBaseline = "top"; timeCtx.fillText("State", 0, 0); timeCtx.restore();

    // Curves
    for (let d = 0; d < stateNames.length; d++) {
      const color = STATE_COLORS[d % STATE_COLORS.length];
      timeCtx.strokeStyle = color; timeCtx.lineWidth = 1.5;
      timeCtx.beginPath();
      for (let i = 0; i < traj.length; i++) {
        const t = traj[i][0], y = traj[i][d + 1];
        const sx = pad + ((t - tMin) / (tMax - tMin)) * plotW;
        const sy = h - pad - ((y - yMin) / (yMax - yMin)) * plotH;
        if (i === 0) timeCtx.moveTo(sx, sy); else timeCtx.lineTo(sx, sy);
      }
      timeCtx.stroke();
    }

    // Legend
    let lx = w - pad - 80, ly = pad + 10;
    for (let d = 0; d < stateNames.length; d++) {
      const color = STATE_COLORS[d % STATE_COLORS.length];
      timeCtx.fillStyle = color; timeCtx.fillRect(lx, ly + d * 16, 12, 3);
      timeCtx.fillStyle = "#333"; timeCtx.font = "11px sans-serif"; timeCtx.textAlign = "left"; timeCtx.textBaseline = "middle";
      timeCtx.fillText(stateNames[d], lx + 16, ly + d * 16 + 1.5);
    }
  }

  function renderSweep() {
    const w = sweepCanvas.width, h = sweepCanvas.height;
    const results = model.get("sweep_results");
    const fps = model.get("sweep_fixed_points");
    sweepCtx.clearRect(0, 0, w, h);
    sweepCtx.fillStyle = "#fafafa"; sweepCtx.fillRect(0, 0, w, h);
    if (!results || results.length === 0) return;

    const paramValues = results.map((r) => r.param_value);
    const pMin = Math.min(...paramValues), pMax = Math.max(...paramValues);
    let yMin = Infinity, yMax = -Infinity;
    if (fps && fps.length > 0) {
      for (const [, x, y] of fps) { yMin = Math.min(yMin, x, y); yMax = Math.max(yMax, x, y); }
    } else { yMin = -1; yMax = 1; }
    const yMargin = (yMax - yMin) * 0.1; yMin -= yMargin; yMax += yMargin;
    const pad = 35, plotW = w - 2 * pad, plotH = h - 2 * pad;

    // Regime shading
    if (results.length > 1) {
      const dp = (pMax - pMin) / (results.length - 1);
      for (let i = 0; i < results.length; i++) {
        const r = results[i];
        const color = REGIME_COLORS[r.regime] || "#f5f5f5";
        const sx1 = pad + ((r.param_value - pMin) / (pMax - pMin)) * plotW;
        const sx2 = pad + ((r.param_value + dp - pMin) / (pMax - pMin)) * plotW;
        sweepCtx.fillStyle = color; sweepCtx.fillRect(sx1, pad, sx2 - sx1, plotH);
      }
    }

    // Grid
    sweepCtx.strokeStyle = "#e9ecef"; sweepCtx.lineWidth = 0.5;
    const np = 5, ny = 4;
    for (let i = 0; i <= np; i++) {
      const p = pMin + (i / np) * (pMax - pMin);
      const sx = pad + ((p - pMin) / (pMax - pMin)) * plotW;
      sweepCtx.beginPath(); sweepCtx.moveTo(sx, pad); sweepCtx.lineTo(sx, h - pad); sweepCtx.stroke();
    }
    for (let i = 0; i <= ny; i++) {
      const y = yMin + (i / ny) * (yMax - yMin);
      const sy = h - pad - ((y - yMin) / (yMax - yMin)) * plotH;
      sweepCtx.beginPath(); sweepCtx.moveTo(pad, sy); sweepCtx.lineTo(w - pad, sy); sweepCtx.stroke();
    }

    // Axes
    sweepCtx.strokeStyle = "#333"; sweepCtx.lineWidth = 1;
    sweepCtx.beginPath(); sweepCtx.moveTo(pad, pad); sweepCtx.lineTo(pad, h - pad); sweepCtx.lineTo(w - pad, h - pad); sweepCtx.stroke();

    // Ticks
    sweepCtx.fillStyle = "#666"; sweepCtx.font = "10px sans-serif"; sweepCtx.textAlign = "center"; sweepCtx.textBaseline = "top";
    for (let i = 0; i <= np; i++) {
      const p = pMin + (i / np) * (pMax - pMin);
      sweepCtx.fillText(p.toFixed(1), pad + ((p - pMin) / (pMax - pMin)) * plotW, h - pad + 2);
    }
    sweepCtx.textAlign = "right"; sweepCtx.textBaseline = "middle";
    for (let i = 0; i <= ny; i++) {
      const y = yMin + (i / ny) * (yMax - yMin);
      sweepCtx.fillText(y.toFixed(1), pad - 4, h - pad - ((y - yMin) / (yMax - yMin)) * plotH);
    }

    sweepCtx.textAlign = "center"; sweepCtx.textBaseline = "bottom"; sweepCtx.font = "bold 11px sans-serif";
    sweepCtx.fillText(model.get("sweep_param") || "Parameter", w / 2, h - 4);
    sweepCtx.save(); sweepCtx.translate(12, h / 2); sweepCtx.rotate(-Math.PI / 2);
    sweepCtx.textAlign = "center"; sweepCtx.textBaseline = "top"; sweepCtx.fillText("Fixed Point State", 0, 0); sweepCtx.restore();

    // Fixed points
    if (fps && fps.length > 0) {
      for (const [pval, x, y, stability] of fps) {
        const sx = pad + ((pval - pMin) / (pMax - pMin)) * plotW;
        const sy1 = h - pad - ((x - yMin) / (yMax - yMin)) * plotH;
        const sy2 = h - pad - ((y - yMin) / (yMax - yMin)) * plotH;
        const color = STABILITY_COLORS[stability] || "#666";
        sweepCtx.fillStyle = color; sweepCtx.beginPath(); sweepCtx.arc(sx, sy1, 3, 0, Math.PI * 2); sweepCtx.fill();
        sweepCtx.fillStyle = color; sweepCtx.beginPath(); sweepCtx.arc(sx, sy2, 3, 0, Math.PI * 2); sweepCtx.fill();
      }
    }
  }

  // ═════════════════════════════════════════════════════════════
  //  INITIALISATION
  // ═════════════════════════════════════════════════════════════

  populateModelSelector();
  createParamSliders(model.get("param_info"), model.get("params"));
  updateLimitInputs();

  // Set up state selectors and clamped sliders for the initial model
  const _initModelName = model.get("model_name");
  const _initDef = MODELS[_initModelName];
  const _initDisplay = model.get("display") || _initDef.display || [0, 1];
  populateStateSelectors(_initDef, _initDisplay);
  const _initClamped = model.get("clamped") || _initDef.stateVars.map((_, i) => {
    const lims = (_initDef.defaultLims && _initDef.defaultLims[i]) || [-1, 1];
    return (lims[0] + lims[1]) / 2;
  });
  createClampedSliders(_initDef, _initClamped, _initDisplay);

  // In standalone mode the model may already have data; otherwise compute it now
  const hasData = (model.get("trajectory") || []).length > 0;
  if (!hasData || isStandalone) {
    computeAll();
  } else {
    renderPhasePlane();
    renderTimeSeries();
    renderSweep();
  }

  return () => {
    // cleanup if needed
  };
}
