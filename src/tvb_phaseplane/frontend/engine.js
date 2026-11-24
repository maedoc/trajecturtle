/** Create a projected RHS for the 2D phase plane using nerdamer compilation.
 *
 *  Non-displayed state variables are substituted as literal numbers.
 *
 *  @param {Object} modelEntry - from compileModelSpec()
 *  @param {number[]} displayIdx - indices of displayed variables
 *  @param {number[]} clamped - full N-dim state vector with clamped values
 *  @returns {Function} f_proj(t, projState, params) -> derivatives of displayed vars
 */
function makeCompiledProjectedRHS(modelEntry, displayIdx, clamped) {
  const stateVars = modelEntry.stateVars;
  const paramNames = modelEntry._paramNames;
  const dim = stateVars.length;

  if (dim <= 2) {
    return (t, projState, params) => modelEntry.f(t, projState, params);
  }

  // Build projected expressions for displayed equations only
  const displayedVars = displayIdx.map((i) => stateVars[i]);
  const allArgs = [...displayedVars, ...paramNames];
  const projectedFns = [];

  for (const idx of displayIdx) {
    let exprStr = modelEntry._rawExprs[stateVars[idx]] || '0';

    // Substitute non-displayed vars with clamped numeric values using word boundaries
    for (let k = 0; k < dim; k++) {
      if (!displayIdx.includes(k)) {
        // Build a regex that matches whole words only
        const vname = stateVars[k];
        // Escape special regex chars in variable names
        const escaped = vname.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const re = new RegExp('\\b' + escaped + '\\b', 'g');
        exprStr = exprStr.replace(re, '(' + clamped[k] + ')');
      }
    }

    try {
      const fn = nerdamer(exprStr).buildFunction(allArgs);
      projectedFns.push(fn);
    } catch (e) {
      throw new Error(`Cannot compile projected equation for ${stateVars[idx]}: ${e.message}`);
    }
  }

  return function (t, projState, params) {
    const args = [...projState];
    for (const pn of paramNames) {
      args.push(params[pn] !== undefined ? params[pn] : 0);
    }
    const result = [];
    for (const fn of projectedFns) {
      result.push(fn(...args));
    }
    return result;
  };
}

// ═══════════════════════════════════════════════════════════════
//  END CUSTOM MODEL COMPILATION
// ═══════════════════════════════════════════════════════════════

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
 *  For custom models (nerdamer-compiled), substitutes clamped non-displayed
 *  vars as literal numbers into the expressions for optimal performance.
 *  For built-in models, uses the original wrapper approach.
 *
 *  @param {string} modelName — key in MODELS (or 'custom')
 *  @param {number[]} displayIdx — indices of displayed state variables
 *  @param {number[]} clamped — full N-dim state vector with clamped values
 *  @returns {Function} f_proj(t, projState, params) → derivatives of displayed vars
 */
function makeProjectedRHS(modelName, displayIdx, clamped, spec = null, resolved = null) {
  const def = resolved || resolveModel(modelName, spec);
  if (modelName === 'custom') return makeCompiledProjectedRHS(def, displayIdx, clamped);
  const fullF = def.f;
  const n = (def.stateVars || def.stateNames || []).length;
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

/** Core fixed-step RK4 ODE solver (bare computation, no safety wrappers).
 *  @param {string|Object} modelName — key in MODELS, or a compiled model entry
 *  Returns array of [t, v0, v1, ...] rows.  */
function _rk4Core(modelName, y0, t0, tf, h, params) {
  const f = typeof modelName === 'object' ? modelName.f : MODELS[modelName].f;
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

/** Clamp a state vector to optional per-variable [lo, hi] bounds. */
function clampState(y, bounds) {
  if (!bounds) return y;
  return y.map((yi, i) => {
    const b = bounds[i];
    if (!b) return yi;
    let v = yi;
    if (b[0] != null && v < b[0]) v = b[0];
    if (b[1] != null && v > b[1]) v = b[1];
    return v;
  });
}

/** Safe RK4 wrapper with NaN/Inf detection, step budget, and exp clamping.
 *  Returns array of [t, v0, v1, ...] rows, truncated on NaN/Inf/budget. */
function rk4(modelName, y0, t0, tf, h, params) {
  const MAX_STEPS = 50000;
  const f = typeof modelName === 'object' ? modelName.f : MODELS[modelName].f;
  const bounds = typeof modelName === 'object' ? modelName.bounds : MODELS[modelName]?.bounds;
  let y = clampState([...y0], bounds);
  const results = [[t0, ...y]];
  let t = t0;
  let steps = 0;
  while (t < tf - 1e-12) {
    if (steps >= MAX_STEPS) break;
    const dt = Math.min(h, tf - t);
    const k1 = f(t, y, params);
    if (k1.some(v => !isFinite(v))) break;
    const k2 = f(t + dt / 2, clampState(y.map((yi, i) => yi + (dt / 2) * k1[i]), bounds), params);
    if (k2.some(v => !isFinite(v))) break;
    const k3 = f(t + dt / 2, clampState(y.map((yi, i) => yi + (dt / 2) * k2[i]), bounds), params);
    if (k3.some(v => !isFinite(v))) break;
    const k4 = f(t + dt, clampState(y.map((yi, i) => yi + dt * k3[i]), bounds), params);
    if (k4.some(v => !isFinite(v))) break;
    y = clampState(y.map((yi, i) => yi + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])), bounds);
    t += dt;
    if (y.some((v) => !isFinite(v))) break;
    results.push([t, ...y]);
    steps++;
  }
  return results;
}

/** Box-Muller transform for standard normal random numbers. */
function randn(random = Math.random) {
  let u1 = random();
  let u2 = random();
  while (u1 === 0) u1 = random();
  while (u2 === 0) u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Fixed-step Stratonovich Heun integrator with additive noise.
 *  noiseSigma is an array of per-variable noise strengths (same length as state).
 *  Returns array of [t, v0, v1, ...] rows.
 *  Max 50,000 step budget.  */
function heun(modelName, y0, t0, tf, h, params, noiseSigma, seed = 0) {
  let rngState = seed >>> 0;
  const random = () => {
    rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0;
    return (rngState + 0.5) / 4294967296;
  };
  const f = typeof modelName === 'object' ? modelName.f : MODELS[modelName].f;
  const bounds = typeof modelName === 'object' ? modelName.bounds : MODELS[modelName]?.bounds;
  const dim = y0.length;
  let y = clampState([...y0], bounds);
  const results = [[t0, ...y]];
  let t = t0;
  let steps = 0;
  const maxSteps = 50000;
  while (t < tf - 1e-12 && steps < maxSteps) {
    const dt = Math.min(h, tf - t);
    const sqrtDt = Math.sqrt(dt);
    // Wiener increments: dW ~ Normal(0, sqrt(dt))
    const dW = new Array(dim);
    for (let i = 0; i < dim; i++) {
      dW[i] = sqrtDt * randn(random);
    }
    // Predictor: Ytilde = Yn + f(Yn)*dt + sigma*dW
    const fn = f(t, y, params);
    const yTilde = clampState(y.map((yi, i) => yi + fn[i] * dt + (noiseSigma[i] || 0) * dW[i]), bounds);
    // Corrector: Yn+1 = Yn + 0.5*(f(Yn)+f(Ytilde))*dt + sigma*dW
    const fTilde = f(t + dt, yTilde, params);
    y = clampState(y.map((yi, i) => yi + 0.5 * (fn[i] + fTilde[i]) * dt + (noiseSigma[i] || 0) * dW[i]), bounds);
    t += dt;
    steps++;
    // Check for NaN / blow-up and abort gracefully
    if (y.some((v) => !isFinite(v))) break;
    results.push([t, ...y]);
  }
  return results;
}

/** Dispatch to the appropriate integrator based on settings.
 *  settings = { integrator: 'rk4'|'heun', noiseSigma: [...]|null }
 */
function simulate(modelName, state0, t0, tf, dt, params, settings) {
  const integrator = (settings && settings.integrator) || 'rk4';
  if (integrator === 'heun') {
    const noiseSigma = (settings && settings.noiseSigma) || state0.map(() => 0);
    return heun(modelName, state0, t0, tf, dt, params, noiseSigma, settings?.seed || 0);
  }
  return rk4(modelName, state0, t0, tf, dt, params);
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

/** Quick RK4 stepper for validation trajectories. */
function _quickRk4(f, y0, t0, tf, dt, params) {
  let t = t0;
  let y = [...y0];
  while (t < tf - 1e-12) {
    const h = Math.min(dt, tf - t);
    const k1 = f(t, y, params);
    const k2 = f(t + h / 2, y.map((yi, i) => yi + (h / 2) * k1[i]), params);
    const k3 = f(t + h / 2, y.map((yi, i) => yi + (h / 2) * k2[i]), params);
    const k4 = f(t + h, y.map((yi, i) => yi + h * k3[i]), params);
    y = y.map((yi, i) => yi + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    t += h;
  }
  return y;
}

/** Validate a fixed point by running a short trajectory from a perturbed IC.
 *  Cross-checks the eigenvalue-based classification against actual dynamics.
 *  Does NOT display the validation trajectory — it is for classification only.
 *
 *  IMPORTANT: Saddles are skipped.  A generic perturbation near a saddle has
 *  a component along the unstable manifold, so forward integration always
 *  diverges; this makes the test useless for saddles.  The eigenvalue-based
 *  "saddle" label (opposite signs on a 2×2 Jacobian) is already robust.
 */
function validateFixedPoint(f, fp, params, classification) {
  // Saddles: skip — forward integration from a random perturbation never
  // converges to a saddle, so the test is structurally incapable of confirming
  // the classification.
  if (classification === "saddle") {
    return classification;
  }

  const eps = 0.02;
  const tEnd = 3.0;
  const dt = 0.01;
  const y0 = [fp[0] + eps, fp[1] + eps];
  const yEnd = _quickRk4(f, y0, 0, tEnd, dt, params);

  const distStart = Math.sqrt(
    (y0[0] - fp[0]) ** 2 + (y0[1] - fp[1]) ** 2
  );
  const distEnd = Math.sqrt(
    (yEnd[0] - fp[0]) ** 2 + (yEnd[1] - fp[1]) ** 2
  );

  // Stable node / focus: verify the trajectory actually converges
  if (classification.startsWith("stable")) {
    if (distEnd < distStart * 0.5) {
      return classification;                 // confirmed
    }
    // Did not converge → recompute (numerical noise near bifurcation?)
    const J = jacobianND(f, [fp[0], fp[1]], params);
    return classifyFixedPoint(eigenvalues2x2(J));
  }

  // Unstable node / focus: verify the trajectory actually diverges
  if (classification.startsWith("unstable")) {
    if (distEnd > distStart * 2.0) {
      return classification;                 // confirmed
    }
    // Did not diverge → recompute
    const J = jacobianND(f, [fp[0], fp[1]], params);
    return classifyFixedPoint(eigenvalues2x2(J));
  }

  return classification;
}

/** Draw a fixed-point marker whose shape encodes stability type. */
function drawFixedPointMarker(ctx, sx, sy, stability, radius = 7) {
  const color = STABILITY_COLORS[stability] || "#666";
  ctx.lineWidth = 1.5;

  switch (stability) {
    case "stable_node": {
      // Filled circle ●
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#333";
      ctx.stroke();
      break;
    }
    case "stable_focus": {
      // Filled circle with white inner dot (target)
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#333";
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(sx, sy, radius * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      break;
    }
    case "unstable_node": {
      // Open circle ○
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      break;
    }
    case "unstable_focus": {
      // Open circle with inner cross
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      const r2 = radius * 0.4;
      ctx.beginPath();
      ctx.moveTo(sx - r2, sy - r2);
      ctx.lineTo(sx + r2, sy + r2);
      ctx.moveTo(sx + r2, sy - r2);
      ctx.lineTo(sx - r2, sy + r2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      break;
    }
    case "saddle": {
      // Diamond ◆
      ctx.beginPath();
      ctx.moveTo(sx, sy - radius);
      ctx.lineTo(sx + radius, sy);
      ctx.lineTo(sx, sy + radius);
      ctx.lineTo(sx - radius, sy);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      break;
    }
    default: {
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = "#333";
      ctx.stroke();
    }
  }
}

/** Grid-search + Newton-Raphson fixed-point finder (1D/2D). */
function findFixedPoints(f, params, xlim, ylim, nGrid = 25) {
  const MAX_ITER_TOTAL = Math.max(625, nGrid * nGrid * 10);
  let iterCount = 0;
  const tol = 0.08;
  const fixedPoints = [];
  const test = f(0, [(xlim[0] + xlim[1]) / 2], params);
  const is1D = test.length === 1;

  if (is1D) {
    const xs = linspace(xlim[0], xlim[1], nGrid);
    for (const xi of xs) {
      let x = xi;
      for (let iter = 0; iter < 50; iter++) {
        if (iterCount >= MAX_ITER_TOTAL) {
          console.warn('Budget exhausted: findFixedPoints');
          return fixedPoints;
        }
        iterCount++;
        const fx = f(0, [x], params)[0];
        if (Math.abs(fx) < 1e-6) {
          if (xlim[0] - 0.5 <= x && x <= xlim[1] + 0.5) {
            let isNew = true;
            for (const ex of fixedPoints) {
              if (Math.abs(x - ex[0]) < tol) {
                isNew = false;
                break;
              }
            }
            if (isNew) {
              const J1D = jacobianND(f, [x], params);
              const deriv = J1D[0][0];
              let stability = deriv < -1e-6 ? 'stable_node' : deriv > 1e-6 ? 'unstable_node' : 'saddle';
              stability = validateFixedPoint(f, [x, 0], params, stability);
              fixedPoints.push([x, 0, stability]);
            }
          }
          break;
        }
        const J1D = jacobianND(f, [x], params, 1e-6);
        const deriv = J1D[0][0];
        if (Math.abs(deriv) < 1e-12) break;
        x = x - fx / deriv;
      }
    }
  } else {
    const xs = linspace(xlim[0], xlim[1], nGrid);
    const ys = linspace(ylim[0], ylim[1], nGrid);

    for (const xi of xs) {
      for (const yi of ys) {
        let x = [xi, yi];
        for (let iter = 0; iter < 50; iter++) {
          if (iterCount >= MAX_ITER_TOTAL) {
            console.warn('Budget exhausted: findFixedPoints');
            return fixedPoints;
          }
          iterCount++;
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
                let stability = classifyFixedPoint(ev);
                stability = validateFixedPoint(f, [x[0], x[1]], params, stability);
                fixedPoints.push([x[0], x[1], stability]);
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
  }
  return fixedPoints;
}

/** Nullcline computation via grid zero-crossings. */
function computeNullclines(f, params, xlim, ylim, nGrid = 60) {
  const MAX_EVAL = Math.max(3600, nGrid * nGrid + 100);
  let evalCount = 0;
  // f is a projected RHS function
  const xs = linspace(xlim[0], xlim[1], nGrid);
  // Detect 1-D: check dimension of the RHS
  const test = f(0, [(xlim[0] + xlim[1]) / 2], params);
  const is1D = test.length === 1;

  if (is1D) {
    // 1-D case: single nullcline f(x)=0
    const zeroX = [];
    for (let i = 0; i < xs.length; i++) {
      if (evalCount >= MAX_EVAL) {
        console.warn('Budget exhausted: computeNullclines');
        return [zeroX, []];
      }
      evalCount++;
      const d = f(0, [xs[i]], params);
      if (i > 0) {
        const dPrev = f(0, [xs[i - 1]], params);
        if (d[0] === 0) zeroX.push([xs[i], 0]);
        else if (dPrev[0] * d[0] < 0) {
          const t = Math.abs(dPrev[0]) / (Math.abs(dPrev[0]) + Math.abs(d[0]));
          zeroX.push([xs[i - 1] + t * (xs[i] - xs[i - 1]), 0]);
        }
      }
    }
    return [zeroX, []];
  }

  // 2-D case
  const ys = linspace(ylim[0], ylim[1], nGrid);
  const dx = Array(nGrid).fill(0).map(() => Array(nGrid).fill(0));
  const dy = Array(nGrid).fill(0).map(() => Array(nGrid).fill(0));

  for (let i = 0; i < nGrid; i++) {
    for (let j = 0; j < nGrid; j++) {
      if (evalCount >= MAX_EVAL) {
        console.warn('Budget exhausted: computeNullclines');
        return [_findZeroCrossings(dx), _findZeroCrossings(dy)];
      }
      evalCount++;
      const d = f(0, [xs[j], ys[i]], params);
      dx[i][j] = d[0];
      dy[i][j] = d[1];
    }
  }

  function _findZeroCrossings(Z) {
    const points = [];
    const n = Z.length;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n - 1; j++) {
        if (Z[i][j] === 0) points.push([xs[j], ys[i]]);
        else if (Z[i][j] * Z[i][j + 1] < 0) {
          const t = Math.abs(Z[i][j]) / (Math.abs(Z[i][j]) + Math.abs(Z[i][j + 1]));
          points.push([xs[j] + t * (xs[j + 1] - xs[j]), ys[i]]);
        }
      }
    }
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n; j++) {
        if (Z[i][j] === 0) points.push([xs[j], ys[i]]);
        else if (Z[i][j] * Z[i + 1][j] < 0) {
          const t = Math.abs(Z[i][j]) / (Math.abs(Z[i][j]) + Math.abs(Z[i + 1][j]));
          points.push([xs[j], ys[i] + t * (ys[i + 1] - ys[i])]);
        }
      }
    }
    return points;
  }

  return [_findZeroCrossings(dx), _findZeroCrossings(dy)];
}

/** Sparse vector field (1D/2D). */
function computeVectorField(f, params, xlim, ylim, nGrid = 12) {
  const MAX_VECTORS = Math.max(144, nGrid * nGrid + 10);
  const xs = linspace(xlim[0], xlim[1], nGrid);
  const test = f(0, [(xlim[0] + xlim[1]) / 2], params);
  const is1D = test.length === 1;
  const vectors = [];
  let count = 0;

  if (is1D) {
    for (const xi of xs) {
      if (count >= MAX_VECTORS) {
        console.warn('Budget exhausted: computeVectorField');
        return vectors;
      }
      count++;
      const d = f(0, [xi], params);
      vectors.push([xi, 0, d[0], 0]);
    }
  } else {
    const ys = linspace(ylim[0], ylim[1], nGrid);
    for (const xi of xs) {
      for (const yi of ys) {
        if (count >= MAX_VECTORS) {
          console.warn('Budget exhausted: computeVectorField');
          return vectors;
        }
        count++;
        const d = f(0, [xi, yi], params);
        vectors.push([xi, yi, d[0], d[1]]);
      }
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

/** Solve N×N linear system A * x = b using Gaussian elimination with partial pivoting.
 *  Returns solution vector x, or null if A is singular or dimensions are invalid. */
export function solveLinearSystem(A, b) {
  const n = Array.isArray(A) ? A.length : 0;
  if (n < 1 || !Array.isArray(b) || b.length !== n) return null;

  const M = A.map((row) => row.slice());
  const rhs = b.slice();

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let pivotAbs = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(M[row][col]);
      if (val > pivotAbs) {
        pivotAbs = val;
        pivotRow = row;
      }
    }
    if (!Number.isFinite(pivotAbs) || pivotAbs < 1e-14) return null;

    if (pivotRow !== col) {
      [M[col], M[pivotRow]] = [M[pivotRow], M[col]];
      [rhs[col], rhs[pivotRow]] = [rhs[pivotRow], rhs[col]];
    }

    const diag = M[col][col];
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / diag;
      if (factor === 0) continue;
      M[row][col] = 0;
      for (let k = col + 1; k < n; k++) {
        M[row][k] -= factor * M[col][k];
      }
      rhs[row] -= factor * rhs[col];
    }
  }

  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = rhs[row];
    for (let k = row + 1; k < n; k++) {
      sum -= M[row][k] * x[k];
    }
    const val = sum / M[row][row];
    if (!Number.isFinite(val)) return null;
    x[row] = val;
  }
  return x;
}

function _vectorNorm(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i];
    if (!Number.isFinite(v)) return Infinity;
    sum += v * v;
  }
  return Math.sqrt(sum);
}

function _newtonPolishND(def, params, startState, tol, maxIter, eps) {
  let u = def.bounds ? clampState([...startState], def.bounds) : [...startState];
  if (u.some((v) => !Number.isFinite(v))) {
    return { u: [...startState], residual: Infinity, converged: false };
  }
  let f0 = def.f(0, u, params);
  let res0 = _vectorNorm(f0);
  if (!Number.isFinite(res0)) {
    return { u, residual: Infinity, converged: false };
  }

  for (let iter = 0; iter < maxIter; iter++) {
    if (res0 < tol) {
      return { u, residual: res0, converged: true };
    }
    const J = jacobianND(def.f, u, params, eps);
    const delta = solveLinearSystem(J, f0.map((v) => -v));
    if (!delta) break;

    let accepted = false;
    for (const alpha of [1, 0.5, 0.25, 0.125, 0.0625]) {
      let cand = u.map((ui, i) => ui + alpha * delta[i]);
      if (def.bounds) cand = clampState(cand, def.bounds);
      const fCand = def.f(0, cand, params);
      const resCand = _vectorNorm(fCand);
      if (Number.isFinite(resCand) && resCand < res0) {
        u = cand;
        f0 = fCand;
        res0 = resCand;
        accepted = true;
        break;
      }
    }
    if (!accepted) break;
  }

  return { u, residual: res0, converged: res0 < tol };
}

/** Find a full-N-D steady state of def.f(0, u, params) = 0 using damped Newton-Raphson,
 *  falling back to deterministic RK4 burn-in and alternative seeds if needed. */
export function findSteadyStateND(def, params, seedState, opts = {}) {
  const tol = opts.tol ?? 1e-8;
  const maxIter = opts.maxIter ?? 40;
  const eps = opts.eps ?? 1e-6;
  const burnInT = opts.burnInT ?? 40;
  const burnInDt = opts.burnInDt ?? 0.05;
  const dim = def.dim ?? seedState?.length ?? def.defaultLims?.length ?? 2;

  const midpoint = Array.from({ length: dim }, (_, i) => {
    const lim = def.defaultLims?.[i];
    return Array.isArray(lim) && lim.length === 2 ? (lim[0] + lim[1]) / 2 : 0;
  });

  const candidates = [];
  if (Array.isArray(seedState) && seedState.length === dim) {
    candidates.push([...seedState]);
  }
  if (Array.isArray(opts.fallbackStates)) {
    for (const fb of opts.fallbackStates) {
      if (Array.isArray(fb) && fb.length === dim) {
        candidates.push([...fb]);
      }
    }
  }
  candidates.push(midpoint);

  let best = { u: [...candidates[0]], residual: Infinity, converged: false };
  const updateBest = (res) => {
    if (Number.isFinite(res.residual) && res.residual < best.residual) {
      best = res;
    }
  };

  // Pass 1: direct damped Newton from primary seed
  const primaryRes = _newtonPolishND(def, params, candidates[0], tol, maxIter, eps);
  updateBest(primaryRes);
  if (primaryRes.converged) return primaryRes;

  // Pass 2: RK4 burn-in from primary seed, then Newton polish
  const primaryBurn = rk4(def, candidates[0], 0, burnInT, burnInDt, params);
  if (primaryBurn.length > 1) {
    const burnedState = primaryBurn[primaryBurn.length - 1].slice(1);
    const burnedRes = _newtonPolishND(def, params, burnedState, tol, maxIter, eps);
    updateBest(burnedRes);
    if (burnedRes.converged) return burnedRes;
  }

  // Pass 3: fallback seeds (direct Newton, then RK4 burn-in + Newton)
  for (let i = 1; i < candidates.length; i++) {
    const cand = candidates[i];
    const directRes = _newtonPolishND(def, params, cand, tol, maxIter, eps);
    updateBest(directRes);
    if (directRes.converged) return directRes;

    const burn = rk4(def, cand, 0, burnInT, burnInDt, params);
    if (burn.length > 1) {
      const burnedState = burn[burn.length - 1].slice(1);
      const burnedRes = _newtonPolishND(def, params, burnedState, tol, maxIter, eps);
      updateBest(burnedRes);
      if (burnedRes.converged) return burnedRes;
    }
  }

  return best;
}

/** Detect non-smooth functions in a custom model_spec that can hinder numerical continuation. */
export function detectNonSmoothExpressions(model_spec) {
  if (!model_spec || typeof model_spec !== 'object') return [];
  const pattern = /\b(step|heaviside|abs|min|max|sign)\s*\(/gi;
  const found = new Set();

  const scanText = (text) => {
    if (typeof text !== 'string') return;
    pattern.lastIndex = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      found.add(`${match[1]}(`);
    }
  };

  const scanContainer = (container) => {
    if (!container || typeof container !== 'object') return;
    for (const val of Object.values(container)) {
      if (typeof val === 'string') {
        scanText(val);
      } else if (val && typeof val === 'object') {
        scanText(val.body);
        scanText(val.expr);
        scanText(val.expression);
      }
    }
  };

  scanContainer(model_spec.equations);
  scanContainer(model_spec.functions);
  scanContainer(model_spec.custom_functions);
  return [...found];
}

// ═══════════════════════════════════════════════════════════════
//  CANVAS RENDERING  (unchanged coordinate helpers & draw calls)
// ═══════════════════════════════════════════════════════════════
