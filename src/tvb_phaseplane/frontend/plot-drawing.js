// Rendering functions have no panel/DOM dependencies; also used by SVG export.
const DEFAULT_PLOT_THEME = { background: '#fafafa', text: '#333', muted: '#666', grid: '#e9ecef', cursor: '#172d46' };
function drawFrozenScene(model, ctx, w, h) {
  if (model.get('show_reference') === false || !(model.get('reference_trajectory') || []).length) return;
  const scene = model.get('reference_scene') || {}, display = model.get('display') || [0, 1];
  if (display.length < 2) return;
  const xlim = model.get('xlim'), ylim = model.get('ylim');
  ctx.save(); ctx.beginPath(); ctx.rect(42, 42, w - 84, h - 84); ctx.clip();
  for (const [key, color] of [['nullcline_x', '#a6b8ca'], ['nullcline_y', '#c7adb3']]) {
    ctx.fillStyle = color;
    for (const [x, y] of scene[key] || []) {
      const [sx, sy] = worldToScreen(x, y, xlim, ylim, w, h);
      ctx.beginPath(); ctx.arc(sx, sy, 2, 0, 2 * Math.PI); ctx.fill();
    }
  }
  ctx.strokeStyle = '#8b929d'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath();
  (model.get('reference_trajectory') || []).forEach((row, i) => {
    const [sx, sy] = worldToScreen(row[display[0] + 1], row[display[1] + 1], xlim, ylim, w, h);
    if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy);
  });
  ctx.stroke(); ctx.setLineDash([]);
  for (const [x, y] of scene.fixed_points || []) {
    const [sx, sy] = worldToScreen(x, y, xlim, ylim, w, h);
    ctx.beginPath(); ctx.arc(sx, sy, 8, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.restore();
}

function drawPhasePlane(model, ctx, w, h, theme = DEFAULT_PLOT_THEME) {
  const xlim = model.get("xlim");
  const ylim = model.get("ylim");
  const stateNames = model.get("state_names");
  const display = model.get("display") || [0, 1];
  const is1D = display.length === 1;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = theme.background;
  ctx.fillRect(0, 0, w, h);

  if (is1D) {
    // 1-D mode: draw dx/dt vs x
    drawAxes(ctx, w, h, xlim, [-2, 2], stateNames[display[0]], "d" + stateNames[display[0]] + "/dt", theme);
  } else {
    drawAxes(ctx, w, h, xlim, ylim, stateNames[display[0]], stateNames[display[1]], theme);
  }

  // Frozen geometry is drawn behind current data, in the current axes.
  drawFrozenScene(model, ctx, w, h);

  // Nullclines
  if (model.get("show_nullclines")) {
    const ncX = model.get("nullcline_x");
    if (ncX && ncX.length > 0) {
      ctx.fillStyle = "#2196F3";
      for (const [wx, wy] of ncX) {
        const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
        ctx.beginPath(); ctx.arc(sx, sy, 1.8, 0, Math.PI * 2); ctx.fill();
      }
    }
    const ncY = model.get("nullcline_y");
    if (ncY && ncY.length > 0) {
      ctx.fillStyle = "#f44336";
      for (const [wx, wy] of ncY) {
        const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
        ctx.beginPath(); ctx.arc(sx, sy, 1.8, 0, Math.PI * 2); ctx.fill();
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
        drawArrow(ctx, sx, sy, sx + ndx * arrowLen, sy + ndy * arrowLen, "#ccc", 3);
      }
    }
  }

  // Fixed points
  if (model.get("show_fixed_points")) {
    const fps = model.get("fixed_points");
    if (fps && fps.length > 0) {
      for (const [wx, wy, stability] of fps) {
        const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
        drawFixedPointMarker(ctx, sx, sy, stability, 7);
      }
    }
  }

  // Trajectory
  if (model.get("show_trajectory")) {
    const traj = model.get("trajectory");
    if (traj && traj.length > 1) {
      ctx.strokeStyle = "#4CAF50"; ctx.lineWidth = 1.5;
      ctx.beginPath();
      const proj0 = [traj[0][display[0] + 1]];
      if (!is1D) proj0.push(traj[0][display[1] + 1]);
      const [sx0, sy0] = is1D
        ? worldToScreen(proj0[0], 0, xlim, [-2, 2], w, h)
        : worldToScreen(proj0[0], proj0[1], xlim, ylim, w, h);
      ctx.moveTo(sx0, sy0);
      for (let i = 1; i < traj.length; i++) {
        const proj = [traj[i][display[0] + 1]];
        if (!is1D) proj.push(traj[i][display[1] + 1]);
        const [sx, sy] = is1D
          ? worldToScreen(proj[0], 0, xlim, [-2, 2], w, h)
          : worldToScreen(proj[0], proj[1], xlim, ylim, w, h);
        ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      // Initial condition marker
      ctx.beginPath(); ctx.arc(sx0, sy0, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#FF9800"; ctx.fill();
      ctx.strokeStyle = theme.text; ctx.lineWidth = 1; ctx.stroke();
      const cursor = traj[Math.round(Math.max(0, Math.min(1, model.get('trajectory_fraction') ?? 1)) * (traj.length - 1))];
      if (cursor && !is1D) {
        const [sx, sy] = worldToScreen(cursor[display[0]+1], cursor[display[1]+1], xlim, ylim, w, h);
        ctx.beginPath(); ctx.arc(sx, sy, 5, 0, 2*Math.PI); ctx.fillStyle = theme.cursor; ctx.fill();
      }
    }
  }
}

function drawTimeSeries(model, timeCtx, w, h, theme = DEFAULT_PLOT_THEME) {
  const traj = model.get("trajectory");
  const stateNames = model.get("state_names");
  timeCtx.clearRect(0, 0, w, h);
  timeCtx.fillStyle = theme.background; timeCtx.fillRect(0, 0, w, h);
  if (!traj || traj.length < 2) return;
  const fraction = Math.max(0, Math.min(1, model.get('trajectory_fraction') ?? 1));
  const cursorRow = traj[Math.round(fraction * (traj.length - 1))];

  const reference = model.get('show_reference') === false ? [] : (model.get('reference_trajectory') || []);
  const tMin = traj[0][0], tMax = Math.max(traj.at(-1)[0], reference.at(-1)?.[0] || 0);
  const pad = 42, plotW = w - 2 * pad, plotH = h - 2 * pad;

  let yMin = Infinity, yMax = -Infinity;
  for (let d = 0; d < stateNames.length; d++) {
    const vals = [...traj, ...reference].map((r) => r[d + 1]);
    yMin = Math.min(yMin, ...vals); yMax = Math.max(yMax, ...vals);
  }
  if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
  const yMargin = (yMax - yMin) * 0.1;
  yMin -= yMargin; yMax += yMargin;

  // Grid
  timeCtx.strokeStyle = theme.grid; timeCtx.lineWidth = 0.5;
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
  timeCtx.strokeStyle = theme.text; timeCtx.lineWidth = 1;
  timeCtx.beginPath(); timeCtx.moveTo(pad, pad); timeCtx.lineTo(pad, h - pad); timeCtx.lineTo(w - pad, h - pad); timeCtx.stroke();

  // Ticks
  timeCtx.fillStyle = theme.muted; timeCtx.font = "10px sans-serif";
  timeCtx.textAlign = "center"; timeCtx.textBaseline = "top";
  for (let i = 0; i <= nt; i++) {
    const t = tMin + (i / nt) * (tMax - tMin);
    timeCtx.fillText(Number(t.toPrecision(3)).toString(), pad + ((t - tMin) / (tMax - tMin)) * plotW, h - pad + 2);
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

  // The same frozen full-system path is visible in the time series.
  timeCtx.save(); timeCtx.setLineDash([5, 4]); timeCtx.strokeStyle = '#8b929d'; timeCtx.lineWidth = 1.5;
  for (let d = 0; d < stateNames.length; d++) {
    timeCtx.beginPath();
    reference.forEach((row, i) => {
      const sx = pad + (row[0] - tMin) / (tMax - tMin) * plotW;
      const sy = h - pad - (row[d + 1] - yMin) / (yMax - yMin) * plotH;
      if (i) timeCtx.lineTo(sx, sy); else timeCtx.moveTo(sx, sy);
    });
    timeCtx.stroke();
  }
  timeCtx.restore();

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

  const cursorX = pad + ((cursorRow[0] - tMin) / (tMax - tMin)) * plotW;
  timeCtx.save(); timeCtx.setLineDash([3, 3]); timeCtx.strokeStyle = theme.cursor;
  timeCtx.beginPath(); timeCtx.moveTo(cursorX, pad); timeCtx.lineTo(cursorX, h-pad); timeCtx.stroke(); timeCtx.restore();
  // Legend
  let lx = w - pad - 80, ly = pad + 10;
  for (let d = 0; d < stateNames.length; d++) {
    const color = STATE_COLORS[d % STATE_COLORS.length];
    timeCtx.fillStyle = color; timeCtx.fillRect(lx, ly + d * 16, 12, 3);
    timeCtx.fillStyle = theme.text; timeCtx.font = "11px sans-serif"; timeCtx.textAlign = "left"; timeCtx.textBaseline = "middle";
    timeCtx.fillText(stateNames[d], lx + 16, ly + d * 16 + 1.5);
  }
}

function drawSweep(model, sweepCtx, w, h, _results = model.get('sweep_results'), _fps = model.get('sweep_fixed_points'), _paramName = model.get('sweep_param'), theme = DEFAULT_PLOT_THEME) {
  sweepCtx.clearRect(0, 0, w, h);
  sweepCtx.fillStyle = theme.background; sweepCtx.fillRect(0, 0, w, h);
  if (!_results || _results.length === 0) return;

  const paramValues = _results.map((r) => r.param_value);
  const pMin = Math.min(...paramValues), pMax = Math.max(...paramValues);
  let yMin = Infinity, yMax = -Infinity;
  if (_fps && _fps.length > 0) {
    for (const [, x, y] of _fps) { yMin = Math.min(yMin, x, y); yMax = Math.max(yMax, x, y); }
  } else { yMin = -1; yMax = 1; }
  if (yMin === yMax) { yMin -= .5; yMax += .5; }
  const yMargin = (yMax - yMin) * 0.1; yMin -= yMargin; yMax += yMargin;
  const pad = 35, plotW = w - 2 * pad, plotH = h - 2 * pad;

  // Regime shading
  if (_results.length > 1) {
    const dp = (pMax - pMin) / (_results.length - 1);
    for (let i = 0; i < _results.length; i++) {
      const r = _results[i];
      const color = REGIME_COLORS[r.regime] || "#f5f5f5";
      const sx1 = pad + ((r.param_value - pMin) / (pMax - pMin)) * plotW;
      const sx2 = pad + ((r.param_value + dp - pMin) / (pMax - pMin)) * plotW;
      sweepCtx.fillStyle = color; sweepCtx.fillRect(sx1, pad, sx2 - sx1, plotH);
    }
  }

  // Grid
  sweepCtx.strokeStyle = theme.grid; sweepCtx.lineWidth = 0.5;
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
  sweepCtx.strokeStyle = theme.text; sweepCtx.lineWidth = 1;
  sweepCtx.beginPath(); sweepCtx.moveTo(pad, pad); sweepCtx.lineTo(pad, h - pad); sweepCtx.lineTo(w - pad, h - pad); sweepCtx.stroke();

  // Ticks
  sweepCtx.fillStyle = theme.muted; sweepCtx.font = "10px sans-serif"; sweepCtx.textAlign = "center"; sweepCtx.textBaseline = "top";
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
  sweepCtx.fillText(_paramName || "Parameter", w / 2, h - 4);
  sweepCtx.save(); sweepCtx.translate(12, h / 2); sweepCtx.rotate(-Math.PI / 2);
  sweepCtx.textAlign = "center"; sweepCtx.textBaseline = "top"; sweepCtx.fillText("Slice equilibrium state", 0, 0); sweepCtx.restore();

  // Fixed points
  if (_fps && _fps.length > 0) {
    for (const [pval, x, y, stability] of _fps) {
      const sx = pad + ((pval - pMin) / (pMax - pMin)) * plotW;
      const sy1 = h - pad - ((x - yMin) / (yMax - yMin)) * plotH;
      const sy2 = h - pad - ((y - yMin) / (yMax - yMin)) * plotH;
      drawFixedPointMarker(sweepCtx, sx, sy1, stability, 3);
      drawFixedPointMarker(sweepCtx, sx, sy2, stability, 3);
    }
  }
}

function bifurcationObservableValue(pt, obsKey, which = 'u') {
  if (!pt) return NaN;
  const vec = which === 'u_max' ? (pt.u_max || pt.u) : which === 'u_min' ? (pt.u_min || pt.u) : pt.u;
  if (!Array.isArray(vec) || vec.length === 0) return NaN;
  if (obsKey === 'L2' || obsKey === 'l2') {
    let sq = 0;
    for (let i = 0; i < vec.length; i++) sq += vec[i] * vec[i];
    return Math.sqrt(sq);
  }
  const idx = Number.isInteger(obsKey) ? obsKey : 0;
  return Number(vec[idx] ?? vec[0]);
}

function _resolveBifurcationObs(model) {
  const cfg = model.get('bifurcation_config') || {};
  const stateNames = model.get('state_names') || [];
  const raw = cfg.observable;
  if (raw === 'L2' || raw === 'l2') return { obsKey: 'L2', obsLabel: 'L2 norm' };
  if (typeof raw === 'string' && stateNames.includes(raw)) {
    const idx = stateNames.indexOf(raw);
    return { obsKey: idx, obsLabel: stateNames[idx] || 'u' + idx };
  }
  const parsed = Number.isInteger(raw) ? raw : (raw !== undefined && raw !== '' && Number.isFinite(Number(raw)) ? Math.trunc(Number(raw)) : 0);
  const idx = Math.max(0, Math.min(Math.max(0, stateNames.length - 1), parsed));
  return { obsKey: idx, obsLabel: stateNames[idx] || ('u' + idx) };
}

const TWO_D_STYLES = {
  LP2:  { key: 'LP2',  strokeStyle: '#245c9b', dash: [],            lineWidth: 2, label: 'Fold (LP2)' },
  HB2:  { key: 'HB2',  strokeStyle: '#c62828', dash: [],            lineWidth: 2, label: 'Hopf (HB2)' },
  LPC2: { key: 'LPC2', strokeStyle: '#2e7d32', dash: [6, 4],        lineWidth: 2, label: 'LC Fold (LPC2)' },
  PD2:  { key: 'PD2',  strokeStyle: '#e65100', dash: [],            lineWidth: 2, label: 'Period Doubling (PD2)' },
  TR2:  { key: 'TR2',  strokeStyle: '#00838f', dash: [3, 3],        lineWidth: 2, label: 'Torus (TR2)' },
  HOM2: { key: 'HOM2', strokeStyle: '#6a1b9a', dash: [8, 3, 2, 3],  lineWidth: 2, label: 'Homoclinic (HOM2)' },
};
const TWO_D_BRANCH_TYPES = new Set(Object.keys(TWO_D_STYLES));

function _padBounds(min, max, frac = 0.05) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1, 1];
  if (min === max) return [min - 0.5, max + 0.5];
  const margin = (max - min) * frac;
  return [min - margin, max + margin];
}

function _resolveAxisBounds(min, max, frac, override) {
  const auto = _padBounds(min, max, frac);
  if (!Array.isArray(override) || override.length !== 2) return auto;
  const lo = override[0] !== null && override[0] !== '' && Number.isFinite(Number(override[0])) ? Number(override[0]) : auto[0];
  const hi = override[1] !== null && override[1] !== '' && Number.isFinite(Number(override[1])) ? Number(override[1]) : auto[1];
  return lo < hi ? [lo, hi] : auto;
}

function bifurcation2DPlanes(branches) {
  const seen = new Set();
  const planes = [];
  for (const br of branches || []) {
    if (!br || !TWO_D_BRANCH_TYPES.has(br.type) || !br.param1 || !br.param2) continue;
    const key = JSON.stringify([br.param1, br.param2]);
    if (seen.has(key)) continue;
    seen.add(key);
    planes.push({ key, param1: br.param1, param2: br.param2 });
  }
  return planes;
}

function _parsePlaneSelection(raw) {
  if (!raw) return null;
  if (Array.isArray(raw) && raw.length === 2 && raw[0] && raw[1]) return JSON.stringify([raw[0], raw[1]]);
  if (typeof raw === 'string') {
    if (raw.startsWith('[')) {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length === 2) return JSON.stringify([parsed[0], parsed[1]]);
      } catch {}
    }
    const parts = raw.split(',').map(s => s.trim()).filter(Boolean);
    if (parts.length === 2) return JSON.stringify(parts);
  }
  return null;
}

function bifurcation1DBounds(model) {
  const cfg = model.get('bifurcation_config') || {};
  const branches = (model.get('bifurcation_branches') || []).filter(b => b && (b.type === 'EP' || b.type === 'LC'));
  const sps = (model.get('bifurcation_special_points') || []).filter(sp => sp && sp.ty !== 'CP' && sp.ty !== 'BT' && sp.ty !== 'GH' && sp.ty !== 'ZH' && !sp.param2);
  const meta = model.get('bifurcation_metadata') || {};
  const param1 = cfg.param1 || branches[0]?.param1 || meta.active_params?.[0] || Object.keys(model.get('params') || {})[0] || 'p';
  const { obsKey, obsLabel } = _resolveBifurcationObs(model);
  let pMin = Infinity, pMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  const includeY = (v) => { if (Number.isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); } };
  const includeP = (v) => { if (Number.isFinite(v)) { pMin = Math.min(pMin, v); pMax = Math.max(pMax, v); } };
  for (const br of branches) {
    for (const pt of br.points || []) {
      includeP(pt.p);
      if (br.type === 'LC') {
        includeY(bifurcationObservableValue(pt, obsKey, 'u_max'));
        includeY(bifurcationObservableValue(pt, obsKey, 'u_min'));
      } else {
        includeY(bifurcationObservableValue(pt, obsKey, 'u'));
      }
    }
  }
  for (const sp of sps) {
    includeP(sp.p);
    includeY(bifurcationObservableValue(sp, obsKey, 'u'));
  }
  return {
    xlim: _resolveAxisBounds(pMin, pMax, 0.05, cfg.xlim1d),
    ylim: _resolveAxisBounds(yMin, yMax, 0.1, cfg.ylim1d),
    autoXlim: _padBounds(pMin, pMax, 0.05),
    autoYlim: _padBounds(yMin, yMax, 0.1),
    param1, obsKey, obsLabel, branches, specialPoints: sps,
  };
}

function bifurcation2DBounds(model) {
  const cfg = model.get('bifurcation_config') || {};
  const all2D = (model.get('bifurcation_branches') || []).filter(b => b && TWO_D_BRANCH_TYPES.has(b.type));
  const planes = bifurcation2DPlanes(all2D);
  const requestedKey = _parsePlaneSelection(cfg.plane2d);
  const activePlane = (requestedKey && planes.find(pl => pl.key === requestedKey)) || planes[planes.length - 1] || null;
  const paramNames = Object.keys(model.get('params') || {});
  const param1 = activePlane?.param1 || cfg.param1 || paramNames[0] || 'p1';
  const param2 = activePlane?.param2 || cfg.param2 || paramNames.find(k => k !== param1) || 'p2';
  const branches = activePlane
    ? all2D.filter(b => b.param1 === param1 && b.param2 === param2)
    : all2D;
  const branchIds = new Set(branches.map(b => b.id));
  const sps = (model.get('bifurcation_special_points') || []).filter(sp => {
    if (!sp) return false;
    if (sp.branch_id && !branchIds.has(sp.branch_id)) return false;
    if (sp.param1 && sp.param2 && (sp.param1 !== param1 || sp.param2 !== param2)) return false;
    return branchIds.has(sp.branch_id) || Number.isFinite(sp.p2) || sp.ty === 'CP' || sp.ty === 'BT' || sp.ty === 'GH' || sp.ty === 'ZH';
  });
  let p1Min = Infinity, p1Max = -Infinity, p2Min = Infinity, p2Max = -Infinity;
  for (const br of branches) {
    for (const pt of br.points || []) {
      if (Number.isFinite(pt.p)) { p1Min = Math.min(p1Min, pt.p); p1Max = Math.max(p1Max, pt.p); }
      if (Number.isFinite(pt.p2)) { p2Min = Math.min(p2Min, pt.p2); p2Max = Math.max(p2Max, pt.p2); }
    }
  }
  for (const sp of sps) {
    if (Number.isFinite(sp.p)) { p1Min = Math.min(p1Min, sp.p); p1Max = Math.max(p1Max, sp.p); }
    if (Number.isFinite(sp.p2)) { p2Min = Math.min(p2Min, sp.p2); p2Max = Math.max(p2Max, sp.p2); }
  }
  return {
    xlim: _resolveAxisBounds(p1Min, p1Max, 0.05, cfg.xlim2d),
    ylim: _resolveAxisBounds(p2Min, p2Max, 0.05, cfg.ylim2d),
    autoXlim: _padBounds(p1Min, p1Max, 0.05),
    autoYlim: _padBounds(p2Min, p2Max, 0.05),
    param1, param2, branches, specialPoints: sps, planes,
  };
}

function _drawBifurcationEmpty(ctx, w, h, theme, message) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = theme.background || theme.bg || DEFAULT_PLOT_THEME.background;
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = theme.muted || DEFAULT_PLOT_THEME.muted;
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, w / 2, h / 2);
}

function _drawStyledCurve(ctx, points, xlim, ylim, w, h, getXY, getStyle) {
  let activeStyle = null, prevScreen = null;
  const flush = () => {
    if (!activeStyle) return;
    ctx.strokeStyle = activeStyle.strokeStyle;
    ctx.lineWidth = activeStyle.lineWidth;
    ctx.setLineDash(activeStyle.dash);
    ctx.stroke();
    activeStyle = null;
  };
  for (const pt of points || []) {
    const [wx, wy] = getXY(pt);
    if (!Number.isFinite(wx) || !Number.isFinite(wy)) {
      flush();
      prevScreen = null;
      continue;
    }
    const [sx, sy] = worldToScreen(wx, wy, xlim, ylim, w, h);
    const style = getStyle(pt);
    if (!activeStyle || activeStyle.key !== style.key) {
      flush();
      activeStyle = style;
      ctx.beginPath();
      if (prevScreen) {
        ctx.moveTo(prevScreen[0], prevScreen[1]);
        ctx.lineTo(sx, sy);
      } else {
        ctx.moveTo(sx, sy);
      }
    } else {
      ctx.lineTo(sx, sy);
    }
    prevScreen = [sx, sy];
  }
  flush();
  ctx.setLineDash([]);
}

function _drawSpecialPointMarker(ctx, sx, sy, sp, theme) {
  const ty = sp.ty || '';
  const r = 5;
  ctx.save();
  ctx.lineWidth = 1.5;
  if (ty === 'HB' || ty === 'GH') {
    ctx.fillStyle = ty === 'HB' ? '#c62828' : '#00838f';
    ctx.beginPath();
    ctx.moveTo(sx, sy - r); ctx.lineTo(sx + r, sy); ctx.lineTo(sx, sy + r); ctx.lineTo(sx - r, sy);
    ctx.closePath(); ctx.fill();
  } else if (ty === 'LP' || ty === 'BT') {
    ctx.fillStyle = ty === 'LP' ? '#245c9b' : '#d84315';
    ctx.beginPath();
    ctx.moveTo(sx, sy - r); ctx.lineTo(sx + r, sy + r); ctx.lineTo(sx - r, sy + r);
    ctx.closePath(); ctx.fill();
  } else if (ty === 'LPC') {
    ctx.fillStyle = '#2e7d32';
    ctx.fillRect(sx - r + 1, sy - r + 1, 2 * r - 2, 2 * r - 2);
  } else if (ty === 'PD') {
    ctx.strokeStyle = '#e65100';
    ctx.beginPath();
    ctx.moveTo(sx, sy - r); ctx.lineTo(sx + r, sy); ctx.lineTo(sx, sy + r); ctx.lineTo(sx - r, sy);
    ctx.closePath(); ctx.stroke();
  } else if (ty === 'BP' || ty === 'CP') {
    ctx.strokeStyle = '#6a1b9a';
    ctx.beginPath(); ctx.arc(sx, sy, r - 0.5, 0, 2 * Math.PI); ctx.stroke();
  } else if (ty === 'ZH') {
    ctx.strokeStyle = '#00695c';
    ctx.beginPath();
    ctx.rect(sx - r + 1, sy - r + 1, 2 * r - 2, 2 * r - 2);
    ctx.stroke();
  } else {
    ctx.fillStyle = theme.text || DEFAULT_PLOT_THEME.text;
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, 2 * Math.PI); ctx.fill();
  }
  const label = sp.label || sp.id || ty;
  if (label) {
    ctx.fillStyle = theme.text || DEFAULT_PLOT_THEME.text;
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(label, sx + 6, sy - 4);
  }
  ctx.restore();
}

export function drawBifurcation1D(model, ctx, w, h, theme = DEFAULT_PLOT_THEME) {
  const allBranches = model.get('bifurcation_branches') || [];
  if (allBranches.length === 0) {
    _drawBifurcationEmpty(ctx, w, h, theme, 'Run continuation to trace equilibria and limit cycles');
    return;
  }
  const { xlim, ylim, param1, obsKey, obsLabel, branches, specialPoints } = bifurcation1DBounds(model);
  if (branches.length === 0) {
    _drawBifurcationEmpty(ctx, w, h, theme, 'Switch to 2D view to inspect 2-parameter curves');
    return;
  }
  const bg = theme.background || theme.bg || DEFAULT_PLOT_THEME.background;
  const accent = theme.accent || theme.cursor || '#245c9b';
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  drawAxes(ctx, w, h, xlim, ylim, param1, obsLabel, theme);

  ctx.save();
  ctx.beginPath();
  ctx.rect(42, 42, w - 84, h - 84);
  ctx.clip();

  const epStyle = (pt) => pt.stable === true
    ? { key: 'ep-s', strokeStyle: accent, dash: [], lineWidth: 2 }
    : { key: 'ep-u', strokeStyle: '#7f8c9a', dash: [6, 4], lineWidth: 1.75 };
  const lcStyle = (pt) => pt.stable === true
    ? { key: 'lc-s', strokeStyle: '#2e7d32', dash: [], lineWidth: 2.5 }
    : { key: 'lc-u', strokeStyle: '#e65100', dash: [3, 3], lineWidth: 2 };

  for (const br of branches) {
    if (br.type === 'EP') {
      _drawStyledCurve(ctx, br.points, xlim, ylim, w, h,
        pt => [pt.p, bifurcationObservableValue(pt, obsKey, 'u')], epStyle);
    } else if (br.type === 'LC') {
      _drawStyledCurve(ctx, br.points, xlim, ylim, w, h,
        pt => [pt.p, bifurcationObservableValue(pt, obsKey, 'u_max')], lcStyle);
      _drawStyledCurve(ctx, br.points, xlim, ylim, w, h,
        pt => [pt.p, bifurcationObservableValue(pt, obsKey, 'u_min')], lcStyle);
    }
  }

  for (const sp of specialPoints) {
    const wy = bifurcationObservableValue(sp, obsKey, 'u');
    if (!Number.isFinite(sp.p) || !Number.isFinite(wy)) continue;
    const [sx, sy] = worldToScreen(sp.p, wy, xlim, ylim, w, h);
    _drawSpecialPointMarker(ctx, sx, sy, sp, theme);
  }

  const currentP = Number(model.get('params')?.[param1]);
  if (Number.isFinite(currentP) && currentP >= xlim[0] && currentP <= xlim[1]) {
    const [sx] = worldToScreen(currentP, ylim[0], xlim, ylim, w, h);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = theme.cursor || accent;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.moveTo(sx, 42);
    ctx.lineTo(sx, h - 42);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

export function drawBifurcation2D(model, ctx, w, h, theme = DEFAULT_PLOT_THEME) {
  const { xlim, ylim, param1, param2, branches, specialPoints } = bifurcation2DBounds(model);
  if (branches.length === 0) {
    _drawBifurcationEmpty(ctx, w, h, theme, 'No 2D continuation curves yet');
    return;
  }
  const bg = theme.background || theme.bg || DEFAULT_PLOT_THEME.background;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  drawAxes(ctx, w, h, xlim, ylim, param1, param2, theme);

  ctx.save();
  ctx.beginPath();
  ctx.rect(42, 42, w - 84, h - 84);
  ctx.clip();

  for (const br of branches) {
    const style = TWO_D_STYLES[br.type] || TWO_D_STYLES.LP2;
    _drawStyledCurve(ctx, br.points, xlim, ylim, w, h,
      pt => [pt.p, pt.p2], () => style);
  }

  for (const sp of specialPoints) {
    if (!Number.isFinite(sp.p) || !Number.isFinite(sp.p2)) continue;
    const [sx, sy] = worldToScreen(sp.p, sp.p2, xlim, ylim, w, h);
    _drawSpecialPointMarker(ctx, sx, sy, sp, theme);
  }

  const params = model.get('params') || {};
  const curP1 = Number(params[param1]), curP2 = Number(params[param2]);
  if (Number.isFinite(curP1) && Number.isFinite(curP2)) {
    const [sx, sy] = worldToScreen(curP1, curP2, xlim, ylim, w, h);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = theme.cursor || theme.accent || '#172d46';
    ctx.lineWidth = 1.25;
    if (curP1 >= xlim[0] && curP1 <= xlim[1]) {
      ctx.beginPath(); ctx.moveTo(sx, 42); ctx.lineTo(sx, h - 42); ctx.stroke();
    }
    if (curP2 >= ylim[0] && curP2 <= ylim[1]) {
      ctx.beginPath(); ctx.moveTo(42, sy); ctx.lineTo(w - 42, sy); ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // Compact in-canvas legend for plotted 2D curve types (also appears in SVG export).
  const activeTypes = [...new Set(branches.map(b => b.type).filter(t => TWO_D_STYLES[t]))];
  if (activeTypes.length > 0 && w >= 220 && h >= 140) {
    const boxW = 148, rowH = 15, boxH = 8 + activeTypes.length * rowH;
    const bx = w - 48 - boxW, by = 48;
    ctx.save();
    ctx.globalAlpha = 0.88;
    ctx.fillStyle = bg;
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.restore();
    ctx.strokeStyle = theme.grid || '#dce3eb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(bx, by, boxW, boxH);
    ctx.stroke();
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    activeTypes.forEach((ty, idx) => {
      const style = TWO_D_STYLES[ty];
      const ry = by + 6 + idx * rowH + rowH / 2 - 2;
      ctx.strokeStyle = style.strokeStyle;
      ctx.lineWidth = style.lineWidth;
      ctx.setLineDash(style.dash);
      ctx.beginPath();
      ctx.moveTo(bx + 8, ry);
      ctx.lineTo(bx + 26, ry);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = theme.text || DEFAULT_PLOT_THEME.text;
      ctx.fillText(style.label, bx + 32, ry);
    });
  }
  ctx.restore();
}
