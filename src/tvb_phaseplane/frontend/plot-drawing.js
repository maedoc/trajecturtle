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
