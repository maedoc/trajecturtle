function worldToScreen(wx, wy, xlim, ylim, w, h) {
  const pad = 42;
  return [pad + ((wx - xlim[0]) / (xlim[1] - xlim[0])) * (w - 2 * pad), h - pad - ((wy - ylim[0]) / (ylim[1] - ylim[0])) * (h - 2 * pad)];
}

function screenToWorld(sx, sy, xlim, ylim, w, h) {
  const pad = 42;
  return [xlim[0] + ((sx - pad) / (w - 2 * pad)) * (xlim[1] - xlim[0]), ylim[0] + ((h - pad - sy) / (h - 2 * pad)) * (ylim[1] - ylim[0])];
}

function drawAxes(ctx, w, h, xlim, ylim, xlabel, ylabel, theme = DEFAULT_PLOT_THEME) {
  const pad = 42;
  ctx.strokeStyle = theme.text;
  ctx.lineWidth = 1;
  const zeroY = worldToScreen(0, 0, xlim, ylim, w, h)[1];
  if (zeroY >= pad && zeroY <= h - pad) {
    ctx.beginPath(); ctx.moveTo(pad, zeroY); ctx.lineTo(w - pad, zeroY); ctx.stroke();
  }
  const zeroX = worldToScreen(0, 0, xlim, ylim, w, h)[0];
  if (zeroX >= pad && zeroX <= w - pad) {
    ctx.beginPath(); ctx.moveTo(zeroX, pad); ctx.lineTo(zeroX, h - pad); ctx.stroke();
  }
  ctx.strokeStyle = theme.grid; ctx.lineWidth = 0.5;
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
  ctx.fillStyle = theme.muted; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
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
