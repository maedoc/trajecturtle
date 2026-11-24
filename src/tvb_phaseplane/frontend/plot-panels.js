function mountPhasePlane({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-phase-container', `
    <div class="ppw-canvas-title">Phase plane</div>
    <canvas class="ppw-phase-canvas" tabindex="0" role="img" aria-label="Phase plane. Click or use arrow keys to set the initial condition. Shift increases the step."></canvas>
    <div class="ppw-canvas-hint">Click or use arrow keys to set initial conditions · Shift for larger steps</div>
    <output class="ppw-initial-state"></output>
    <div class="ppw-legend"><span class="ppw-nullcline-x" style="color:#2196f3"></span><span class="ppw-nullcline-y" style="color:#f44336"></span><span style="color:#4caf50">● Stable</span><span style="color:#9c27b0">◆ Saddle</span><span style="color:#f44336">○ Unstable</span><span style="color:#4caf50">— Trajectory</span><span>┄ Reference</span></div>`);
  const canvas = element.querySelector('canvas');
  const surface = canvasSurface(canvas, scope, (ctx, w, h, theme) => drawPhasePlane(model, ctx, w, h, theme));
  scope.on(canvas, 'click', event => {
    const rect = canvas.getBoundingClientRect(), w = surface.width, h = surface.height;
    const sx = (event.clientX - rect.left) * w / rect.width, sy = (event.clientY - rect.top) * h / rect.height;
    if (sx < 42 || sy < 42 || sx > w - 42 || sy > h - 42) return;
    const [x0, y0] = screenToWorld(sx, sy, model.get('xlim'), model.get('ylim'), w, h);
    controller.commit({ x0, y0 }); canvas.focus();
  });
  scope.on(canvas, 'keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const [xmin, xmax] = model.get('xlim'), [ymin, ymax] = model.get('ylim');
    const fraction = event.shiftKey ? .1 : .01;
    const x0 = model.get('x0') + (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0) * (xmax - xmin) * fraction;
    const y0 = model.get('y0') + (event.key === 'ArrowUp' ? 1 : event.key === 'ArrowDown' ? -1 : 0) * (ymax - ymin) * fraction;
    controller.commit({ x0: Math.max(xmin, Math.min(xmax, x0)), y0: Math.max(ymin, Math.min(ymax, y0)) });
  });
  return mountedPanel(element, scope, () => {
    surface.render();
    const display = model.get('display') || [0, 1], names = model.get('state_names') || [];
    element.querySelector('.ppw-nullcline-x').textContent = `d${names[display[0]]}/dt = 0`;
    element.querySelector('.ppw-nullcline-y').textContent = display.length > 1 ? `d${names[display[1]]}/dt = 0` : '';
    element.querySelector('.ppw-initial-state').textContent = display.map((index, i) => `${names[index]}₀ = ${Number(model.get(i ? 'y0' : 'x0')).toFixed(3)}`).join(' · ');
  });
}

function mountTimeSeries({ model, controller, document: doc }) {
  const scope = panelScope(), owner = {}, element = panelElement(doc, 'ppw-time-container', `
    <div class="ppw-canvas-title">Time series</div><canvas class="ppw-time-canvas" role="img" aria-label="Time series"></canvas>
    <div class="ppw-playback"><button class="ppw-play">Play trajectory</button><input class="ppw-time-cursor" aria-label="Trajectory time" type="range" min="0" max="1" step="0.001"><output class="ppw-cursor-time"></output></div>`);
  const surface = canvasSurface(element.querySelector('canvas'), scope, (ctx, w, h, theme) => drawTimeSeries(model, ctx, w, h, theme), { ratio: .56, minHeight: 230 });
  const cursor = element.querySelector('.ppw-time-cursor'), button = element.querySelector('.ppw-play');
  scope.on(cursor, 'input', () => controller.commit({ trajectory_fraction: Number(cursor.value) }));
  scope.on(button, 'click', () => controller.play(owner));
  scope.cleanup(() => controller.pause(owner));
  return mountedPanel(element, scope, () => {
    surface.render();
    const trajectory = model.get('trajectory') || [], fraction = Math.max(0, Math.min(1, model.get('trajectory_fraction') ?? 1));
    cursor.value = fraction; cursor.disabled = trajectory.length < 2; button.disabled = trajectory.length < 2;
    const row = trajectory[Math.round(fraction * (trajectory.length - 1))];
    element.querySelector('.ppw-cursor-time').textContent = row ? 't = ' + row[0].toFixed(2) : 'No trajectory';
    button.textContent = controller.playing ? 'Pause' : 'Play trajectory'; button.setAttribute('aria-pressed', String(controller.playing));
  });
}

function mountSweep({ model, controller, document: doc }) {
  const scope = panelScope(), owner = {}, element = panelElement(doc, 'ppw-sweep', `
    <div class="ppw-sweep-header">Parameter sweep</div>
    <div class="ppw-sweep-controls"><label>Parameter <select class="ppw-sweep-select" aria-label="Sweep parameter"></select></label>
    <label>Minimum <input class="ppw-sweep-min" type="number" step="any"></label><label>Maximum <input class="ppw-sweep-max" type="number" step="any"></label>
    <label>Samples <input class="ppw-sweep-n" type="number" value="50" min="2" max="200" step="1"></label>
    <button class="ppw-sweep-btn">Run sweep</button><button class="ppw-sweep-cancel">Cancel</button></div>
    <div class="ppw-sweep-progress"><div class="ppw-sweep-progress-bar"></div><span class="ppw-sweep-progress-text"></span></div>
    <div class="ppw-sweep-status" role="status"></div><p class="ppw-sweep-empty ppw-hint">Run a sweep to compare regimes and slice equilibria across parameter values.</p><canvas class="ppw-sweep-canvas" role="img" aria-label="Parameter sweep"></canvas>
    <div class="ppw-legend ppw-regime-legend" hidden><span><i style="background:#e3f2fd"></i>Fixed-point regime</span><span><i style="background:#fff3e0"></i>Oscillatory regime</span><span><i style="background:#f5f5f5"></i>Other</span></div>`);
  const select = element.querySelector('select'), min = element.querySelector('.ppw-sweep-min'), max = element.querySelector('.ppw-sweep-max'), count = element.querySelector('.ppw-sweep-n'), run = element.querySelector('.ppw-sweep-btn'), cancel = element.querySelector('.ppw-sweep-cancel');
  let signature;
  const surface = canvasSurface(element.querySelector('canvas'), scope, (ctx, w, h, theme) => drawSweep(model, ctx, w, h, undefined, undefined, undefined, theme), { width: 700, ratio: 3 / 7, minHeight: 240 });
  function bounds() { const info = model.get('param_info')?.[select.value]; if (info) { min.value = info[0]; max.value = info[1]; } }
  scope.on(select, 'change', bounds);
  scope.on(run, 'click', () => controller.startSweep({ parameter: select.value,
    min: min.value === '' ? NaN : Number(min.value), max: max.value === '' ? NaN : Number(max.value), count: Number(count.value),
  }, owner));
  scope.on(cancel, 'click', () => controller.cancelSweep());
  scope.cleanup(() => controller.cancelSweep(owner));
  return mountedPanel(element, scope, () => {
    const info = model.get('param_info') || {}, next = JSON.stringify(info);
    if (signature !== next) {
      signature = next; const previous = select.value; select.replaceChildren();
      for (const name of Object.keys(info)) { const option = doc.createElement('option'); option.value = name; option.textContent = name; select.append(option); }
      if (previous in info) select.value = previous; else bounds();
    }
    const running = !!model.get('sweep_running'); run.disabled = running || !select.value; cancel.disabled = !running;
    element.querySelector('.ppw-sweep-status').textContent = controller.sweep.status;
    element.querySelector('.ppw-sweep-progress').classList.toggle('visible', running);
    element.querySelector('.ppw-sweep-progress-bar').style.width = controller.sweep.progress + '%';
    element.querySelector('.ppw-sweep-progress-text').textContent = controller.sweep.progress + '%';
    const hasResults = !!model.get('sweep_results')?.length;
    element.querySelector('.ppw-sweep-empty').hidden = hasResults;
    element.querySelector('.ppw-regime-legend').hidden = !hasResults;
    element.querySelector('canvas').hidden = !hasResults;
    if (hasResults) surface.render();
  });
}
