export function mountBifurcation({ model, controller, document: doc, el } = {}) {
  const targetDoc = doc || el?.ownerDocument || globalThis.document;
  const scope = panelScope(), owner = {};
  const element = panelElement(targetDoc, 'ppw-bifurcation', `
    <div class="ppw-bif-header">Bifurcation continuation</div>
    <div class="ppw-bif-toolbar">
      <label>Parameter <select class="ppw-bif-param1" aria-label="Continuation parameter"></select></label>
      <label>Observable <select class="ppw-bif-obs" aria-label="Observable"></select></label>
      <button type="button" class="ppw-bif-trace">Trace equilibria</button>
      <button type="button" class="ppw-bif-cancel" disabled>Cancel</button>
      <button type="button" class="ppw-bif-clear">Clear</button>
      <div class="ppw-bif-view-switch" hidden>
        <button type="button" class="ppw-bif-mode-1d" aria-pressed="true">1D</button>
        <button type="button" class="ppw-bif-mode-2d" aria-pressed="false">2D</button>
      </div>
    </div>
    <div class="ppw-bif-presets" hidden></div>
    <div class="ppw-bif-stale" role="status" hidden>
      <span>Other parameters changed — Re-run</span>
      <button type="button" class="ppw-bif-rerun">Re-run</button>
    </div>
    <div class="ppw-bif-nonsmooth" role="note" hidden></div>
    <div class="ppw-bif-nd-note" hidden>Full N-D continuation; clicking a branch updates clamped variables to match.</div>
    <div class="ppw-bif-error" role="alert" hidden></div>
    <canvas class="ppw-bif-canvas" role="img" aria-label="Bifurcation diagram"></canvas>
    <output class="ppw-bif-status"></output>
    <div class="ppw-bif-chips"></div>
    <details class="ppw-bif-advanced">
      <summary>Advanced settings &amp; diagnostics</summary>
      <div class="ppw-bif-settings">
        <label>rl0 <input type="number" step="any" data-cfg="rl0" placeholder="auto"></label>
        <label>rl1 <input type="number" step="any" data-cfg="rl1" placeholder="auto"></label>
        <label>ds <input type="number" step="any" data-cfg="ds" value="0.05"></label>
        <label>dsmax <input type="number" step="any" data-cfg="dsmax" value="0.2"></label>
        <label>nmx <input type="number" step="1" min="10" data-cfg="nmx" value="200"></label>
        <label>ntst <input type="number" step="1" min="5" data-cfg="ntst" value="50"></label>
        <label>ncol <input type="number" step="1" min="2" max="7" data-cfg="ncol" value="4"></label>
      </div>
      <pre class="ppw-bif-diagnostics"></pre>
    </details>`);

  const param1Select = element.querySelector('.ppw-bif-param1');
  const obsSelect = element.querySelector('.ppw-bif-obs');
  const traceBtn = element.querySelector('.ppw-bif-trace');
  const cancelBtn = element.querySelector('.ppw-bif-cancel');
  const clearBtn = element.querySelector('.ppw-bif-clear');
  const viewSwitch = element.querySelector('.ppw-bif-view-switch');
  const mode1DBtn = element.querySelector('.ppw-bif-mode-1d');
  const mode2DBtn = element.querySelector('.ppw-bif-mode-2d');
  const presetsBox = element.querySelector('.ppw-bif-presets');
  const staleBanner = element.querySelector('.ppw-bif-stale');
  const rerunBtn = element.querySelector('.ppw-bif-rerun');
  const nonSmoothBox = element.querySelector('.ppw-bif-nonsmooth');
  const ndNote = element.querySelector('.ppw-bif-nd-note');
  const errorBox = element.querySelector('.ppw-bif-error');
  const canvas = element.querySelector('.ppw-bif-canvas');
  const statusOut = element.querySelector('.ppw-bif-status');
  const chipsBox = element.querySelector('.ppw-bif-chips');
  const diagPre = element.querySelector('.ppw-bif-diagnostics');
  const advInputs = [...element.querySelectorAll('.ppw-bif-settings input[data-cfg]')];

  let viewMode = '1d';
  let paramSig = '', obsSig = '', presetsSig = '', chipsSig = '';

  const surface = canvasSurface(
    canvas,
    scope,
    (ctx, w, h, theme) => (viewMode === '2d'
      ? drawBifurcation2D(model, ctx, w, h, theme)
      : drawBifurcation1D(model, ctx, w, h, theme)),
    { width: 560, ratio: 0.62, minHeight: 240 },
  );

  function readAdvancedConfig() {
    const out = {};
    for (const input of advInputs) {
      const key = input.dataset.cfg;
      const raw = input.value.trim();
      if (raw === '') continue;
      const num = Number(raw);
      if (Number.isFinite(num)) out[key] = num;
    }
    return out;
  }

  function updateConfig(patch) {
    const prev = model.get('bifurcation_config') || {};
    controller.commit({ bifurcation_config: { ...prev, ...patch } });
  }

  function canvasCoords(event) {
    const rect = canvas.getBoundingClientRect();
    const w = surface.width, h = surface.height;
    const rw = rect.width || w, rh = rect.height || h;
    return [
      (event.clientX - rect.left) * w / rw,
      (event.clientY - rect.top) * h / rh,
      w,
      h,
    ];
  }

  function findNearest1D(sx, sy, w, h) {
    const bounds = bifurcation1DBounds(model);
    let best = null, bestDist = Infinity;
    for (const br of bounds.branches) {
      for (const pt of br.points || []) {
        const yVal = br.type === 'LC'
          ? bifurcationObservableValue(pt, bounds.obsKey, 'u_max')
          : bifurcationObservableValue(pt, bounds.obsKey, 'u');
        if (!Number.isFinite(pt.p) || !Number.isFinite(yVal)) continue;
        const [px, py] = worldToScreen(pt.p, yVal, bounds.xlim, bounds.ylim, w, h);
        const d = Math.hypot(px - sx, py - sy);
        if (d < bestDist) { bestDist = d; best = { point: pt, branch: br, yVal, dist: d }; }
        if (br.type === 'LC') {
          const yMin = bifurcationObservableValue(pt, bounds.obsKey, 'u_min');
          if (Number.isFinite(yMin)) {
            const [px2, py2] = worldToScreen(pt.p, yMin, bounds.xlim, bounds.ylim, w, h);
            const d2 = Math.hypot(px2 - sx, py2 - sy);
            if (d2 < bestDist) { bestDist = d2; best = { point: pt, branch: br, yVal: yMin, dist: d2 }; }
          }
        }
      }
    }
    return { best, bounds };
  }

  function findNearest2D(sx, sy, w, h) {
    const bounds = bifurcation2DBounds(model);
    let best = null, bestDist = Infinity;
    for (const br of bounds.branches) {
      for (const pt of br.points || []) {
        if (!Number.isFinite(pt.p) || !Number.isFinite(pt.p2)) continue;
        const [px, py] = worldToScreen(pt.p, pt.p2, bounds.xlim, bounds.ylim, w, h);
        const d = Math.hypot(px - sx, py - sy);
        if (d < bestDist) { bestDist = d; best = { point: pt, branch: br, dist: d }; }
      }
    }
    return { best, bounds };
  }

  scope.on(param1Select, 'change', () => updateConfig({ param1: param1Select.value }));
  scope.on(obsSelect, 'change', () => {
    const val = obsSelect.value;
    updateConfig({ observable: val === 'L2' ? 'L2' : Number(val) });
  });
  scope.on(traceBtn, 'click', () => {
    controller.startEquilibriumContinuation({ param1: param1Select.value, ...readAdvancedConfig() }, owner);
  });
  scope.on(rerunBtn, 'click', () => {
    controller.startEquilibriumContinuation({ param1: param1Select.value, ...readAdvancedConfig() }, owner);
  });
  scope.on(cancelBtn, 'click', () => controller.cancelBifurcation(owner));
  scope.on(clearBtn, 'click', () => controller.clearBifurcation());
  scope.on(mode1DBtn, 'click', () => { viewMode = '1d'; update(); });
  scope.on(mode2DBtn, 'click', () => { viewMode = '2d'; update(); });

  for (const input of advInputs) {
    scope.on(input, 'change', () => {
      const key = input.dataset.cfg;
      const raw = input.value.trim();
      const prev = { ...(model.get('bifurcation_config') || {}) };
      if (raw === '') delete prev[key];
      else if (Number.isFinite(Number(raw))) prev[key] = Number(raw);
      controller.commit({ bifurcation_config: prev });
    });
  }

  scope.on(canvas, 'click', (event) => {
    const [sx, sy, w, h] = canvasCoords(event);
    if (sx < 42 || sy < 42 || sx > w - 42 || sy > h - 42) return;
    if (viewMode === '2d') {
      const { best, bounds } = findNearest2D(sx, sy, w, h);
      if (best && best.dist <= 30) {
        controller.jumpToBifurcationPoint(best.point, best.branch);
      } else {
        const [p1Val, p2Val] = screenToWorld(sx, sy, bounds.xlim, bounds.ylim, w, h);
        controller.jumpToBifurcationPoint(
          { p: p1Val, p2: p2Val },
          { type: 'LP2', param1: bounds.param1, param2: bounds.param2 },
        );
      }
      return;
    }
    const { best, bounds } = findNearest1D(sx, sy, w, h);
    if (best && best.dist <= 30) {
      controller.jumpToBifurcationPoint(best.point, best.branch);
    } else {
      const [pClick] = screenToWorld(sx, sy, bounds.xlim, bounds.ylim, w, h);
      controller.jumpToBifurcationPoint({ p: pClick }, { type: 'EP', param1: bounds.param1 });
    }
  });

  scope.on(canvas, 'mousemove', (event) => {
    const branches = model.get('bifurcation_branches') || [];
    if (branches.length === 0) return;
    const [sx, sy, w, h] = canvasCoords(event);
    if (sx < 42 || sy < 42 || sx > w - 42 || sy > h - 42) {
      statusOut.textContent = '';
      return;
    }
    if (viewMode === '2d') {
      const { best, bounds } = findNearest2D(sx, sy, w, h);
      if (best) {
        statusOut.textContent = `${bounds.param1} = ${Number(best.point.p).toPrecision(4)} · ${bounds.param2} = ${Number(best.point.p2).toPrecision(4)} · ${best.branch.type}`;
      }
      return;
    }
    const { best } = findNearest1D(sx, sy, w, h);
    if (!best) return;
    const pt = best.point;
    const stab = pt.stable ? 'stable' : 'unstable';
    if (best.branch.type === 'LC') {
      const period = Number(pt.period);
      const freq = Number.isFinite(period) && period > 0 ? 1 / period : NaN;
      statusOut.textContent = `p = ${Number(pt.p).toPrecision(4)} · period = ${Number.isFinite(period) ? period.toPrecision(4) : '—'} · freq = ${Number.isFinite(freq) ? freq.toPrecision(4) : '—'} · ${stab} LC`;
    } else {
      statusOut.textContent = `p = ${Number(pt.p).toPrecision(4)} · y = ${Number(best.yVal).toPrecision(4)} · ${stab} EP`;
    }
  });

  scope.on(canvas, 'mouseleave', () => { statusOut.textContent = ''; });

  function renderChips(specialPoints, branches, paramNames, running) {
    const branchById = new Map(branches.map(b => [b.id, b]));
    chipsBox.replaceChildren();
    for (const sp of specialPoints) {
      const chip = targetDoc.createElement('div');
      chip.className = 'ppw-bif-chip';
      chip.dataset.spId = sp.id;
      chip.dataset.spTy = sp.ty;

      const jumpBtn = targetDoc.createElement('button');
      jumpBtn.type = 'button';
      jumpBtn.className = 'ppw-bif-chip-jump';
      jumpBtn.textContent = `${sp.label || sp.id} (p=${Number(sp.p).toPrecision(4)})`;
      jumpBtn.addEventListener('click', () => controller.jumpToBifurcationPoint(sp, branchById.get(sp.branch_id) || null));
      chip.append(jumpBtn);

      if (sp.ty === 'HB') {
        const lcBtn = targetDoc.createElement('button');
        lcBtn.type = 'button';
        lcBtn.className = 'ppw-bif-switch-lc';
        lcBtn.textContent = '→ Limit cycle';
        lcBtn.disabled = running;
        lcBtn.addEventListener('click', () => controller.switchToLimitCycle(sp.id, readAdvancedConfig(), owner));
        chip.append(lcBtn);
      }

      if (sp.ty === 'HB' || sp.ty === 'LP') {
        const otherParams = paramNames.filter(p => p !== sp.param1);
        if (otherParams.length > 0) {
          const p2Select = targetDoc.createElement('select');
          p2Select.className = 'ppw-bif-param2';
          p2Select.setAttribute('aria-label', `Second parameter for ${sp.label || sp.id}`);
          for (const p of otherParams) {
            const opt = targetDoc.createElement('option');
            opt.value = p; opt.textContent = p;
            p2Select.append(opt);
          }
          const twoDBtn = targetDoc.createElement('button');
          twoDBtn.type = 'button';
          twoDBtn.className = 'ppw-bif-trace-2d';
          twoDBtn.textContent = '→ 2D curve';
          twoDBtn.disabled = running;
          twoDBtn.addEventListener('click', () => controller.startTwoParamContinuation(sp.id, p2Select.value, readAdvancedConfig(), owner));
          chip.append(p2Select, twoDBtn);
        }
      }

      if (sp.ty === 'LPC' || sp.ty === 'PD') {
        const badge = targetDoc.createElement('span');
        badge.className = 'ppw-bif-chip-badge';
        badge.textContent = 'LC-to-2D (v2)';
        badge.title = 'LC-to-2D continuation is planned for v2';
        chip.append(badge);
      }

      chipsBox.append(chip);
    }
  }

  function update() {
    const paramNames = Object.keys(model.get('param_info') || model.get('params') || {});
    const stateNames = model.get('state_names') || [];
    const cfg = model.get('bifurcation_config') || {};
    const branches = model.get('bifurcation_branches') || [];
    const specialPoints = model.get('bifurcation_special_points') || [];
    const presets = model.get('bifurcation_presets') || [];
    const meta = model.get('bifurcation_metadata') || {};
    const running = !!model.get('bifurcation_running');

    const nextParamSig = JSON.stringify(paramNames);
    if (nextParamSig !== paramSig) {
      paramSig = nextParamSig;
      const prev = param1Select.value;
      param1Select.replaceChildren();
      for (const p of paramNames) {
        const opt = targetDoc.createElement('option');
        opt.value = p; opt.textContent = p;
        param1Select.append(opt);
      }
      if (cfg.param1 && paramNames.includes(cfg.param1)) param1Select.value = cfg.param1;
      else if (prev && paramNames.includes(prev)) param1Select.value = prev;
    } else if (cfg.param1 && paramNames.includes(cfg.param1) && param1Select.value !== cfg.param1) {
      param1Select.value = cfg.param1;
    }

    const nextObsSig = JSON.stringify(stateNames);
    if (nextObsSig !== obsSig) {
      obsSig = nextObsSig;
      obsSelect.replaceChildren();
      stateNames.forEach((name, idx) => {
        const opt = targetDoc.createElement('option');
        opt.value = String(idx); opt.textContent = name;
        obsSelect.append(opt);
      });
      const l2Opt = targetDoc.createElement('option');
      l2Opt.value = 'L2'; l2Opt.textContent = 'L2 norm';
      obsSelect.append(l2Opt);
    }
    const desiredObs = cfg.observable === 'L2' || cfg.observable === 'l2'
      ? 'L2'
      : String(Number.isInteger(cfg.observable) ? cfg.observable : 0);
    if ([...obsSelect.options].some(o => o.value === desiredObs)) obsSelect.value = desiredObs;

    const nextPresetsSig = JSON.stringify(presets.map(p => p.label || p.name || p.param1));
    if (nextPresetsSig !== presetsSig) {
      presetsSig = nextPresetsSig;
      presetsBox.replaceChildren();
      presetsBox.hidden = presets.length === 0;
      presets.forEach((preset, idx) => {
        const btn = targetDoc.createElement('button');
        btn.type = 'button';
        btn.className = 'ppw-bif-preset-btn';
        btn.textContent = preset.label || preset.name || `Preset ${idx + 1}`;
        btn.addEventListener('click', () => controller.runBifurcationPreset(idx, owner));
        presetsBox.append(btn);
      });
    }
    for (const btn of presetsBox.querySelectorAll('button')) btn.disabled = running;

    traceBtn.disabled = running || !param1Select.value;
    cancelBtn.disabled = !running;
    clearBtn.disabled = running && branches.length === 0;

    const has2D = branches.some(b => b && (b.type === 'LP2' || b.type === 'HB2'));
    viewSwitch.hidden = !has2D;
    if (!has2D && viewMode === '2d') viewMode = '1d';
    mode1DBtn.setAttribute('aria-pressed', String(viewMode === '1d'));
    mode2DBtn.setAttribute('aria-pressed', String(viewMode === '2d'));

    staleBanner.hidden = !meta.stale;
    rerunBtn.disabled = running;

    const nonSmoothTokens = detectNonSmoothExpressions(model.get('model_spec'));
    nonSmoothBox.hidden = nonSmoothTokens.length === 0;
    if (nonSmoothTokens.length > 0) {
      nonSmoothBox.textContent = `Non-smooth expression detected (${nonSmoothTokens.join(', ')}): continuation may fail or miss bifurcations.`;
    }

    ndNote.hidden = stateNames.length <= 2;

    errorBox.hidden = !meta.error;
    errorBox.textContent = meta.error || '';

    for (const input of advInputs) {
      const key = input.dataset.cfg;
      if (key in cfg && targetDoc.activeElement !== input) input.value = cfg[key];
    }
    diagPre.textContent = meta.fort9 || meta.status || '';

    const nextChipsSig = JSON.stringify({ specialPoints, paramNames, running });
    if (nextChipsSig !== chipsSig) {
      chipsSig = nextChipsSig;
      renderChips(specialPoints, branches, paramNames, running);
    }

    surface.render();
  }

  scope.cleanup(() => controller.cancelBifurcation(owner));
  if (el) {
    el.append(element);
    update();
  }
  return mountedPanel(element, scope, update);
}
