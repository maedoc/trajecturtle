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
      <label class="ppw-bif-plane-label" hidden>2D plane <select class="ppw-bif-plane2d" aria-label="2D parameter plane"></select></label>
    </div>
    <div class="ppw-bif-presets" hidden></div>
    <div class="ppw-bif-axes">
      <span class="ppw-bif-axes-title">Plot axes</span>
      <label>x min <input type="number" step="any" class="ppw-bif-xmin" aria-label="Bifurcation X minimum" placeholder="auto"></label>
      <label>x max <input type="number" step="any" class="ppw-bif-xmax" aria-label="Bifurcation X maximum" placeholder="auto"></label>
      <label>y min <input type="number" step="any" class="ppw-bif-ymin" aria-label="Bifurcation Y minimum" placeholder="auto"></label>
      <label>y max <input type="number" step="any" class="ppw-bif-ymax" aria-label="Bifurcation Y maximum" placeholder="auto"></label>
      <button type="button" class="ppw-bif-axes-auto">Auto</button>
      <span class="ppw-bif-axes-error" role="status" hidden></span>
    </div>
    <div class="ppw-bif-stale" role="status" hidden>
      <span>Other parameters changed — Re-run</span>
      <button type="button" class="ppw-bif-rerun">Re-run</button>
    </div>
    <div class="ppw-bif-nonsmooth" role="note" hidden></div>
    <div class="ppw-bif-nd-note" hidden>Full N-D continuation; clicking a branch updates clamped variables to match.</div>
    <div class="ppw-bif-error" role="alert" hidden></div>
    <canvas class="ppw-bif-canvas" role="img" aria-label="Bifurcation diagram"></canvas>
    <div class="ppw-bif-legend"></div>
    <div class="ppw-bif-support-note">Supported: 1D EP (LP, HB) · 1D LC (LPC, PD, TR) · 2D (LP2, HB2, LPC2, PD2, HOM2) · Codim-2 (CP, BT, GH, ZH).</div>
    <output class="ppw-bif-status"></output>
    <div class="ppw-bif-chips"></div>
    <details class="ppw-bif-advanced">
      <summary>Advanced settings &amp; diagnostics</summary>
      <div class="ppw-bif-settings">
        <label>rl0 <input type="number" step="any" data-cfg="rl0" placeholder="auto"></label>
        <label>rl1 <input type="number" step="any" data-cfg="rl1" placeholder="auto"></label>
        <label>rl2_0 <input type="number" step="any" data-cfg="rl2_0" placeholder="auto"></label>
        <label>rl2_1 <input type="number" step="any" data-cfg="rl2_1" placeholder="auto"></label>
        <label>stopPeriod <input type="number" step="any" min="1" data-cfg="stopPeriod" placeholder="auto (e.g. 30)"></label>
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
  const planeLabel = element.querySelector('.ppw-bif-plane-label');
  const plane2DSelect = element.querySelector('.ppw-bif-plane2d');
  const xminInput = element.querySelector('.ppw-bif-xmin');
  const xmaxInput = element.querySelector('.ppw-bif-xmax');
  const yminInput = element.querySelector('.ppw-bif-ymin');
  const ymaxInput = element.querySelector('.ppw-bif-ymax');
  const autoAxesBtn = element.querySelector('.ppw-bif-axes-auto');
  const axesError = element.querySelector('.ppw-bif-axes-error');
  const presetsBox = element.querySelector('.ppw-bif-presets');
  const staleBanner = element.querySelector('.ppw-bif-stale');
  const rerunBtn = element.querySelector('.ppw-bif-rerun');
  const nonSmoothBox = element.querySelector('.ppw-bif-nonsmooth');
  const ndNote = element.querySelector('.ppw-bif-nd-note');
  const errorBox = element.querySelector('.ppw-bif-error');
  const canvas = element.querySelector('.ppw-bif-canvas');
  const legendBox = element.querySelector('.ppw-bif-legend');
  const statusOut = element.querySelector('.ppw-bif-status');
  const chipsBox = element.querySelector('.ppw-bif-chips');
  const diagPre = element.querySelector('.ppw-bif-diagnostics');
  const advInputs = [...element.querySelectorAll('.ppw-bif-settings input[data-cfg]')];
  const twoDTypes = typeof TWO_D_BRANCH_TYPES !== 'undefined'
    ? TWO_D_BRANCH_TYPES
    : new Set(['LP2', 'HB2', 'LPC2', 'PD2', 'HOM2']);

  let viewMode = '1d';
  let paramSig = '', obsSig = '', presetsSig = '', chipsSig = '', planesSig = '', legendMode = '';
  let lastModelSig = '', last2DCount = 0, lastLatest2DKey = '';

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

  function axisKeys() {
    return viewMode === '2d' ? ['xlim2d', 'ylim2d'] : ['xlim1d', 'ylim1d'];
  }

  function parseAxisPair(minInput, maxInput, autoPair) {
    const minRaw = minInput.value.trim(), maxRaw = maxInput.value.trim();
    if (minRaw === '' && maxRaw === '') return { value: null };
    const lo = minRaw === '' ? null : Number(minRaw);
    const hi = maxRaw === '' ? null : Number(maxRaw);
    if ((lo !== null && !Number.isFinite(lo)) || (hi !== null && !Number.isFinite(hi))) {
      return { error: 'Enter finite axis bounds' };
    }
    const effLo = lo !== null ? lo : autoPair[0];
    const effHi = hi !== null ? hi : autoPair[1];
    if (effLo >= effHi) return { error: 'Axis minimum must be less than maximum' };
    return { value: [lo, hi] };
  }

  function commitAxisInputs() {
    const bounds = viewMode === '2d' ? bifurcation2DBounds(model) : bifurcation1DBounds(model);
    const xRes = parseAxisPair(xminInput, xmaxInput, bounds.autoXlim || bounds.xlim);
    const yRes = parseAxisPair(yminInput, ymaxInput, bounds.autoYlim || bounds.ylim);
    const err = xRes.error || yRes.error || '';
    axesError.hidden = !err;
    axesError.textContent = err;
    if (err) return;
    const [xKey, yKey] = axisKeys();
    const next = { ...(model.get('bifurcation_config') || {}) };
    if (xRes.value === null) delete next[xKey];
    else next[xKey] = xRes.value;
    if (yRes.value === null) delete next[yKey];
    else next[yKey] = yRes.value;
    controller.commit({ bifurcation_config: next });
  }

  function renderLegend(mode) {
    if (mode === legendMode) return;
    legendMode = mode;
    if (mode === '2d') {
      legendBox.innerHTML = [
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line" style="border-color:#245c9b"></i>Fold (LP2)</span>',
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line" style="border-color:#c62828"></i>Hopf (HB2)</span>',
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line ppw-bif-dash" style="border-color:#2e7d32"></i>LC Fold (LPC2)</span>',
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line" style="border-color:#e65100"></i>Period Doubling (PD2)</span>',
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line ppw-bif-dashdot" style="border-color:#6a1b9a"></i>Homoclinic (HOM2)</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#6a1b9a">○</span> CP</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#d84315">▲</span> BT</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#00838f">◆</span> GH</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#00695c">□</span> ZH</span>',
      ].join('');
    } else {
      legendBox.innerHTML = [
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line" style="border-color:#245c9b"></i>Stable EP</span>',
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line ppw-bif-dash" style="border-color:#7f8c9a"></i>Unstable EP</span>',
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line" style="border-color:#2e7d32"></i>Stable LC</span>',
        '<span class="ppw-bif-leg-item"><i class="ppw-bif-line ppw-bif-dot" style="border-color:#e65100"></i>Unstable LC</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#245c9b">▲</span> LP</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#c62828">◆</span> HB</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#2e7d32">■</span> LPC</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#e65100">◇</span> PD</span>',
        '<span class="ppw-bif-leg-item"><span style="color:#6a1b9a">○</span> BP</span>',
      ].join('');
    }
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

  function inPlotScreen(px, py, w, h) {
    return px >= 36 && px <= w - 36 && py >= 36 && py <= h - 36;
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
        if (inPlotScreen(px, py, w, h)) {
          const d = Math.hypot(px - sx, py - sy);
          if (d < bestDist) { bestDist = d; best = { point: pt, branch: br, yVal, dist: d }; }
        }
        if (br.type === 'LC') {
          const yMin = bifurcationObservableValue(pt, bounds.obsKey, 'u_min');
          if (Number.isFinite(yMin)) {
            const [px2, py2] = worldToScreen(pt.p, yMin, bounds.xlim, bounds.ylim, w, h);
            if (inPlotScreen(px2, py2, w, h)) {
              const d2 = Math.hypot(px2 - sx, py2 - sy);
              if (d2 < bestDist) { bestDist = d2; best = { point: pt, branch: br, yVal: yMin, dist: d2 }; }
            }
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
        if (!inPlotScreen(px, py, w, h)) continue;
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
  scope.on(plane2DSelect, 'change', () => updateConfig({ plane2d: plane2DSelect.value }));
  scope.on(traceBtn, 'click', () => {
    controller.startEquilibriumContinuation({ param1: param1Select.value, ...readAdvancedConfig() }, owner);
  });
  scope.on(rerunBtn, 'click', () => {
    controller.startEquilibriumContinuation({ param1: param1Select.value, ...readAdvancedConfig() }, owner);
  });
  scope.on(cancelBtn, 'click', () => controller.cancelBifurcation(owner));
  scope.on(clearBtn, 'click', () => controller.clearBifurcation());
  scope.on(mode1DBtn, 'click', () => { viewMode = '1d'; updateConfig({ viewMode: '1d' }); });
  scope.on(mode2DBtn, 'click', () => { viewMode = '2d'; updateConfig({ viewMode: '2d' }); });

  for (const input of [xminInput, xmaxInput, yminInput, ymaxInput]) {
    scope.on(input, 'change', commitAxisInputs);
  }
  scope.on(autoAxesBtn, 'click', () => {
    axesError.hidden = true;
    axesError.textContent = '';
    const [xKey, yKey] = axisKeys();
    const next = { ...(model.get('bifurcation_config') || {}) };
    delete next[xKey];
    delete next[yKey];
    controller.commit({ bifurcation_config: next });
  });

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

      const canTrace2D = Boolean(sp.can_trace_2d || sp.ty === 'HB' || sp.ty === 'LP' || sp.ty === 'LPC' || sp.ty === 'PD' || sp.ty === 'TR');
      if (canTrace2D || sp.can_trace_homoclinic) {
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
          if (canTrace2D) {
            const twoDBtn = targetDoc.createElement('button');
            twoDBtn.type = 'button';
            twoDBtn.className = 'ppw-bif-trace-2d';
            twoDBtn.textContent = '→ 2D curve';
            twoDBtn.disabled = running;
            twoDBtn.addEventListener('click', () => controller.startTwoParamContinuation(sp.id, p2Select.value, readAdvancedConfig(), owner));
            chip.append(p2Select, twoDBtn);
          } else if (sp.can_trace_homoclinic) {
            const homBtn = targetDoc.createElement('button');
            homBtn.type = 'button';
            homBtn.className = 'ppw-bif-trace-hom';
            homBtn.textContent = `→ Homoclinic (2D${Number.isFinite(sp.period) ? `, T=${Number(sp.period).toPrecision(3)}` : ''})`;
            homBtn.disabled = running;
            homBtn.addEventListener('click', () => controller.startTwoParamContinuation(
              sp.id,
              p2Select.value,
              { ...readAdvancedConfig(), homoclinic: true },
              owner,
            ));
            chip.append(p2Select, homBtn);
          }
        }
      }

      chipsBox.append(chip);
    }
  }

  function update() {
    const paramNames = Object.keys(model.get('param_info') || model.get('params') || {});
    const stateNames = model.get('state_names') || [];
    const nextModelSig = JSON.stringify([model.get('model_name'), model.get('model_spec'), stateNames]);
    if (lastModelSig && lastModelSig !== nextModelSig) {
      const prevCfg = model.get('bifurcation_config') || {};
      if ('xlim1d' in prevCfg || 'ylim1d' in prevCfg || 'xlim2d' in prevCfg || 'ylim2d' in prevCfg || 'plane2d' in prevCfg) {
        const cleaned = { ...prevCfg };
        delete cleaned.xlim1d; delete cleaned.ylim1d;
        delete cleaned.xlim2d; delete cleaned.ylim2d;
        delete cleaned.plane2d;
        model.set('bifurcation_config', cleaned);
      }
    }
    lastModelSig = nextModelSig;

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

    const twoDBranches = branches.filter(b => b && twoDTypes.has(b.type));
    const has2D = twoDBranches.length > 0;
    const latest2D = twoDBranches[twoDBranches.length - 1];
    const latest2DKey = latest2D?.param1 && latest2D?.param2
      ? JSON.stringify([latest2D.param1, latest2D.param2])
      : '';
    if (twoDBranches.length > last2DCount && latest2DKey && latest2DKey !== lastLatest2DKey) {
      if (cfg.plane2d && cfg.plane2d !== latest2DKey) {
        model.set('bifurcation_config', { ...cfg, plane2d: latest2DKey });
      }
    }
    last2DCount = twoDBranches.length;
    lastLatest2DKey = latest2DKey;

    if (cfg.viewMode === '2d' && has2D) viewMode = '2d';
    else if (cfg.viewMode === '1d') viewMode = '1d';
    if (!has2D && viewMode === '2d') viewMode = '1d';

    viewSwitch.hidden = !has2D;
    mode1DBtn.setAttribute('aria-pressed', String(viewMode === '1d'));
    mode2DBtn.setAttribute('aria-pressed', String(viewMode === '2d'));

    const bounds2D = has2D ? bifurcation2DBounds(model) : { planes: [] };
    const planes = bounds2D.planes || [];
    const nextPlanesSig = JSON.stringify(planes);
    if (nextPlanesSig !== planesSig) {
      planesSig = nextPlanesSig;
      plane2DSelect.replaceChildren();
      for (const pl of planes) {
        const opt = targetDoc.createElement('option');
        opt.value = pl.key;
        opt.textContent = `${pl.param1} vs ${pl.param2}`;
        plane2DSelect.append(opt);
      }
    }
    planeLabel.hidden = viewMode !== '2d' || planes.length === 0;
    if (planes.length > 0) {
      const activeKey = JSON.stringify([bounds2D.param1, bounds2D.param2]);
      if ([...plane2DSelect.options].some(o => o.value === activeKey)) plane2DSelect.value = activeKey;
    }

    const activeCfg = model.get('bifurcation_config') || {};
    const [xKey, yKey] = axisKeys();
    const xPair = activeCfg[xKey], yPair = activeCfg[yKey];
    const syncAxisInput = (input, val) => {
      if (targetDoc.activeElement === input) return;
      input.value = val !== undefined && val !== null && Number.isFinite(Number(val)) ? String(val) : '';
    };
    syncAxisInput(xminInput, Array.isArray(xPair) ? xPair[0] : null);
    syncAxisInput(xmaxInput, Array.isArray(xPair) ? xPair[1] : null);
    syncAxisInput(yminInput, Array.isArray(yPair) ? yPair[0] : null);
    syncAxisInput(ymaxInput, Array.isArray(yPair) ? yPair[1] : null);

    renderLegend(viewMode);

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
      if (targetDoc.activeElement === input) continue;
      if (key in activeCfg && activeCfg[key] !== null && activeCfg[key] !== undefined) input.value = activeCfg[key];
      else if (input.defaultValue !== undefined) input.value = input.defaultValue;
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
