function mountParameters({ model, controller, document: doc, configuration }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-params');
  let signature, controls = new Map(), fields = panelScope();
  scope.cleanup(() => fields.dispose());
  function update() {
    const info = model.get('param_info') || {}, params = model.get('params') || {};
    const visible = configuration.get('visible_parameters');
    const next = JSON.stringify([info, visible, model.get('model_spec')?.parameters]);
    if (signature !== next) {
      signature = next; fields.dispose(); fields = panelScope(); controls = new Map(); element.replaceChildren();
      const entries = visible ? [...new Set(visible)].filter(name => name in info).map(name => [name, info[name]]) : Object.entries(info);
      for (const [name, [min, max, fallback, description]] of entries) {
        const control = numberControl(doc, fields, { label: name, min, max, value: params[name] ?? fallback,
          step: model.get('model_spec')?.parameters?.[name]?.step || 'any', delay: 30,
          onChange: value => controller.commit({ params: { ...model.get('params'), [name]: value } }),
        });
        control.slider.dataset.param = name; control.element.title = description || name;
        controls.set(name, control); element.append(control.element);
      }
      if (!controls.size) { const hint = doc.createElement('p'); hint.className = 'ppw-hint'; hint.textContent = 'No parameters selected'; element.append(hint); }
    }
    for (const [name, control] of controls) control.update(params[name] ?? info[name][2]);
  }
  return mountedPanel(element, scope, update);
}

function mountModel({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-control-row', '<label class="ppw-label">Model <select class="ppw-model-select" aria-label="Model"></select></label>');
  const select = element.querySelector('select');
  const names = { wilson_cowan: 'Wilson–Cowan', fitzhugh_nagumo: 'FitzHugh–Nagumo', mpr: 'MPR (QIF)', hindmarsh_rose: 'Hindmarsh–Rose', custom: 'Custom' };
  for (const [value, label] of Object.entries(names)) { const option = doc.createElement('option'); option.value = value; option.textContent = label; select.append(option); }
  scope.on(select, 'change', () => controller.selectModel(select.value));
  return mountedPanel(element, scope, () => { select.value = model.get('model_name'); });
}

function mountProjection({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-projection', '<div class="ppw-control-row ppw-state-selector"><label>X axis <select class="ppw-state-x" aria-label="X axis"></select></label><label>Y axis <select class="ppw-state-y" aria-label="Y axis"></select></label></div><div class="ppw-clamped-sliders"></div>');
  const row = element.querySelector('.ppw-state-selector'), x = element.querySelector('.ppw-state-x'), y = element.querySelector('.ppw-state-y'), clamped = element.querySelector('.ppw-clamped-sliders');
  let signature, controls = new Map(), fields = panelScope();
  scope.cleanup(() => fields.dispose());
  for (const [changed, other] of [[x, y], [y, x]]) scope.on(changed, 'change', () => {
    if (x.value === y.value) other.value = (Number(changed.value) + 1) % model.get('state_names').length;
    controller.commit({ display: [Number(x.value), Number(y.value)] });
  });
  function update() {
    const names = model.get('state_names') || [], display = model.get('display') || [0, 1];
    const def = controller.definition;
    if (!def) return;
    const next = JSON.stringify([names, display, def.defaultLims]);
    if (next !== signature) {
      signature = next; fields.dispose(); fields = panelScope(); controls = new Map();
      row.hidden = names.length <= 2;
      for (const select of [x, y]) {
        select.replaceChildren();
        names.forEach((name, i) => { const option = doc.createElement('option'); option.value = i; option.textContent = name; select.append(option); });
      }
      x.value = display[0]; y.value = display[1]; clamped.replaceChildren();
      names.forEach((name, i) => {
        if (display.includes(i)) return;
        const [min, max] = def.defaultLims?.[i] || [-5, 5];
        const control = numberControl(doc, fields, { label: `Fixed ${name}`, min, max,
          value: model.get('clamped')?.[i] ?? (min + max) / 2, sliderClass: 'ppw-clamped-slider', valueClass: 'ppw-clamped-value',
          onChange: value => { const values = [...model.get('clamped')]; values[i] = value; controller.commit({ clamped: values }); },
        });
        control.slider.dataset.var = i; controls.set(i, control); clamped.append(control.element);
      });
    }
    for (const [i, control] of controls) control.update(model.get('clamped')?.[i] ?? 0);
  }
  return mountedPanel(element, scope, update);
}

function mountLimits({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-control-row ppw-limits', `
    <label>X minimum <input class="ppw-xmin" aria-label="X minimum" type="number" step="any"></label>
    <label>X maximum <input class="ppw-xmax" aria-label="X maximum" type="number" step="any"></label>
    <label>Y minimum <input class="ppw-ymin" aria-label="Y minimum" type="number" step="any"></label>
    <label>Y maximum <input class="ppw-ymax" aria-label="Y maximum" type="number" step="any"></label>
    <label>Duration <input class="ppw-tmax" aria-label="Duration" type="number" step="any" min="0.01"></label>
    <span class="ppw-input-error" role="status" hidden></span>`);
  const inputs = [...element.querySelectorAll('input')], message = element.querySelector('[role=status]');
  function update() {
    const values = [...model.get('xlim'), ...model.get('ylim'), model.get('t_max')];
    inputs.forEach((input, i) => { if (doc.activeElement !== input) input.value = values[i]; });
  }
  for (const input of inputs) scope.on(input, 'change', () => {
    const [xmin, xmax, ymin, ymax, duration] = inputs.map(input => input.value === '' ? NaN : Number(input.value));
    if (![xmin, xmax, ymin, ymax, duration].every(Number.isFinite) || xmin >= xmax || ymin >= ymax || duration <= 0) {
      message.hidden = false; message.textContent = 'Use increasing axis bounds and a positive duration'; return;
    }
    message.hidden = true;
    controller.commit({ xlim: [xmin, xmax], ylim: [ymin, ymax], t_max: duration });
  });
  return mountedPanel(element, scope, update);
}

function mountNoise({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-noise-section', `
    <div class="ppw-control-row"><label><input class="ppw-noise-enable" type="checkbox"> Enable noise</label>
    <label>Integrator <select class="ppw-integrator-select"><option value="rk4">RK4</option><option value="heun">Heun</option></select></label></div>
    <div class="ppw-noise-sliders"></div>
    <label>Noise seed <input class="ppw-noise-seed" aria-label="Noise seed" type="number" step="1"></label>
    <button class="ppw-resample">New realization</button>`);
  const enable = element.querySelector('.ppw-noise-enable'), integrator = element.querySelector('select'), seed = element.querySelector('.ppw-noise-seed'), sliders = element.querySelector('.ppw-noise-sliders');
  let signature, controls = [], fields = panelScope(); scope.cleanup(() => fields.dispose());
  scope.on(enable, 'change', () => controller.commit({ noise_enable: enable.checked, integrator: enable.checked ? 'heun' : model.get('integrator') }));
  scope.on(integrator, 'change', () => controller.commit({ integrator: integrator.value, noise_enable: integrator.value === 'rk4' ? false : model.get('noise_enable') }));
  scope.on(seed, 'change', () => {
    const value = Number(seed.value);
    if (seed.value && Number.isSafeInteger(value) && value >= -2147483648 && value <= 2147483647) controller.commit({ noise_seed: value });
    else seed.value = model.get('noise_seed') || 0;
  });
  scope.on(element.querySelector('.ppw-resample'), 'click', () => controller.commit({ noise_seed: ((model.get('noise_seed') || 0) + 1) | 0 }));
  function update() {
    enable.checked = !!model.get('noise_enable'); integrator.value = model.get('integrator') || 'rk4';
    if (doc.activeElement !== seed) seed.value = model.get('noise_seed') || 0;
    sliders.hidden = !enable.checked;
    const names = model.get('state_names') || [], next = JSON.stringify(names);
    if (next !== signature) {
      signature = next; fields.dispose(); fields = panelScope(); sliders.replaceChildren();
      controls = names.map((name, i) => {
        const control = numberControl(doc, fields, { label: name + ' noise', min: 0, max: 1, step: .01,
          value: model.get('noise_sigma')?.[i] || 0, sliderClass: 'ppw-noise-slider', valueClass: 'ppw-noise-value-' + i,
          onChange: value => { const sigma = names.map((_, k) => model.get('noise_sigma')?.[k] || 0); sigma[i] = value; controller.commit({ noise_sigma: sigma }); },
        });
        control.slider.dataset.idx = i; sliders.append(control.element); return control;
      });
    }
    controls.forEach((control, i) => control.update(model.get('noise_sigma')?.[i] || 0));
  }
  return mountedPanel(element, scope, update);
}

function mountLayers({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-control-row ppw-toggles');
  const controls = [['show_nullclines', 'Nullclines', 'ppw-show-null'], ['show_vector_field', 'Vector field', 'ppw-show-vf'], ['show_trajectory', 'Trajectory', 'ppw-show-traj'], ['show_fixed_points', 'Fixed points', 'ppw-show-fp']].map(([trait, text, className]) => {
    const label = doc.createElement('label'), input = doc.createElement('input'); input.type = 'checkbox'; input.className = className;
    label.append(input, doc.createTextNode(text)); element.append(label);
    scope.on(input, 'change', () => controller.commit({ [trait]: input.checked }));
    return [trait, input];
  });
  return mountedPanel(element, scope, () => { for (const [trait, input] of controls) input.checked = !!model.get(trait); });
}

function mountComparison({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-comparison', '<button class="ppw-reference">Freeze scene</button><button class="ppw-clear-reference">Clear reference</button><label><input class="ppw-show-reference" type="checkbox"> Show frozen scene</label><button class="ppw-reset">Reset experiment</button><span class="ppw-reference-status" role="status"></span>');
  scope.on(element.querySelector('.ppw-reference'), 'click', () => controller.captureReference());
  scope.on(element.querySelector('.ppw-clear-reference'), 'click', () => controller.commit({ reference_trajectory: [], reference_scene: {}, reference_metadata: {} }));
  scope.on(element.querySelector('.ppw-reset'), 'click', () => controller.reset());
  scope.on(element.querySelector('.ppw-show-reference'), 'change', event => controller.commit({ show_reference: event.target.checked }));
  return mountedPanel(element, scope, () => {
    const hasReference = !!model.get('reference_trajectory')?.length;
    element.querySelector('.ppw-clear-reference').disabled = !hasReference;
    element.querySelector('.ppw-reference').disabled = !model.get('trajectory')?.length;
    element.querySelector('.ppw-reference').textContent = hasReference ? 'Replace frozen scene' : 'Freeze scene';
    element.querySelector('.ppw-show-reference').checked = model.get('show_reference') !== false;
    element.querySelector('.ppw-show-reference').disabled = !hasReference;
    const params = Object.entries(model.get('reference_metadata')?.params || {}).map(([key, value]) => `${key}=${Number(value).toPrecision(3)}`).join(' · ');
    element.querySelector('.ppw-reference-status').textContent = hasReference ? `Frozen scene retained: ${params}` : 'Freeze nullclines, equilibria and trajectory, then move a slider.';
  });
}

function mountExport({ model, controller, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-export-row', '<button class="ppw-export-svg">Export SVG</button><button class="ppw-export-bif-svg">Export Bifurcation SVG</button><button class="ppw-export-tikz">Export TikZ</button>');
  scope.on(element.querySelector('.ppw-export-svg'), 'click', () => {
    if (controller.error) return;
    const canvas = new C2S(500, 500); drawPhasePlane(model, canvas, 500, 500, plotTheme(element));
    downloadBlob(doc, new Blob([canvas.getSerializedSvg(true)], { type: 'image/svg+xml' }), 'phase_plane.svg');
  });
  const bifSvg = element.querySelector('.ppw-export-bif-svg');
  scope.on(bifSvg, 'click', () => {
    const branches = model.get('bifurcation_branches') || [];
    if (controller.error || !branches.length) return;
    const has1d = branches.some(b => b && (b.type === 'EP' || b.type === 'LC'));
    const has2d = branches.some(b => b && ['LP2', 'HB2', 'LPC2', 'PD2', 'TR2', 'HOM2'].includes(b.type));
    const use2d = (model.get('bifurcation_config')?.viewMode === '2d' && has2d) || (!has1d && has2d);
    const canvas = new C2S(560, 420);
    if (use2d) drawBifurcation2D(model, canvas, 560, 420, plotTheme(element));
    else drawBifurcation1D(model, canvas, 560, 420, plotTheme(element));
    downloadBlob(doc, new Blob([canvas.getSerializedSvg(true)], { type: 'image/svg+xml' }), 'bifurcation.svg');
  });
  const tikz = element.querySelector('.ppw-export-tikz');
  const standalone = model._isMock || typeof model.send !== 'function';
  tikz.disabled = standalone; if (standalone) tikz.title = 'TikZ export requires a Jupyter kernel';
  scope.on(tikz, 'click', () => model.send({ type: 'export_tikz' }));
  const receive = message => { if (message?.type === 'tikz_data') downloadBlob(doc, new Blob([message.content], { type: 'application/x-tex' }), message.filename || 'phase_plane.tex'); };
  if (!standalone) { model.on('msg:custom', receive); scope.cleanup(() => model.off('msg:custom', receive)); }
  return mountedPanel(element, scope, () => {
    element.querySelector('.ppw-export-svg').disabled = !!controller.error;
    bifSvg.disabled = !!controller.error || !(model.get('bifurcation_branches') || []).length;
  });
}
