// Bounded scientific-task prototypes. Preview models never acquire controllers
// or mutate the canonical experiment. Only explicit Use/Restore actions commit.
const PRESENTATION_LABELS = {
  figure: 'Interactive figure', bench: '16:9 experiment bench', drawer: 'Instrument drawer',
  comparison: 'Comparison lens', lesson: 'Guided lesson',
  multiples: 'Small multiples', navigator: 'Bifurcation navigator', basin: 'Initial-condition explorer',
  evidence: 'Evidence notebook', prediction: 'Prediction & reveal', ensemble: 'Stochastic ensemble',
  bifurcation: 'Bifurcation & phase plane',
};
const RESEARCH_KINDS = ['multiples', 'navigator', 'basin', 'evidence', 'prediction', 'ensemble'];
const TRACE_COLORS = ['#1769aa', '#d04b27', '#008577', '#963ab0', '#a37300', '#cf3973', '#4d6599', '#527c23'];
const RESEARCH_PRESENTATION_STATE = new WeakMap();
function researchSession(model) { return experimentSession(model).research ||= {}; }
function researchPresentationState(configuration) {
  if (!RESEARCH_PRESENTATION_STATE.has(configuration)) RESEARCH_PRESENTATION_STATE.set(configuration, {});
  return RESEARCH_PRESENTATION_STATE.get(configuration);
}
function experimentSnapshot(model) { return Object.fromEntries(RESET_KEYS.map(key => [key, structuredClone(model.get(key))])); }
function previewModel(model, overrides = {}) {
  const values = { ...experimentSnapshot(model), show_nullclines: model.get('show_nullclines'),
    show_vector_field: model.get('show_vector_field'), show_fixed_points: model.get('show_fixed_points'),
    show_trajectory: true, trajectory_fraction: 1, show_reference: false, ...overrides };
  return { get: key => values[key], set: (key, value) => { values[key] = value; } };
}
function previewTrajectory(model, overrides = {}) {
  const copy = previewModel(model, overrides), name = copy.get('model_name');
  const def = resolveModel(name, copy.get('model_spec')), display = copy.get('display');
  const initial = [...(copy.get('clamped') || def.stateVars.map(() => 0))];
  initial[display[0]] = copy.get('x0'); if (display.length > 1) initial[display[1]] = copy.get('y0');
  const noise = copy.get('noise_enable') && copy.get('integrator') === 'heun' ? copy.get('noise_sigma') : null;
  const duration = Math.min(120, copy.get('t_max'));
  const trajectory = simulate(name === 'custom' ? def : name, initial, 0, duration, .01,
    copy.get('params'), { integrator: copy.get('integrator'), noiseSigma: noise, seed: copy.get('noise_seed') || 0 });
  if (!trajectory.length || trajectory.at(-1)[0] < duration - .011 || trajectory.some(row => row.some(value => !Number.isFinite(value)))) throw new Error('Preview diverged; reduce duration, noise or parameter range.');
  const step = Math.max(1, Math.ceil(trajectory.length / 1000));
  return trajectory.filter((_, index) => index % step === 0 || index === trajectory.length - 1);
}
function drawTracePaths(model, ctx, w, h, traces, labelStarts = false) {
  const display = model.get('display'); if (display.length < 2) return;
  ctx.save(); ctx.beginPath(); ctx.rect(42, 42, w - 84, h - 84); ctx.clip();
  traces.forEach((trajectory, index) => {
    ctx.strokeStyle = TRACE_COLORS[index % TRACE_COLORS.length]; ctx.lineWidth = 1.8; ctx.beginPath();
    trajectory.forEach((row, i) => {
      const [x, y] = worldToScreen(row[display[0] + 1], row[display[1] + 1], model.get('xlim'), model.get('ylim'), w, h);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }); ctx.stroke();
    if (labelStarts && trajectory.length) {
      const [x, y] = worldToScreen(trajectory[0][display[0] + 1], trajectory[0][display[1] + 1], model.get('xlim'), model.get('ylim'), w, h);
      ctx.fillStyle = TRACE_COLORS[index % TRACE_COLORS.length]; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill(); ctx.font = '12px sans-serif'; ctx.fillText(String(index + 1), x + 6, y - 6);
    }
  }); ctx.restore();
}
function drawEnsembleTime(model, ctx, w, h, traces, theme) {
  ctx.fillStyle = theme.background; ctx.fillRect(0, 0, w, h);
  const d = model.get('display')[0] + 1, pad = 42;
  let low = Infinity, high = -Infinity, end = 0;
  for (const path of traces) for (const row of path) { low = Math.min(low, row[d]); high = Math.max(high, row[d]); end = Math.max(end, row[0]); }
  if (!Number.isFinite(low)) return;
  const margin = Math.max(.1, (high - low) * .1); low -= margin; high += margin;
  ctx.strokeStyle = theme.grid; ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);
  ctx.fillStyle = theme.text; ctx.font = '11px sans-serif'; ctx.fillText(`${model.get('state_names')[d - 1]}: ${low.toFixed(2)} … ${high.toFixed(2)}`, pad, 20);
  ctx.fillText(`Time: 0 … ${end.toFixed(1)}`, pad, h - 10);
  traces.forEach((path, index) => { ctx.strokeStyle = TRACE_COLORS[index]; ctx.lineWidth = 1.5; ctx.beginPath();
    path.forEach((row, i) => { const x = pad + row[0] / (end || 1) * (w - 2 * pad), y = h - pad - (row[d] - low) / (high - low) * (h - 2 * pad); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke(); });
}

function mountResearchView(context, kind) {
  const { model, controller, document: doc } = context, scope = panelScope();
  const element = panelElement(doc, 'ppw-research-view');
  const tools = panelElement(doc, 'ppw-research-tools'), plots = panelElement(doc, 'ppw-research-plots');
  const note = doc.createElement('p'); note.className = 'ppw-research-note';
  const status = doc.createElement('p'); status.className = 'ppw-research-status'; status.setAttribute('role', 'status');
  element.append(tools, note, plots, status);
  const sessions = researchPresentationState(context.presentationOwner || context.configuration), state = sessions[kind] ||= {};
  const surfaces = []; let signature, redraw = () => {};
  function button(text, cls, fn, parent = tools) { const node = doc.createElement('button'); node.type = 'button'; node.textContent = text; node.className = cls; parent.append(node); scope.on(node, 'click', fn); return node; }
  function input(label, cls, value, min, max) { const wrapper = doc.createElement('label'); wrapper.append(doc.createTextNode(label + ' ')); const node = doc.createElement('input'); node.type = 'number'; node.step = 'any'; node.className = cls; node.value = value; node.min = min; node.max = max; node.setAttribute('aria-label', label); wrapper.append(node); tools.append(wrapper); return node; }
  function canvas(label, parent = plots) { const wrapper = panelElement(doc, 'ppw-research-plot'); const title = doc.createElement('strong'); title.textContent = label; const node = doc.createElement('canvas'); node.setAttribute('role', 'img'); node.setAttribute('aria-label', label); wrapper.append(title, node); parent.append(wrapper); return node; }
  function surface(node, draw, ratio = .7) { const s = canvasSurface(node, scope, (ctx, w, h, theme) => { ctx.fillStyle = theme.background; ctx.fillRect(0, 0, w, h); draw(ctx, w, h, theme); }, { ratio, minHeight: 200 }); surfaces.push(s); return s; }
  function run(fn) { try { if (controller.error) throw controller.error; fn(); status.textContent = ''; } catch (error) { status.textContent = error.message; } }
  function parameterPicker() {
    const wrapper = doc.createElement('label'); wrapper.append(doc.createTextNode('Parameter '));
    const select = doc.createElement('select'); select.setAttribute('aria-label', 'Comparison parameter'); select.className = 'ppw-research-parameter'; wrapper.append(select); tools.append(wrapper);
    let infoKey;
    function sync() { const info = model.get('param_info') || {}, key = JSON.stringify(info); if (key === infoKey) return; infoKey = key;
      select.replaceChildren(); for (const name of Object.keys(info)) { const option = doc.createElement('option'); option.value = name; option.textContent = name; select.append(option); }
      const preferred = context.configuration.get('visible_parameters')?.[0];
      select.value = state.parameter in info ? state.parameter : preferred in info ? preferred : 'I' in info ? 'I' : Object.keys(info)[0] || '';
      state.parameter = select.value;
    }
    sync(); return { select, sync };
  }
  if (kind === 'multiples') {
    const { select, sync } = parameterPicker(); const values = [], tiles = []; let scenes = [];
    function defaults() { const info = model.get('param_info')[select.value]; if (!info) { state.values = [0, 0, 0]; return; } const [min, max] = info, value = model.get('params')[select.value];
      state.values = [-1, 0, 1].map(i => Number(Math.max(min, Math.min(max, value + i * (max - min) * .1)).toPrecision(10)));
      values.forEach((node, index) => { node.min = min; node.max = max; node.value = state.values[index]; });
    }
    for (let index = 0; index < 3; index++) {
      const node = input(`Tile ${index + 1}`, 'ppw-multiple-value', state.values?.[index] ?? 0, -1e6, 1e6); values.push(node);
      const tile = panelElement(doc, 'ppw-multiple-tile'); plots.append(tile);
      const use = button('Use tile', 'ppw-use-tile', () => controller.commit({ params: { ...model.get('params'), [select.value]: state.values[index] } }), tile); tiles.push(use);
      surface(canvas(`Parameter tile ${index + 1}`, tile), (ctx, w, h, theme) => { if (scenes[index]) drawPhasePlane(scenes[index], ctx, w, h, theme); });
      scope.on(node, 'change', () => { const info = model.get('param_info')[select.value], value = Number(node.value);
        if (!node.value || !Number.isFinite(value) || value < info[0] || value > info[1]) { status.textContent = 'Tile values must be within the parameter bounds.'; node.setAttribute('aria-invalid', 'true'); return; }
        node.removeAttribute('aria-invalid'); state.values[index] = value; signature = null; update();
      });
    }
    if (!state.values || state.identity !== referenceIdentity(model)) defaults(); else values.forEach((node, index) => { const info = model.get('param_info')[select.value]; if (!info) return; node.min = info[0]; node.max = info[1]; node.value = state.values[index]; });
    state.identity = referenceIdentity(model);
    scope.on(select, 'change', () => { state.parameter = select.value; defaults(); signature = null; update(); });
    note.textContent = 'Three isolated previews, shared axes and initial condition. Use tile explicitly changes the live parameter. Preview duration is capped at 120; no extra controllers.';
    redraw = () => { const previous = select.value; sync(); if (previous !== select.value || state.identity !== referenceIdentity(model)) { defaults(); state.identity = referenceIdentity(model); }
      const available = !!model.get('param_info')[select.value]; select.disabled = !available;
      for (const node of [...values, ...tiles]) node.disabled = !available;
      if (!available) { scenes = []; status.textContent = 'This model has no parameters to compare.'; return; }
      const key = JSON.stringify([experimentSnapshot(model), state.parameter, state.values]);
      if (key !== signature) { signature = key; scenes = []; run(() => { scenes = state.values.map(value => { const copy = previewModel(model, { params: { ...model.get('params'), [select.value]: value }, t_max: Math.min(120, model.get('t_max')) }); computeExperiment(copy); if (!copy.get('trajectory').length || copy.get('trajectory').at(-1)[0] < copy.get('t_max') - .011 || copy.get('trajectory').some(row => row.some(x => !Number.isFinite(x)))) throw new Error('A tile diverged; choose a narrower range.'); return copy; }); }); }
      tiles.forEach((node, index) => { node.textContent = `Use ${select.value} = ${Number(state.values[index]).toPrecision(3)}`; node.setAttribute('aria-pressed', String(Math.abs(model.get('params')[select.value] - state.values[index]) < 1e-10)); });
    };
  } else if (kind === 'basin' || kind === 'ensemble') {
    let traces = [];
    const background = { get: key => key === 'show_trajectory' || key === 'show_reference' ? false : model.get(key) };
    const phase = canvas(kind === 'basin' ? 'Initial-condition trajectories' : 'Seeded phase trajectories');
    const phaseSurface = surface(phase, (ctx, w, h, theme) => { drawPhasePlane(background, ctx, w, h, theme); drawTracePaths(model, ctx, w, h, traces, kind === 'basin'); }, kind === 'basin' ? .38 : .7);
    const list = panelElement(doc, 'ppw-trace-list'); element.append(list); let listKey;
    if (kind === 'basin') {
      const identity = () => JSON.stringify([referenceIdentity(model), model.get('display')]);
      const compatible = () => (model.get('initial_conditions') || []).filter(point => point.identity === identity());
      const add = (x, y) => { const points = compatible(); if (points.length >= 6) { status.textContent = 'Maximum six initial conditions. Remove one or clear the list.'; return; } controller.commit({ initial_conditions: [...points, { x, y, identity: identity() }] }); };
      const addButton = button('Add current initial condition', 'ppw-add-start', () => add(model.get('x0'), model.get('y0')));
      button('Clear starts', 'ppw-clear-starts', () => controller.commit({ initial_conditions: [] }));
      phase.tabIndex = 0; phase.setAttribute('aria-label', 'Initial-condition trajectories. Click to add a start; use Add current initial condition for keyboard access.');
      scope.on(phase, 'click', event => { if (model.get('display').length < 2) return; const rect = phase.getBoundingClientRect(), x = (event.clientX - rect.left) * phaseSurface.width / rect.width, y = (event.clientY - rect.top) * phaseSurface.height / rect.height;
        if (x < 42 || y < 42 || x > phaseSurface.width - 42 || y > phaseSurface.height - 42) return;
        add(...screenToWorld(x, y, model.get('xlim'), model.get('ylim'), phaseSurface.width, phaseSurface.height));
      });
      note.textContent = 'Click to add up to six starts, or add the current start set in the figure view. Colored paths are re-integrated when settings change (up to 120 time units). This is not a computed basin classification.';
      let listScope = panelScope(); scope.cleanup(() => listScope.dispose());
      redraw = () => { const points = compatible(), key = JSON.stringify([experimentSnapshot(model), points]);
        addButton.disabled = points.length >= 6 || model.get('display').length < 2;
        if (key !== signature) { signature = key; traces = []; run(() => { traces = points.map(point => previewTrajectory(model, { x0: point.x, y0: point.y })); }); }
        const next = JSON.stringify(points); if (next !== listKey) { listKey = next; listScope.dispose(); listScope = panelScope(); list.replaceChildren();
          points.forEach((point, index) => { const remove = doc.createElement('button'); remove.textContent = `${index + 1}: (${point.x.toFixed(2)}, ${point.y.toFixed(2)}) — remove`; remove.style.color = TRACE_COLORS[index]; list.append(remove); listScope.on(remove, 'click', () => controller.commit({ initial_conditions: points.filter((_, i) => i !== index) })); });
        }
        if (model.get('display').length < 2) status.textContent = 'This view requires two displayed variables.';
      };
    } else {
      state.count ??= 4; state.sigma ??= Math.max(.1, ...(model.get('noise_sigma') || [0]));
      const count = input('Members', 'ppw-ensemble-count', state.count, 2, 8); count.step = '1';
      const sigma = input('Noise strength (all variables)', 'ppw-ensemble-sigma', state.sigma, 0, 2);
      surface(canvas('Ensemble time series (displayed X variable)'), (ctx, w, h, theme) => drawEnsembleTime(model, ctx, w, h, traces, theme));
      for (const node of [count, sigma]) scope.on(node, 'change', () => { const n = Number(count.value), s = Number(sigma.value);
        if (!count.value || !sigma.value || !Number.isInteger(n) || n < 2 || n > 8 || !Number.isFinite(s) || s < 0 || s > 2) { status.textContent = 'Choose 2–8 members and noise strength 0–2.'; return; }
        state.count = n; state.sigma = s; signature = null; update();
      });
      button('Next seed set', 'ppw-next-seeds', () => controller.commit({ noise_seed: (model.get('noise_seed') + state.count) % 2147483647 }));
      note.textContent = '2–8 seeded Heun traces with equal additive noise strength in every variable, up to 120 time units. Preview noise does not alter live settings. Time plot shows X only; these traces are not confidence intervals. Use member applies its noise settings explicitly.';
      let listScope = panelScope(); scope.cleanup(() => listScope.dispose());
      redraw = () => { const key = JSON.stringify([experimentSnapshot(model), state.count, state.sigma]);
        if (key !== signature) { signature = key; traces = []; run(() => { traces = Array.from({ length: state.count }, (_, index) => previewTrajectory(model, { integrator: 'heun', noise_enable: true, noise_sigma: model.get('state_names').map(() => state.sigma), noise_seed: model.get('noise_seed') + index })); }); }
        const next = JSON.stringify([model.get('noise_seed'), state.count, state.sigma]); if (next !== listKey) { listKey = next; listScope.dispose(); listScope = panelScope(); list.replaceChildren();
          for (let index = 0; index < state.count; index++) { const seed = model.get('noise_seed') + index, use = doc.createElement('button'); use.className = 'ppw-use-member'; use.textContent = `Use seed ${seed}`; use.style.color = TRACE_COLORS[index]; list.append(use);
            listScope.on(use, 'click', () => controller.commit({ noise_seed: seed, integrator: 'heun', noise_enable: true, noise_sigma: model.get('state_names').map(() => state.sigma) })); }
        }
      };
    }
  } else if (kind === 'prediction') {
    const label = doc.createElement('label'); label.textContent = 'Predict the outcome before revealing the trajectory '; const answer = doc.createElement('textarea'); answer.className = 'ppw-prediction-answer'; answer.setAttribute('aria-label', 'Your prediction'); answer.value = state.answer || ''; label.append(answer); tools.append(label);
    scope.on(answer, 'input', () => { state.answer = answer.value; });
    const reveal = button('Reveal trajectory', 'ppw-reveal', () => { state.revealed = !state.revealed; update(); });
    const masked = { get: key => (!state.revealed && (key === 'trajectory' || key === 'reference_trajectory')) ? [] : key === 'show_reference' ? false : model.get(key) };
    surface(canvas('Predict: phase plane'), (ctx, w, h, theme) => {
      drawPhasePlane(masked, ctx, w, h, theme);
      if (!state.revealed) {
        const one = model.get('display').length < 2;
        const [x, y] = worldToScreen(model.get('x0'), one ? 0 : model.get('y0'), model.get('xlim'), one ? [-2, 2] : model.get('ylim'), w, h);
        ctx.fillStyle = '#ff9800'; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = theme.text; ctx.font = '12px sans-serif'; ctx.fillText('start', x + 8, y - 8);
      }
    });
    surface(canvas('Reveal: time series'), (ctx, w, h, theme) => {
      if (state.revealed) drawTimeSeries(masked, ctx, w, h, theme);
      else { ctx.fillStyle = theme.background; ctx.fillRect(0, 0, w, h); ctx.fillStyle = theme.text; ctx.font = '13px sans-serif'; ctx.fillText('Trajectory deliberately hidden — make a prediction.', 15, 45); }
    });
    note.textContent = 'Use the nullclines, equilibria and initial condition to predict the outcome. Reveal/hide affects only this view, not shared layer settings. A changed experiment hides the answer again. This is an open prediction, not automatic grading.';
    redraw = () => { const key = JSON.stringify([experimentSnapshot(model), researchSession(model).revision || 0]); if (state.identity !== key) { state.identity = key; state.revealed = false; state.answer = ''; answer.value = ''; }
      reveal.textContent = state.revealed ? 'Hide trajectory' : 'Reveal trajectory'; reveal.setAttribute('aria-pressed', String(!!state.revealed));
      status.textContent = state.revealed ? 'Trajectory revealed. Compare it with your prediction.' : 'Trajectory intentionally hidden, not a missing numerical result.';
    };
  } else if (kind === 'evidence') {
    plots.classList.add('ppw-evidence-preview');
    let selected = null, cardScope = panelScope(), cardsKey;
    scope.cleanup(() => cardScope.dispose());
    const name = doc.createElement('input'); name.type = 'text'; name.maxLength = 80; name.placeholder = 'Observation label'; name.setAttribute('aria-label', 'Observation label'); tools.append(name);
    const capture = button('Capture observation', 'ppw-capture-observation', () => {
      if (controller.error) { status.textContent = controller.error.message; return; }
      const cards = model.get('observation_cards') || []; if (cards.length >= 6) return;
      const card = { label: name.value.trim() || `Observation ${cards.length + 1}`, captured_at: new Date().toISOString(), state: experimentSnapshot(model),
        scene: Object.fromEntries(['trajectory', 'nullcline_x', 'nullcline_y', 'vector_field', 'fixed_points'].map(key => [key, structuredClone(model.get(key))])) };
      selected = card; controller.commit({ observation_cards: [...cards, card] });
    });
    button('Clear observations', 'ppw-clear-observations', () => { selected = null; controller.commit({ observation_cards: [] }); });
    button('Download evidence JSON', 'ppw-download-evidence', () => downloadBlob(doc, new Blob([JSON.stringify({ format: 'trajecturtle-evidence-v1', observations: model.get('observation_cards') || [] }, null, 2)], { type: 'application/json' }), 'trajecturtle-evidence.json'));
    const preview = canvas('Selected observation (stored scene)');
    surface(preview, (ctx, w, h, theme) => { if (selected) drawPhasePlane(previewModel(model, { ...selected.state, ...selected.scene }), ctx, w, h, theme); else { ctx.fillStyle = theme.background; ctx.fillRect(0, 0, w, h); ctx.fillStyle = theme.text; ctx.fillText('Capture an observation to start a record.', 20, 45); } });
    const cardsList = panelElement(doc, 'ppw-evidence-cards'); plots.append(cardsList);
    note.textContent = 'Capture up to six labeled, reproducible scenes. Preview does not change the experiment; Restore explicitly applies the saved model and settings. Cards sync to Python and standalone exports; download JSON for a portable record. Reset clears the record.';
    redraw = () => { const cards = model.get('observation_cards') || []; capture.disabled = cards.length >= 6 || !!controller.error;
      if (selected && !cards.includes(selected)) selected = cards.find(card => card.captured_at === selected.captured_at) || null;
      selected ||= cards.at(-1) || null;
      const key = JSON.stringify(cards); if (key !== cardsKey) { cardsKey = key; cardScope.dispose(); cardScope = panelScope(); cardsList.replaceChildren();
        cards.forEach(card => { const item = doc.createElement('article'), title = doc.createElement('strong'), metadata = doc.createElement('p'); title.textContent = card.label;
          metadata.textContent = `${card.state.model_name} · ${Object.entries(card.state.params).map(([key, value]) => `${key}=${Number(value).toPrecision(3)}`).join(' · ')} · (${Number(card.state.x0).toFixed(2)}, ${Number(card.state.y0).toFixed(2)}) · seed ${card.state.noise_seed}`;
          item.append(title, metadata); cardsList.append(item);
          const previewButton = doc.createElement('button'); previewButton.textContent = 'Preview'; item.append(previewButton); cardScope.on(previewButton, 'click', () => { selected = card; update(); });
          const restore = doc.createElement('button'); restore.className = 'ppw-restore-observation'; restore.textContent = 'Restore'; item.append(restore);
          cardScope.on(restore, 'click', () => { selected = card; controller.commit({ ...structuredClone(card.state), reference_trajectory: [], reference_scene: {}, reference_metadata: {}, trajectory_fraction: 1 }); });
        });
      }
    };
  }
  function update() { redraw(); for (const s of surfaces) s.render(); }
  return mountedPanel(element, scope, update);
}
