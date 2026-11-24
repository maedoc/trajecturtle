// Experimental task-oriented compositions. All controls/plots are the same panel
// factories as the full explorer; switching layouts never clones the experiment.
function mountReadout({ model, document: doc }) {
  const scope = panelScope(), element = panelElement(doc, 'ppw-readout');
  const builtinEquations = {
    fitzhugh_nagumo: ['dv/dt = v − v³/3 − w + I', 'dw/dt = ε (v + a − b w)'],
    wilson_cowan: ['dE/dt = −E + Sₑ(aee E − aei I + Pe)', 'dI/dt = −I + Sᵢ(aie E − aii I + Pi)'],
    mpr: ['dr/dt = Δ/π + 2 r v', 'dv/dt = v² + η̄ + J r + I − (πr)²'],
    hindmarsh_rose: ['dx/dt = y − a x³ + b x² − z + I', 'dy/dt = c − d x² − y', 'dz/dt = r (s (x − xr) − z)'],
  };
  return mountedPanel(element, scope, () => {
    element.replaceChildren();
    function text(tag, value) { const node = doc.createElement(tag); node.textContent = value; element.append(node); return node; }
    text('h3', 'Model & current state');
    const spec = model.get('model_spec');
    const equations = model.get('model_name') === 'custom' ? Object.entries(spec?.equations || {}).map(([key, value]) => `d${key}/dt = ${value}`) : builtinEquations[model.get('model_name')] || [];
    for (const equation of equations) text('code', equation);
    text('p', Object.entries(model.get('params') || {}).map(([key, value]) => `${key} = ${Number(value).toPrecision(3)}`).join(' · '));
    const display = model.get('display') || [0, 1], names = model.get('state_names') || [];
    text('p', `Initial condition: ${names[display[0]]} = ${Number(model.get('x0')).toFixed(3)}${display.length > 1 ? `, ${names[display[1]]} = ${Number(model.get('y0')).toFixed(3)}` : ''}`);
    text('h3', names.length > 2 ? 'Equilibria of the displayed slice' : 'Equilibria');
    const points = model.get('fixed_points') || [];
    if (!points.length) text('p', 'No equilibria found in the current window.');
    for (const [x, y, stability] of points.slice(0, 12)) text('p', `${Number(x).toFixed(3)}, ${Number(y).toFixed(3)} — ${stability.replaceAll('_', ' ')}`);
    if (points.length > 12) text('p', `${points.length - 12} more points in this window`);
    text('small', `${model.get('integrator') || 'rk4'} · dt = ${model.get('dt') ?? 0.01} · duration ${model.get('t_max')}${model.get('noise_enable') ? ` · noise seed ${model.get('noise_seed')}` : ' · deterministic'}`);
    if (names.length > 2) text('p', 'Nullclines/equilibria describe a clamped slice; trajectories integrate the full system.');
  });
}

function mountPresentation(context, kind = context.configuration.get('presentation')) {
  const { model, controller, document: doc } = context;
  const scope = panelScope(), groups = new Set();
  const element = panelElement(doc, `ppw-composition ppw-${kind === 'comparison' ? 'lens' : kind}`);
  element.dataset.presentation = kind;
  let expanded = null, popover = null;
  function group(container, names, { primary = false, fit = false } = {}) {
    const components = [];
    const configuration = { get: key => {
      if (key !== 'visible_parameters') return context.configuration.get(key);
      if (!primary) return null;
      const chosen = context.configuration.get(key);
      if (chosen != null) return chosen;
      const keys = Object.keys(model.get('param_info') || {});
      const preferred = ['I', 'Pe', 'a'].filter(key => keys.includes(key));
      return preferred.length ? preferred.slice(0, 2) : keys.slice(0, 2);
    } };
    for (const name of names) {
      const factory = name === 'details' ? mountReadout : name === 'navigator' ? mountNavigatorView : RESEARCH_KINDS.includes(name) ? context => mountResearchView(context, name) : PANEL_FACTORIES[name];
      const component = factory({ ...context, configuration, presentationOwner: context.presentationOwner || context.configuration });
      component.element.classList.add('ppw-panel'); component.element.dataset.panel = name;
      if (fit && component.element.querySelector('canvas')) component.element.setAttribute('data-fit-plot', '');
      components.push(component); container.append(component.element);
    }
    const instance = {
      update() { for (const component of components) component.update(); },
      dispose() { groups.delete(instance); for (const component of components) component.dispose(); },
    };
    groups.add(instance); instance.update(); return instance;
  }
  function button(label, className, action, container = toolbar) {
    const node = doc.createElement('button'); node.type = 'button'; node.textContent = label; node.className = className;
    scope.on(node, 'click', () => action(node)); container.append(node); return node;
  }
  const toolbar = doc.createElement('header'); toolbar.className = 'ppw-stage-toolbar'; element.append(toolbar);
  const title = doc.createElement('strong'); title.className = 'ppw-stage-title'; toolbar.append(title);
  const badges = doc.createElement('span'); badges.className = 'ppw-stage-badges'; toolbar.append(badges);
  function closePopover(restore = false) {
    if (!popover) return;
    const { container, instance, trigger, listeners } = popover; popover = null;
    listeners.dispose(); instance.dispose(); container.remove(); trigger.setAttribute('aria-expanded', 'false'); if (restore) trigger.focus();
  }
  const inspect = button('Inspect', 'ppw-inspect', trigger => {
    if (popover) { closePopover(true); return; }
    const container = panelElement(doc, 'ppw-details-popover'); container.setAttribute('role', 'region'); container.setAttribute('aria-label', 'Experiment details');
    element.append(container);
    const close = doc.createElement('button'); close.textContent = 'Close details'; container.append(close);
    const listeners = panelScope(); listeners.on(close, 'click', () => closePopover(true));
    const instance = group(container, ['details']);
    popover = { container, instance, trigger, listeners }; trigger.setAttribute('aria-expanded', 'true'); close.focus();
  });
  inspect.setAttribute('aria-expanded', 'false');
  scope.on(doc, 'pointerdown', event => { if (popover && !popover.container.contains(event.target) && !inspect.contains(event.target)) closePopover(); });
  scope.on(doc, 'keydown', event => { if (event.key === 'Escape' && popover) { event.preventDefault(); closePopover(true); } });
  function closeExpanded() {
    if (!expanded) return;
    const { dialog, composition, trigger, listeners } = expanded; expanded = null;
    listeners.dispose(); composition.dispose(); if (dialog.open && dialog.close) dialog.close(); dialog.remove();
    if (trigger.isConnected) trigger.focus();
  }
  if (!context.inDialog) button('Expand', 'ppw-expand', trigger => {
    if (expanded) return;
    closePopover();
    const dialog = doc.createElement('dialog'); dialog.className = 'ppw-widget ppw-expanded-workspace'; dialog.setAttribute('aria-label', 'Expanded experiment bench');
    const close = doc.createElement('button'); close.textContent = 'Close workspace'; close.className = 'ppw-close-workspace'; dialog.append(close);
    const composition = mountPresentation({ ...context, inDialog: true }, 'bench'); dialog.append(composition.element);
    const listeners = panelScope(); expanded = { dialog, composition, trigger, listeners }; doc.body.append(dialog);
    listeners.on(close, 'click', closeExpanded); listeners.on(dialog, 'close', closeExpanded);
    if (dialog.showModal) dialog.showModal(); else { dialog.setAttribute('open', ''); dialog.setAttribute('role', 'dialog'); }
    composition.update(); close.focus();
  });
  button('Reset', 'ppw-stage-reset', () => controller.reset());
  const error = doc.createElement('div'); error.className = 'ppw-presentation-error'; error.setAttribute('role', 'alert'); error.hidden = true; element.append(error);
  const body = doc.createElement('div'); body.className = 'ppw-stage-body'; element.append(body);
  const footer = doc.createElement('footer'); footer.className = 'ppw-stage-footer'; element.append(footer);
  function slot(parent, className) { const node = doc.createElement('div'); node.className = className; parent.append(node); return node; }
  function inspector(container, initial = 'details') {
    const label = doc.createElement('label'); label.className = 'ppw-tool-picker'; label.append(doc.createTextNode('Tool '));
    const select = doc.createElement('select'); select.setAttribute('aria-label', 'Inspector section'); label.append(select); container.append(label);
    const tools = { details: 'Model & equilibria', parameters: 'All parameters', noise: 'Noise & integrator', layers: 'Plot layers', projection: 'Projection / fixed state', limits: 'Axes & duration', sweep: 'Parameter sweep', editor: 'Equation editor', export: 'Export', model: 'Choose model' };
    for (const [value, text] of Object.entries(tools)) { const option = doc.createElement('option'); option.value = value; option.textContent = text; select.append(option); }
    const content = slot(container, 'ppw-inspector-content'); let current;
    function choose() { current?.dispose(); current = group(content, [select.value]); }
    select.value = initial; choose(); scope.on(select, 'change', choose);
    return () => current?.dispose();
  }
  if (kind === 'figure' || kind === 'drawer') {
    const plot = slot(body, 'ppw-primary-plot'); group(plot, ['phase_plane'], { fit: true });
    group(footer, ['parameters'], { primary: true });
    if (kind === 'drawer') {
      const drawer = slot(body, 'ppw-instrument-drawer'); drawer.hidden = true; let disposeInspector;
      const toggle = button('Controls', 'ppw-drawer-toggle', () => {
        const open = drawer.hidden; drawer.hidden = !open;
        toggle.setAttribute('aria-expanded', String(open)); element.classList.toggle('ppw-drawer-open', open);
        if (open) disposeInspector = inspector(drawer, 'parameters');
        else { disposeInspector?.(); drawer.replaceChildren(); }
        update();
      });
      toggle.setAttribute('aria-expanded', 'false');
    }
  } else if (kind === 'bench') {
    const plot = slot(body, 'ppw-primary-plot'); group(plot, ['phase_plane'], { fit: true });
    const right = slot(body, 'ppw-bench-secondary');
    group(slot(right, 'ppw-secondary-plot'), ['time_series'], { fit: true });
    inspector(slot(right, 'ppw-bench-inspector'));
    group(footer, ['parameters'], { primary: true });
  } else if (kind === 'comparison') {
    group(slot(body, 'ppw-primary-plot'), ['phase_plane'], { fit: true });
    group(slot(body, 'ppw-secondary-plot'), ['time_series'], { fit: true });
    group(footer, ['parameters'], { primary: true }); group(footer, ['comparison']);
    const caption = doc.createElement('p'); caption.className = 'ppw-comparison-caption'; caption.textContent = 'Freeze a scene, then move a slider. Muted nullclines, ringed equilibria and dashed paths stay fixed; saturated colors are the live experiment.'; footer.append(caption);
  } else if (kind === 'lesson') {
    const nav = doc.createElement('nav'); nav.className = 'ppw-task-tabs'; nav.setAttribute('aria-label', 'Lesson task'); element.insertBefore(nav, body);
    let taskGroups = [], active = 'Explore';
    const descriptions = { Explore: 'Explore excitability: move I, or click the phase plane to change initial conditions.', Observe: 'Observe the full-system trajectory and use playback to connect the two plots.', Compare: 'Freeze the current scene, then adjust I. Gray geometry and dashed paths retain the baseline.', Sweep: 'Scan a parameter range. Switching away cancels a running sweep; completed results are retained.' };
    const description = doc.createElement('p'); description.className = 'ppw-task-description'; element.insertBefore(description, body);
    const buttons = [];
    function choose(task) {
      active = task; for (const current of taskGroups) current.dispose(); taskGroups = [];
      body.replaceChildren(); footer.replaceChildren(); description.textContent = descriptions[task]; element.dataset.task = task;
      for (const node of buttons) node.setAttribute('aria-pressed', String(node.textContent === task));
      const names = task === 'Sweep' ? ['sweep'] : task === 'Explore' ? ['phase_plane'] : ['phase_plane', 'time_series'];
      for (const name of names) taskGroups.push(group(slot(body, name === 'phase_plane' ? 'ppw-primary-plot' : 'ppw-secondary-plot'), [name], { fit: name !== 'sweep' }));
      if (task !== 'Sweep') taskGroups.push(group(footer, ['parameters'], { primary: true }));
      if (task === 'Compare') taskGroups.push(group(footer, ['comparison']));
      update();
    }
    for (const task of Object.keys(descriptions)) buttons.push(button(task, 'ppw-task-tab', () => choose(task), nav));
    choose(active);
  } else if (RESEARCH_KINDS.includes(kind)) {
    group(body, [kind]); group(footer, ['parameters'], { primary: true });
  } else { throw new Error('Unknown presentation: ' + kind); }
  function update() {
    title.textContent = PRESENTATION_LABELS[kind];
    badges.textContent = `${(model.get('model_name') || '').replaceAll('_', ' ')}${model.get('noise_enable') ? ' · noise on' : ''}${model.get('reference_trajectory')?.length ? ' · frozen reference' : ''}${model.get('sweep_running') ? ' · sweep running' : ''}`;
    error.textContent = controller.error?.message || ''; error.hidden = !controller.error;
    for (const instance of [...groups]) instance.update();
    expanded?.composition.update();
  }
  scope.cleanup(closePopover); scope.cleanup(closeExpanded);
  return { element, update, dispose() { scope.dispose(); for (const instance of [...groups]) instance.dispose(); element.remove(); } };
}
