export async function render({ model, el, options = null, signal }) {
  const configuration = options || model;
  try {
    const ref = model.get('experiment');
    const id = typeof ref === 'string' ? ref.replace(/^IPY_MODEL_/, '') : model.get('experiment_id');
    let shared = ref && typeof ref.get === 'function' ? ref : model;
    if (shared === model && id) {
      if (!model.widget_manager) throw new Error('Widget manager unavailable; cannot resolve the shared experiment');
      shared = await model.widget_manager.get_model(id);
      if (!shared || typeof shared.get !== 'function') throw new Error('Shared experiment was not found');
    } else if (shared === model && ref) {
      throw new Error('Invalid shared experiment reference');
    }
    if (signal?.aborted) return () => {};
    return mountConfigured(shared, el, configuration);
  } catch (error) {
    if (signal?.aborted) return () => {};
    el.replaceChildren();
    const message = el.ownerDocument.createElement('div');
    message.setAttribute('role', 'alert');
    message.textContent = `Cannot display experiment: ${error.message}`;
    el.appendChild(message);
    return () => el.replaceChildren();
  }
}


// A view owns presentation only. Computation and async work live on its controller.
const PANEL_FACTORIES = {
  model: mountModel, parameters: mountParameters, projection: mountProjection,
  limits: mountLimits, noise: mountNoise, layers: mountLayers,
  comparison: mountComparison, export: mountExport, editor: mountEditor,
  phase_plane: mountPhasePlane, time_series: mountTimeSeries, sweep: mountSweep,
};
const DEFAULT_PANELS = ['model', 'parameters', 'projection', 'limits', 'noise', 'layers', 'comparison', 'export', 'editor', 'phase_plane', 'time_series', 'sweep'];

function mountConfigured(model, el, configuration) {
  const controller = acquireExperiment(model);
  const doc = el.ownerDocument;
  let components = [], unsubscribe = () => {}, root, errorMessage, disposed = false, localPresentation = null;
  function update() {
    if (disposed) return;
    let error = controller.error;
    for (const component of components) {
      try { component.update(); }
      catch (failure) { error ||= failure; }
    }
    errorMessage.textContent = error?.message || '';
    errorMessage.hidden = !error;
  }
  function unmount() {
    unsubscribe();
    for (const component of components.splice(0)) component.dispose();
    el.replaceChildren();
  }
  function mount() {
    unmount();
    const requested = configuration.get('panels') || [];
    const displayMode = configuration.get('display_mode') || 'full';
    const names = requested.length ? requested : displayMode === 'phase_plane' ? ['phase_plane'] : DEFAULT_PANELS;
    root = doc.createElement('div');
    const presentation = localPresentation || configuration.get('presentation') || 'panels';
    root.className = presentation === 'panels' ? 'ppw-widget ppw-component-layout' : 'ppw-widget ppw-presentation-root';
    root.classList.toggle('ppw-phase-only', !requested.length && displayMode === 'phase_plane');
    root.classList.toggle('ppw-vertical', configuration.get('panel_layout') === 'vertical');
    root.classList.toggle('ppw-single-panel', names.length === 1);
    root.classList.toggle('ppw-single-plot', presentation === 'panels' && names.includes('phase_plane') && !names.includes('time_series'));
    errorMessage = doc.createElement('div'); errorMessage.className = 'ppw-error';
    errorMessage.setAttribute('role', 'alert'); errorMessage.hidden = true; root.append(errorMessage);
    el.append(root);
    if (configuration.get('view_switcher')) {
      const scope = panelScope(), picker = panelElement(doc, 'ppw-view-picker');
      const label = doc.createElement('label'); label.append(doc.createTextNode('View '));
      const select = doc.createElement('select'); select.className = 'ppw-view-selector'; select.setAttribute('aria-label', 'View layout'); label.append(select);
      const choices = { figure: 'Interactive figure', bench: '16:9 experiment bench', drawer: 'Instrument drawer', comparison: 'Comparison lens', lesson: 'Guided lesson' };
      if (presentation === 'panels') choices.panels = 'All panels';
      for (const [value, text] of Object.entries(choices)) { const option = doc.createElement('option'); option.value = value; option.textContent = text; select.append(option); }
      select.value = presentation; picker.append(label);
      const hint = doc.createElement('small'); hint.textContent = 'Experimental layouts · same experiment across views'; picker.append(hint);
      scope.on(select, 'change', () => {
        const value = select.value;
        if (typeof configuration.set === 'function') {
          configuration.set('presentation', value); configuration.save_changes?.();
        } else { localPresentation = value; mount(); }
        root.querySelector('.ppw-view-selector')?.focus();
      });
      components.push(mountedPanel(picker, scope, () => {})); root.append(picker);
    }
    if (presentation !== 'panels') {
      const composition = mountPresentation({ model, controller, document: doc, configuration }, presentation);
      components.push(composition); root.append(composition.element);
    } else for (const name of names) {
      const factory = PANEL_FACTORIES[name];
      if (!factory) throw new Error('Unknown panel: ' + name);
      const component = factory({ model, controller, document: doc, configuration });
      component.element.classList.add('ppw-panel');
      component.element.dataset.panel = name;
      component.element.setAttribute('aria-label', name.replaceAll('_', ' '));
      components.push(component); root.append(component.element);
    }
    unsubscribe = controller.subscribe(update);
    update();
  }
  const events = ['panels', 'visible_parameters', 'panel_layout', 'display_mode', 'presentation', 'view_switcher'].map(key => 'change:' + key);
  try { mount(); }
  catch (error) { unmount(); controller.release(); throw error; }
  for (const event of events) configuration.on?.(event, mount);
  return () => {
    if (disposed) return;
    disposed = true;
    for (const event of events) configuration.off?.(event, mount);
    unmount(); controller.release();
  };
}

export default { render };
