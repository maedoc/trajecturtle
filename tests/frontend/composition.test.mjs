import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><body></body>', { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
// These tests check state, controls and lifecycle, not canvas pixel output.
dom.window.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (_, key) => key === 'measureText' ? () => ({ width: 10 }) : () => {},
  set: () => true,
});
const { render, computeExperiment, experimentSession } = await import('../../src/tvb_phaseplane/static/widget.js');
const pause = (ms = 80) => new Promise(resolve => setTimeout(resolve, ms));
const disposers = [];
afterEach(() => {
  for (const dispose of disposers.splice(0)) dispose();
  document.body.replaceChildren();
});

function model(overrides = {}) {
  const state = {
    model_name: 'fitzhugh_nagumo', params: { a: .7, b: .8, epsilon: .08, I: .5 },
    param_info: { a: [0, 1, .7, 'a'], I: [-1, 1, .5, 'I'] },
    state_names: ['v', 'w'], xlim: [-3, 3], ylim: [-1.5, 2], x0: .1, y0: .1,
    t_max: 2, display: [0, 1], clamped: [0, 0], integrator: 'rk4',
    noise_enable: false, noise_sigma: [0, 0], noise_seed: 0,
    show_nullclines: true, show_vector_field: true, show_trajectory: true, show_fixed_points: true,
    panels: [], reference_trajectory: [], reference_metadata: {}, trajectory_fraction: 1,
    sweep_results: [], sweep_fixed_points: [], sweep_metadata: {}, sweep_running: false,
    ...overrides,
  };
  const callbacks = new Map();
  const result = {
    get: key => state[key],
    set(key, value) {
      if (Object.is(state[key], value)) return;
      state[key] = value;
      for (const cb of [...(callbacks.get('change:' + key) || [])]) cb(result, value);
    },
    on(event, cb) { callbacks.set(event, [...(callbacks.get(event) || []), cb]); },
    off(event, cb) { callbacks.set(event, (callbacks.get(event) || []).filter(f => f !== cb)); },
    save_changes() {}, send() {},
    listenerCount: () => [...callbacks.values()].reduce((sum, list) => sum + list.length, 0),
  };
  return result;
}
async function mount(m, options, extra = {}) {
  const el = document.createElement('div'); document.body.append(el);
  const dispose = await render({ model: m, el, options, ...extra });
  disposers.push(dispose);
  return el;
}
function sweep(el, count = 8) {
  el.querySelector('.ppw-sweep-min').value = '-.2';
  el.querySelector('.ppw-sweep-max').value = '.2';
  el.querySelector('.ppw-sweep-n').value = count;
  el.querySelector('.ppw-sweep-btn').click();
}

test('focused panels allocate no hidden canvases or editor and honor author order', async () => {
  const original = window.HTMLCanvasElement.prototype.getContext;
  let contexts = 0;
  window.HTMLCanvasElement.prototype.getContext = function (...args) { contexts++; return original.apply(this, args); };
  try {
    const m = model({ panels: ['parameters'], visible_parameters: ['I'] });
    const el = await mount(m);
    assert.equal(contexts, 0);
    assert.equal(el.querySelector('canvas'), null);
    assert.equal(el.querySelector('.ppw-editor-panel'), null);
    assert.equal(el.querySelector('.ppw-sweep-select'), null);
    assert.equal(el.querySelectorAll('input[type=range]').length, 1);
    m.set('panels', ['time_series', 'phase_plane', 'parameters']);
    assert.deepEqual([...el.querySelectorAll('[data-panel]')].map(node => node.dataset.panel), ['time_series', 'phase_plane', 'parameters']);
    assert.equal(contexts, 2);
    assert.equal(el.querySelectorAll('canvas').length, 2);
  } finally { window.HTMLCanvasElement.prototype.getContext = original; }
});

test('multiple views share one controller and compute once per input change', async () => {
  const m = model({ panels: ['parameters'], visible_parameters: ['I'] });
  let geometryComputations = 0;
  const set = m.set.bind(m);
  m.set = (key, value) => { if (key === 'fixed_points') geometryComputations++; set(key, value); };
  const first = await mount(m), second = await mount(m), third = await mount(m);
  assert.equal(geometryComputations, 1);
  const number = first.querySelector('.ppw-param-number');
  number.value = '.8'; number.dispatchEvent(new window.Event('change'));
  await pause();
  assert.equal(geometryComputations, 2);
  assert.equal(second.querySelector('.ppw-param-number').value, '0.8');
  assert.equal(third.querySelector('.ppw-param-number').value, '0.8');
});

test('keyboard initial conditions and isolated comparison controls work without export', async () => {
  const m = model({ panels: ['phase_plane', 'comparison'] });
  const el = await mount(m), canvas = el.querySelector('canvas');
  assert.equal(el.querySelector('.ppw-export-svg'), null);
  canvas.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  assert.ok(Math.abs(m.get('x0') - .16) < 1e-9);
  el.querySelector('.ppw-reference').click();
  assert.ok(m.get('reference_trajectory').length > 1);
  el.querySelector('.ppw-reset').click();
  assert.equal(m.get('x0'), .1);
  assert.deepEqual(m.get('reference_trajectory'), []);
});

test('numeric errors are explicit and do not change the experiment', async () => {
  const m = model({ panels: ['parameters'], visible_parameters: ['I'] });
  const el = await mount(m), input = el.querySelector('.ppw-param-number');
  input.value = '2'; input.dispatchEvent(new window.Event('change'));
  assert.equal(m.get('params').I, .5);
  assert.equal(input.getAttribute('aria-invalid'), 'true');
  assert.equal(el.querySelector('.ppw-input-error').hidden, false);
  input.value = '.2'; input.dispatchEvent(new window.Event('change'));
  assert.equal(m.get('params').I, .2);
  assert.equal(el.querySelector('.ppw-input-error').hidden, true);
});

test('editor-only views apply coherent metadata and cancel pending edits on disposal', async () => {
  const spec = { state_vars: { x: { range: [-1, 1], default: 0 }, y: { range: [-1, 1], default: 0 } }, equations: { x: '-x', y: '-y' }, parameters: {} };
  const m = model({ model_name: 'custom', model_spec: spec, panels: ['editor'] });
  const el = await mount(m);
  assert.equal(el.querySelector('canvas'), null);
  el.querySelector('.ppw-editor-toggle').click();
  const equation = el.querySelector('.ppw-ed-expr');
  equation.value = '-2*x'; equation.dispatchEvent(new window.Event('input', { bubbles: true }));
  el.querySelector('.ppw-editor-apply').click();
  assert.equal(m.get('model_spec').equations.x, '-2*x');
  assert.deepEqual(m.get('state_names'), ['x', 'y']);
  assert.deepEqual(m.get('params'), {});
  assert.match(el.querySelector('.ppw-editor-status').textContent, /Valid/);
  equation.value = '-3*x'; equation.dispatchEvent(new window.Event('input', { bubbles: true }));
  m.set('panels', ['parameters']);
  await pause(550);
  assert.equal(m.get('model_spec').equations.x, '-2*x');
});

test('invalid custom model shows an error, clears results, and recovers', async () => {
  const m = model();
  const el = await mount(m);
  m.set('model_name', 'custom'); m.set('model_spec', { equations: {} });
  await pause();
  assert.equal(el.querySelector('[role=alert]').hidden, false);
  assert.match(el.querySelector('[role=alert]').textContent, /invalid or empty spec/);
  assert.deepEqual(m.get('trajectory'), []);
  assert.deepEqual(m.get('fixed_points'), []);
  m.set('model_name', 'fitzhugh_nagumo');
  await pause();
  assert.ok(m.get('trajectory').length > 1);
  assert.equal(el.querySelector('[role=alert]').hidden, true);
});

test('equation compilation errors are not replaced with zero dynamics', () => {
  const m = model({ model_name: 'custom', model_spec: {
    state_vars: { x: { range: [-1, 1] }, y: { range: [-1, 1] } },
    equations: { x: 'x + )', y: '-y' }, parameters: {},
  } });
  assert.throws(() => computeExperiment(m), /Cannot compile equation for x/);
  assert.deepEqual(m.get('trajectory'), []);
});

test('invalid custom model can mount and a failed sweep does not stay running', async () => {
  const m = model({ model_name: 'custom', model_spec: { equations: {} } });
  const el = await mount(m);
  sweep(el);
  assert.equal(m.get('sweep_running'), false);
  assert.equal(el.querySelector('.ppw-sweep-btn').disabled, false);
  assert.match(el.querySelector('.ppw-sweep-status').textContent, /Sweep failed/);
});

test('an exception in a later sweep chunk restores state and permits retry', async () => {
  const m = model();
  const el = await mount(m);
  let broken = false;
  const params = new Proxy(m.get('params'), { get(target, key) {
    if (broken && key === 'I') throw new Error('injected RHS failure');
    return target[key];
  } });
  m.set('params', params);
  await pause();
  sweep(el);
  assert.equal(m.get('sweep_running'), true);
  broken = true;
  await pause();
  assert.equal(m.get('sweep_running'), false);
  assert.match(el.querySelector('.ppw-sweep-status').textContent, /injected RHS failure/);
  broken = false;
  sweep(el, 2);
  assert.equal(m.get('sweep_running'), false);
  assert.equal(m.get('sweep_results').length, 2);
});

test('clamped values and metadata refresh preserve selected sweep settings', async () => {
  const m = model({
    model_name: 'custom', state_names: ['x', 'y', 'z'], clamped: [0, 0, .5],
    model_spec: {
      state_vars: { x: { range: [-1, 1] }, y: { range: [-1, 1] }, z: { range: [0, 1] } },
      equations: { x: '-x + z', y: '-y', z: '-z' },
      parameters: { a: { range: [0, 1], default: .7 }, I: { range: [-1, 1], default: .5 } },
    },
  });
  const el = await mount(m);
  el.querySelector('.ppw-sweep-select').value = 'I';
  el.querySelector('.ppw-sweep-min').value = '-.3';
  el.querySelector('.ppw-sweep-max').value = '.9';
  const clampedSlider = el.querySelector('.ppw-clamped-slider');
  clampedSlider.value = '.7';
  clampedSlider.dispatchEvent(new window.Event('input'));
  await pause();
  assert.equal(m.get('clamped')[2], .7);
  assert.equal(clampedSlider.isConnected, true);
  m.set('clamped', [0, 0, .8]);
  await pause();
  assert.equal(clampedSlider.value, '0.8');
  m.set('param_info', { ...m.get('param_info') });
  await pause();
  assert.equal(el.querySelector('.ppw-sweep-select').value, 'I');
  assert.equal(el.querySelector('.ppw-sweep-min').value, '-.3');
  assert.equal(el.querySelector('.ppw-sweep-max').value, '.9');
});

test('per-view proxies resolve one canonical experiment and share cache and baseline', async () => {
  const shared = model();
  const manager = { get_model: async id => { assert.equal(id, 'experiment'); return shared; } };
  const proxy = () => ({ get: key => key === 'experiment_id' ? 'experiment' : shared.get(key), widget_manager: manager });
  const controls = await mount(proxy(), { get: key => ({ panels: ['parameters'], visible_parameters: ['I'] })[key] });
  const baseline = experimentSession(shared).baseline;
  const trajectory = shared.get('trajectory');
  const view = model({ experiment: 'IPY_MODEL_experiment', panels: ['phase_plane'] });
  view.widget_manager = manager;
  await mount(view);
  assert.equal(shared.get('trajectory'), trajectory);
  const number = controls.querySelector('.ppw-param-number');
  number.value = '.8'; number.dispatchEvent(new window.Event('change'));
  await pause();
  assert.equal(shared.get('params').I, .8);
  assert.notEqual(shared.get('trajectory'), trajectory);
  assert.equal(experimentSession(shared).baseline, baseline);
  assert.equal(baseline.params.I, .5);
  assert.equal(document.querySelectorAll('.ppw-phase-canvas').length, 1);
});

test('missing/rejected managers show errors instead of mounting a view as an experiment', async () => {
  for (const manager of [undefined, { get_model: async () => { throw new Error('manager failed'); } }]) {
    const view = model({ experiment: 'IPY_MODEL_missing' });
    view.widget_manager = manager;
    const el = await mount(view);
    assert.match(el.textContent, /Cannot display experiment/);
    assert.equal(el.querySelector('.ppw-widget'), null);
  }
});

test('an aborted async resolution never mounts', async () => {
  const controller = new AbortController();
  const m = model({ experiment_id: 'experiment' });
  m.widget_manager = { get_model: async () => { controller.abort(); return model(); } };
  const el = await mount(m, null, { signal: controller.signal });
  assert.equal(el.children.length, 0);
});

test('disposing a cancelled sweep owner does not cancel a newer shared sweep', async () => {
  const m = model();
  const first = await mount(m);
  const second = await mount(m);
  sweep(first, 100);
  const number = first.querySelector('.ppw-param-number');
  number.value = '.8';
  number.dispatchEvent(new window.Event('change')); // immediately invalidates/cancels the first sweep
  assert.equal(m.get('sweep_running'), false);
  sweep(second, 8);
  assert.equal(m.get('sweep_running'), true);
  disposers.shift()(); // dispose the old job owner before its pending chunk can run
  assert.equal(m.get('sweep_running'), true);
  await pause(300);
  assert.equal(m.get('sweep_results').length, 8);
  assert.equal(m.get('sweep_running'), false);
});

test('remount/disposal cleans listeners and cancels only the owning sweep', async () => {
  const a = model(), b = model();
  const elA = await mount(a), elB = await mount(b);
  const count = a.listenerCount();
  a.set('panels', ['phase_plane', 'sweep']);
  assert.equal(a.listenerCount(), count - 1); // export's msg:custom listener is not mounted
  a.set('panels', ['phase_plane', 'sweep']);
  assert.equal(a.listenerCount(), count - 1);
  sweep(elA, 100); sweep(elB, 8);
  a.set('panels', ['phase_plane']);
  assert.equal(a.get('sweep_running'), false);
  assert.equal(b.get('sweep_running'), true);
  await pause(300);
  assert.equal(b.get('sweep_running'), false);
  assert.equal(b.get('sweep_results').length, 8);
  for (const dispose of disposers.splice(0)) dispose();
  assert.equal(a.listenerCount(), 0);
  assert.equal(b.listenerCount(), 0);
});

test('notebook view picker syncs presentation, retains state/focus and cleans listeners', async () => {
  const m = model({ presentation: 'figure', view_switcher: true });
  const el = await mount(m), trajectory = m.get('trajectory'), count = m.listenerCount();
  for (const value of ['comparison', 'bench', 'drawer', 'lesson', 'figure']) {
    const picker = el.querySelector('.ppw-view-selector'); picker.value = value; picker.dispatchEvent(new window.Event('change'));
    assert.equal(m.get('presentation'), value);
    assert.equal(el.querySelector('[data-presentation]').dataset.presentation, value);
    assert.equal(document.activeElement, el.querySelector('.ppw-view-selector'));
    assert.equal(m.get('trajectory'), trajectory);
    assert.equal(m.listenerCount(), count);
  }
  m.set('presentation', 'comparison'); el.querySelector('.ppw-reference').click();
  const snapshot = m.get('reference_scene');
  m.set('presentation', 'figure');
  assert.equal(el.querySelector('.ppw-view-selector').value, 'figure');
  assert.equal(m.get('reference_scene'), snapshot);
  for (const dispose of disposers.splice(0)) dispose();
  assert.equal(m.listenerCount(), 0);
});

test('all six scientific-task layouts use isolated previews and clean up listeners', async () => {
  for (const presentation of ['multiples', 'navigator', 'basin', 'evidence', 'prediction', 'ensemble']) {
    const m = model({ presentation, view_switcher: true, visible_parameters: ['I'] });
    computeExperiment(m); const trajectory = m.get('trajectory'), params = m.get('params');
    const el = await mount(m);
    assert.equal(el.querySelector('[data-presentation]').dataset.presentation, presentation);
    assert.equal(m.get('trajectory'), trajectory);
    assert.equal(m.get('params'), params);
    assert.equal(m.get('noise_enable'), false);
    for (const dispose of disposers.splice(0)) dispose();
    assert.equal(m.listenerCount(), 0);
  }
});

test('small multiples validate tile bounds and change live parameters only on Use', async () => {
  const m = model({ presentation: 'multiples', visible_parameters: ['I'] }), el = await mount(m);
  assert.equal(el.querySelectorAll('canvas').length, 3);
  const input = el.querySelectorAll('.ppw-multiple-value')[2];
  input.value = '.9'; input.dispatchEvent(new window.Event('change'));
  assert.equal(m.get('params').I, .5);
  el.querySelectorAll('.ppw-use-tile')[2].click(); assert.equal(m.get('params').I, .9);
  input.value = '10'; input.dispatchEvent(new window.Event('change'));
  assert.equal(input.getAttribute('aria-invalid'), 'true'); assert.equal(m.get('params').I, .9);
});

test('initial-condition collection survives layout changes, filters projections and resets', async () => {
  const m = model({ presentation: 'basin' }), el = await mount(m);
  el.querySelector('.ppw-add-start').click();
  m.set('x0', .4); await pause(); el.querySelector('.ppw-add-start').click();
  assert.equal(m.get('initial_conditions').length, 2);
  m.set('presentation', 'figure'); m.set('presentation', 'basin');
  assert.equal(el.querySelectorAll('.ppw-trace-list button').length, 2);
  m.set('display', [1, 0]); await pause(); assert.equal(el.querySelectorAll('.ppw-trace-list button').length, 0);
  m.set('display', [0, 1]); await pause(); assert.equal(el.querySelectorAll('.ppw-trace-list button').length, 2);
  el.querySelector('.ppw-stage-reset').click(); assert.deepEqual(m.get('initial_conditions'), []);
});

test('evidence stores deep snapshots, previews safely and restores explicitly', async () => {
  const m = model({ presentation: 'evidence' }), el = await mount(m);
  el.querySelector('[aria-label="Observation label"]').value = '<b>first</b>';
  el.querySelector('.ppw-capture-observation').click();
  const card = m.get('observation_cards')[0]; assert.equal(card.state.params.I, .5);
  assert.notEqual(card.scene.trajectory, m.get('trajectory'));
  assert.equal(el.querySelector('.ppw-evidence-cards b'), null);
  m.set('params', { ...m.get('params'), I: .8 }); await pause();
  el.querySelector('.ppw-evidence-cards button').click(); assert.equal(m.get('params').I, .8);
  m.set('presentation', 'figure'); m.set('presentation', 'evidence');
  el.querySelector('.ppw-restore-observation').click(); assert.equal(m.get('params').I, .5);
  assert.equal(m.get('observation_cards').length, 1);
  el.querySelector('.ppw-stage-reset').click(); assert.deepEqual(m.get('observation_cards'), []);
});

test('prediction reveal is local, survives layout switches and hides again on changed experiment', async () => {
  const m = model({ presentation: 'prediction' }), el = await mount(m);
  const answer = el.querySelector('.ppw-prediction-answer'); answer.value = 'A stable resting state'; answer.dispatchEvent(new window.Event('input'));
  el.querySelector('.ppw-reveal').click(); assert.equal(el.querySelector('.ppw-reveal').getAttribute('aria-pressed'), 'true');
  assert.equal(m.get('show_trajectory'), true);
  m.set('presentation', 'figure'); m.set('presentation', 'prediction');
  assert.equal(el.querySelector('.ppw-prediction-answer').value, 'A stable resting state');
  m.set('params', { ...m.get('params'), I: .1 }); await pause();
  assert.equal(el.querySelector('.ppw-reveal').getAttribute('aria-pressed'), 'false');
  assert.equal(el.querySelector('.ppw-prediction-answer').value, '');
});

test('prediction drafts/reveal remain independent across separately configured outputs', async () => {
  const m = model(), config = () => ({ get: key => key === 'presentation' ? 'prediction' : undefined });
  const first = await mount(m, config()), second = await mount(m, config());
  first.querySelector('.ppw-reveal').click();
  m.set('trajectory_fraction', .5);
  assert.equal(first.querySelector('.ppw-reveal').getAttribute('aria-pressed'), 'true');
  assert.equal(second.querySelector('.ppw-reveal').getAttribute('aria-pressed'), 'false');
  first.querySelector('.ppw-stage-reset').click();
  assert.equal(first.querySelector('.ppw-reveal').getAttribute('aria-pressed'), 'false');
});

test('ensemble previews leave live noise unchanged until a seed is explicitly applied', async () => {
  const m = model({ presentation: 'ensemble' }), el = await mount(m);
  assert.equal(el.querySelectorAll('.ppw-use-member').length, 4);
  assert.equal(m.get('noise_enable'), false);
  const count = el.querySelector('.ppw-ensemble-count'); count.value = '20'; count.dispatchEvent(new window.Event('change'));
  assert.match(el.querySelector('.ppw-research-status').textContent, /2–8/);
  el.querySelectorAll('.ppw-use-member')[1].click();
  assert.equal(m.get('noise_seed'), 1); assert.equal(m.get('noise_enable'), true);
  assert.equal(m.get('integrator'), 'heun'); assert.deepEqual(m.get('noise_sigma'), [.1, .1]);
});

test('navigator retains a completed scan when selecting its parameter, invalidates other changes', async () => {
  const m = model({ presentation: 'navigator' }), el = await mount(m);
  const select = el.querySelector('.ppw-sweep-select'); select.value = 'I'; select.dispatchEvent(new window.Event('change'));
  el.querySelector('.ppw-sweep-min').value = '0'; el.querySelector('.ppw-sweep-max').value = '1';
  el.querySelector('.ppw-sweep-n').value = '4'; el.querySelector('.ppw-sweep-btn').click(); await pause(400);
  const seek = el.querySelector('.ppw-scan-seek'); assert.equal(seek.disabled, false);
  seek.value = '1'; seek.dispatchEvent(new window.Event('input'));
  assert.equal(m.get('params').I, 1 / 3); assert.equal(seek.disabled, false);
  assert.equal(el.querySelector('.ppw-sweep-canvas').hidden, false);
  m.set('presentation', 'figure'); m.set('presentation', 'navigator');
  assert.equal(el.querySelector('.ppw-scan-seek').disabled, false);
  m.set('x0', .6); await pause(); assert.equal(el.querySelector('.ppw-scan-seek').disabled, true);
  el.querySelector('.ppw-stage-reset').click();
  sweep(el, 4); await pause(400);
  assert.equal(el.querySelector('.ppw-scan-seek').disabled, false);
  el.querySelector('.ppw-stage-reset').click();
  assert.equal(el.querySelector('.ppw-scan-seek').disabled, true);
  sweep(el, 100); assert.equal(m.get('sweep_running'), true);
  el.querySelector('.ppw-stage-reset').click();
  assert.equal(m.get('sweep_running'), false);
  assert.deepEqual(m.get('sweep_results'), []);
});

test('parameter-free 3D custom experiments have safe empty multiples and projected trajectories', async () => {
  const spec = { state_vars: { x: { range: [-1, 1] }, y: { range: [-1, 1] }, z: { range: [-1, 1] } }, equations: { x: '-x', y: '-y', z: '-z' }, parameters: {} };
  const m = model({ model_name: 'custom', model_spec: spec, params: {}, param_info: {}, state_names: ['x', 'y', 'z'], display: [2, 0], clamped: [0, .5, 0], presentation: 'multiples' });
  const el = await mount(m);
  assert.match(el.querySelector('.ppw-research-status').textContent, /no parameters/);
  assert.equal(el.querySelector('.ppw-use-tile').disabled, true);
  m.set('presentation', 'figure'); m.set('presentation', 'multiples');
  assert.match(el.querySelector('.ppw-research-status').textContent, /no parameters/);
  m.set('presentation', 'basin'); el.querySelector('.ppw-add-start').click();
  assert.equal(m.get('initial_conditions').length, 1);
  assert.equal(el.querySelector('.ppw-research-status').textContent, '');
  m.set('presentation', 'ensemble'); assert.equal(el.querySelector('.ppw-research-status').textContent, '');
});

test('truncated divergent preview paths are reported rather than presented as complete', async () => {
  const spec = { state_vars: { x: { range: [-1, 1] }, y: { range: [-1, 1] } }, equations: { x: 'x*x*x', y: '-y' }, parameters: {} };
  const m = model({ model_name: 'custom', model_spec: spec, params: {}, param_info: {}, state_names: ['x', 'y'], x0: 10, presentation: 'ensemble' });
  const el = await mount(m);
  assert.match(el.querySelector('.ppw-research-status').textContent, /diverged/);
});

test('view picker works in static per-view configurations without a mutable model proxy', async () => {
  const m = model(), options = { get: key => ({ presentation: 'figure', view_switcher: true, visible_parameters: ['I'] })[key] };
  const el = await mount(m, options);
  const picker = el.querySelector('.ppw-view-selector'); picker.value = 'bench'; picker.dispatchEvent(new window.Event('change'));
  assert.equal(el.querySelector('[data-presentation]').dataset.presentation, 'bench');
  assert.equal(el.querySelectorAll('canvas').length, 2);
  assert.equal(m.get('presentation'), undefined);
});

test('experimental presentations switch without recomputing or resetting the experiment', async () => {
  const m = model({ presentation: 'figure', visible_parameters: ['I'] });
  const el = await mount(m), original = m.get('trajectory');
  assert.equal(el.querySelectorAll('canvas').length, 1);
  for (const presentation of ['bench', 'drawer', 'comparison', 'lesson', 'figure']) {
    m.set('presentation', presentation);
    assert.equal(el.querySelector('[data-presentation]').dataset.presentation, presentation);
    assert.equal(m.get('trajectory'), original);
  }
  assert.equal(el.querySelectorAll('canvas').length, 1);
  for (const dispose of disposers.splice(0)) dispose();
  assert.equal(m.listenerCount(), 0);
});

test('frozen scenes retain geometry and metadata across parameter changes but reject different projections', async () => {
  const m = model({ presentation: 'comparison' });
  const el = await mount(m);
  el.querySelector('.ppw-reference').click();
  const scene = m.get('reference_scene'), trajectory = m.get('reference_trajectory');
  assert.deepEqual(scene.nullcline_x, m.get('nullcline_x'));
  assert.notEqual(scene.nullcline_x, m.get('nullcline_x'));
  assert.deepEqual(scene.clamped, m.get('clamped'));
  assert.equal(m.get('reference_metadata').params.I, .5);
  m.set('params', { ...m.get('params'), I: .1 }); await pause();
  assert.equal(m.get('reference_scene'), scene);
  assert.equal(m.get('reference_trajectory'), trajectory);
  assert.notDeepEqual(m.get('nullcline_x'), scene.nullcline_x);
  const toggle = el.querySelector('.ppw-show-reference'); toggle.checked = false; toggle.dispatchEvent(new window.Event('change'));
  assert.equal(m.get('show_reference'), false);
  assert.equal(m.get('reference_scene'), scene);
  m.set('display', [1, 0]); await pause();
  assert.deepEqual(m.get('reference_scene'), {});
  assert.deepEqual(m.get('reference_trajectory'), []);
});

test('lesson tasks and the instrument drawer mount only active tools and retain completed state', async () => {
  const m = model({ presentation: 'lesson', visible_parameters: ['I'] });
  const el = await mount(m);
  const choose = name => [...el.querySelectorAll('.ppw-task-tab')].find(button => button.textContent === name).click();
  choose('Compare'); el.querySelector('.ppw-reference').click();
  const reference = m.get('reference_scene');
  choose('Sweep');
  assert.equal(el.querySelector('.ppw-phase-canvas'), null);
  sweep(el, 4); await pause(300);
  assert.equal(m.get('sweep_results').length, 4);
  choose('Explore'); choose('Sweep');
  assert.match(el.querySelector('.ppw-sweep-status').textContent, /4 samples complete/);
  assert.equal(m.get('reference_scene'), reference);
  sweep(el, 100); choose('Observe');
  assert.equal(m.get('sweep_running'), false);
  assert.equal(el.querySelectorAll('canvas').length, 2);
  m.set('presentation', 'drawer');
  assert.equal(el.querySelector('.ppw-instrument-drawer').hidden, true);
  el.querySelector('.ppw-drawer-toggle').click();
  assert.equal(el.querySelector('.ppw-instrument-drawer').hidden, false);
  assert.equal(el.querySelectorAll('.ppw-param-number').length, 3);
  el.querySelector('.ppw-drawer-toggle').click();
  assert.equal(el.querySelectorAll('.ppw-param-number').length, 1);
});

test('dt, bounds clamping, safe math helpers, and theme observer cleanup', async () => {
  const m = model({
    model_name: 'custom',
    model_spec: {
      name: 'BoundedHelper',
      state_vars: {
        r: { default: 0.5, range: [0, 2], bounds: [0, null] },
        v: { default: 0, range: [-2, 2] },
      },
      parameters: { c: { default: 1.0, range: [0, 2], step: 0.1 } },
      equations: { r: '-10 * r - c + step(v)', v: '1.0 / (1.0 + safe_exp(-1000)) - rww_H(v, 0.154)' },
      display: [0, 1],
      custom_functions: {},
      integrator: 'rk4',
    },
    state_names: ['r', 'v'],
    params: { c: 1.0 },
    param_info: { c: [0, 2, 1.0, 'c'] },
    x0: 0.1,
    y0: 0.0,
    dt: 0.05,
    t_max: 1.0,
  });
  const el = await mount(m);
  const traj = m.get('trajectory');
  assert.ok(traj.length > 5);
  assert.ok(traj.every(([, r, v]) => Number.isFinite(r) && Number.isFinite(v) && r >= 0));
  assert.ok(Math.abs(traj[traj.length - 1][0] - 1.0) < 1e-6);

  document.documentElement.classList.add('dark');
  await pause(20);
  document.documentElement.classList.remove('dark');
});
