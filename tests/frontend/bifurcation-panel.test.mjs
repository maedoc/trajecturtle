import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><body></body>', { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
dom.window.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
  get: (_, key) => (key === 'measureText' ? () => ({ width: 10 }) : () => {}),
  set: () => true,
});

const {
  acquireExperiment,
  drawBifurcation1D,
  drawBifurcation2D,
  mountBifurcation,
  render,
} = await import('../../src/tvb_phaseplane/static/widget.js');

const cleanups = [];
afterEach(() => {
  for (const fn of cleanups.splice(0)) fn();
  document.body.replaceChildren();
});

function createRecordingCtx() {
  const calls = [];
  const state = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    font: '',
    textAlign: '',
    textBaseline: '',
    dash: [],
  };
  const ctx = {
    get calls() { return calls; },
    set fillStyle(v) { state.fillStyle = v; },
    get fillStyle() { return state.fillStyle; },
    set strokeStyle(v) { state.strokeStyle = v; },
    get strokeStyle() { return state.strokeStyle; },
    set lineWidth(v) { state.lineWidth = v; },
    get lineWidth() { return state.lineWidth; },
    set font(v) { state.font = v; },
    get font() { return state.font; },
    set textAlign(v) { state.textAlign = v; },
    get textAlign() { return state.textAlign; },
    set textBaseline(v) { state.textBaseline = v; },
    get textBaseline() { return state.textBaseline; },
    clearRect: (...a) => calls.push({ op: 'clearRect', a }),
    fillRect: (...a) => calls.push({ op: 'fillRect', a, fillStyle: state.fillStyle }),
    beginPath: () => calls.push({ op: 'beginPath' }),
    closePath: () => calls.push({ op: 'closePath' }),
    moveTo: (...a) => calls.push({ op: 'moveTo', a }),
    lineTo: (...a) => calls.push({ op: 'lineTo', a }),
    arc: (...a) => calls.push({ op: 'arc', a }),
    rect: (...a) => calls.push({ op: 'rect', a }),
    clip: () => calls.push({ op: 'clip' }),
    save: () => calls.push({ op: 'save' }),
    restore: () => calls.push({ op: 'restore' }),
    translate: (...a) => calls.push({ op: 'translate', a }),
    rotate: (...a) => calls.push({ op: 'rotate', a }),
    setLineDash: (d) => { state.dash = [...d]; calls.push({ op: 'setLineDash', d: [...d] }); },
    stroke: () => calls.push({ op: 'stroke', strokeStyle: state.strokeStyle, lineWidth: state.lineWidth, dash: [...state.dash] }),
    fill: () => calls.push({ op: 'fill', fillStyle: state.fillStyle }),
    fillText: (text, x, y) => calls.push({ op: 'fillText', text, x, y, fillStyle: state.fillStyle }),
    measureText: () => ({ width: 10 }),
  };
  return ctx;
}

function createTestModel(overrides = {}) {
  const state = {
    model_name: 'fitzhugh_nagumo',
    model_spec: null,
    params: { a: 0.7, b: 0.8, epsilon: 0.08, I: 0.5 },
    param_info: { a: [0, 1, 0.7, 'a'], b: [0, 2, 0.8, 'b'], epsilon: [0.01, 1, 0.08, 'eps'], I: [-1, 2, 0.5, 'I'] },
    state_names: ['v', 'w'],
    xlim: [-3, 3],
    ylim: [-1.5, 2],
    x0: 0.1,
    y0: 0.1,
    t_max: 2,
    dt: 0.05,
    display: [0, 1],
    clamped: [0, 0],
    integrator: 'rk4',
    noise_enable: false,
    noise_sigma: [0, 0],
    noise_seed: 0,
    show_nullclines: false,
    show_vector_field: false,
    show_trajectory: true,
    show_fixed_points: true,
    trajectory_fraction: 1,
    sweep_results: [],
    sweep_fixed_points: [],
    sweep_metadata: {},
    sweep_running: false,
    bifurcation_branches: [],
    bifurcation_special_points: [],
    bifurcation_config: { param1: 'I', observable: 0 },
    bifurcation_running: false,
    bifurcation_metadata: {},
    bifurcation_presets: [],
    ...overrides,
  };
  const callbacks = new Map();
  return {
    get: (key) => state[key],
    set(key, value) {
      if (Object.is(state[key], value)) return;
      state[key] = value;
      for (const cb of [...(callbacks.get('change:' + key) || [])]) cb(this, value);
    },
    on(event, cb) {
      callbacks.set(event, [...(callbacks.get(event) || []), cb]);
    },
    off(event, cb) {
      callbacks.set(event, (callbacks.get(event) || []).filter((f) => f !== cb));
    },
    save_changes() {},
    send() {},
  };
}

test('drawBifurcation1D renders empty state, EP/LC stability styles, special points, and crosshair', () => {
  const m = createTestModel();
  const emptyCtx = createRecordingCtx();
  drawBifurcation1D(m, emptyCtx, 400, 300, { bg: '#ffffff', text: '#111', muted: '#777', grid: '#eee', accent: '#245c9b' });
  const emptyTexts = emptyCtx.calls.filter((c) => c.op === 'fillText').map((c) => c.text);
  assert.ok(emptyTexts.some((t) => /continuation/i.test(t)));

  m.set('bifurcation_branches', [
    {
      id: 'ep_1',
      type: 'EP',
      param1: 'I',
      points: [
        { p: 0.0, u: [-1.0, -0.5], stable: true },
        { p: 0.5, u: [0.0, 0.0], stable: false },
        { p: 1.0, u: [1.0, 0.5], stable: true },
      ],
    },
    {
      id: 'lc_1',
      type: 'LC',
      param1: 'I',
      points: [
        { p: 0.4, u: [0.6, 0.2], u_max: [0.6, 0.2], u_min: [-0.6, -0.2], period: 5.0, stable: true },
        { p: 0.8, u: [0.9, 0.3], u_max: [0.9, 0.3], u_min: [-0.9, -0.3], period: 6.0, stable: false },
      ],
    },
  ]);
  m.set('bifurcation_special_points', [
    { id: 'HB1', label: 'HB1', ty: 'HB', p: 0.5, u: [0.0, 0.0], branch_id: 'ep_1', param1: 'I' },
    { id: 'LP1', label: 'LP1', ty: 'LP', p: 0.2, u: [-0.5, -0.2], branch_id: 'ep_1', param1: 'I' },
    { id: 'LPC1', label: 'LPC1', ty: 'LPC', p: 0.7, u: [0.8, 0.25], branch_id: 'lc_1', param1: 'I' },
    { id: 'PD1', label: 'PD1', ty: 'PD', p: 0.75, u: [0.85, 0.28], branch_id: 'lc_1', param1: 'I' },
    { id: 'BP1', label: 'BP1', ty: 'BP', p: 0.9, u: [0.95, 0.4], branch_id: 'ep_1', param1: 'I' },
  ]);

  const ctx = createRecordingCtx();
  drawBifurcation1D(m, ctx, 500, 320, { background: '#fafafa', text: '#222', muted: '#666', grid: '#e9ecef', accent: '#245c9b' });
  const strokes = ctx.calls.filter((c) => c.op === 'stroke');
  assert.ok(strokes.some((s) => s.strokeStyle === '#245c9b' && s.dash.length === 0), 'solid stable EP');
  assert.ok(strokes.some((s) => s.strokeStyle === '#7f8c9a' && s.dash[0] === 6), 'dashed unstable EP');
  assert.ok(strokes.some((s) => s.strokeStyle === '#2e7d32' && s.dash.length === 0), 'solid stable LC');
  assert.ok(strokes.some((s) => s.strokeStyle === '#e65100' && s.dash[0] === 3), 'dashed unstable LC');
  assert.ok(strokes.some((s) => s.dash[0] === 4 && s.dash[1] === 4), 'live crosshair');

  const texts = ctx.calls.filter((c) => c.op === 'fillText').map((c) => c.text);
  for (const label of ['HB1', 'LP1', 'LPC1', 'PD1', 'BP1']) {
    assert.ok(texts.includes(label), `expected special-point label ${label}`);
  }

  // Test L2 observable mode
  m.set('bifurcation_config', { param1: 'I', observable: 'L2' });
  const l2Ctx = createRecordingCtx();
  drawBifurcation1D(m, l2Ctx, 500, 320);
  const l2Texts = l2Ctx.calls.filter((c) => c.op === 'fillText').map((c) => c.text);
  assert.ok(l2Texts.includes('L2 norm'));
});

test('drawBifurcation2D renders LP2/HB2 curves, codim-2 markers, and 2D crosshair', () => {
  const m = createTestModel({
    bifurcation_branches: [
      {
        id: 'lp2_1',
        type: 'LP2',
        param1: 'I',
        param2: 'a',
        points: [{ p: 0.1, p2: 0.5, u: [0, 0] }, { p: 0.6, p2: 0.8, u: [0.2, 0.1] }],
      },
      {
        id: 'hb2_1',
        type: 'HB2',
        param1: 'I',
        param2: 'a',
        points: [{ p: 0.3, p2: 0.6, u: [0.1, 0] }, { p: 0.9, p2: 0.9, u: [0.3, 0.2] }],
      },
    ],
    bifurcation_special_points: [
      { id: 'CP1', label: 'CP1', ty: 'CP', p: 0.4, p2: 0.7, u: [0, 0], branch_id: 'lp2_1', param1: 'I', param2: 'a' },
      { id: 'BT1', label: 'BT1', ty: 'BT', p: 0.5, p2: 0.75, u: [0, 0], branch_id: 'hb2_1', param1: 'I', param2: 'a' },
      { id: 'GH1', label: 'GH1', ty: 'GH', p: 0.7, p2: 0.82, u: [0, 0], branch_id: 'hb2_1', param1: 'I', param2: 'a' },
    ],
  });
  const ctx = createRecordingCtx();
  drawBifurcation2D(m, ctx, 500, 320);
  const strokes = ctx.calls.filter((c) => c.op === 'stroke');
  assert.ok(strokes.some((s) => s.strokeStyle === '#245c9b'), 'LP2 blue curve');
  assert.ok(strokes.some((s) => s.strokeStyle === '#c62828'), 'HB2 red curve');
  const texts = ctx.calls.filter((c) => c.op === 'fillText').map((c) => c.text);
  for (const label of ['CP1', 'BT1', 'GH1']) {
    assert.ok(texts.includes(label), `expected codim-2 label ${label}`);
  }
});

test('mountBifurcation renders toolbar, presets, stale/non-smooth/N>2 banners, chips, and disposes cleanly', async () => {
  const m = createTestModel({
    model_name: 'custom',
    model_spec: {
      states: [{ name: 'x' }, { name: 'y' }, { name: 'z' }],
      params: [{ name: 'I', value: 0.5, min: -1, max: 2 }, { name: 'a', value: 0.7, min: 0, max: 1 }],
      equations: ['abs(x) - y', 'x - y', '-z'],
    },
    state_names: ['x', 'y', 'z'],
    params: { I: 0.5, a: 0.7 },
    param_info: { I: [-1, 2, 0.5, 'I'], a: [0, 1, 0.7, 'a'] },
    display: [0, 1],
    clamped: [0, 0, 0],
    bifurcation_presets: [{ label: 'Sweep I', param1: 'I', range: [-1, 2] }],
    bifurcation_branches: [
      {
        id: 'ep_1',
        type: 'EP',
        param1: 'I',
        points: [{ p: 0.2, u: [0.1, 0.2, 0.3], stable: true }, { p: 0.8, u: [0.4, 0.5, 0.6], stable: false }],
      },
      {
        id: 'lc_1',
        type: 'LC',
        param1: 'I',
        points: [{ p: 0.6, u: [0.3, 0.4, 0.5], u_max: [0.3, 0.4, 0.5], u_min: [-0.3, -0.4, -0.5], period: 4.0, stable: true }],
      },
    ],
    bifurcation_special_points: [
      { id: 'HB1', label: 'HB1', ty: 'HB', p: 0.5, u: [0.2, 0.3, 0.4], branch_id: 'ep_1', param1: 'I' },
      { id: 'LP1', label: 'LP1', ty: 'LP', p: 0.2, u: [0.1, 0.2, 0.3], branch_id: 'ep_1', param1: 'I' },
      { id: 'LPC1', label: 'LPC1', ty: 'LPC', p: 0.7, u: [0.3, 0.4, 0.5], branch_id: 'lc_1', param1: 'I', can_trace_2d: true },
      { id: 'UZ1', label: 'UZ1', ty: 'UZ', p: 0.85, u: [0.3, 0.4, 0.5], period: 30, branch_id: 'lc_1', param1: 'I', can_trace_homoclinic: true },
    ],
    bifurcation_metadata: { stale: true, status: 'complete', fort9: 'AUTO fort.9 summary' },
  });

  const calls = [];
  const baseController = acquireExperiment(m);
  const controller = {
    ...baseController,
    startEquilibriumContinuation: (...a) => { calls.push(['startEP', ...a]); },
    switchToLimitCycle: (...a) => { calls.push(['switchLC', ...a]); },
    startTwoParamContinuation: (...a) => { calls.push(['start2D', ...a]); },
    runBifurcationPreset: (...a) => { calls.push(['preset', ...a]); },
    jumpToBifurcationPoint: (...a) => { calls.push(['jump', ...a]); },
    cancelBifurcation: (...a) => { calls.push(['cancel', ...a]); baseController.cancelBifurcation(...a); },
    clearBifurcation: (...a) => { calls.push(['clear', ...a]); baseController.clearBifurcation(); },
  };
  cleanups.push(() => baseController.release());

  const host = document.createElement('div');
  document.body.append(host);
  const panel = mountBifurcation({ model: m, controller, el: host });

  // Stale banner + Re-run
  const stale = host.querySelector('.ppw-bif-stale');
  assert.equal(stale.hidden, false);
  assert.match(stale.textContent, /Other parameters changed/);
  host.querySelector('.ppw-bif-rerun').click();
  assert.equal(calls[0][0], 'startEP');

  // Non-smooth advisory & N > 2 note
  const nonSmooth = host.querySelector('.ppw-bif-nonsmooth');
  assert.equal(nonSmooth.hidden, false);
  assert.match(nonSmooth.textContent, /abs/i);
  const ndNote = host.querySelector('.ppw-bif-nd-note');
  assert.equal(ndNote.hidden, false);
  assert.match(ndNote.textContent, /clamped variables/i);

  // Presets
  const presetBtn = host.querySelector('.ppw-bif-preset-btn');
  assert.ok(presetBtn);
  presetBtn.click();
  assert.deepEqual(calls.at(-1).slice(0, 2), ['preset', 0]);

  // Special-point chips
  const hbChip = host.querySelector('[data-sp-id="HB1"]');
  hbChip.querySelector('.ppw-bif-chip-jump').click();
  assert.equal(calls.at(-1)[0], 'jump');
  hbChip.querySelector('.ppw-bif-switch-lc').click();
  assert.deepEqual(calls.at(-1).slice(0, 2), ['switchLC', 'HB1']);
  hbChip.querySelector('.ppw-bif-trace-2d').click();
  assert.deepEqual(calls.at(-1).slice(0, 3), ['start2D', 'HB1', 'a']);

  const lpChip = host.querySelector('[data-sp-id="LP1"]');
  lpChip.querySelector('.ppw-bif-trace-2d').click();
  assert.deepEqual(calls.at(-1).slice(0, 3), ['start2D', 'LP1', 'a']);

  const lpcBtn = host.querySelector('[data-sp-id="LPC1"] .ppw-bif-trace-2d');
  assert.ok(lpcBtn);
  lpcBtn.click();
  assert.deepEqual(calls.at(-1).slice(0, 3), ['start2D', 'LPC1', 'a']);

  const homBtn = host.querySelector('[data-sp-id="UZ1"] .ppw-bif-trace-hom');
  assert.ok(homBtn);
  homBtn.click();
  assert.equal(calls.at(-1)[0], 'start2D');
  assert.equal(calls.at(-1)[1], 'UZ1');
  assert.equal(calls.at(-1)[3].homoclinic, true);

  // 1D/2D switch visibility
  assert.equal(host.querySelector('.ppw-bif-view-switch').hidden, true);
  m.set('bifurcation_branches', [
    ...m.get('bifurcation_branches'),
    { id: 'lp2_1', type: 'LP2', param1: 'I', param2: 'a', points: [{ p: 0.2, p2: 0.5, u: [0, 0, 0] }] },
  ]);
  panel.update();
  assert.equal(host.querySelector('.ppw-bif-view-switch').hidden, false);

  // Cancel button enabled when running
  assert.equal(host.querySelector('.ppw-bif-cancel').disabled, true);
  m.set('bifurcation_running', true);
  panel.update();
  assert.equal(host.querySelector('.ppw-bif-cancel').disabled, false);

  // Canvas hover & click
  const canvas = host.querySelector('.ppw-bif-canvas');
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 560, height: 347 });
  canvas.dispatchEvent(new window.MouseEvent('mousemove', { clientX: 250, clientY: 150 }));
  assert.ok(host.querySelector('.ppw-bif-status').textContent.length > 0);
  canvas.dispatchEvent(new window.MouseEvent('click', { clientX: 250, clientY: 150 }));
  assert.equal(calls.at(-1)[0], 'jump');

  // Diagnostics
  assert.match(host.querySelector('.ppw-bif-diagnostics').textContent, /fort\.9/);

  // Dispose cleans up DOM and cancels owner's bifurcation job
  panel.dispose();
  assert.equal(host.querySelector('.ppw-bifurcation'), null);
  assert.equal(calls.at(-1)[0], 'cancel');
});

test('1D/2D axis bounds, 2D plane filtering, v2 curve/ZH styling, and bifurcation presentation controls', async () => {
  const m = createTestModel({
    presentation: 'bifurcation',
    bifurcation_branches: [
      { id: 'ep_1', type: 'EP', param1: 'I', points: [{ p: -1, u: [-1, 0], stable: true }, { p: 2, u: [2, 0], stable: false }] },
      { id: 'lp2_old', type: 'LP2', param1: 'I', param2: 'a', points: [{ p: 0.1, p2: 0.2, u: [0, 0] }, { p: 0.9, p2: 0.9, u: [0.5, 0.2] }] },
      { id: 'hb2_new', type: 'HB2', param1: 'I', param2: 'b', points: [{ p: -0.2, p2: 0.4, u: [0, 0] }, { p: 0.4, p2: 1.2, u: [0.1, 0.1] }] },
      { id: 'lpc2_new', type: 'LPC2', param1: 'I', param2: 'b', points: [{ p: -0.1, p2: 0.5, u: [0, 0] }, { p: 0.3, p2: 1.0, u: [0.1, 0.1] }] },
      { id: 'pd2_new', type: 'PD2', param1: 'I', param2: 'b', points: [{ p: 0.0, p2: 0.6, u: [0, 0] }, { p: 0.2, p2: 0.9, u: [0.1, 0.1] }] },
      { id: 'hom2_new', type: 'HOM2', param1: 'I', param2: 'b', points: [{ p: -0.15, p2: 0.45, u: [0, 0] }, { p: 0.35, p2: 1.1, u: [0.1, 0.1] }] },
    ],
    bifurcation_special_points: [
      { id: 'CP1', label: 'CP1', ty: 'CP', p: 0.5, p2: 0.6, u: [0, 0], branch_id: 'lp2_old', param1: 'I', param2: 'a' },
      { id: 'ZH1', label: 'ZH1', ty: 'ZH', p: 0.1, p2: 0.7, u: [0, 0], branch_id: 'hb2_new', param1: 'I', param2: 'b' },
    ],
  });

  // Default 2D plane is the most recently added 2D branch's (I, b), filtering out CP1 on (I, a)
  const ctx2d = createRecordingCtx();
  drawBifurcation2D(m, ctx2d, 520, 340);
  const texts2d = ctx2d.calls.filter((c) => c.op === 'fillText').map((c) => c.text);
  assert.ok(texts2d.includes('b'), 'y-axis should label active param2 b');
  assert.ok(texts2d.includes('ZH1'), 'ZH1 on active plane should be drawn');
  assert.ok(!texts2d.includes('CP1'), 'CP1 on inactive plane should be filtered out');
  const strokes2d = ctx2d.calls.filter((c) => c.op === 'stroke');
  assert.ok(strokes2d.some((s) => s.strokeStyle === '#2e7d32' && s.dash[0] === 6), 'LPC2 green dashed');
  assert.ok(strokes2d.some((s) => s.strokeStyle === '#e65100'), 'PD2 orange');
  assert.ok(strokes2d.some((s) => s.strokeStyle === '#6a1b9a'), 'HOM2 purple');

  // Mount full bifurcation presentation and verify all parameters + model + editor + axes controls
  const host = document.createElement('div');
  document.body.append(host);
  const dispose = await render({ model: m, el: host });
  cleanups.push(dispose);

  assert.equal(host.querySelectorAll('.ppw-stage-footer .ppw-param').length, 4, 'all 4 parameters shown, not primary-only');
  assert.ok(host.querySelector('.ppw-model-select'), 'model switcher mounted directly');
  assert.ok(host.querySelector('.ppw-editor-toggle'), 'equation editor toggle mounted directly');
  assert.match(host.querySelector('.ppw-bif-support-note').textContent, /1D EP.*LPC2/i);

  // 1D axis override + validation + Auto reset
  const xmin = host.querySelector('.ppw-bif-xmin');
  const xmax = host.querySelector('.ppw-bif-xmax');
  xmin.value = '0.5'; xmax.value = '0.2';
  xmax.dispatchEvent(new window.Event('change'));
  assert.equal(host.querySelector('.ppw-bif-axes-error').hidden, false);
  assert.equal(m.get('bifurcation_config').xlim1d, undefined);

  xmin.value = '-0.2'; xmax.value = '0.8';
  xmax.dispatchEvent(new window.Event('change'));
  assert.equal(host.querySelector('.ppw-bif-axes-error').hidden, true);
  assert.deepEqual(m.get('bifurcation_config').xlim1d, [-0.2, 0.8]);

  host.querySelector('.ppw-bif-axes-auto').click();
  assert.equal(m.get('bifurcation_config').xlim1d, undefined);

  // Switch to 2D mode, switch plane selector to (I, a), and set 2D y bounds
  host.querySelector('.ppw-bif-mode-2d').click();
  assert.equal(m.get('bifurcation_config').viewMode, '2d');
  const planeSelect = host.querySelector('.ppw-bif-plane2d');
  assert.equal(planeSelect.options.length, 2);
  planeSelect.value = JSON.stringify(['I', 'a']);
  planeSelect.dispatchEvent(new window.Event('change'));
  assert.equal(m.get('bifurcation_config').plane2d, JSON.stringify(['I', 'a']));

  const ymin = host.querySelector('.ppw-bif-ymin');
  const ymax = host.querySelector('.ppw-bif-ymax');
  ymin.value = '0.1'; ymax.value = '0.6';
  ymax.dispatchEvent(new window.Event('change'));
  assert.deepEqual(m.get('bifurcation_config').ylim2d, [0.1, 0.6]);
});
