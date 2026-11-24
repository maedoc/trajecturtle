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
  solveLinearSystem,
  findSteadyStateND,
  detectNonSmoothExpressions,
} = await import('../../src/tvb_phaseplane/static/widget.js');

const pause = (ms = 40) => new Promise((resolve) => setTimeout(resolve, ms));

const cleanups = [];
afterEach(() => {
  for (const fn of cleanups.splice(0)) fn();
  document.body.replaceChildren();
});

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
    bifurcation_config: {},
    bifurcation_running: false,
    bifurcation_metadata: {},
    bifurcation_presets: [],
    auto_wasm_url: 'http://localhost/auto2000_core.wasm',
    auto_worker_url: 'http://localhost/auto2000-worker.js',
    ...overrides,
  };
  const callbacks = new Map();
  const model = {
    _isMock: true,
    get: (k) => state[k],
    set(k, v) {
      if (Object.is(state[k], v)) return;
      state[k] = v;
      for (const cb of [...(callbacks.get('change:' + k) || [])]) cb(model, v);
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
  return model;
}

function installMockWorker(handler) {
  const OrigWorker = globalThis.Worker;
  let constructed = 0;
  let terminated = 0;
  const calls = [];

  class MockWorker {
    constructor(url) {
      constructed++;
      this.url = url;
      this._listeners = new Map();
      this._terminated = false;
    }
    addEventListener(type, fn) {
      const list = this._listeners.get(type) || [];
      list.push(fn);
      this._listeners.set(type, list);
    }
    removeEventListener(type, fn) {
      this._listeners.set(type, (this._listeners.get(type) || []).filter((f) => f !== fn));
    }
    _emit(type, data) {
      if (this._terminated) return;
      for (const fn of [...(this._listeners.get(type) || [])]) {
        fn({ data });
      }
    }
    postMessage(msg) {
      if (this._terminated) return;
      if (msg.type === 'init') {
        queueMicrotask(() => this._emit('message', { type: 'ready' }));
        return;
      }
      if (msg.type === 'run') {
        calls.push(msg);
        Promise.resolve(handler(msg, this)).then((res) => {
          if (!this._terminated && res) {
            this._emit('message', { type: 'result', jobId: msg.jobId, elapsedMs: 1, ...res });
          }
        }).catch((err) => {
          if (!this._terminated) {
            this._emit('message', { type: 'error', jobId: msg.jobId, message: err?.message || String(err) });
          }
        });
      }
    }
    terminate() {
      this._terminated = true;
      terminated++;
    }
  }

  globalThis.Worker = MockWorker;
  cleanups.push(() => {
    globalThis.Worker = OrigWorker;
  });

  return {
    get constructed() { return constructed; },
    get terminated() { return terminated; },
    calls,
  };
}

test('solveLinearSystem and findSteadyStateND work on 2D and 6D models', () => {
  const x = solveLinearSystem([[2, 1], [1, 3]], [1, 2]);
  assert.ok(x);
  assert.ok(Math.abs(x[0] - 0.2) < 1e-12);
  assert.ok(Math.abs(x[1] - 0.6) < 1e-12);
  assert.equal(solveLinearSystem([[1, 2], [2, 4]], [3, 6]), null);

  // 2D FitzHugh-Nagumo-like system: v' = v - v^3/3 - w + I, w' = eps*(v + a - b*w)
  const def2D = {
    dim: 2,
    defaultLims: [[-3, 3], [-2, 2]],
    f: (_t, u, p) => [
      u[0] - (u[0] ** 3) / 3 - u[1] + p.I,
      p.epsilon * (u[0] + p.a - p.b * u[1]),
    ],
  };
  const res2D = findSteadyStateND(def2D, { a: 0.7, b: 0.8, epsilon: 0.08, I: 0.5 }, [0.5, 0.5]);
  assert.equal(res2D.converged, true);
  assert.ok(res2D.residual < 1e-8);
  const f2D = def2D.f(0, res2D.u, { a: 0.7, b: 0.8, epsilon: 0.08, I: 0.5 });
  assert.ok(Math.hypot(...f2D) < 1e-8);

  // 6D nonlinear coupled oscillator/damped system with unique steady state
  const def6D = {
    dim: 6,
    defaultLims: Array.from({ length: 6 }, () => [-2, 2]),
    f: (_t, u, p) => [
      -u[0] + 0.1 * u[1] + p.drive,
      -2 * u[1] + 0.2 * Math.tanh(u[0]) + 0.1 * u[2],
      -1.5 * u[2] + 0.1 * u[3] - 0.05 * u[2] ** 3,
      -3 * u[3] + 0.2 * u[4] + 0.1 * u[0],
      -2.5 * u[4] + 0.15 * u[5],
      -1.2 * u[5] + 0.1 * u[4] + 0.25,
    ],
  };
  const res6D = findSteadyStateND(def6D, { drive: 0.4 }, [1.5, -1.2, 0.8, -0.5, 1.1, -0.9]);
  assert.equal(res6D.converged, true);
  assert.equal(res6D.u.length, 6);
  assert.ok(res6D.residual < 1e-8);
});

test('detectNonSmoothExpressions identifies non-smooth operators case-insensitively', () => {
  assert.deepEqual(detectNonSmoothExpressions(null), []);
  assert.deepEqual(detectNonSmoothExpressions({
    equations: { x: 'mu * x - y', y: 'x + mu * y' },
  }), []);

  const tokens = detectNonSmoothExpressions({
    equations: {
      x: 'step(v - 1) + Heaviside(w) - Abs(v)',
      y: 'Min(x, 1) + Max(y, 0) + sign(x)',
    },
    custom_functions: {
      g: 'STEP(x)',
    },
  });
  const lower = tokens.map((t) => t.toLowerCase());
  for (const expected of ['step(', 'heaviside(', 'abs(', 'min(', 'max(', 'sign(']) {
    assert.ok(lower.includes(expected), `Expected ${expected} in ${JSON.stringify(tokens)}`);
  }
});

test('lazy worker creation, bidirectional EP stitching, LC switch, 2D continuation, and stale tracking', async () => {
  const mock = installMockWorker((msg) => {
    const { continuation } = msg;
    if (continuation.ips === 1 && continuation.isw === 1) {
      if (continuation.ds > 0) {
        return {
          branch: [
            { p: 0.5, u: [0.1, 0.2], stable: true, ty: null, lab: 0 },
            { p: 0.8, u: [0.3, 0.4], stable: false, ty: null, lab: 0 },
          ],
          specialPoints: [
            { lab: 2, ty: 'HB', p: 0.7, u: [0.25, 0.35], can_switch_lc: true, can_trace_2d: true, restartSp: { itp: 3, lab: 2, par: [0.7], u1Based: [0.25, 0.35] } },
          ],
          restartHandles: { 2: { itp: 3, lab: 2, par: [0.7], u1Based: [0.25, 0.35] } },
        };
      }
      return {
        branch: [
          { p: 0.5, u: [0.1, 0.2], stable: true, ty: null, lab: 0 },
          { p: 0.1, u: [-0.2, -0.1], stable: true, ty: null, lab: 0 },
        ],
        specialPoints: [
          { lab: 3, ty: 'LP', p: 0.2, u: [-0.1, 0.0], can_switch_lc: false, can_trace_2d: true, restartSp: { itp: 2, lab: 3, par: [0.2], u1Based: [-0.1, 0.0] } },
        ],
        restartHandles: { 3: { itp: 2, lab: 3, par: [0.2], u1Based: [-0.1, 0.0] } },
      };
    }
    if (continuation.ips === 2) {
      return {
        branch: [
          { p: 0.7, u: [0.25, 0.35], u_max: [0.25, 0.35], u_min: [0.25, 0.35], period: 6.28, stable: true, ty: null, lab: 0 },
          { p: 0.9, u: [0.8, 0.7], u_max: [0.8, 0.7], u_min: [-0.4, -0.3], period: 7.1, stable: true, ty: null, lab: 0 },
        ],
        specialPoints: [
          { lab: 4, ty: 'LPC', p: 0.95, u: [0.9, 0.8], period: 7.5, can_switch_lc: false, can_trace_2d: false, restartSp: { itp: 5, lab: 4, par: [0.95], u1Based: [0.9, 0.8] } },
        ],
        restartHandles: { 4: { itp: 5, lab: 4, par: [0.95], u1Based: [0.9, 0.8] } },
      };
    }
    if (continuation.isw === 2) {
      const p2Base = continuation.ds > 0 ? 0.9 : 0.7;
      return {
        branch: [
          { p: 0.2, p2: 0.8, u: [-0.1, 0.0], stable: true, ty: null, lab: 0 },
          { p: 0.3, p2: p2Base, u: [0.0, 0.1], stable: true, ty: null, lab: 0 },
        ],
        specialPoints: continuation.ds > 0
          ? [{ lab: 5, ty: 'CP', p: 0.3, p2: 0.9, u: [0.0, 0.1], can_switch_lc: false, can_trace_2d: false, restartSp: { itp: 1, lab: 5, par: [0.3, 0.9], u1Based: [0.0, 0.1] } }]
          : [],
        restartHandles: {},
      };
    }
    return { branch: [], specialPoints: [], restartHandles: {} };
  });

  const m = createTestModel();
  const ctrl = acquireExperiment(m);
  cleanups.push(() => ctrl.release());

  // 0 Workers created before any continuation is started
  assert.equal(mock.constructed, 0);

  const epRes = await ctrl.startEquilibriumContinuation({ param1: 'I', range: [-1, 2] });
  assert.equal(mock.constructed, 1);
  assert.ok(epRes);
  assert.equal(epRes.branch.id, 'ep_1');
  assert.equal(epRes.branch.type, 'EP');
  // Backward reversed [0.1, 0.5] + forward [0.5, 0.8] with seed deduplicated => 3 points
  assert.deepEqual(epRes.branch.points.map((pt) => pt.p), [0.1, 0.5, 0.8]);
  assert.deepEqual(epRes.specialPoints.map((sp) => sp.id), ['LP1', 'HB1']);
  assert.ok(ctrl.restartCache.has('HB1'));
  assert.ok(ctrl.restartCache.has('LP1'));
  assert.equal(m.get('bifurcation_running'), false);
  assert.equal(m.get('bifurcation_metadata').stale, false);

  // Dragging continued parameter p1 ('I') preserves branches and keeps stale === false
  ctrl.commit({ params: { ...m.get('params'), I: 0.9 } });
  assert.equal(m.get('bifurcation_branches').length, 1);
  assert.equal(m.get('bifurcation_metadata').stale, false);

  // Dragging uncontinued parameter ('a') sets stale === true without erasing branches
  ctrl.commit({ params: { ...m.get('params'), a: 0.75 } });
  assert.equal(m.get('bifurcation_branches').length, 1);
  assert.equal(m.get('bifurcation_metadata').stale, true);

  // Dragging uncontinued parameter back restores stale === false
  ctrl.commit({ params: { ...m.get('params'), a: 0.7 } });
  assert.equal(m.get('bifurcation_metadata').stale, false);

  // Switch to limit cycle from HB1
  const lcRes = await ctrl.switchToLimitCycle('HB1');
  assert.ok(lcRes);
  assert.equal(lcRes.branch.id, 'lc_1');
  assert.equal(lcRes.branch.type, 'LC');
  assert.equal(lcRes.branch.points[1].period, 7.1);
  assert.equal(m.get('bifurcation_branches').length, 2);
  assert.ok(m.get('bifurcation_special_points').some((sp) => sp.id === 'LPC1'));

  // Two-parameter continuation from LP1 with rl2 clipping
  ctrl.commit({ bifurcation_config: { rl2_0: 0.75, rl2_1: 0.95 } });
  const lp2Res = await ctrl.startTwoParamContinuation('LP1', 'b');
  assert.ok(lp2Res);
  assert.equal(lp2Res.branch.id, 'lp2_1');
  assert.equal(lp2Res.branch.type, 'LP2');
  assert.equal(lp2Res.branch.param2, 'b');
  assert.deepEqual(lp2Res.branch.points.map((pt) => pt.p2), [0.8, 0.9]);
  assert.ok(m.get('bifurcation_special_points').some((sp) => sp.id === 'CP1'));

  // Engine v2: LPC -> LPC2 via collocation mesh restart
  const lpcRestart = ctrl.restartCache.get('LPC1');
  lpcRestart.restartSp.ups = [[0, 0.9, 0.8]];
  const lpc2Res = await ctrl.startTwoParamContinuation('LPC1', 'b', { rl2_0: null, rl2_1: null });
  assert.ok(lpc2Res);
  assert.equal(lpc2Res.branch.type, 'LPC2');

  // Changing model clears branches and restartCache
  ctrl.selectModel('wilson_cowan');
  assert.deepEqual(m.get('bifurcation_branches'), []);
  assert.deepEqual(m.get('bifurcation_special_points'), []);
  assert.equal(ctrl.restartCache.size, 0);
});

test('non-convergent steady state reports error cleanly without spawning Worker', async () => {
  const mock = installMockWorker(() => ({ branch: [], specialPoints: [], restartHandles: {} }));
  const noFixedPointSpec = {
    state_vars: { x: { range: [-1, 1], default: 0 }, y: { range: [-1, 1], default: 0 } },
    equations: { x: '1 + x^2', y: '1 + y^2' },
    parameters: { mu: { range: [-1, 1], default: 0 } },
  };
  const m = createTestModel({
    model_name: 'custom',
    model_spec: noFixedPointSpec,
    params: { mu: 0 },
    state_names: ['x', 'y'],
  });
  const ctrl = acquireExperiment(m);
  cleanups.push(() => ctrl.release());

  const res = await ctrl.startEquilibriumContinuation({ param1: 'mu' });
  assert.equal(res, null);
  assert.equal(mock.constructed, 0);
  assert.equal(m.get('bifurcation_running'), false);
  assert.equal(m.get('bifurcation_metadata').status, 'error');
  assert.match(m.get('bifurcation_metadata').error, /steady state/i);
});

test('owner-scoped cancelBifurcation cancels only its own job and model change cancels active job', async () => {
  let resolveSlowRun;
  installMockWorker(() => new Promise((resolve) => {
    resolveSlowRun = resolve;
  }));

  const m = createTestModel();
  const ctrl = acquireExperiment(m);
  cleanups.push(() => ctrl.release());

  const ownerA = { name: 'panelA' };
  const ownerB = { name: 'panelB' };

  const jobPromise = ctrl.startEquilibriumContinuation({ param1: 'I' }, ownerA);
  await pause(15);
  assert.equal(m.get('bifurcation_running'), true);

  // Disposing/cancelling a different owner does NOT cancel ownerA's job
  ctrl.cancelBifurcation(ownerB);
  assert.equal(m.get('bifurcation_running'), true);

  // Cancelling ownerA cancels its job
  ctrl.cancelBifurcation(ownerA);
  assert.equal(m.get('bifurcation_running'), false);
  assert.equal(m.get('bifurcation_metadata').status, 'cancelled');
  if (resolveSlowRun) resolveSlowRun({ branch: [], specialPoints: [], restartHandles: {} });
  const resA = await jobPromise;
  assert.equal(resA, null);

  // Active job is cancelled when model changes
  const jobPromise2 = ctrl.startEquilibriumContinuation({ param1: 'I' }, ownerA);
  await pause(15);
  assert.equal(m.get('bifurcation_running'), true);
  ctrl.selectModel('wilson_cowan');
  assert.equal(m.get('bifurcation_running'), false);
  if (resolveSlowRun) resolveSlowRun({ branch: [], specialPoints: [], restartHandles: {} });
  const resB = await jobPromise2;
  assert.equal(resB, null);
});

test('jumpToBifurcationPoint updates params and clamped for N > 2', () => {
  const m = createTestModel({
    model_name: 'hindmarsh_rose',
    params: { a: 1, b: 3, c: 1, d: 5, r: 0.006, s: 4, x1: -1.6, I: 2.0 },
    state_names: ['x', 'y', 'z'],
    display: [0, 2],
    clamped: [0, 1.5, 0],
    x0: 0.5,
    y0: 0.5,
  });
  const ctrl = acquireExperiment(m);
  cleanups.push(() => ctrl.release());

  ctrl.jumpToBifurcationPoint(
    { p: 3.2, p2: 0.01, u: [5.1, 6.2, 13.3] },
    { id: 'ep_1', type: 'EP', param1: 'I', param2: 'r' },
  );

  assert.equal(m.get('params').I, 3.2);
  assert.equal(m.get('params').r, 0.01);
  // display is [0, 2], so index 1 is non-displayed and must be updated to 6.2
  assert.deepEqual(m.get('clamped'), [0, 6.2, 0]);
  // x0/y0 unchanged unless requested
  assert.equal(m.get('x0'), 0.5);
  assert.equal(m.get('y0'), 0.5);

  // With updateInitialCondition: true
  ctrl.jumpToBifurcationPoint(
    { p: 2.5, u: [7.0, 8.0, 17.0] },
    { id: 'ep_1', type: 'EP', param1: 'I' },
    { updateInitialCondition: true },
  );
  assert.equal(m.get('params').I, 2.5);
  assert.deepEqual(m.get('clamped'), [0, 8.0, 0]);
  assert.equal(m.get('x0'), 7.0);
  assert.equal(m.get('y0'), 17.0);
});
