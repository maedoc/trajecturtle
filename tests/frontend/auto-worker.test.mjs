import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { chromium } from 'playwright';

const STATIC_DIR = path.resolve(import.meta.dirname, '../../src/tvb_phaseplane/static');
const FRONTEND_DIR = path.resolve(import.meta.dirname, '../../src/tvb_phaseplane/frontend');
const WORKER_SOURCES = ['worker-shim', 'vendor', 'models-core', 'vendor-auto2000', 'auto-worker'];

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
};

let server;
let baseUrl;
let browser;
let skipReason = '';

before(async () => {
  const snapChromium = '/snap/bin/chromium';
  const defaultExecutable = typeof chromium.executablePath === 'function' ? chromium.executablePath() : '';
  const executablePath = fs.existsSync(snapChromium)
    ? snapChromium
    : (defaultExecutable && fs.existsSync(defaultExecutable) ? defaultExecutable : '');

  if (!executablePath) {
    skipReason = 'No Chromium executable found for Playwright';
    return;
  }

  server = http.createServer((req, res) => {
    const reqPath = (req.url || '/').split('?')[0];
    if (reqPath === '/test.html') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end('<!doctype html><html><body></body></html>');
      return;
    }
    if (reqPath === '/dev-auto2000-worker.js') {
      const assembled = WORKER_SOURCES
        .map((name) => fs.readFileSync(path.join(FRONTEND_DIR, `${name}.js`), 'utf8').trimEnd())
        .join('\n') + '\n';
      res.writeHead(200, { 'Content-Type': MIME_TYPES['.js'] });
      res.end(assembled);
      return;
    }
    const filePath = path.join(STATIC_DIR, reqPath.replace(/^\/+/, ''));
    if (!filePath.startsWith(STATIC_DIR) || !fs.existsSync(filePath)) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'Content-Type': MIME_TYPES[ext] || 'application/octet-stream' });
    fs.createReadStream(filePath).pipe(res);
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address();
  baseUrl = `http://127.0.0.1:${addr.port}`;

  browser = await chromium.launch({
    headless: true,
    executablePath,
  });
});

after(async () => {
  if (browser) await browser.close();
  if (server) await new Promise((resolve) => server.close(resolve));
});

test('auto-client + auto2000-worker runs SupHopf (EP+LC), MPR (bidirectional EP+2D LP), custom Nerdamer spec, and cancel()', async (t) => {
  if (skipReason) {
    t.skip(skipReason);
    return;
  }
  const page = await browser.newPage();
  try {
    await page.goto(`${baseUrl}/test.html`);

    const results = await page.evaluate(async (origin) => {
      const { createAutoClient } = await import(`${origin}/widget.js`);

      function makeMockModel(overrides = {}) {
        const data = {
          _isMock: true,
          auto_wasm_url: `${origin}/auto2000_core.wasm`,
          auto_worker_url: '',
          ...overrides,
        };
        return {
          _isMock: data._isMock,
          get(k) { return data[k]; },
          on() {},
          off() {},
        };
      }

      // 0. Lazy Worker check & lightweight unavailable error check
      let workerConstructed = 0;
      const OrigWorker = globalThis.Worker;
      globalThis.Worker = class extends OrigWorker {
        constructor(...args) {
          workerConstructed++;
          super(...args);
        }
      };

      const unavailClient = createAutoClient(makeMockModel({ auto_wasm_url: '', auto_worker_url: '' }));
      assertWorkerCount0: if (workerConstructed !== 0) throw new Error(`Expected 0 workers before run, got ${workerConstructed}`);
      let unavailMsg = '';
      try {
        await unavailClient.run({ model: { model_name: 'mpr', params: {} }, continuation: {} });
      } catch (e) {
        unavailMsg = e.message;
      }

      // 1. SupHopf (EP + LC) via custom model_spec
      const supHopfSpec = {
        state_vars: {
          x: { range: [-2, 2], default: 0 },
          y: { range: [-2, 2], default: 0 },
        },
        equations: {
          x: 'mu * x - y - x * (x^2 + y^2)',
          y: 'x + mu * y - y * (x^2 + y^2)',
        },
        parameters: {
          mu: { range: [-1, 1], default: -0.5 },
        },
      };
      const client = createAutoClient(makeMockModel());
      if (workerConstructed !== 0) throw new Error('Expected lazy Worker creation');

      const hopfEp = await client.run({
        model: { model_name: 'custom', model_spec: supHopfSpec, params: { mu: -0.5 } },
        continuation: {
          ips: 1,
          isw: 1,
          activeParams: ['mu'],
          rl0: -1.0,
          rl1: 1.0,
          ds: 0.05,
          dsmin: 1e-4,
          dsmax: 0.1,
          nmx: 200,
          stpnt: { u: [0, 0], p: [-0.5] },
        },
      });
      if (workerConstructed !== 1) throw new Error(`Expected 1 worker after first run, got ${workerConstructed}`);

      const hb = hopfEp.specialPoints.find((sp) => sp.ty === 'HB');
      if (!hb) throw new Error('SupHopf EP did not detect HB');

      const hopfLc = await client.run({
        model: { model_name: 'custom', model_spec: supHopfSpec, params: { mu: hb.p } },
        continuation: {
          ips: 2,
          isw: 1,
          activeParams: ['mu'],
          rl0: -0.5,
          rl1: 1.0,
          ds: 0.05,
          dsmin: 1e-4,
          dsmax: 0.1,
          nmx: 100,
          ntst: 30,
          ncol: 4,
          restartSp: hb.restartSp,
        },
      });
      if (workerConstructed !== 1) throw new Error('Expected worker reuse across runs');

      // 2. MPR (bidirectional EP + 2D LP)
      const r0 = 0.05;
      const v0 = -1.0 / (2 * Math.PI * r0);
      const eta0 = -(v0 * v0 + 15.0 * r0 - (Math.PI * r0) ** 2);
      const mprFwd = await client.run({
        model: { model_name: 'mpr', params: { delta: 1.0, eta_bar: eta0, J: 15.0, I: 0.0 } },
        continuation: {
          ips: 1,
          isw: 1,
          activeParams: ['eta_bar'],
          rl0: -15.0,
          rl1: 0.0,
          ds: 0.05,
          dsmin: 1e-4,
          dsmax: 0.2,
          nmx: 300,
          stpnt: { u: [r0, v0], p: [eta0] },
        },
      });
      const mprBwd = await client.run({
        model: { model_name: 'mpr', params: { delta: 1.0, eta_bar: eta0, J: 15.0, I: 0.0 } },
        continuation: {
          ips: 1,
          isw: 1,
          activeParams: ['eta_bar'],
          rl0: -15.0,
          rl1: 0.0,
          ds: -0.05,
          dsmin: 1e-4,
          dsmax: 0.2,
          nmx: 300,
          stpnt: { u: [r0, v0], p: [eta0] },
        },
      });
      const lp = [...mprFwd.specialPoints, ...mprBwd.specialPoints].find((sp) => sp.ty === 'LP');
      if (!lp) throw new Error('MPR bidirectional EP did not detect LP');

      const mpr2d = await client.run({
        model: { model_name: 'mpr', params: { delta: 1.0, eta_bar: lp.p, J: 15.0, I: 0.0 } },
        continuation: {
          ips: 1,
          isw: 2,
          activeParams: ['eta_bar', 'J'],
          rl0: -20.0,
          rl1: 5.0,
          ds: 0.05,
          dsmin: 1e-4,
          dsmax: 0.2,
          nmx: 100,
          restartSp: lp.restartSp,
        },
      });

      // 3. Custom Nerdamer model_spec (saddle-node normal form: dx = r + x^2, dy = -y)
      const snSpec = {
        state_vars: {
          x: { range: [-3, 3], default: -1 },
          y: { range: [-3, 3], default: 0 },
        },
        equations: {
          x: 'r + x^2',
          y: '-y',
        },
        parameters: {
          r: { range: [-2, 2], default: -1 },
        },
      };
      const snRes = await client.run({
        model: { model_name: 'custom', model_spec: snSpec, params: { r: -1.0 } },
        continuation: {
          ips: 1,
          isw: 1,
          activeParams: ['r'],
          rl0: -2.0,
          rl1: 1.0,
          ds: 0.05,
          dsmin: 1e-4,
          dsmax: 0.1,
          nmx: 150,
          stpnt: { u: [-1.0, 0.0], p: [-1.0] },
        },
      });

      // 4. cancel() + clean recovery
      const pendingPromise = client.run({
        model: { model_name: 'mpr', params: { delta: 1.0, eta_bar: eta0, J: 15.0, I: 0.0 } },
        continuation: {
          ips: 1,
          isw: 1,
          activeParams: ['eta_bar'],
          rl0: -15.0,
          rl1: 0.0,
          ds: 0.05,
          nmx: 200,
          stpnt: { u: [r0, v0], p: [eta0] },
        },
      });
      client.cancel();
      let cancelMsg = '';
      try {
        await pendingPromise;
      } catch (e) {
        cancelMsg = e.message;
      }

      const recovered = await client.run({
        model: { model_name: 'custom', model_spec: snSpec, params: { r: -1.0 } },
        continuation: {
          ips: 1,
          isw: 1,
          activeParams: ['r'],
          rl0: -2.0,
          rl1: 1.0,
          ds: 0.05,
          nmx: 100,
          stpnt: { u: [-1.0, 0.0], p: [-1.0] },
        },
      });

      return {
        unavailMsg,
        workerConstructed,
        hb,
        hopfLcCount: hopfLc.branch.length,
        hopfLcLast: hopfLc.branch[hopfLc.branch.length - 1],
        lp,
        mpr2dCount: mpr2d.branch.length,
        mpr2dFirst: mpr2d.branch[0],
        snLp: snRes.specialPoints.find((sp) => sp.ty === 'LP'),
        cancelMsg,
        recoveredCount: recovered.branch.length,
      };
    }, baseUrl);

    assert.match(results.unavailMsg, /AUTO2000 WASM assets are unavailable/);
    assert.equal(results.workerConstructed, 2, '1st worker reused, 2nd spawned after cancel()');

    // SupHopf assertions
    assert.ok(Math.abs(results.hb.p) < 1e-4, `Expected HB at mu=0, got ${results.hb.p}`);
    assert.equal(results.hb.can_switch_lc, true);
    assert.equal(results.hb.can_trace_2d, true);
    assert.ok(results.hopfLcCount > 5, 'Expected LC branch points');
    assert.ok(results.hopfLcLast.u_max[0] > 0.1, 'Expected u_max > 0.1 on LC');
    assert.ok(results.hopfLcLast.u_min[0] < -0.1, 'Expected u_min < -0.1 on LC');
    assert.ok(Math.abs(results.hopfLcLast.period - 2 * Math.PI) < 0.2, `Expected period ~2pi, got ${results.hopfLcLast.period}`);

    // MPR assertions
    assert.equal(results.lp.can_switch_lc, false);
    assert.equal(results.lp.can_trace_2d, true);
    assert.ok(results.mpr2dCount > 5, 'Expected 2D LP branch points');
    assert.equal(typeof results.mpr2dFirst.p2, 'number', 'Expected p2 on 2D LP branch');

    // Custom Nerdamer saddle-node assertions
    assert.ok(results.snLp, 'Expected LP on saddle-node normal form');
    assert.ok(Math.abs(results.snLp.p) < 1e-4, `Expected LP at r=0, got ${results.snLp.p}`);

    // Cancel & recovery assertions
    assert.match(results.cancelMsg, /cancelled/);
    assert.ok(results.recoveredCount > 5, 'Expected clean recovery after cancel()');
  } finally {
    await page.close();
  }
});

test('auto-worker v2: LPC2, PD2, HOM2, and codim-2 labels (CP, BT, GH, ZH) in Web Worker', async (t) => {
  if (skipReason) {
    t.skip(skipReason);
    return;
  }
  const page = await browser.newPage();
  try {
    await page.goto(`${baseUrl}/test.html`);

    const results = await page.evaluate(async (origin) => {
      const worker = new Worker(`${origin}/dev-auto2000-worker.js`);
      let nextJobId = 1;
      const pending = new Map();
      let initResolve, initReject;
      const readyPromise = new Promise((res, rej) => { initResolve = res; initReject = rej; });

      worker.onmessage = (e) => {
        const msg = e.data || {};
        if (msg.type === 'ready') { initResolve(); return; }
        if (msg.type === 'error' && msg.jobId === undefined) { initReject(new Error(msg.message)); return; }
        const p = pending.get(msg.jobId);
        if (!p) return;
        pending.delete(msg.jobId);
        if (msg.type === 'error') p.reject(new Error(msg.message));
        else p.resolve(msg);
      };

      worker.postMessage({ type: 'init', wasmUrl: `${origin}/auto2000_core.wasm` });
      await readyPromise;

      const runJob = (payload) => new Promise((resolve, reject) => {
        const jobId = nextJobId++;
        pending.set(jobId, { resolve, reject });
        worker.postMessage({ type: 'run', jobId, ...payload });
      });

      // 1. Bautin HB2 (GH) + LC -> LPC (can_trace_2d) -> LPC2
      const bautinSpec = {
        state_vars: { x: { range: [-2, 2], default: 0 }, y: { range: [-2, 2], default: 0 } },
        equations: {
          x: 'a * x - y + c * x * (x^2 + y^2) - x * (x^2 + y^2)^2',
          y: 'a * y + x + c * y * (x^2 + y^2) - y * (x^2 + y^2)^2',
        },
        parameters: { a: { range: [-2, 1], default: -0.5 }, c: { range: [-2, 2], default: 1.0 } },
      };
      const bautinEp = await runJob({
        model: { model_name: 'custom', model_spec: bautinSpec, params: { a: -0.5, c: 1.0 } },
        continuation: { ips: 1, isw: 1, activeParams: ['a'], rl0: -1.0, rl1: 0.5, ds: 0.037, dsmin: 1e-4, dsmax: 0.05, stpnt: { u: [0, 0], p: [-0.5] } },
      });
      const bautinHb = bautinEp.specialPoints.find((sp) => sp.ty === 'HB');
      const bautinHb2 = await runJob({
        model: { model_name: 'custom', model_spec: bautinSpec, params: { a: bautinHb.p, c: 1.0 } },
        continuation: { ips: 1, isw: 2, ilp: 0, activeParams: ['a', 'c'], rl0: -1.0, rl1: 1.0, ds: 0.05, nmx: 80, restartSp: bautinHb.restartSp },
      });
      const bautinLc = await runJob({
        model: { model_name: 'custom', model_spec: bautinSpec, params: { a: bautinHb.p, c: 1.0 } },
        continuation: { ips: 2, isw: 1, ilp: 1, isp: 2, activeParams: ['a'], rl0: -0.5, rl1: 0.5, ds: 0.03, dsmin: 1e-4, dsmax: 0.08, nmx: 120, ntst: 40, ncol: 4, restartSp: bautinHb.restartSp },
      });
      const lpcSp = bautinLc.specialPoints.find((sp) => sp.ty === 'LPC');
      const lpc2 = await runJob({
        model: { model_name: 'custom', model_spec: bautinSpec, params: { a: lpcSp.p, c: 1.0 } },
        continuation: { ips: 2, isw: 2, ilp: 0, isp: 0, activeParams: ['a', 'c'], rl0: -2.0, rl1: 0.5, ds: -0.05, dsmin: 1e-4, dsmax: 0.1, nmx: 50, ntst: 40, ncol: 4, restartSp: lpcSp.restartSp },
      });

      // 2. DBT fast subsystem: LP2 -> CP & BT, HB -> LC (stopPeriod=30) -> HOM2
      const dbtSpec = {
        state_vars: { x: { range: [-3, 3], default: -1.5 }, y: { range: [-3, 3], default: 0 } },
        equations: {
          x: 'y',
          y: '-x^3 + w * x - z - y * (w + 1.5 * x + x^2)',
        },
        parameters: { z: { range: [-2, 4], default: 3.075 }, w: { range: [-1, 2], default: 0.2 } },
      };
      const dbtEp = await runJob({
        model: { model_name: 'custom', model_spec: dbtSpec, params: { z: 3.075, w: 0.2 } },
        continuation: { ips: 1, isw: 1, activeParams: ['z'], rl0: -2.0, rl1: 4.0, ds: -0.05, dsmin: 1e-4, dsmax: 0.1, nmx: 300, stpnt: { u: [-1.5, 0], p: [3.075] } },
      });
      const dbtLp = dbtEp.specialPoints.find((sp) => sp.ty === 'LP');
      const dbtHb = dbtEp.specialPoints.find((sp) => sp.ty === 'HB');
      const dbtLp2Fwd = await runJob({
        model: { model_name: 'custom', model_spec: dbtSpec, params: { z: dbtLp.p, w: 0.2 } },
        continuation: { ips: 1, isw: 2, activeParams: ['z', 'w'], rl0: -1.5, rl1: 1.5, ds: 0.03, dsmin: 1e-4, dsmax: 0.08, nmx: 200, restartSp: dbtLp.restartSp },
      });
      const dbtLp2Bwd = await runJob({
        model: { model_name: 'custom', model_spec: dbtSpec, params: { z: dbtLp.p, w: 0.2 } },
        continuation: { ips: 1, isw: 2, activeParams: ['z', 'w'], rl0: -1.5, rl1: 1.5, ds: -0.03, dsmin: 1e-4, dsmax: 0.08, nmx: 200, restartSp: dbtLp.restartSp },
      });
      const dbtLc = await runJob({
        model: { model_name: 'custom', model_spec: dbtSpec, params: { z: dbtHb.p, w: 0.2 } },
        continuation: { ips: 2, isw: 1, ilp: 0, isp: 0, activeParams: ['z'], rl0: -2.0, rl1: 4.0, ds: 0.03, dsmin: 1e-5, dsmax: 0.1, nmx: 200, ntst: 60, ncol: 4, stopPeriod: 30.0, restartSp: dbtHb.restartSp },
      });
      const homSeed = dbtLc.specialPoints.find((sp) => sp.can_trace_homoclinic && (sp.period ?? 0) >= 15);
      const dbtHom2 = await runJob({
        model: { model_name: 'custom', model_spec: dbtSpec, params: { z: homSeed.p, w: 0.2 } },
        continuation: { ips: 2, isw: 2, ilp: 0, isp: 0, activeParams: ['z', 'w'], rl0: -1.5, rl1: 1.5, ds: 0.03, dsmin: 1e-4, dsmax: 0.08, nmx: 60, ntst: 60, ncol: 4, restartSp: homSeed.restartSp },
      });

      // 3. 3D Zero-Hopf (ZH)
      const zhSpec = {
        state_vars: { x: { range: [-2, 2], default: 0 }, y: { range: [-2, 2], default: 0 }, z: { range: [-2, 2], default: 0.5 } },
        equations: { x: '(mu2 + z) * x - y', y: 'x + (mu2 + z) * y', z: 'mu1 - z^2' },
        parameters: { mu1: { range: [-1, 1], default: 0.25 }, mu2: { range: [-1, 1], default: -0.3 } },
      };
      const zhEp = await runJob({
        model: { model_name: 'custom', model_spec: zhSpec, params: { mu1: 0.25, mu2: -0.3 } },
        continuation: { ips: 1, isw: 1, activeParams: ['mu1'], rl0: -0.5, rl1: 1.0, ds: -0.05, nmx: 200, stpnt: { u: [0, 0, 0.5], p: [0.25] } },
      });
      const zhLp = zhEp.specialPoints.find((sp) => sp.ty === 'LP');
      const zhLp2Fwd = await runJob({
        model: { model_name: 'custom', model_spec: zhSpec, params: { mu1: zhLp.p, mu2: -0.3 } },
        continuation: { ips: 1, isw: 2, ilp: 0, activeParams: ['mu1', 'mu2'], rl0: -1.0, rl1: 1.0, ds: 0.05, nmx: 60, restartSp: zhLp.restartSp },
      });
      const zhLp2Bwd = await runJob({
        model: { model_name: 'custom', model_spec: zhSpec, params: { mu1: zhLp.p, mu2: -0.3 } },
        continuation: { ips: 1, isw: 2, ilp: 0, activeParams: ['mu1', 'mu2'], rl0: -1.0, rl1: 1.0, ds: -0.05, nmx: 60, restartSp: zhLp.restartSp },
      });

      worker.terminate();
      return {
        gh: bautinHb2.specialPoints.find((sp) => sp.ty === 'GH'),
        lpcSp,
        lpc2Branch: lpc2.branch,
        dbtLp2Sps: [...dbtLp2Fwd.specialPoints, ...dbtLp2Bwd.specialPoints],
        homSeed,
        dbtHom2Branch: dbtHom2.branch,
        zhSps: [...zhLp2Fwd.specialPoints, ...zhLp2Bwd.specialPoints],
      };
    }, baseUrl);

    assert.ok(results.gh, 'Expected GH special point from worker');
    assert.ok(results.lpcSp.can_trace_2d, 'Expected LPC can_trace_2d=true');
    assert.ok(results.lpcSp.restartSp.ups && results.lpcSp.restartSp.udotps, 'Expected BVP mesh on LPC restartSp');
    assert.ok(results.lpc2Branch.length >= 10, 'Expected >=10 LPC2 points');
    assert.equal(typeof results.lpc2Branch[0].p2, 'number', 'Expected p2 on 2D LPC2 branch');
    for (const pt of results.lpc2Branch) {
      if (pt.p2 > 0.05) {
        assert.ok(Math.abs(pt.p - (-pt.p2 * pt.p2 / 4)) < 2e-3, `Worker LPC2 mismatch: ${pt.p} vs ${-pt.p2 * pt.p2 / 4}`);
      }
    }

    assert.ok(results.dbtLp2Sps.some((sp) => sp.ty === 'CP'), 'Expected CP label on DBT LP2');
    assert.ok(results.dbtLp2Sps.some((sp) => sp.ty === 'BT'), 'Expected BT label on DBT LP2');
    assert.ok(results.homSeed.can_trace_homoclinic, 'Expected can_trace_homoclinic on high-period LC point');
    assert.ok(results.dbtHom2Branch.length >= 10, 'Expected >=10 HOM2 points');
    assert.equal(typeof results.dbtHom2Branch[0].p2, 'number', 'Expected p2 on HOM2 branch');
    assert.ok(Math.abs(results.dbtHom2Branch[0].period - 30.0) < 1e-2, 'Expected fixed period ~30 on HOM2');
    assert.ok(results.zhSps.some((sp) => sp.ty === 'ZH'), 'Expected ZH label on 3D LP2');
  } finally {
    await page.close();
  }
});
