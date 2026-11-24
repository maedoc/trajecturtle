import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { chromium } from 'playwright';

const STATIC_DIR = path.resolve(import.meta.dirname, '../../src/tvb_phaseplane/static');

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
