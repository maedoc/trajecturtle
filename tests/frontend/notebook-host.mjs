// Opt-in integration test: starts a temporary local JupyterLab, uses a real
// widget manager/kernel, then tears down the browser, server and notebook.
// Requires jupyterlab in $PYTHON and `npx playwright install chromium`.
import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { mkdtemp, writeFile, rm, realpath, mkdir, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import net from 'node:net';
import { createServer } from 'node:http';
import { chromium } from 'playwright';
import { checkDesignGallery } from './design-gallery.mjs';
import { checkResearchViews } from './research-views.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const directory = await realpath(await mkdtemp(join(tmpdir(), 'trajecturtle-host-')));
const python = process.env.PYTHON || 'python';
const token = randomUUID();
const socket = net.createServer();
await new Promise(resolve => socket.listen(0, '127.0.0.1', resolve));
const port = socket.address().port;
await new Promise(resolve => socket.close(resolve));
const base = `http://127.0.0.1:${port}`;
const env = { ...process.env, PYTHONPATH: [resolve(root, 'src'), process.env.PYTHONPATH].filter(Boolean).join(process.platform === 'win32' ? ';' : ':') };
const code = [
  "from IPython.display import display\nfrom tvb_phaseplane import PhasePlaneWidget, phase_plane\nw = PhasePlaneWidget(model_name='fitzhugh_nagumo', params={'I': .5}, t_max=2)\nother = PhasePlaneWidget(model_name='fitzhugh_nagumo', params={'I': -.2}, t_max=2)",
  "display(w.view(['parameters'], parameters=['I']))",
  "display(w.view(['parameters'], parameters=['I']))",
  "display(w.view(['phase_plane', 'time_series']))",
  "display(other.view(['parameters'], parameters=['I']))",
];
await writeFile(join(directory, 'shared.ipynb'), JSON.stringify({
  nbformat: 4, nbformat_minor: 5,
  metadata: { kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' } },
  cells: code.map(source => ({ cell_type: 'code', id: randomUUID(), metadata: {}, source, outputs: [], execution_count: null })),
}));
const server = spawn(python, ['-m', 'jupyterlab', '--no-browser',
  `--ServerApp.root_dir=${directory}`, '--ServerApp.ip=127.0.0.1',
  `--ServerApp.port=${port}`, '--ServerApp.port_retries=0',
  `--IdentityProvider.token=${token}`, '--LabApp.expose_app_in_browser=True'], { env });
let log = '', launchError;
server.stdout.on('data', data => { log += data; });
server.stderr.on('data', data => { log += data; });
server.on('error', error => { launchError = error; });
let browser, staticServer;
try {
  const deadline = Date.now() + 60000;
  for (;;) {
    if (launchError) throw launchError;
    if (server.exitCode !== null) throw new Error(`JupyterLab exited: ${log}`);
    try { if ((await fetch(`${base}/api?token=${token}`)).ok) break; } catch {}
    if (Date.now() > deadline) throw new Error(`JupyterLab startup timed out: ${log}`);
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 960 }, deviceScaleFactor: 2 });
  page.setDefaultTimeout(30000);
  const pageErrors = [];
  page.on('pageerror', error => { pageErrors.push(error.message); console.error('Browser error:', error.message); });
  await page.goto(`${base}/lab/tree/shared.ipynb?token=${token}`);
  await page.waitForFunction(() => window.jupyterapp?.shell?.currentWidget?.sessionContext?.session?.kernel);
  await page.evaluate(() => window.jupyterapp.commands.execute('notebook:run-all-cells'));
  await page.waitForFunction(() => document.querySelectorAll('.ppw-param-number').length === 3);
  async function execute(code) {
    return page.evaluate(async code => {
      const kernel = window.jupyterapp.shell.currentWidget.sessionContext.session.kernel;
      const future = kernel.requestExecute({ code });
      const output = [];
      let error;
      future.onIOPub = msg => {
        if (msg.header.msg_type === 'stream') output.push(msg.content.text);
        if (msg.header.msg_type === 'error') error = msg.content.evalue;
      };
      await future.done;
      if (error) throw new Error(error);
      return output.join('');
    }, code);
  }
  const canvasBefore = await page.locator('.ppw-phase-canvas').evaluate(c => c.toDataURL());
  await execute("w.params = {**w.params, 'I': .8}");
  await page.waitForFunction(() => [...document.querySelectorAll('.ppw-param-number')].slice(0, 2).every(n => n.value === '0.8'));
  assert.equal(await page.locator('.ppw-param-number').nth(2).inputValue(), '-0.2');
  assert.notEqual(await page.locator('.ppw-phase-canvas').evaluate(c => c.toDataURL()), canvasBefore);
  await page.locator('.ppw-param-number').first().fill('0.3');
  await page.locator('.ppw-param-number').first().dispatchEvent('change');
  await page.waitForFunction(() => document.querySelectorAll('.ppw-param-number')[1].value === '0.3');
  assert.match(await execute("print(w.params['I'], other.params['I'])"), /0.3 -0.2/);
  console.log('PASS: Python assignment, linked controls/plots, independent experiments, JS-to-kernel sync');

  await page.evaluate(async () => {
    const app = window.jupyterapp;
    app.shell.currentWidget.content.activeCellIndex = 1;
    await app.commands.execute('notebook:run-cell');
  });
  await page.waitForFunction(() => document.querySelectorAll('.ppw-param-number').length === 3);
  await page.locator('.ppw-param-number').first().fill('0.6');
  await page.locator('.ppw-param-number').first().dispatchEvent('change');
  await page.waitForFunction(() => document.querySelectorAll('.ppw-param-number')[1].value === '0.6');
  assert.match(await execute("print(w.params['I'])"), /0.6/);
  assert.equal(await page.locator('.ppw-error:not([hidden])').count(), 0);
  console.log('PASS: separate-cell output disposal/remount and late view synchronization');

  await page.evaluate(async () => {
    const app = window.jupyterapp;
    const notebook = app.shell.currentWidget.content;
    notebook.model.sharedModel.insertCell(5, { cell_type: 'code', source: "custom = phase_plane(['-x', '-y'], state_vars={'x': (-1,1), 'y': (-1,1)}, integrator='heun', noise_per_var=[.1,.2])\ncustom.noise_seed = 42\ndisplay(custom.view(['noise','phase_plane']))" });
    notebook.activeCellIndex = 5;
    await app.commands.execute('notebook:run-cell');
  });
  await page.waitForFunction(() => document.querySelectorAll('.ppw-phase-canvas').length === 2);
  await page.waitForFunction(() => document.querySelectorAll('.ppw-noise-slider').length === 2);
  for (let i = 0; i < 30; i++) {
    if (Number(await execute('print(len(custom.trajectory))')) > 1) break;
    if (i === 29) throw new Error('Custom trajectory did not sync to the kernel');
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  const path = await execute('import json\nprint(json.dumps(custom.trajectory))');
  await execute('custom.noise_seed = 43');
  await page.waitForFunction(() => document.querySelector('.ppw-noise-seed').value === '43');
  // Await the frontend's derived-state comm, not merely the control update.
  let changed;
  for (let i = 0; i < 30; i++) {
    changed = await execute('print(json.dumps(custom.trajectory))');
    if (changed !== path) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.notEqual(changed, path);
  await execute('custom.noise_seed = 42');
  for (let i = 0; i < 30; i++) {
    if ((await execute('print(json.dumps(custom.trajectory))')) === path) break;
    if (i === 29) throw new Error('Seed did not reproduce the custom trajectory');
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  console.log('PASS: custom model, Python seed changes and reproducible noise');

  await page.evaluate(async () => {
    const app = window.jupyterapp, notebook = app.shell.currentWidget.content;
    notebook.model.sharedModel.insertCell(6, { cell_type: 'code', source: "feedback = w.explore(parameters=['I'])\ndisplay(feedback)" });
    notebook.activeCellIndex = 6;
    await app.commands.execute('notebook:run-cell');
  });
  const notebookFigure = page.locator('.ppw-composition[data-presentation=figure]');
  await notebookFigure.locator('.ppw-phase-canvas').waitFor();
  await page.locator('.ppw-view-selector').selectOption('bench');
  assert.match(await execute('print(feedback.presentation)'), /bench/);
  await page.locator('.ppw-view-selector').selectOption('figure');
  await notebookFigure.locator('.ppw-expand').click();
  const notebookDialog = page.locator('dialog.ppw-expanded-workspace');
  const notebookNumber = notebookDialog.locator('.ppw-stage-footer .ppw-param-number');
  await notebookNumber.fill('0.65'); await notebookNumber.dispatchEvent('change');
  assert.match(await execute("print(w.params['I'])"), /0.65/);
  await notebookDialog.locator('.ppw-close-workspace').click();
  assert.equal(await notebookFigure.locator('.ppw-param-number').inputValue(), '0.65');
  console.log('PASS: notebook explorer picker syncs to Python; figure and expanded bench share live experiment state');
  if (process.env.TT_SCREENSHOTS) await mkdir(process.env.TT_SCREENSHOTS, { recursive: true });
  await checkResearchViews(page, {
    root: page.locator('.ppw-presentation-root'),
    choose: async value => { await page.locator('.ppw-view-selector').selectOption(value); await page.locator(`[data-presentation=${value}]`).waitFor(); },
    screenshots: process.env.TT_SCREENSHOTS, prefix: 'notebook-research',
    afterCapture: async () => { assert.match(await execute('print(len(w.observation_cards), len(w.initial_conditions))'), /1 2/); },
  });

  // Use a plain static server, matching offline/Pages exports. Jupyter's /files
  // handler deliberately sandboxes HTML and disallows download links.
  staticServer = createServer(async (request, response) => {
    const name = request.url?.split('?')[0];
    if (!['/standalone.html', '/ux.html', '/design-lab.html'].includes(name)) { response.writeHead(404).end(); return; }
    try {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end(await readFile(join(directory, name.slice(1))));
    } catch { response.end(); }
  });
  await new Promise(resolve => staticServer.listen(0, '127.0.0.1', resolve));
  const staticBase = `http://127.0.0.1:${staticServer.address().port}`;
  // Export with Python, serve over HTTP, and exercise the same bundle without a kernel.
  execFileSync(python, ['-c', `from tvb_phaseplane import PhasePlaneWidget\nw = PhasePlaneWidget(model_name='fitzhugh_nagumo', params={'I':.5}, t_max=2)\nw.to_standalone_html(${JSON.stringify(join(directory, 'standalone.html'))}, views=[{'panels':['parameters'], 'visible_parameters':['I']}, {'panels':['parameters'], 'visible_parameters':['I']}, {'panels':['phase_plane']}])`], { env });
  await page.goto(`${staticBase}/standalone.html`);
  await page.locator('.ppw-param-number').nth(1).waitFor();
  const staticBefore = await page.locator('.ppw-phase-canvas').evaluate(c => c.toDataURL());
  await page.locator('.ppw-param-number').first().fill('0.8');
  await page.locator('.ppw-param-number').first().dispatchEvent('change');
  await page.waitForFunction(() => document.querySelectorAll('.ppw-param-number')[1].value === '0.8');
  assert.notEqual(await page.locator('.ppw-phase-canvas').evaluate(c => c.toDataURL()), staticBefore);
  console.log('PASS: standalone linked views and rendered plot updates');

  execFileSync(python, ['-c', `from tvb_phaseplane import phase_plane\nu = phase_plane(['-a*x+y', '-y', '0.1*(x-z)'], state_vars={'x':(-2,2), 'y':(-2,2), 'z':(-2,2)}, params={'a':(.7,0,2)}, display=['x','y'])\nu.t_max=4\nu.x0=.4\nu.panels=['time_series','phase_plane','parameters','projection','comparison','export','noise','layers','limits','editor','sweep']\nu.to_standalone_html(${JSON.stringify(join(directory, 'ux.html'))})`], { env });
  await page.goto(`${staticBase}/ux.html`);
  await page.locator('.ppw-phase-canvas').waitFor();
  assert.deepEqual(await page.locator('[data-panel]').evaluateAll(nodes => nodes.slice(0,3).map(node => node.dataset.panel)), ['time_series', 'phase_plane', 'parameters']);
  const phase = page.locator('.ppw-phase-canvas');
  const beforeKeyboard = await phase.evaluate(c => c.toDataURL());
  await phase.focus(); await phase.press('ArrowRight');
  assert.notEqual(await phase.evaluate(c => c.toDataURL()), beforeKeyboard);
  assert.match(await page.locator('.ppw-initial-state').textContent(), /0.440/);
  await page.locator('.ppw-reference').click();
  assert.match(await page.locator('.ppw-reference-status').textContent(), /retained/);
  await page.locator('.ppw-reset').click();
  assert.match(await page.locator('.ppw-initial-state').textContent(), /0.400/);
  const invalid = page.locator('[data-panel=parameters] .ppw-param-number');
  await invalid.fill('3'); await invalid.dispatchEvent('change');
  assert.equal(await invalid.getAttribute('aria-invalid'), 'true');
  await invalid.fill('0.8'); await invalid.dispatchEvent('change');
  assert.equal(await invalid.getAttribute('aria-invalid'), null);
  const download = page.waitForEvent('download');
  await page.locator('.ppw-export-svg').click();
  assert.match(await readFile(await (await download).path(), 'utf8'), /<svg/);
  await page.locator('.ppw-editor-toggle').click();
  await page.locator('.ppw-ed-expr').first().fill('-2*a*x+y');
  await page.locator('.ppw-editor-apply').click();
  assert.match(await page.locator('.ppw-editor-status').textContent(), /Valid/);
  assert.equal(await page.locator('.ppw-error:not([hidden])').count(), 0);
  assert.equal(await page.locator('.ppw-sweep-canvas').isVisible(), false);
  await page.locator('.ppw-sweep-min').fill('0.2');
  await page.locator('.ppw-sweep-max').fill('1');
  await page.locator('.ppw-sweep-n').fill('4');
  await page.locator('.ppw-sweep-btn').click();
  await page.waitForFunction(() => document.querySelector('.ppw-sweep-status').textContent.includes('4 samples complete'));
  assert.equal(await page.locator('.ppw-sweep-canvas').isVisible(), true);
  console.log('PASS: ordered panels, keyboard initial conditions, comparison/reset, validation, SVG export, isolated editor and sweep results');

  if (process.env.TT_SCREENSHOTS) {
    await mkdir(process.env.TT_SCREENSHOTS, { recursive: true });
    await page.screenshot({ path: join(process.env.TT_SCREENSHOTS, 'desktop.png'), fullPage: true });
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => [...document.querySelectorAll('canvas')].filter(canvas => canvas.offsetWidth > 0).every(canvas => Math.abs(canvas.width - canvas.getBoundingClientRect().width * devicePixelRatio) < 3));
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'narrow layout must not overflow horizontally');
  const panelBounds = await page.locator('[data-panel=time_series], [data-panel=phase_plane]').evaluateAll(nodes => nodes.map(node => ({ left: node.getBoundingClientRect().left, top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom })));
  assert.equal(panelBounds[0].left, panelBounds[1].left);
  assert.ok(panelBounds[1].top >= panelBounds[0].bottom);
  if (process.env.TT_SCREENSHOTS) await page.screenshot({ path: join(process.env.TT_SCREENSHOTS, 'mobile.png'), fullPage: true });
  await page.addStyleTag({ content: '.ppw-widget { --ppw-plot-bg: #14202e; --ppw-text: #eef4fa; --ppw-grid: #334455; }' });
  await phase.focus(); await phase.press('ArrowRight');
  const background = await phase.evaluate(canvas => [...canvas.getContext('2d').getImageData(4, 4, 1, 1).data].slice(0,3));
  assert.deepEqual(background, [20, 32, 46]);
  console.log('PASS: narrow responsive layout, 2× high-DPI backing canvases and canvas theme tokens');
  execFileSync(python, ['-c', `import sys\nsys.path.insert(0, ${JSON.stringify(join(root, 'scripts'))})\nfrom build_design_gallery import build_design_gallery\nbuild_design_gallery(${JSON.stringify(join(directory, 'design-lab.html'))})`], { env });
  await checkDesignGallery(page, staticBase, process.env.TT_SCREENSHOTS);
  execFileSync(python, ['-c', `from tvb_phaseplane import explore\nexplore(params={'I':.5}, parameters=['I'], t_max=4).to_standalone_html(${JSON.stringify(join(directory, 'standalone.html'))})`], { env });
  await page.goto(`${staticBase}/standalone.html`);
  await page.locator('.ppw-view-selector').selectOption('comparison');
  await page.locator('.ppw-reference').click();
  for (const value of ['bench', 'drawer', 'lesson', 'figure', 'comparison']) {
    await page.locator('.ppw-view-selector').selectOption(value);
    await page.locator(`[data-presentation=${value}]`).waitFor();
  }
  assert.match(await page.locator('.ppw-reference-status').textContent(), /retained/);
  assert.equal(await page.locator('.ppw-param-number').inputValue(), '0.5');
  console.log('PASS: Python explore() export includes a functional view picker preserving frozen state');
  await page.setViewportSize({ width: 1280, height: 960 });
  await checkResearchViews(page, {
    root: page.locator('.ppw-presentation-root'),
    choose: async value => { await page.locator('.ppw-view-selector').selectOption(value); await page.locator(`[data-presentation=${value}]`).waitFor(); },
    screenshots: process.env.TT_SCREENSHOTS,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  for (const kind of ['multiples', 'navigator', 'basin', 'evidence', 'prediction', 'ensemble']) {
    await page.locator('.ppw-view-selector').selectOption(kind);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${kind} should not overflow horizontally`);
  }
  assert.deepEqual(pageErrors, []);
} catch (error) {
  console.error(log);
  throw error;
} finally {
  if (browser) await browser.close();
  if (staticServer) await new Promise(resolve => staticServer.close(resolve));
  if (server.exitCode === null && !launchError) {
    const exited = new Promise(resolve => server.once('exit', resolve));
    server.kill('SIGTERM');
    await exited;
  }
  await rm(directory, { recursive: true, force: true });
}
