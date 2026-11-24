import assert from 'node:assert/strict';
import { join } from 'node:path';
import { readFile } from 'node:fs/promises';

export async function checkDesignGallery(page, base, screenshotDirectory) {
  await page.setViewportSize({ width: 1280, height: 960 });
  await page.goto(`${base}/design-lab.html`);
  const main = page.locator('.ppw-presentation-root');
  const number = (root, key) => root.locator(`.ppw-param:has([data-param="${key}"]) .ppw-param-number`);
  const setNumber = async (root, key, value) => { const input = number(root, key); await input.fill(String(value)); await input.dispatchEvent('change'); };
  const choose = async mode => { await page.locator(`.design-lab-nav [data-view=${mode}]`).click(); await main.locator(`[data-presentation=${mode}]`).waitFor(); };
  const snapshot = async name => { if (screenshotDirectory) await page.screenshot({ path: join(screenshotDirectory, `design-${name}.png`), fullPage: true }); };
  const fits = async () => {
    await page.waitForFunction(() => [...document.querySelectorAll('.ppw-composition canvas')].filter(c => c.offsetWidth).every(c => Math.abs(c.width - c.getBoundingClientRect().width * devicePixelRatio) < 3));
    const bounds = await main.locator('.ppw-composition').evaluate(element => {
      const stage = element.getBoundingClientRect(), body = element.querySelector('.ppw-stage-body').getBoundingClientRect(), footer = element.querySelector('.ppw-stage-footer').getBoundingClientRect();
      return { ratio: stage.width / stage.height, bodyBottom: body.bottom, footerTop: footer.top, footerBottom: footer.bottom, stageBottom: stage.bottom, overflow: element.scrollHeight - element.clientHeight };
    });
    assert.ok(Math.abs(bounds.ratio - 16 / 9) < .02, JSON.stringify(bounds));
    assert.ok(bounds.bodyBottom <= bounds.footerTop + 2, JSON.stringify(bounds));
    assert.ok(bounds.footerBottom <= bounds.stageBottom + 2, JSON.stringify(bounds));
    assert.ok(bounds.overflow < 3, JSON.stringify(bounds));
  };
  await main.locator('.ppw-phase-canvas').waitFor();
  assert.equal(await main.locator('canvas').count(), 1);
  await main.locator('.ppw-inspect').click();
  assert.equal(await main.locator('.ppw-details-popover').isVisible(), true);
  await page.keyboard.press('Escape');
  assert.equal(await main.locator('.ppw-details-popover').count(), 0);
  assert.equal(await main.locator('.ppw-inspect').evaluate(node => node === document.activeElement), true);
  await setNumber(main, 'I', .8);
  await snapshot('figure');
  await main.locator('.ppw-expand').click();
  const dialog = page.locator('dialog.ppw-expanded-workspace'); await dialog.waitFor();
  assert.equal(await dialog.locator('.ppw-composition').getAttribute('data-presentation'), 'bench');
  await setNumber(dialog.locator('.ppw-stage-footer'), 'I', .2);
  await page.keyboard.press('Escape'); await dialog.waitFor({ state: 'detached' });
  assert.equal(await number(main, 'I').inputValue(), '0.2');
  assert.equal(await main.locator('.ppw-expand').evaluate(node => node === document.activeElement), true);

  await choose('bench'); await fits();
  assert.equal(await main.locator('canvas').count(), 2);
  const picker = main.locator('[aria-label="Inspector section"]');
  await picker.selectOption('parameters');
  await setNumber(main.locator('.ppw-inspector-content'), 'a', .6);
  await picker.selectOption('noise');
  await main.locator('.ppw-noise-enable').check();
  assert.equal(await main.locator('.ppw-integrator-select').inputValue(), 'heun');
  await main.locator('.ppw-noise-enable').uncheck();
  await picker.selectOption('details');
  assert.match(await main.locator('.ppw-readout').textContent(), /a = 0.600/);
  await snapshot('bench');

  await choose('drawer');
  assert.equal(await main.locator('.ppw-instrument-drawer').isVisible(), false);
  await main.locator('.ppw-drawer-toggle').click();
  assert.equal(await main.locator('.ppw-instrument-drawer').isVisible(), true);
  await setNumber(main.locator('.ppw-instrument-drawer'), 'a', .7);
  await snapshot('drawer');
  await main.locator('.ppw-drawer-toggle').click();
  assert.equal(await main.locator('.ppw-instrument-drawer .ppw-param-number').count(), 0);

  await choose('comparison');
  await main.locator('.ppw-reference').click();
  assert.match(await main.locator('.ppw-reference-status').textContent(), /I=0.200/);
  await setNumber(main, 'I', .8);
  await fits();
  assert.match(await main.locator('.ppw-reference-status').textContent(), /I=0.200/);
  const frozen = await main.locator('.ppw-phase-canvas').evaluate(c => c.toDataURL());
  await main.locator('.ppw-show-reference').uncheck();
  assert.notEqual(await main.locator('.ppw-phase-canvas').evaluate(c => c.toDataURL()), frozen);
  await main.locator('.ppw-show-reference').check();
  assert.equal(await main.locator('.ppw-phase-canvas').evaluate(c => c.toDataURL()), frozen);
  await snapshot('comparison');
  await main.locator('.ppw-expand').click();
  await dialog.locator('[aria-label="Inspector section"]').selectOption('export');
  const download = page.waitForEvent('download');
  await dialog.locator('.ppw-export-svg').click();
  const svg = await readFile(await (await download).path(), 'utf8');
  assert.match(svg, /stroke-dasharray="6 4"/);
  assert.match(svg, /clipPath/);
  await dialog.locator('.ppw-close-workspace').click();

  await choose('lesson');
  await main.locator('.ppw-task-tab', { hasText: 'Observe' }).click();
  assert.equal(await main.locator('canvas').count(), 2);
  await main.locator('.ppw-task-tab', { hasText: 'Compare' }).click(); await fits();
  assert.match(await main.locator('.ppw-reference-status').textContent(), /I=0.200/);
  await snapshot('lesson');
  await main.locator('.ppw-task-tab', { hasText: 'Sweep' }).click();
  await main.locator('.ppw-sweep-select').selectOption('I');
  await main.locator('.ppw-sweep-min').fill('0'); await main.locator('.ppw-sweep-max').fill('1');
  await main.locator('.ppw-sweep-n').fill('4'); await main.locator('.ppw-sweep-btn').click();
  await page.waitForFunction(() => document.querySelector('.ppw-sweep-status').textContent.includes('4 samples complete'));
  await main.locator('.ppw-task-tab', { hasText: 'Explore' }).click();
  assert.equal(await number(main, 'I').inputValue(), '0.8');
  await main.locator('.ppw-task-tab', { hasText: 'Sweep' }).click();
  assert.match(await main.locator('.ppw-sweep-status').textContent(), /4 samples complete/);
  await main.locator('.ppw-sweep-n').fill('100'); await main.locator('.ppw-sweep-btn').click();
  await main.locator('.ppw-task-tab', { hasText: 'Explore' }).click();
  await choose('figure');
  assert.equal(await main.locator('canvas').count(), 1);
  assert.equal(await number(main, 'I').inputValue(), '0.8');
  await main.locator('.ppw-stage-reset').click();
  assert.equal(await number(main, 'I').inputValue(), '0.5');

  await page.setViewportSize({ width: 390, height: 844 });
  for (const mode of ['figure', 'bench', 'drawer', 'comparison', 'lesson']) {
    await choose(mode);
    if (mode === 'drawer') await main.locator('.ppw-drawer-toggle').click();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `${mode} must not overflow horizontally`);
    assert.ok(await main.locator('.ppw-phase-canvas').evaluate(c => c.clientHeight >= 100));
  }
  await snapshot('mobile');
  console.log('PASS: five gallery compositions, frozen-scene toggles, expansion/focus, bounded 16:9 layouts, inspector/drawer, task/sweep state and narrow fallback');
}
