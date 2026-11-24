import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

export async function checkResearchViews(page, { choose, root, screenshots, prefix = 'research', afterCapture }) {
  const number = () => root.locator('.ppw-stage-footer .ppw-param:has([data-param="I"]) .ppw-param-number');
  const setI = async value => { await number().fill(String(value)); await number().dispatchEvent('change'); };
  const capture = async name => { if (screenshots) await root.screenshot({ path: join(screenshots, `${prefix}-${name}.png`) }); };
  await choose('figure'); await root.locator('.ppw-stage-reset').click();
  await choose('multiples');
  assert.equal(await root.locator('canvas').count(), 3);
  const tile = root.locator('.ppw-multiple-value').nth(2); await tile.fill('0.8'); await tile.dispatchEvent('change');
  assert.equal(await number().inputValue(), '0.5');
  await root.locator('.ppw-use-tile').nth(2).click(); assert.equal(await number().inputValue(), '0.8');
  await capture('multiples');

  await choose('basin'); await root.locator('.ppw-add-start').click();
  const phase = root.locator('canvas').first(); const bounds = await phase.boundingBox();
  await phase.click({ position: { x: bounds.width * .65, y: bounds.height * .6 } });
  assert.equal(await root.locator('.ppw-trace-list button').count(), 2);
  await capture('basin');
  await choose('figure'); await choose('basin'); assert.equal(await root.locator('.ppw-trace-list button').count(), 2);

  await choose('evidence');
  await root.locator('[aria-label="Observation label"]').fill('Oscillatory candidate');
  await root.locator('.ppw-capture-observation').click();
  assert.equal(await root.locator('.ppw-restore-observation').count(), 1);
  if (afterCapture) await afterCapture();
  await setI(.2);
  await root.locator('.ppw-evidence-cards button', { hasText: 'Preview' }).click(); assert.equal(await number().inputValue(), '0.2');
  await choose('bench'); await root.locator('[aria-label="Inspector section"]').selectOption('model');
  await root.locator('.ppw-model-select').selectOption('wilson_cowan');
  await choose('evidence');
  await root.locator('.ppw-restore-observation').click(); assert.equal(await number().inputValue(), '0.8');
  assert.match(await root.locator('.ppw-stage-badges').textContent(), /fitzhugh nagumo/);
  const download = page.waitForEvent('download'); await root.locator('.ppw-download-evidence').click();
  const data = JSON.parse(await readFile(await (await download).path(), 'utf8'));
  assert.equal(data.format, 'trajecturtle-evidence-v1'); assert.equal(data.observations[0].state.params.I, .8);
  assert.ok(data.observations[0].scene.trajectory.length > 1);
  await capture('evidence');

  await choose('prediction');
  const hidden = await root.locator('canvas').first().evaluate(canvas => canvas.toDataURL());
  await root.locator('.ppw-prediction-answer').fill('I predict oscillation.'); await root.locator('.ppw-reveal').click();
  assert.notEqual(await root.locator('canvas').first().evaluate(canvas => canvas.toDataURL()), hidden);
  await choose('figure'); await choose('prediction');
  assert.equal(await root.locator('.ppw-prediction-answer').inputValue(), 'I predict oscillation.');
  await setI(.4); assert.equal(await root.locator('.ppw-reveal').getAttribute('aria-pressed'), 'false');
  await root.locator('.ppw-reveal').click(); await capture('prediction');

  await choose('ensemble');
  assert.equal(await root.locator('.ppw-use-member').count(), 4);
  const seeded = await root.locator('canvas').first().evaluate(canvas => canvas.toDataURL());
  await root.locator('.ppw-use-member').first().click();
  assert.equal(await root.locator('canvas').first().evaluate(canvas => canvas.toDataURL()), seeded, 'Applying the preview seed must reproduce its ensemble');
  await root.locator('.ppw-next-seeds').click();
  assert.notEqual(await root.locator('canvas').first().evaluate(canvas => canvas.toDataURL()), seeded);
  await capture('ensemble');

  await choose('navigator');
  await root.locator('.ppw-sweep-select').selectOption('I');
  await root.locator('.ppw-sweep-min').fill('0'); await root.locator('.ppw-sweep-max').fill('1');
  await root.locator('.ppw-sweep-n').fill('4'); await root.locator('.ppw-sweep-btn').click();
  await root.locator('.ppw-scan-seek').waitFor();
  await page.waitForFunction(() => !document.querySelector('.ppw-scan-seek').disabled);
  await root.locator('.ppw-scan-seek').evaluate(node => { node.value = '2'; node.dispatchEvent(new Event('input', { bubbles: true })); });
  assert.ok(Math.abs(Number(await number().inputValue()) - 2 / 3) < 1e-10);
  await root.locator('.ppw-sweep-canvas').focus(); await page.keyboard.press('ArrowRight');
  assert.equal(await number().inputValue(), '1');
  await choose('figure'); await choose('navigator');
  assert.equal(await root.locator('.ppw-scan-seek').isEnabled(), true);
  assert.equal(await root.locator('.ppw-sweep-select').inputValue(), 'I');
  await capture('navigator');
  assert.equal(await root.locator('.ppw-research-status').count(), 0);
  await root.locator('.ppw-stage-reset').click();
  assert.equal(await root.locator('.ppw-scan-seek').isEnabled(), false);
  await choose('figure');
  console.log('PASS: six scientific-task views, isolated tiles, multiple starts, durable evidence/JSON/restore, local prediction, reproducible ensemble and retained interactive scan');
}
