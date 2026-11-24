// Small DOM/lifecycle primitives shared by independently mounted panels.
let controlSequence = 0;
function panelScope() {
  let disposed = false;
  const cleanups = [], timers = new Set();
  return {
    on(target, event, callback) {
      target.addEventListener(event, callback);
      cleanups.push(() => target.removeEventListener(event, callback));
    },
    later(callback, delay = 0) {
      if (disposed) return null;
      const timer = setTimeout(() => { timers.delete(timer); if (!disposed) callback(); }, delay);
      timers.add(timer); return timer;
    },
    cancel(timer) { clearTimeout(timer); timers.delete(timer); },
    cleanup(callback) { cleanups.push(callback); },
    dispose() {
      if (disposed) return;
      disposed = true;
      for (const timer of timers) clearTimeout(timer);
      for (const cleanup of cleanups.splice(0).reverse()) cleanup();
      timers.clear();
    },
  };
}
function panelElement(doc, className, html = '') {
  const element = doc.createElement('section');
  element.className = className;
  element.innerHTML = html;
  return element;
}
function mountedPanel(element, scope, update = () => {}) {
  return { element, update, dispose() { scope.dispose(); element.remove(); } };
}
function numberControl(doc, scope, { label, min, max, value, step = 'any', sliderClass = 'ppw-param-slider', valueClass = 'ppw-param-value', onChange, delay = 0 }) {
  const element = doc.createElement('div'); element.className = 'ppw-param';
  const id = 'ppw-control-' + ++controlSequence;
  element.innerHTML = `<label for="${id}"></label><input id="${id}" type="range"><input type="number" class="ppw-param-number"><span class="ppw-input-error" role="status" hidden></span>`;
  const title = element.querySelector('label'); title.append(doc.createTextNode(label + ': '));
  const output = doc.createElement('span'); output.className = valueClass; title.append(output);
  const slider = element.querySelector('[type=range]'), number = element.querySelector('[type=number]'), error = element.querySelector('[role=status]');
  slider.className = sliderClass;
  slider.setAttribute('aria-label', label);
  number.setAttribute('aria-label', label + ' value');
  for (const input of [slider, number]) { input.min = min; input.max = max; input.step = step; }
  let timer, currentValue;
  function update(next) {
    slider.value = next;
    if (next !== currentValue || doc.activeElement !== number) number.value = next;
    if (next !== currentValue) { error.hidden = true; number.removeAttribute('aria-invalid'); }
    currentValue = next;
    output.textContent = Number(next).toFixed(3);
  }
  function commit(next) { error.hidden = true; number.removeAttribute('aria-invalid'); onChange(next); }
  scope.on(slider, 'input', () => {
    const next = Number(slider.value); number.value = next; output.textContent = next.toFixed(3);
    scope.cancel(timer);
    if (delay) timer = scope.later(() => commit(next), delay); else commit(next);
  });
  scope.on(number, 'change', () => {
    const next = Number(number.value);
    if (!number.value || !Number.isFinite(next) || next < Number(min) || next > Number(max)) {
      error.textContent = `Enter a number between ${min} and ${max}`;
      error.hidden = false; number.setAttribute('aria-invalid', 'true'); return;
    }
    scope.cancel(timer); slider.value = next; output.textContent = next.toFixed(3); commit(next);
  });
  update(value);
  return { element, slider, number, update };
}
function downloadBlob(doc, blob, filename) {
  const url = URL.createObjectURL(blob), anchor = doc.createElement('a');
  anchor.href = url; anchor.download = filename; doc.body.append(anchor); anchor.click(); anchor.remove();
  URL.revokeObjectURL(url);
}

function plotTheme(element) {
  const style = element.ownerDocument.defaultView.getComputedStyle(element);
  const read = (token, fallback) => style.getPropertyValue(token).trim() || fallback;
  return {
    background: read('--ppw-plot-bg', DEFAULT_PLOT_THEME.background),
    text: read('--ppw-text', DEFAULT_PLOT_THEME.text),
    muted: read('--ppw-muted', DEFAULT_PLOT_THEME.muted),
    grid: read('--ppw-grid', DEFAULT_PLOT_THEME.grid),
    cursor: read('--ppw-accent', DEFAULT_PLOT_THEME.cursor),
  };
}

// Drawing coordinates remain in CSS pixels; backing resolution follows devicePixelRatio.
function canvasSurface(canvas, scope, draw, { width = 500, ratio = 1, minHeight = 0 } = {}) {
  const ctx = canvas.getContext('2d');
  let w = width, h = Math.max(minHeight, width * ratio);
  function render() {
    const available = canvas.getBoundingClientRect().width;
    w = Math.max(180, available || width); h = Math.max(minHeight, w * ratio);
    const parent = canvas.parentElement;
    if (parent?.hasAttribute('data-fit-plot') && parent.clientHeight > 0) {
      const view = canvas.ownerDocument.defaultView, style = view.getComputedStyle(parent);
      const siblings = [...parent.children].filter(child => child !== canvas && !child.hidden);
      const occupied = siblings.reduce((sum, child) => {
        const css = view.getComputedStyle(child);
        return sum + child.getBoundingClientRect().height + (parseFloat(css.marginTop) || 0) + (parseFloat(css.marginBottom) || 0);
      }, 0);
      h = Math.max(100, parent.clientHeight - occupied - parseFloat(style.paddingTop || 0) - parseFloat(style.paddingBottom || 0) - 4);
    }
    const dpr = canvas.ownerDocument.defaultView.devicePixelRatio || 1;
    const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, w, h, plotTheme(canvas));
  }
  const Observer = canvas.ownerDocument.defaultView.ResizeObserver;
  if (Observer) { const observer = new Observer(render); observer.observe(canvas); if (canvas.parentElement) observer.observe(canvas.parentElement); scope.cleanup(() => observer.disconnect()); }
  const ThemeObserver = canvas.ownerDocument.defaultView.MutationObserver;
  if (ThemeObserver && canvas.ownerDocument.documentElement) {
    const themeObserver = new ThemeObserver(render);
    themeObserver.observe(canvas.ownerDocument.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'data-color-mode'] });
    scope.cleanup(() => themeObserver.disconnect());
  }
  return { render, get width() { return w; }, get height() { return h; } };
}
