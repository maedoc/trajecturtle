function mountNavigatorView(context) {
  const { document: doc } = context, scope = panelScope();
  const element = panelElement(doc, 'ppw-research-view ppw-navigator-view');
  const note = doc.createElement('p'); note.className = 'ppw-research-note';
  note.textContent = 'Sampled deterministic regime scan, not numerical continuation. Equilibria refer to the displayed clamped slice, not full-system stability. Select a sampled value to update both live plots. A completed scan survives changes to its scanned parameter; other experiment changes invalidate it.';
  const grid = panelElement(doc, 'ppw-navigator-grid'); element.append(note, grid);
  const components = [mountSweep({ ...context, navigation: true }), mountPhasePlane(context), mountTimeSeries(context)];
  for (const component of components) { component.element.classList.add('ppw-panel'); component.element.setAttribute('data-fit-plot', ''); grid.append(component.element); }
  scope.cleanup(() => { for (const component of components) component.dispose(); });
  return mountedPanel(element, scope, () => { for (const component of components) component.update(); });
}
