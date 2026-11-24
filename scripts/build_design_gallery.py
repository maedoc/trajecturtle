"""Build the supplementary, interactive view-design lab (no runtime dependencies)."""
from pathlib import Path

from tvb_phaseplane import PhasePlaneWidget

GALLERY_JS = r"""
const modes = {
  figure: ['Interactive figure', 'A notebook-sized viewfinder with two sliders. Inspect for a quick readout; Expand for the same experiment in a bench.'],
  bench: ['16:9 experiment bench', 'Phase plane and time series remain visible. Choose one advanced tool in the bounded inspector instead of stacking all tools.'],
  drawer: ['Instrument drawer', 'Controls opens an instrument strip beside the plot. Close it to return to an uncluttered figure.'],
  comparison: ['Comparison lens', 'Freeze the scene, change I or epsilon, and compare muted nullclines, equilibrium rings and dashed paths with the live colors.'],
  lesson: ['Guided lesson', 'Explore, Observe, Compare and Sweep change the task, not the underlying experiment. Completed results and parameters survive task switches.'],
  multiples: ['Small multiples', 'Compare three parameter values with linked axes and initial conditions. Use a tile explicitly to make it the live experiment.'],
  navigator: ['Bifurcation navigator', 'Run a sampled scan, then click or navigate its parameter values to update the live phase plane and time series. This is not numerical continuation.'],
  basin: ['Initial-condition explorer', 'Click to add several starts at fixed parameters. Compare labeled trajectories; this is not an automatic basin classification.'],
  evidence: ['Evidence notebook', 'Capture labeled scenes, preview them without changing the experiment, restore explicitly, or download a reproducible JSON record.'],
  prediction: ['Prediction & reveal', 'Predict an outcome from the nullclines and initial condition, then reveal the deliberately hidden trajectory.'],
  ensemble: ['Stochastic ensemble', 'Compare 2–8 seeded Heun traces, then explicitly use a member. These are individual realizations, not confidence intervals.'],
  bifurcation: ['AUTO2000 bifurcation', 'Client-side numerical continuation in WebAssembly (1D equilibria, HB -> LC limit cycles, and 2D LP/HB curves).'],
};
const style = document.createElement('style');
style.textContent = `
body { margin: 0; padding: 10px; background: #f0f4f8; font-family: system-ui, sans-serif; color: #253247; }
#ppw-root { max-width: 1100px; margin: 0 auto; }
.ppw-presentation-root { background: transparent; }
.design-lab-header { max-width: 1090px; margin: 0 auto 8px; }
.design-lab-header h1 { font-size: 18px; margin: 2px 0 8px; }
.design-lab-nav { display: flex; flex-wrap: wrap; gap: 5px; }
.design-lab-nav button { border: 1px solid #c9d5e3; border-radius: 6px; padding: 6px 9px; cursor: pointer; background: white; color: #253247; }
.design-lab-nav button[aria-pressed=true] { background: #245c9b; color: white; border-color: #245c9b; }
.design-lab-description { font-size: 12px; margin: 7px 0 4px; line-height: 1.5; }
.design-lab-note { font-size: 11px; color: #5f6c7b; margin: 4px 0; }
`;
document.head.append(style);
const header = document.createElement('header'); header.className = 'design-lab-header';
header.innerHTML = '<h1>View design lab</h1><nav class="design-lab-nav" aria-label="View design"></nav><p class="design-lab-description"></p><p class="design-lab-note">One live experiment across all eleven designs. Switching a design preserves parameters and frozen references; Reset restores the starting experiment. These are working prototypes, not a settled layout API.</p>';
document.body.prepend(header);
const nav = header.querySelector('nav');
for (const [value, [label]] of Object.entries(modes)) {
  const button = document.createElement('button'); button.textContent = label; button.dataset.view = value;
  button.addEventListener('click', () => select(value)); nav.append(button);
}
function select(value) {
  if (!modes[value]) value = 'figure';
  mockModel.set('presentation', value);
  for (const button of nav.children) button.setAttribute('aria-pressed', String(button.dataset.view === value));
  header.querySelector('.design-lab-description').textContent = modes[value][1];
  const url = new URL(location.href); url.searchParams.set('view', value); history.replaceState(null, '', url);
}
const query = new URL(location.href).searchParams;
if (query.get('embed') === '1') header.hidden = true;
select(query.get('view') || 'figure');
"""


def build_design_gallery(output=None):
    output = Path(output) if output else Path(__file__).resolve().parents[1] / "docs/demos/design-lab.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    widget = PhasePlaneWidget(
        model_name="fitzhugh_nagumo", params={"I": .5}, x0=-1., y0=-.5,
        t_max=80., presentation="figure", visible_parameters=["I", "epsilon"],
    )
    widget.to_standalone_html(output, title="Trajecturtle — View design lab", on_render_js=GALLERY_JS, include_auto=True)
    print(f"Generated {output.name} ({output.stat().st_size:,} bytes)")
    # The notebook's built-in picker, without the gallery-specific navigation.
    widget.view_switcher = True
    feedback = output.with_name("view-feedback.html")
    widget.to_standalone_html(feedback, title="Trajecturtle — Switchable explorer", include_auto=True)
    print(f"Generated {feedback.name} ({feedback.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_design_gallery()
