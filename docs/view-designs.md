---
hide:
  - toc
---

# View design lab — experimental

The components are reusable now, but a full catalogue of panels is not necessarily
a good interface. This supplementary lab explores **five task-oriented compositions**:
a notebook figure, a wide experiment bench, a temporary instrument drawer, a
comparison lens, and a guided lesson.

These are working alternatives, not a final design decision. They all reuse the
same plot/control factories and experiment controller. Within the lab below,
**switching designs preserves the current experiment**. Reset deliberately returns
to its initial state. Each separately opened lab starts its own experiment.

## Try it in a notebook

```python
from tvb_phaseplane import explore

lab = explore(
    "fitzhugh_nagumo", params={"I": 0.5},
    parameters=["I", "epsilon"], t_max=80,
)
lab
```

The widget includes a **View** picker for all five layouts. Changes to parameters,
initial conditions and frozen references survive switching. The picker is synced
to `lab.presentation`, so Python can also select a layout, for example
`lab.presentation = "bench"`. Reset changes the experiment, not the selected view.
Leaving a running sweep cancels it; completed results remain available.

For an existing built-in or custom experiment, display
`experiment.explore(parameters=["I"])` instead. This creates a separately
configurable view sharing that experiment. Existing fixed `present(...)` views
remain unchanged.

**[Download the feedback notebook](notebooks/view-feedback.ipynb){: download="view-feedback.ipynb" }**
contains a short test route and a feedback template. With the current checkout:

```sh
uv sync --extra dev
uv run jupyter lab docs/notebooks/view-feedback.ipynb
```

Colleagues need the current code installed in their notebook environment; these
new APIs are not assumed to be available in an older published release. There
is no automatic feedback collection or telemetry. For a colleague without a
Python environment, `lab.to_standalone_html("view-feedback.html")` exports the
same switchable widget for offline use. You can also
**[try the built-in view picker in your browser](demos/view-feedback.html)**
or [download that ready-made HTML example](demos/view-feedback.html){: download="view-feedback.html" }.

## Browser gallery

**[Open the lab at full browser width](demos/design-lab.html){: target="_blank" rel="noopener" }**
— recommended for judging the 16:9 compositions. The embedded lab is useful for
judging notebook-width layouts. On narrow containers, the bench switches to a
readable stacked fallback instead of squeezing plots and controls.

<iframe src="../demos/design-lab.html" title="Interactive view design gallery" width="100%" height="810" frameborder="0" style="border:1px solid #dce3eb; border-radius:12px;"></iframe>

## 1. Interactive figure: the notebook default

**[Try the figure](demos/design-lab.html?view=figure){: target="_blank" rel="noopener" }**

**Analogy:** a camera viewfinder, or an interactive figure in a paper. The thing
being observed dominates; only the most useful controls are permanently visible.

- Phase plane above a compact strip of two author-selected sliders (`I`, `epsilon`).
- Click the plot or use arrow keys to move the initial condition.
- **Inspect** opens a temporary readout of equations, parameters and equilibria.
  Click outside or press Escape to dismiss it.
- **Expand** opens the same experiment in a larger bench, not a copied experiment.
  Escape or Close workspace returns to the figure and restores keyboard focus.

**Try:** change `I`, expand, adjust another parameter in the inspector, then close
the workspace. The compact controls and plot should show that same change.

**Evaluate:** does this expose enough for a paragraph-sized notebook interaction?
Are the inspector and expansion actions discoverable without competing with the plot?

## 2. Experiment bench: a bounded 16:9 workspace

**[Try the bench](demos/design-lab.html?view=bench){: target="_blank" rel="noopener" }**

**Analogy:** a scientific instrument or an audio workstation. Keep the signal and
its most important controls visible; choose a tool instead of adding another row.

- Phase plane on the left; time series at upper right.
- One bounded inspector at lower right: model readout, all parameters, noise,
  plot layers, projection, axes/duration, sweep, editor or export.
- Primary sliders remain along the bottom; playback lives with the time series.
- Inspector content may scroll, but selecting another tool does not push the
  main plots off the stage. Selecting another tool disposes the previous tool;
  leaving a running sweep cancels it rather than hiding background work.

**Try:** choose Noise & integrator, enable noise and raise a strength; then inspect
Plot layers or Model & equilibria without losing the two plots. Use the time
slider to connect the trajectory marker to the time series.

**Evaluate:** is the balance between phase plane, time series and inspector useful?
Would you trade some inspector height for a taller time series?

## 3. Instrument drawer: controls only when needed

**[Try the drawer](demos/design-lab.html?view=drawer){: target="_blank" rel="noopener" }**

**Analogy:** map layers, a camera's quick-settings panel, or an audio effect's
expanded controls. Routine interaction stays simple; deeper adjustment is modeless.

- Normally a plot and the primary sliders.
- **Controls** opens an instrument strip beside the plot. Choose one control
  group, manipulate it while watching the plot, then close it.
- **Inspect** remains a lightweight read-only popover, distinct from sustained
  manipulation in the drawer.
- On narrow containers, the drawer moves below the plot instead of obscuring it.

**Try:** open Controls, select All parameters, adjust `a`, and close the drawer.
The plot keeps the result. Reopening controls does not reset the experiment.

**Evaluate:** is reclaiming that space worthwhile, or is the resize distracting?
Should the drawer be docked, overlay the plot, or become a small floating palette?
This prototype uses a docked strip so it never covers the plotted geometry.

## 4. Comparison lens: freeze a scene, then change it

**[Try the comparison lens](demos/design-lab.html?view=comparison){: target="_blank" rel="noopener" }**

**Analogy:** tracing paper, an oscilloscope's stored trace, or an image editor's
before/after overlay. The baseline is a **frozen scene**, not just an old line.

1. Click **Freeze scene** to capture the current nullclines, equilibria and full
   trajectory, together with parameter/projection metadata.
2. Change `I` or `epsilon`. Saturated colors update; dulled nullclines, ringed
   equilibria and dashed trajectories retain the frozen experiment. The time
   series also shows the stored trajectory in gray.
3. Toggle **Show frozen scene** to judge the live plot without the overlay.
4. **Replace frozen scene** takes a new snapshot. **Clear reference** removes it.

The overlay uses the current axes, so changing the window is a zoom/reframe,
not a recomputation of the baseline. Frozen geometry is clipped to the plotting
window. Changing the model, variable names or projection clears an incompatible
snapshot rather than drawing it against unrelated axes. A clamped-state change
can deliberately be compared with the previous slice; the captured clamped
values remain in the snapshot metadata.

**Try:** freeze at `I = 0.5`, reduce it toward zero, then toggle the stored scene.
Look for both the nullcline movement and the trajectory change. Parameters used
for the snapshot are printed beneath the controls.

**Evaluate:** is the muted overlay legible without dominating? Would a later
side-by-side or draggable before/after split be preferable for dense scenes?
The current prototype implements the overlay, not those additional variants.

## 5. Guided lesson: change tasks, not experiments

**[Try the lesson](demos/design-lab.html?view=lesson){: target="_blank" rel="noopener" }**

**Analogy:** a presentation stage or an instrument's operating modes. The author
makes the relevant task obvious while keeping unrelated controls out of the way.

- **Explore:** phase plane and primary sliders.
- **Observe:** phase plane plus time series/playback.
- **Compare:** the paired plots and frozen-scene actions.
- **Sweep:** only the sweep controls and result, with the other plots unmounted.

Short instructions change with the task. Parameters, frozen reference and
completed sweep results survive task switches. Leaving a running sweep cancels
it; switching back allows a fresh run. The lesson does not automatically change
parameters or silently pick a new initial condition.

**Try:** choose a parameter in Explore, freeze it in Compare, then move the slider
and visit Observe. Finally run a small sweep (e.g. 8 samples) and switch tasks
once it completes.

**Evaluate:** is the reduced cognitive load worth hiding simultaneous information?
Would an authored sequence of questions be better than freely selectable tasks?

## Reusing the experiments

The experimental layout API is intentionally small:

```python
from tvb_phaseplane import PhasePlaneWidget

experiment = PhasePlaneWidget(model_name="fitzhugh_nagumo", params={"I": 0.5})
display(experiment.present("figure", parameters=["I", "epsilon"]))
# Another output of that same experiment, not an independent copy:
display(experiment.present("comparison", parameters=["I"]))

experiment.to_standalone_html("bench.html", views=[{
    "presentation": "bench", "visible_parameters": ["I", "epsilon"],
}])
```

Valid experimental presentations are `figure`, `bench`, `drawer`, `comparison`
and `lesson`. Existing `panels=[...]` compositions remain available unchanged.
The public API and exact proportions may evolve based on this lab's feedback.
Other host environments remain deferred while we explore these compositions.
