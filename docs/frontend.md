# Frontend development

The public Python API remains `PhasePlaneWidget`, its linked `view(...)` outputs,
and `to_standalone_html(..., views=[...])`. The internal frontend now separates
experiment ownership from independently mounted panels. The JavaScript factory
interfaces described here are internal, not a separately versioned public API.

## Source layout

Edit `src/tvb_phaseplane/frontend/`, then run:

```sh
uv run python scripts/build_frontend.py
uv run python scripts/build_frontend.py --check
```

The build concatenates an explicit source list into `static/widget.js`. There
are no runtime imports, npm dependencies or network assets in an exported widget.
Npm dependencies are only for DOM/browser testing.

| Source | Responsibility |
| --- | --- |
| `engine.js` | Models, custom compilation and numerical algorithms |
| `experiment.js` | Derived-state caches and first-mount reset/reference identities |
| `controller.js` | One controller per canonical model: input subscriptions, computation, playback and asynchronous sweeps |
| `components.js` | DOM scopes, numeric controls, cleanup, responsive/high-DPI canvas surfaces |
| `control-panels.js` | Model, parameters, projection, limits, noise, layers, comparison and export |
| `plot-panels.js` | Phase plane, time series/playback and sweep interface |
| `editor-panel.js` | Debounced equation editor and validation |
| `plot-drawing.js` | DOM-independent canvas/SVG drawing, including frozen scenes |
| `presentations.js` | Experimental figure, bench, drawer, comparison lens and lesson compositions |
| `coordinates.js` | Plot coordinate transforms, axes and arrows |
| `view.js` | Resolve shared model, select/order factories, subscribe/update and dispose |

Vendor CAS and canvas-to-SVG code remain in separate source files.

## Panel contract

A panel factory receives `{model, controller, document, configuration}` and
returns `{element, update, dispose}`:

- Construct only the panel's own DOM. Do not query another panel's elements.
- `update()` reflects current experiment state without starting numerical work
  or rebuilding active inputs unnecessarily. The view subscribes once to the
  controller and updates its mounted panels.
- Use `controller.commit({...traits})` for changes, rather than calling numerical
  routines from event handlers. Input changes are coalesced once per experiment,
  not recomputed for every linked output.
- `dispose()` releases event listeners, timers and observers, and removes DOM.
  `panelScope()` and `mountedPanel()` implement this pattern. Rebuilt subcontrols
  should have their own replaceable scopes.
- A panel may have local draft state (such as an uncommitted sweep range). Other
  panels must not reset that state during an unrelated update.

The factory registry in `view.js` and `PhasePlaneWidget.PANEL_NAMES` must agree.
Only requested factories are invoked; a parameters-only view does not create
canvases, an editor, sweep controls, or playback handlers. An export-only view
uses the same drawing function as a phase-plane panel without needing a hidden
phase-plane canvas. List order is actual DOM/tab order.

## Experiment ownership

Anywidget outputs can expose per-view model proxies. Resolve them to the
canonical experiment through the widget manager before acquiring a controller.
The controller is reference-counted across mounted outputs and released after
the last view is disposed. Presentation remounts retain that controller.

Geometry and trajectory caches, and the reset baseline, remain associated with
the shared model. Reset still means **the first frontend mount's state**, not a
persistent Python constructor snapshot.

A sweep belongs to its initiating panel; removing that panel cancels its work.
Progress/results are shared with other sweep panels. Inputs invalidate a sweep,
exceptions restore an operable state, and old jobs cannot overwrite newer jobs.
Playback also belongs to its initiating time-series panel; seeking or reset
pauses it. Cursor updates redraw linked views without recomputing trajectories.

## Experimental presentations

The supplementary [View design lab](view-designs.md) exercises eleven compositions
through `widget.present("figure", parameters=["I"])` and the synced
`presentation` setting. `explore(...)` creates a notebook-ready experiment with
`view_switcher=True`; `widget.explore(...)` creates a linked switchable view.
The picker changes the presentation trait on its configuration model, not the
canonical experiment when mounted as a separate view. Static per-view exports
without a mutable configuration model keep that selection locally. The default remains `panels`, preserving existing views.
These layouts are experimental, rather than a general-purpose layout DSL.

`mountPresentation` groups the existing panel factories using the same model and
controller. Task changes and inspector selections dispose inactive panels instead
of hiding a catalogue. Expansion mounts another composition inside a native
modal dialog, using that same controller and reset baseline. It restores focus
when closed. Inspect is a modeless, read-only region dismissed by Escape or an
outside pointer event. Dialog/popover listener scopes are released on each close.

Wide stages use a 16:9 grid; narrow containers use a stacked fallback. Plot slots
marked `data-fit-plot` give their canvases the space remaining after titles,
legends and playback controls. Canvas surfaces observe both themselves and their
parent so drawer changes and workspace resizing redraw at the correct density.

Freeze scene captures deep copies of nullclines, slice equilibria, the full
trajectory and its parameters/projection/clamped-state metadata. The snapshot
lives in the shared `reference_scene`, `reference_trajectory` and
`reference_metadata` traits; `show_reference` temporarily hides it without
deleting it. Both phase plane and time series render frozen paths. Different
models/variable names/projections clear incompatible snapshots. SVG export uses
the same frozen geometry and clips it to the current axes; the bundled SVG
adapter has a small dash-array extension for stored paths.

### Scientific-task prototypes

`research-views.js` supplies the shared presentation labels and six new task
names. Small multiples use detached preview models with no controller or trait
subscriptions; initial-condition and ensemble paths call the existing simulator
without writing shared results. Preview horizons are capped at 120 time units,
with 3 tiles, 6 starts/cards and 2–8 ensemble members. Non-finite paths display
an explicit divergence error. Hidden/failed tile canvases are cleared.

`initial_conditions` and `observation_cards` are shared, synced Python traits and
standalone state. Evidence captures deep settings/scene snapshots, supports
explicit restore and JSON download, and never interpolates user labels as HTML.
Reset clears both collections, cancels active sweeps and clears scan results.
Preview options and prediction drafts are held in a WeakMap keyed by presentation
configuration, so separately configured outputs do not share reveal state. Prediction masks paths/references locally
and invalidates its reveal when experiment settings change.

`navigator-view.js` reuses phase/time/sweep panels. The navigation-enabled sweep
stores a completed scan in the frontend session, with an identity excluding only
the scanned parameter's value. This allows navigation without keeping a stale
scan after other settings change. Scan clicks, arrow keys and a range control
commit sampled values to the canonical experiment. The scan is sampled,
deterministic and slice-based, not a continuation/stability solver.

## UI and theme tokens

`comparison` holds keep/clear reference and reset; `export` holds SVG/TikZ only.
The full explorer includes both. Focused lessons can choose either independently:

```python
widget = PhasePlaneWidget(
    panels=["parameters", "phase_plane", "time_series", "comparison"],
    visible_parameters=["I"],
    model_name="fitzhugh_nagumo",
)
```

Panels use container-aware grid layout; narrow containers stack automatically.
Canvases draw in CSS-pixel coordinates with device-pixel backing resolution, so
click/keyboard coordinates and SVG exports do not depend on display density.
Arrow keys set initial conditions; Shift uses a larger step. Numeric controls
share bounded validation and accessible labels/status messages.

Override theme tokens on the widget root, for example in a containing lesson's
stylesheet:

```css
.lesson .ppw-widget {
  --ppw-bg: #17202b;
  --ppw-surface: #202c3a;
  --ppw-plot-bg: #14202e;
  --ppw-text: #eef4fa;
  --ppw-muted: #a7b6c9;
  --ppw-border: #405269;
  --ppw-grid: #334455;
  --ppw-accent: #7eb8f5;
  --ppw-error: #ffabab;
}
```

Plot backgrounds, text/grid/cursor colors and SVG exports use these tokens at
render time. Scientific layer colors remain consistent across figures. Changing
a stylesheet alone does not currently request a redraw: resize or interact with
the figure after a dynamic theme change.

## Tests

```sh
uv sync --extra dev
uv run pytest -q
npm ci
npm test
npx playwright install chromium
uv run npm run test:host
```

DOM tests mock canvas drawing and verify that absent components are not created,
linked outputs compute once, inputs retain state, and disposal removes work.
The Playwright suite starts a temporary JupyterLab and a plain static HTML
server, exercising real widget-manager communication and rendered plot changes.
It also tests panel order, keyboard controls, numeric errors, comparison/reset,
SVG download, custom editing, sweep results, a narrow viewport, high-DPI backing
canvases and plot-theme colors. The design-gallery checks cover shared state
across layouts, 16:9 bounds, frozen overlays and SVG dash arrays, inspector/drawer
lifecycle, lesson/sweep state, expansion and focus restoration. Notebook checks
also expand a figure and verify the resulting edits reach the Python kernel.
Jupyter's own HTML-file handler sandboxes downloads,
so standalone exports are tested on the plain static server instead.

Set `TT_SCREENSHOTS=/tmp/widget-ui` when running the host test to retain desktop
and narrow-layout screenshots. The regular user browser is not controlled.

Static HTML and JupyterLab are the current test targets. Other notebook hosts
will be revisited after the extraction and teaching UX have matured.
