# Phase Plane Widget

Interactive phase plane widget for neural mass models, usable in Jupyter notebooks, VS Code, and static HTML exports.

📖 **[Full Documentation & Live Demos](https://maedoc.github.io/trajecturtle/)**

🧪 **[Try the Custom Model Editor →](https://maedoc.github.io/trajecturtle/demos/custom-model-demo.html)** — pre-loaded with the MPR (QIF) model; tweak equations and watch the phase plane update in real time.

## Features

- **Interactive phase plane visualization** with nullclines, vector field, fixed points, and trajectories
- **N-dimensional state space** — project any 2 variables, clamp the rest with live sliders
- **Click-to-set initial conditions** directly on the phase plane
- **Real-time time series** display of all state variables
- **Stochastic integration** — Stratonovich Heun with per-variable noise strength sliders
- **Parameter sweeps** with bifurcation diagram visualization
- **Regime detection** — automatically classifies dynamics as fixed point, limit cycle, or other
- **Custom models** — define arbitrary ODEs in Python (SymPy), compile to JavaScript via inlined Nerdamer CAS
- **Standalone HTML** — fully self-contained exports (no server, no kernel)

## Installation

```bash
pip install tvb-phaseplane
```

For development with Jupyter:
```bash
pip install tvb-phaseplane[dev]
```

Or install from source:
```bash
# Using uv
uv venv
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

### Jupyter / VS Code — Built-in Models

```python
from tvb_phaseplane import PhasePlaneWidget

widget = PhasePlaneWidget()
widget  # display interactive widget
```

### A notebook explorer for trying the views

```python
from tvb_phaseplane import explore

lab = explore("fitzhugh_nagumo", params={"I": 0.5}, parameters=["I", "epsilon"])
lab  # display in a notebook cell
```

The built-in **View** picker switches between all eleven experimental layouts
without resetting the experiment. `lab.presentation = "bench"` selects a layout
from Python. For an existing experiment, use `experiment.explore()` to create a
linked, switchable output. You can share an offline version with
`lab.to_standalone_html("view-feedback.html")`.

A [feedback notebook](docs/notebooks/view-feedback.ipynb) provides a short trial
and a feedback template for colleagues. It requires the current checkout installed
in the notebook environment; no feedback is sent automatically.

### Composable teaching figures

The supplementary [View design lab](docs/view-designs.md) includes working
prototypes of a compact interactive figure, 16:9 experiment bench, instrument
drawer, frozen-scene comparison lens and guided lesson, plus small multiples,
a bifurcation-scan navigator, initial-condition explorer, evidence notebook,
prediction/reveal and seeded stochastic ensembles. These experimental
presentations share the same experiment and controls instead of stacking every
panel into one long output.

Select only the panels needed for a lesson. The same configuration works in the offline HTML export:

```python
widget = PhasePlaneWidget(
    model_name="fitzhugh_nagumo",
    panels=["parameters", "phase_plane", "time_series"],
    visible_parameters=["I"],
)
widget.to_standalone_html("excitability.html")
```

In notebooks, display separate views of the same experiment in different cells:

```python
display(widget.view(["parameters"], parameters=["I"]))
# In another cell, after explanatory text:
display(widget.view(["phase_plane"]))
display(widget.view(["time_series"]))
```

Available panels: `model`, `parameters`, `projection`, `limits`, `noise`,
`layers`, `editor`, `comparison`, `export`, `phase_plane`, `time_series`, `sweep`.
Panels mount in the supplied order; unrequested panels are not instantiated.
Use `panel_layout="vertical"` (or `layout="vertical"` on a view) to stack panels.
Horizontal layouts automatically stack on narrow containers. Omitting panels
preserves the full explorer.

The time-series panel includes playback and a time slider linked to a marker on
the phase plane. The `comparison` panel offers a dashed reference trajectory and
reset independently of `export` (which contains SVG/TikZ downloads only).
The noise panel offers a seed and a
new-realization button; a fixed seed produces the same additive-noise path for
the same initial conditions and integration settings.

Reset restores the experiment state captured by its first browser mount, shared
by all its views. It is not a persistent constructor-default snapshot: reloading
the page establishes a new baseline. Sweeps are owned by the view that starts
them and are cancelled when that view is disposed or its inputs change.
Compilation failures are shown as errors; another model is never silently used
in place of an invalid custom model. Phase-plane canvases support arrow-key
initial conditions (Shift for larger steps), adapt to their container and render
at device-pixel resolution. Numeric controls report invalid values without
changing the experiment.

To export separate linked views in a single HTML page:

```python
widget.to_standalone_html("lesson.html", views=[
    {"panels": ["parameters"], "visible_parameters": ["I"]},
    {"panels": ["phase_plane", "time_series"]},
])
```

### Custom Model

```python
from tvb_phaseplane import phase_plane

# Define any ODE system
pp = phase_plane(
    equations=["a*x - x**3 - y", "x - b*y"],
    state_vars={"x": (-3, 3), "y": (-3, 3)},
    params={"a": (0.7, 0, 2), "b": (0.8, 0, 2)},
)
pp  # Nerdamer-compiled, runs entirely in the browser
```

### 3-D Projection

```python
pp = phase_plane(
    equations=["a*x - x**3 - y", "x - b*y", "c*(x - z)"],
    state_vars={"x": (-3, 3), "y": (-3, 3), "z": (0, 5)},
    params={"a": (0.7, 0, 2), "b": (0.8, 0, 2), "c": (10, 0, 20)},
    display=["x", "y"],  # project onto x-y plane
)
```

### Stochastic Dynamics

```python
pp = phase_plane(
    equations=["a*x - x**3 - y", "x - b*y"],
    state_vars={"x": (-3, 3), "y": (-3, 3)},
    params={"a": (0.7, 0, 2), "b": (0.8, 0, 2)},
    integrator="heun",
    noise_per_var=[0.1, 0.05],
)
```

### Standalone HTML Export

```python
from tvb_phaseplane import PhasePlaneWidget

widget = PhasePlaneWidget()
widget.to_standalone_html("widget.html", title="My Model")
```

The resulting `.html` file is fully self-contained — it bundles the Nerdamer CAS, all built-in models, the computation engine, CSS, and current widget state. Works in any modern browser with **no Python runtime and no Jupyter kernel**.

## Built-in Models

| Model | State Vars | Parameters | Key Dynamics |
|-------|-----------|------------|-------------|
| **Wilson-Cowan** | E, I | aee, aei, aie, aii, Pe, Pi, ke, ki, thetae, thetai | E-I population oscillations, bistability |
| **FitzHugh-Nagumo** | v, w | a, b, epsilon, I | Excitable spiking, limit cycles |
| **MPR (QIF)** | r, v | delta, eta_bar, J, I | Mean-field firing rate, macroscopic chaos |

## Architecture

Frontend sources live in `src/tvb_phaseplane/frontend/`. `engine.js` contains
numerics; `experiment.js` caches derived state; `controller.js` owns computation,
playback and sweeps once per shared model. `view.js` mounts only requested panel
factories with explicit `update`/`dispose` lifecycles. Control, plot and editor
panels live separately, sharing DOM/numeric-input/canvas primitives in
`components.js`. DOM-independent drawing functions also serve SVG exports.
Run `python scripts/build_frontend.py` after editing these sources; `--check`
verifies the checked-in bundle matches. The assembled `static/widget.js` stays
import-free for offline embedding. See [frontend development](docs/frontend.md)
for the component contract and theme tokens.

- **Client-side computation** — All ODE solving, fixed-point finding, and rendering runs in JavaScript (no server round-trips)
- **Nerdamer CAS inlined** — ~100 KB minified core bundled into `widget.js`; custom SymPy models compile directly to JS
- **Dual-mode** — Jupyter widget (`anywidget`) + fully self-contained HTML (`to_standalone_html()`)
- **Safety guards** — NaN/Inf short-circuit, 50K step budget, exp clamping (`[-709, 709]`), computation budgets on nullclines and fixed-point search

## Development and tests

```sh
uv sync --extra dev
uv run pytest -q
npm ci
npm test
uv run python scripts/build_frontend.py --check
```

Node dependencies are for tests only; building/exporting the widget does not
require npm. DOM tests use a mock canvas and do not validate rendered pixels.
For real JupyterLab/kernel/widget-manager and standalone browser checks:

```sh
npx playwright install chromium
uv run npm run test:host
```

The host test starts a temporary loopback-only JupyterLab, exercises linked views,
Python/JavaScript synchronization, output remounting, custom seeded noise and
standalone plot updates, then removes its temporary notebook and stops the server.
Use `PYTHON=/path/to/python npm run test:host` to select another environment with
JupyterLab and the package dependencies. CI tests both anywidget 0.9.0 and the
current release. VS Code, classic Notebook and JupyterLite still need separate
host verification; see the deployment guide.

## Documentation

- **[Live demos](https://maedoc.github.io/trajecturtle/)** — Embedded standalone widgets
- **[Deployment guide](https://maedoc.github.io/trajecturtle/deployment/)** — Jupyter, VS Code, standalone HTML, mkdocs
- **[Model reference](https://maedoc.github.io/trajecturtle/models/)** — Equations, parameters, dynamics
- **[API reference](https://maedoc.github.io/trajecturtle/api/)** — Auto-generated from docstrings
