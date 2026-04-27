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
# Using uv
uv venv
uv pip install -e .

# Or using pip
pip install -e .
```

For development with Jupyter:
```bash
uv pip install -e ".[dev]"
```

## Quick Start

### Jupyter / VS Code — Built-in Models

```python
from phase_plane_widget import PhasePlaneWidget

widget = PhasePlaneWidget()
widget  # display interactive widget
```

### Custom Model

```python
from phase_plane_widget import phase_plane

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
from phase_plane_widget import PhasePlaneWidget

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

- **Client-side computation** — All ODE solving, fixed-point finding, and rendering runs in JavaScript (no server round-trips)
- **Nerdamer CAS inlined** — ~100 KB minified core bundled into `widget.js`; custom SymPy models compile directly to JS
- **Dual-mode** — Jupyter widget (`anywidget`) + fully self-contained HTML (`to_standalone_html()`)
- **Safety guards** — NaN/Inf short-circuit, 50K step budget, exp clamping (`[-709, 709]`), computation budgets on nullclines and fixed-point search

## Documentation

- **[Live demos](https://maedoc.github.io/trajecturtle/)** — Embedded standalone widgets
- **[Deployment guide](https://maedoc.github.io/trajecturtle/deployment/)** — Jupyter, VS Code, standalone HTML, mkdocs
- **[Model reference](https://maedoc.github.io/trajecturtle/models/)** — Equations, parameters, dynamics
- **[API reference](https://maedoc.github.io/trajecturtle/api/)** — Auto-generated from docstrings
