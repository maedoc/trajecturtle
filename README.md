# Phase Plane Widget

Interactive phase plane widget for neural mass models, usable in Jupyter notebooks, VS Code, and static HTML exports.

📖 **[Full Documentation & Live Demos](https://maedoc.github.io/trajecturtle/)**

## Features

- **Interactive phase plane visualization** with nullclines, vector field, fixed points, and trajectories
- **Click-to-set initial conditions** directly on the phase plane
- **Real-time time series** display alongside the phase plane
- **Parameter sweeps** with bifurcation diagram visualization
- **Regime detection**: automatically classifies dynamics as fixed point, limit cycle, or other
- **Multiple neural mass models**: Wilson-Cowan (E-I populations), FitzHugh-Nagumo (excitable neuron), MPR (QIF firing-rate)

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

## Usage

### Jupyter Notebook

```python
from phase_plane_widget import PhasePlaneWidget

widget = PhasePlaneWidget()
widget
```

### Programmatic Parameter Sweep

```python
# Run a sweep over the external current I
widget.run_sweep("I", [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
```

### VS Code

Works directly in VS Code's Jupyter extension — no extra configuration needed.

### Static HTML Export

```python
from ipywidgets.embed import embed_minimal_html

embed_minimal_html("export.html", views=[widget], title="Phase Plane Widget")
```

## Supported Models

| Model | Description | Parameters |
|-------|-------------|------------|
| **Wilson-Cowan** | Excitatory-inhibitory population dynamics | aee, aei, aie, aii, Pe, Pi, ke, ki, thetae, thetai |
| **FitzHugh-Nagumo** | Excitable neuron, simplified Hodgkin-Huxley | a, b, epsilon, I |

## Architecture

- **Client-side computation**: All ODE solving, fixed-point finding, and rendering runs in JavaScript — no server round-trips
- **Dual-mode architecture**: Works as a Jupyter widget *and* as a standalone self-contained HTML file
- **anywidget**: Jupyter/VS Code bridge; `to_standalone_html()` for blogs, docs, and courses

## Documentation

See the [full documentation site](https://maedoc.github.io/trajecturtle/) for:

- **Live interactive demos** (embedded standalone widgets)
- [Deployment guide](https://maedoc.github.io/trajecturtle/deployment/) — Jupyter, VS Code, standalone HTML, mkdocs
- [Model reference](https://maedoc.github.io/trajecturtle/models/) — equations, parameters, dynamics
- [API reference](https://maedoc.github.io/trajecturtle/api/) — auto-generated from docstrings
