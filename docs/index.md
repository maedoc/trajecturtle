# Phase Plane Widget

Interactive phase plane widget for dynamical systems, built with [`anywidget`](https://anywidget.dev/) and rendered on HTML5 Canvas. All computation runs **client-side** — no server round-trips after the initial load.

## What It Does

Explore dynamical systems in real time, directly in the browser — in Jupyter, VS Code, or static HTML pages:

- **Click on the phase plane** to set initial conditions and watch the trajectory evolve
- **Drag parameter sliders** to see nullclines, fixed points, and vector fields update instantly
- **Switch between state variables** for N-dimensional systems — clamp the rest with sliders
- **Add noise** to trajectories with the stochastic Heun integrator
- **Run parameter sweeps** to detect bifurcations — fixed points, limit cycles, and bistability appear automatically
- **Define your own models** in Python (SymPy) and they compile to JavaScript automatically
- **Export to standalone HTML** — works offline, no kernel, no server

## Quick Start

### Jupyter / VS Code

```bash
pip install tvb_phaseplane
```

```python
from tvb_phaseplane import PhasePlaneWidget

widget = PhasePlaneWidget(model_name="mpr")
widget
```

### Custom Model

```python
from tvb_phaseplane import phase_plane

pp = phase_plane(
    equations=["a*x - x**3 - y", "x - b*y"],
    state_vars={"x": (-3, 3), "y": (-3, 3)},
    params={"a": (0.7, 0, 2), "b": (0.8, 0, 2)},
)
pp  # fully interactive, compiled in the browser via Nerdamer
```

### 3-D Projection

```python
pp = phase_plane(
    equations=["a*x - x**3 - y", "x - b*y", "c*(x - z)"],
    state_vars={"x": (-3, 3), "y": (-3, 3), "z": (0, 5)},
    params={"a": (0.7, 0, 2), "b": (0.8, 0, 2), "c": (10, 0, 20)},
    display=["x", "y"],
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

### Static HTML (no kernel)

```python
widget.to_standalone_html("widget.html")
# Open widget.html in any browser — works offline
```

## Features

| Feature | Description |
|---------|-------------|
| **Phase plane** | Nullclines, vector field, fixed points, trajectories on HTML5 Canvas |
| **N-dimensional** | Project any 2 variables; sliders clamp the rest |
| **Click-to-set IC** | Click anywhere on the phase plane to launch a trajectory |
| **Real-time parameters** | Slider controls update dynamics instantly |
| **Time series** | Simultaneous overlay of all state variables vs. time |
| **Stochastic integration** | Stratonovich Heun with per-variable noise sliders |
| **Bifurcation sweep** | Vary one parameter, detect regimes automatically |
| **Regime detection** | Auto-classifies: fixed point, limit cycle, or other |
| **Custom models** | Define ODEs in SymPy → compile to JS via inlined Nerdamer CAS |
| **Standalone HTML** | Self-contained `.html` export, no Python runtime |
| **Safety guards** | NaN/Inf short-circuit, step budgets, exp clamping |

## Supported Models

| Model | Variables | Key Parameters | Dynamics |
|-------|-----------|--------------|----------|
| [Wilson-Cowan](models.md#wilson-cowan) | `E`, `I` | aee, aei, aie, aii, Pe, Pi | E-I population rate model |
| [FitzHugh-Nagumo](models.md#fitzhugh-nagumo) | `v`, `w` | a, b, ε, I | Excitable neuron |
| [MPR (QIF)](models.md#mpr-quadratic-integrate-and-fire) | `r`, `v` | Δ, η̄, J, I | Firing-rate + mean voltage (Montbrió, Pazó & Roxin 2015) |
| [Custom](deployment.md#custom-models) | Arbitrary | Arbitrary | Any ODE system via SymPy / Nerdamer |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Python (thin wrapper)                    │
│  • Model metadata, initial state, standalone exporter         │
│  • ModelSpec validation and SymPy → JS transpiler            │
└────────────────────────┬────────────────────────────────────┘
                         │  anywidget bridge  (traitlets)
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     JavaScript (full engine)                 │
│  • RK4 (deterministic) + Heun (stochastic) integrators         │
│  • Nerdamer CAS inlined — compiles custom models client-side │
│  • Newton-Raphson fixed-point finder + budget guards         │
│  • Nullcline grid-search with computation budgets            │
│  • Regime detection (trajectory variance analysis)           │
│  • Parameter sweep engine                                    │
│  • HTML5 Canvas rendering                                    │
└─────────────────────────────────────────────────────────────┘
```

All heavy computation lives in the browser. The Python layer is a thin wrapper that sets up initial state and provides a programmatic API.

## Citation

If you use the MPR model, cite:

> Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028.
