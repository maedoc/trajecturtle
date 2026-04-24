# Phase Plane Widget

Interactive phase plane widget for neural mass models, built with [`anywidget`](https://anywidget.dev/) and rendered on HTML5 Canvas.

## What It Does

This library provides **client-side interactive phase plane visualizations** for 2D dynamical systems commonly used in computational neuroscience:

- **Click on the phase plane** to set initial conditions and watch the trajectory evolve
- **Drag parameter sliders** to see nullclines, fixed points, and vector fields update in real time
- **Run parameter sweeps** to detect bifurcations — fixed points, limit cycles, and bistability appear automatically
- **View time series** alongside the phase plane

All computation (ODE integration, fixed-point search, nullclines, regime classification) runs **in the browser** via JavaScript. No Python kernel is needed after the initial load.

## Quick Start

### Jupyter / VS Code

```bash
pip install phase-plane-widget
```

```python
from phase_plane_widget import PhasePlaneWidget

widget = PhasePlaneWidget(model_name="mpr")
widget
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
| **Click-to-set IC** | Click anywhere on the phase plane to launch a trajectory |
| **Real-time parameters** | Slider controls update the dynamics instantly (no server round-trip) |
| **Time series** | Simultaneous plot of state variables vs. time |
| **Bifurcation sweep** | Vary one parameter, detect fixed-point / limit-cycle regimes automatically |
| **Regime detection** | Auto-classifies: fixed point, limit cycle, or other |

## Supported Models

| Model | Variables | Key Parameters | Dynamics |
|-------|-----------|--------------|----------|
| [Wilson-Cowan](models.md#wilson-cowan) | `E`, `I` | aee, aei, aie, aii, Pe, Pi | E-I population rate model |
| [FitzHugh-Nagumo](models.md#fitzhugh-nagumo) | `v`, `w` | a, b, ε, I | Excitable neuron |
| [MPR (QIF)](models.md#mpr-quadratic-integrate-and-fire) | `r`, `v` | Δ, η̄, J, I | Firing-rate + mean voltage (Montbrió, Pazó & Roxin 2015) |

## Architecture

The widget has a **dual-mode architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                     Python (thin wrapper)                    │
│  • Model metadata (param names, defaults, state names)       │
│  • Initial state setup                                       │
│  • to_standalone_html() exporter                             │
└────────────────────────┬────────────────────────────────────┘
                         │  anywidget bridge  (traitlets)
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     JavaScript (full engine)                 │
│  • RK4 ODE solver                                            │
│  • Newton-Raphson fixed-point finder                       │
│  • Nullcline grid-search                                     │
│  • Regime detection (trajectory variance analysis)         │
│  • Parameter sweep engine                                  │
│  • HTML5 Canvas rendering                                    │
└─────────────────────────────────────────────────────────────┘
```

All heavy computation lives in the browser. The Python layer is a thin wrapper that sets up initial state and provides a programmatic API.

## Citation

If you use the MPR model, cite:

> Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028.
