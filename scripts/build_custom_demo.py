#!/usr/bin/env python3
"""Build a standalone HTML demo page that pre-loads the MPR (QIF) model
as an editable custom model with the live editor open by default.

Usage:
    uv run python scripts/build_custom_demo.py

Output is written to ``docs/custom-model-demo.html``.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from tvb_phaseplane import PhasePlaneWidget


def build():
    # MPR (QIF) model spec with nerdamer-friendly equations.
    # Nerdamer understands ``pi`` as a named constant and the transpiler
    # maps it to ``Math.PI`` in the generated JS function body.
    spec = {
        "name": "custom",
        "state_vars": {
            "r": {"default": 0.5, "range": [0.0, 2.0]},
            "v": {"default": 0.0, "range": [-4.0, 2.0]},
        },
        "parameters": {
            "delta":   {"default": 1.0,  "range": [0.01, 5.0],  "step": 0.01},
            "eta_bar": {"default": -5.0, "range": [-20.0, 10.0], "step": 0.1},
            "J":       {"default": 15.0, "range": [-20.0, 30.0], "step": 0.1},
            "I":       {"default": 0.0,  "range": [-10.0, 10.0], "step": 0.1},
        },
        "equations": {
            "r": "delta/pi + 2*r*v",
            "v": "v^2 + eta_bar + J*r + I - (pi*r)^2",
        },
        "display": [0, 1],
        "integrator": "rk4",
    }

    w = PhasePlaneWidget()
    w.model_spec = spec
    w.model_name = "custom"
    # Sync param info so sliders populate correctly
    w.param_info = {
        "delta":   [0.01, 5.0, 1.0,  "Half-width Δ"],
        "eta_bar": [-20.0, 10.0, -5.0, "Mean excitability η̄"],
        "J":       [-20.0, 30.0, 15.0, "Synaptic coupling J"],
        "I":       [-10.0, 10.0, 0.0,  "External input I"],
    }
    w.state_names = ["r", "v"]
    w.params = {
        "delta": 1.0,
        "eta_bar": -5.0,
        "J": 15.0,
        "I": 0.0,
    }
    w.xlim = [0.0, 2.0]
    w.ylim = [-4.0, 2.0]
    w.display = [0, 1]
    w.clamped = [0.5, 0.0]
    w.x0 = 0.5
    w.y0 = -1.0

    docs_dir = pathlib.Path(__file__).parent.parent / "docs" / "demos"
    docs_dir.mkdir(exist_ok=True)
    out = docs_dir / "custom-model-demo.html"

    # Auto-open the editor 100 ms after render so the user sees the
    # equation textareas immediately.
    w.to_standalone_html(
        out,
        title="Phase Plane Widget — Custom Model Demo (MPR)",
        on_render_js="setTimeout(() => { document.querySelector('.ppw-editor-toggle')?.click(); }, 100);",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
