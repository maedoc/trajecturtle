#!/usr/bin/env python3
"""Generate standalone HTML demos for the documentation site.

Usage:
    python scripts/generate_demos.py

Generates four demo widgets in docs/demos/:
    - wilson_cowan.html
    - fitzhugh_nagumo.html
    - mpr_bistable.html
    - mpr_limit_cycle.html
"""

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tvb_phaseplane import PhasePlaneWidget


def build_custom_demo():
    """Generate the interactive custom-model demo with the MPR (QIF) model
    pre-loaded and the live editor open by default."""
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

    out_dir = Path(__file__).parent.parent / "docs" / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "custom-model-demo.html"

    w.to_standalone_html(
        str(out),
        title="Phase Plane Widget — Custom Model Demo (MPR)",
        on_render_js="setTimeout(() => { document.querySelector('.ppw-editor-toggle')?.click(); }, 100);",
    )
    print(f"Generated {out.name} ({out.stat().st_size:,} bytes)")


DEMOS = {
    "wilson_cowan": {
        "params": {
            "aee": 12.0, "aei": 10.0, "aie": 10.0, "aii": 2.0,
            "Pe": -2.0, "Pi": -8.0,
            "ke": 1.0, "ki": 1.0,
            "thetae": 4.0, "thetai": 4.0,
        },
        "x0": 0.3,
        "y0": 0.1,
        "t_max": 100.0,
    },
    "fitzhugh_nagumo": {
        "params": {"a": 0.7, "b": 0.8, "epsilon": 0.08, "I": 0.5},
        "x0": -1.0,
        "y0": -0.5,
        "t_max": 100.0,
    },
    "mpr_bistable": {
        "params": {"delta": 1.0, "eta_bar": -5.0, "J": 15.0, "I": 0.0},
        "x0": 0.1,
        "y0": -2.0,
        "t_max": 100.0,
    },
    "mpr_limit_cycle": {
        "params": {"delta": 1.0, "eta_bar": -2.0, "J": 18.0, "I": 0.0},
        "x0": 0.5,
        "y0": -1.0,
        "t_max": 100.0,
    },
    "hindmarsh_rose": {
        "params": {"a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0, "I": 3.0, "r": 0.001, "s": 4.0, "x_r": -1.6},
        "x0": -1.5,
        "y0": -5.0,
        "t_max": 500.0,
    },
}


def build_bifurcation_demo():
    """Generate an interactive AUTO2000 WASM bifurcation continuation demo
    using the DBT fast-subsystem equations (exhibits Fold LP + Hopf HB +
    limit cycles + 2D cusp/Bogdanov-Takens structure)."""
    out_dir = Path(__file__).parent.parent / "docs" / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bifurcation_auto2000.html"

    dbt_spec = {
        "name": "DBT Fast Subsystem",
        "state_vars": {
            "x": {"default": 0.5, "range": [-2.0, 2.0]},
            "y": {"default": 0.0, "range": [-2.0, 2.0]},
        },
        "parameters": {
            "z":    {"default": -0.5, "range": [-2.0, 3.5], "step": 0.01},
            "w":    {"default": 0.2,  "range": [-0.5, 1.5], "step": 0.01},
            "mu2C": {"default": 1.0,  "range": [0.0, 3.0],  "step": 0.05},
            "mu1B": {"default": 1.0,  "range": [0.0, 3.0],  "step": 0.05},
            "nuC":  {"default": 1.0,  "range": [-2.0, 3.0], "step": 0.05},
            "b":    {"default": 1.5,  "range": [-3.0, 3.0], "step": 0.05},
        },
        "equations": {
            "x": "y",
            "y": "-x^3 + mu2C*w*x - mu1B*z - y*(nuC*w + b*x + x^2)",
        },
        "display": [0, 1],
        "integrator": "rk4",
    }

    widget = PhasePlaneWidget(
        model_spec=dbt_spec,
        x0=0.888,
        y0=0.0,
        t_max=50.0,
        presentation="bifurcation",
        bifurcation_config={
            "param1": "z",
            "observable": 0,
            "rl0": -2.0,
            "rl1": 3.5,
            "rl2_0": -0.1,
            "rl2_1": 1.2,
            "stopPeriod": 30,
            "ntst": 60,
            "ds": 0.02,
            "dsmax": 0.1,
            "nmx": 300,
        },
        bifurcation_presets=[
            {"label": "1D: Fold + Hopf in z", "param1": "z", "range": [-2.0, 3.5]},
            {"label": "1D: EP + Limit cycles in z", "param1": "z", "range": [-2.0, 3.5], "auto_switch_lc": True, "stop_period": 30},
            {"label": "2D: (z, w) Fold curve", "param1": "z", "range": [-2.0, 3.5], "param2": "w", "range2": [-0.1, 1.2], "auto_trace_2d": True, "view_2d": True},
            {
                "label": "2D: (z, w) Fold + Hopf + Homoclinic",
                "param1": "z",
                "range": [-2.0, 3.5],
                "param2": "w",
                "range2": [-0.1, 1.2],
                "xlim2d": [-0.6, 0.6],
                "ylim2d": [-0.05, 0.8],
                "stop_period": 30,
                "auto_trace_2d": "all",
                "auto_trace_homoclinic": True,
                "view_2d": True,
            },
        ],
    )
    widget.to_standalone_html(
        str(out),
        title="Trajecturtle — AUTO2000 Bifurcation Continuation",
        include_auto=True,
    )
    print(f"Generated {out.name} ({out.stat().st_size:,} bytes)")


def build_bifurcation_mpr_demo():
    """Generate a companion AUTO2000 bifurcation demo for the MPR (QIF) model."""
    out_dir = Path(__file__).parent.parent / "docs" / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bifurcation_mpr.html"

    widget = PhasePlaneWidget(
        model_name="mpr",
        params={"delta": 1.0, "eta_bar": -5.0, "J": 15.0, "I": 0.0},
        x0=0.5,
        y0=-0.3,
        t_max=100.0,
        presentation="bifurcation",
        bifurcation_config={
            "param1": "eta_bar",
            "observable": 0,
            "rl0": -15.0,
            "rl1": 2.0,
            "ds": 0.05,
            "dsmax": 0.2,
            "nmx": 300,
        },
        bifurcation_presets=[
            {"label": "1D: Fold in eta_bar", "param1": "eta_bar", "range": [-15.0, 2.0]},
            {"label": "2D: (eta_bar, J) Fold cusp", "param1": "eta_bar", "range": [-20.0, 5.0], "param2": "J", "range2": [0.0, 25.0], "auto_trace_2d": True},
        ],
    )
    widget.to_standalone_html(
        str(out),
        title="Trajecturtle — MPR Bifurcation Continuation",
        include_auto=True,
    )
    print(f"Generated {out.name} ({out.stat().st_size:,} bytes)")


def main():
    out_dir = Path(__file__).parent.parent / "docs" / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, config in DEMOS.items():
        model_name = name.split("_")[0] if name.startswith("mpr_") else name
        widget = PhasePlaneWidget(model_name=model_name)

        for k, v in config["params"].items():
            widget.params[k] = v
        widget.x0 = config["x0"]
        widget.y0 = config["y0"]
        widget.t_max = config["t_max"]

        out_path = out_dir / f"{name}.html"
        widget.to_standalone_html(str(out_path), title=f"Phase Plane: {name}")
        print(f"Generated {out_path.name} ({out_path.stat().st_size:,} bytes)")

    build_bifurcation_demo()
    build_bifurcation_mpr_demo()
    build_custom_demo()
    from build_design_gallery import build_design_gallery
    build_design_gallery()
    print("Done.")


if __name__ == "__main__":
    main()
