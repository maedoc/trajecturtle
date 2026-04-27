"""Interactive phase plane widget for neural mass models.

Two usage modes:
  1. Jupyter / VS Code  – anywidget wrapper; JS computes everything client-side.
  2. Standalone HTML    – export via ``to_standalone_html()`` for mkdocs,
     GitHub Pages, or any static site (no kernel, no Python runtime).
"""

import json
import pathlib

import anywidget
import numpy as np
import traitlets

_STATIC_DIR = pathlib.Path(__file__).parent / "static"


class PhasePlaneWidget(anywidget.AnyWidget):
    """Interactive phase plane widget.

    In Jupyter / VS Code the widget is backed by anywidget.  All numerical
    work (ODE integration, fixed-point search, nullclines, sweeps) is done
    in the browser by the JS front-end, so interactivity is instantaneous.

    For static sites use :meth:`to_standalone_html` to obtain a self-contained
    ``.html`` file that can be dropped into mkdocs, GitHub Pages, etc.
    """

    # Inline JS/CSS so the widget is self-contained and standalone HTML exports work.
    _esm = (_STATIC_DIR / "widget.js").read_text()
    _css = (_STATIC_DIR / "widget.css").read_text()

    # ── Initial state (synced to JS front-end) ──
    model_name = traitlets.Unicode("wilson_cowan").tag(sync=True)
    params = traitlets.Dict({}).tag(sync=True)
    param_info = traitlets.Dict({}).tag(sync=True)
    state_names = traitlets.List(["x", "y"]).tag(sync=True)

    x0 = traitlets.Float(0.1).tag(sync=True)
    y0 = traitlets.Float(0.1).tag(sync=True)

    xlim = traitlets.List([-0.5, 1.5]).tag(sync=True)
    ylim = traitlets.List([-0.5, 1.5]).tag(sync=True)
    t_max = traitlets.Float(100.0).tag(sync=True)

    # Pre-computed data (populated by JS, kept for inspection / export)
    nullcline_x = traitlets.List([]).tag(sync=True)
    nullcline_y = traitlets.List([]).tag(sync=True)
    vector_field = traitlets.List([]).tag(sync=True)
    fixed_points = traitlets.List([]).tag(sync=True)
    trajectory = traitlets.List([]).tag(sync=True)

    sweep_results = traitlets.List([]).tag(sync=True)
    sweep_fixed_points = traitlets.List([]).tag(sync=True)
    sweep_running = traitlets.Bool(False).tag(sync=True)

    # Display toggles
    show_nullclines = traitlets.Bool(True).tag(sync=True)
    show_vector_field = traitlets.Bool(True).tag(sync=True)
    show_trajectory = traitlets.Bool(True).tag(sync=True)
    show_fixed_points = traitlets.Bool(True).tag(sync=True)

    # Integrator / noise
    integrator = traitlets.Unicode("rk4").tag(sync=True)
    noise_enable = traitlets.Bool(False).tag(sync=True)
    noise_sigma = traitlets.List([]).tag(sync=True)

    # Custom model specification (JSON-serialisable dict for JS)
    model_spec = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)

    # Display indices for multi-variable projections
    display = traitlets.List([0, 1]).tag(sync=True)

    # Clamped values for non-displayed state variables
    clamped = traitlets.List(default_value=None, allow_none=True).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_model()

    def _get_model(self):
        from .models import MODEL_REGISTRY

        cls = MODEL_REGISTRY.get(self.model_name, MODEL_REGISTRY["wilson_cowan"])
        return cls()

    def _update_model(self):
        """Push model metadata to the JS front-end."""
        model = self._get_model()
        self.param_info = model.param_info
        self.state_names = model.state_names
        self.params = {k: v[2] for k, v in model.param_info.items()}
        self.xlim = model.default_xlim
        self.ylim = model.default_ylim

    @traitlets.observe("model_name")
    def _on_model_change(self, change):
        if self.model_name != "custom":
            self._update_model()

    def set_model_spec(self, spec: dict):
        """Load a custom model from a ``ModelSpec`` dict.

        Parameters
        ----------
        spec : dict
            JSON-serialisable model specification (see
            :meth:`ModelSpec.to_widget_state`).
        """
        self.model_spec = spec
        self.model_name = "custom"
        # Derive initial params / limits from the spec so the widget
        # has sensible defaults before JS takes over.
        params = {n: v["default"] for n, v in spec.get("parameters", {}).items()}
        self.params = params
        # Sync param_info for the existing slider infrastructure
        param_info = {}
        for n, v in spec.get("parameters", {}).items():
            lo, hi = v["range"]
            step = v.get("step", (hi - lo) / 500)
            param_info[n] = [lo, hi, v["default"], f"Parameter {n}"]
        self.param_info = param_info
        state_names = list(spec.get("state_vars", {}).keys())
        self.state_names = state_names
        # Sync display indices
        display = spec.get("display", [0, min(1, len(state_names) - 1)])
        self.display = display
        # Set default display limits from state variable ranges
        state_vars = spec.get("state_vars", {})
        if state_names:
            first = state_names[0]
            lo, hi = state_vars[first]["range"]
            self.xlim = [lo, hi]
            self.x0 = state_vars[first]["default"]
        if len(state_names) > 1:
            second = state_names[display[1]] if len(display) > 1 else state_names[1]
            lo, hi = state_vars[second]["range"]
            self.ylim = [lo, hi]
            self.y0 = state_vars[second]["default"]
        # Initialize clamped values for non-displayed vars
        n = len(state_names)
        clamped = []
        for i, name in enumerate(state_names):
            if i in display:
                clamped.append(None)  # displayed vars are not clamped
            else:
                lo, hi = state_vars[name]["range"]
                clamped.append((lo + hi) / 2.0)
        self.clamped = clamped

    # ── Python-side helpers (for programmatic use / validation) ──

    def run_sweep(self, param_name: str, values: list):
        """Run a parameter sweep from Python (delegates to JS in the widget).

        Parameters
        ----------
        param_name : str
            Parameter to vary.
        values : list of float
            Values to evaluate.
        """
        # The JS front-end handles sweeps interactively.  This method is a
        # convenience for programmatic access; it simply ensures the sweep
        # traitlets are in a consistent state.  For actual computation in
        # a headless environment use the model classes in ``models.py``.
        pass  # sweeps are computed client-side by the JS front-end

    # ── Standalone HTML export ──

    def to_standalone_html(
        self,
        filename: str | pathlib.Path,
        title: str = "Phase Plane Widget",
    ):
        """Export the widget to a self-contained HTML file.

        The resulting ``.html`` file contains the full JS computation engine,
        all model definitions, the CSS, and the current widget state.  It works
        in any modern browser with **no Python runtime and no Jupyter kernel**.

        Parameters
        ----------
        filename : str or pathlib.Path
            Output path (e.g. ``"widget.html"``).
        title : str
            Page ``<title>``.
        """
        js_code = self._esm
        css_code = self._css

        state = {
            "model_name": self.model_name,
            "params": self.params,
            "param_info": self.param_info,
            "state_names": self.state_names,
            "x0": self.x0,
            "y0": self.y0,
            "xlim": self.xlim,
            "ylim": self.ylim,
            "t_max": self.t_max,
            "show_nullclines": self.show_nullclines,
            "show_vector_field": self.show_vector_field,
            "show_trajectory": self.show_trajectory,
            "show_fixed_points": self.show_fixed_points,
            "nullcline_x": self.nullcline_x,
            "nullcline_y": self.nullcline_y,
            "vector_field": self.vector_field,
            "fixed_points": self.fixed_points,
            "trajectory": self.trajectory,
            "sweep_results": self.sweep_results,
            "sweep_fixed_points": self.sweep_fixed_points,
            "sweep_param": "",
            "sweep_running": False,
            "model_spec": self.model_spec,
            "display": list(self.display) if self.display else [0, 1],
            "clamped": list(self.clamped) if self.clamped else None,
            "integrator": self.integrator,
            "noise_enable": self.noise_enable,
            "noise_sigma": self.noise_sigma,
        }

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
{css_code}
    </style>
</head>
<body>
<div id="ppw-root"></div>
<script type="module">
{js_code}

const initialState = {json.dumps(state, indent=2)};

const mockModel = {{
  _isMock: true,
  _data: initialState,
  _callbacks: {{}},
  get(name) {{ return this._data[name]; }},
  set(name, value) {{ this._data[name] = value; }},
  save_changes() {{}},
  on(event, cb) {{
    if (!this._callbacks[event]) this._callbacks[event] = [];
    this._callbacks[event].push(cb);
  }},
  send() {{}},
}};

render({{ model: mockModel, el: document.getElementById('ppw-root') }});
</script>
</body>
</html>"""

        pathlib.Path(filename).write_text(html, encoding="utf-8")
