"""Demo: pass a Python model instance to the widget and read back tuned parameters.

This demonstrates the ``model=`` argument to ``PhasePlaneWidget``,
which accepts any ``BaseModel`` subclass instance.  After the user
interacts with the widget (dragging sliders, clicking the phase plane),
the current parameter values can be read back from the ``.params``
traitlet.

If the model name is registered in ``MODEL_REGISTRY`` the JavaScript
front-end will recognise it and carry out all computation client-side.
For arbitrary custom models that are *not* in the built-in registry use
:func:`phase_plane` instead (see ``custom_model_demo.py``).
"""

from tvb_phaseplane import PhasePlaneWidget, MPRModel

# ------------------------------------------------------------------
# 1. Instantiate a model class and optionally override defaults
# ------------------------------------------------------------------
model = MPRModel()

# You can pre-configure parameters before creating the widget
# (the widget will pick these up as its initial slider positions)
initial_params = {
    "delta": 1.0,
    "eta_bar": -5.0,
    "J": 15.0,
    "I": 0.0,
}

# Override the model's default parameter values with your own
for k, v in initial_params.items():
    model.default_params[k] = v

# ------------------------------------------------------------------
# 2. Pass the instance to PhasePlaneWidget
# ------------------------------------------------------------------
widget = PhasePlaneWidget(model=model)

# The widget is now live — display it in Jupyter / VS Code
widget

# ------------------------------------------------------------------
# 3. After user interaction, read back tuned parameters
# ------------------------------------------------------------------
# Run this cell *after* adjusting sliders in the widget:
print("Current parameter values after tuning:")
for name, value in widget.params.items():
    print(f"  {name:12s} = {value:.4f}")

# You can also programmatically set parameters and trigger updates:
# widget.params["J"] = 20.0

# ------------------------------------------------------------------
# 4. Read back computed data (nullclines, fixed points, trajectory)
# ------------------------------------------------------------------
print(f"\nFixed points detected: {len(widget.fixed_points)}")
for fp in widget.fixed_points[:5]:
    print(f"  x={fp[0]:.4f}, y={fp[1]:.4f}, type={fp[2]}")

print(f"\nTrajectory points: {len(widget.trajectory)}")

# ------------------------------------------------------------------
# 5. Export the tuned configuration to standalone HTML
# ------------------------------------------------------------------
widget.to_standalone_html("mpr_tuned.html", title="MPR Model – Tuned Parameters")
print("\nExported to 'mpr_tuned.html' with current parameter values.")
