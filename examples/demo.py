# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase Plane Widget Demo
#
# Interactive visualization of neural mass models using `phase_plane_widget`.
# This notebook works in Jupyter, VS Code, and can be exported to static HTML.

# %%
from phase_plane_widget import PhasePlaneWidget

# Create and display the widget
widget = PhasePlaneWidget()
widget

# %% [markdown]
# ## Interactive Features
#
# - **Click on the phase plane** to set initial conditions
# - **Adjust sliders** to change model parameters in real time
# - **Toggle visibility** of nullclines, vector field, fixed points, and trajectories
# - **Run parameter sweeps** from the UI or programmatically (see below)

# %% [markdown]
# ## Programmatic Parameter Sweep
#
# Run a sweep over a parameter range and see the bifurcation diagram.

# %%
# Example: sweep external current I in FitzHugh-Nagumo
widget.model_name = "fitzhugh_nagumo"
widget.run_sweep("I", [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

# %% [markdown]
# ## Switch Model
#
# Change to Wilson-Cowan model to explore E-I population dynamics.

# %%
widget.model_name = "wilson_cowan"

# %% [markdown]
# ## Export to Static HTML
#
# Save the widget state for embedding in a static website.

# %%
from ipywidgets.embed import embed_minimal_html

embed_minimal_html(
    "phase_plane_export.html",
    views=[widget],
    title="Phase Plane Widget"
)
print("Exported to phase_plane_export.html")
