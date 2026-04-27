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
# # van der Pol Oscillator
#
# The van der Pol oscillator is a classic nonlinear dynamical system:
#
# $$\dot{x} = y$$
# $$\dot{y} = \mu (1 - x^2) y - x$$
#
# where $\mu > 0$ controls the strength of nonlinear damping.

# %%
from phase_plane_widget import phase_plane

# mu = 1.0  → relaxation oscillator
pp = phase_plane(
    equations=["y", "mu * (1 - x**2) * y - x"],
    state_vars={"x": (-3, 3), "y": (-3, 3)},
    params={"mu": (1.0, 0.0, 3.0)},
    name="van_der_pol",
)
pp

# %% [markdown]
# ## Standalone Export
#
# Save as a self-contained `.html` file for embedding in documentation.

# %%
pp.to_standalone_html("van_der_pol.html", title="van der Pol Oscillator")
print("Exported to van_der_pol.html")
