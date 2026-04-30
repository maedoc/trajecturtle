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
# # Stochastic Dynamics in a Bistable System
#
# A simple bistable system driven by additive noise.
#
# $$\dot{x} = x - x^3 + \sigma \xi(t)$$
#
# With $\sigma = 0$ the system has stable fixed points at $x = \pm 1$ and an
# unstable fixed point at $x = 0$.  With noise the trajectory jumps between
# the two wells.

# %%
from tvb_phaseplane import phase_plane

pp = phase_plane(
    equations=["x - x**3", "-y"],
    state_vars={"x": (-2, 2), "y": (-1, 1)},
    params={"a": (1.0, 0.0, 2.0)},
    integrator="heun",
    noise_per_var=[0.15, 0.0],
    display=["x", "y"],
    name="stochastic_bistable",
)
pp

# %% [markdown]
# Try toggling **Enable noise** in the widget UI and switching between
# the **RK4** and **Heun** integrators to see how noise affects the
# trajectory.
