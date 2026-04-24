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
# # Montbrió-Pazó-Roxin (MPR) Model Demo
#
# Reproduces Figures 1 and 2 from:
# > E. Montbrió, D. Pazó, and A. Roxin, *Macroscopic Description for Networks of Spiking Neurons*,
# > Phys. Rev. X **5**, 021028 (2015). DOI: [10.1103/PhysRevX.5.021028](https://doi.org/10.1103/PhysRevX.5.021028)
#
# The MPR model is an exact firing-rate description of a population of QIF neurons:
#
# $$\dot{r} = \frac{\Delta}{\pi} + 2rv$$
# $$\dot{v} = v^2 + \bar{\eta} + Jr + I(t) - (\pi r)^2$$
#
# where $r$ is the firing rate and $v$ is the mean membrane potential.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from phase_plane_widget import PhasePlaneWidget
from phase_plane_widget.models import MPRModel

plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# ## Helper: Analytical Fixed-Point Solver
# The fixed-point condition reduces to a quartic in $v$, which we solve exactly.

# %%
def mpr_fixed_points(delta, eta_bar, J, I=0.0):
    """Find all fixed points of the MPR model via quartic root finding.

    From dr/dt=0: r = -Δ/(2πv)  (requires v < 0 for r > 0)
    From dv/dt=0: v² = (πr)² - η̄ - Jr - I
    Substituting r gives a quartic in v:
        4v⁴ + 4(η̄+I)v² - (2JΔ/π)v - Δ² = 0
    """
    coeffs = [4.0, 0.0, 4.0 * (eta_bar + I), -2.0 * J * delta / np.pi, -delta ** 2]
    roots = np.roots(coeffs)

    fps = []
    for v in roots:
        if np.isreal(v) and np.real(v) < -1e-10:
            v = float(np.real(v))
            r = -delta / (2.0 * np.pi * v)
            if r > 1e-10:
                # Jacobian: [[2v, 2r], [J - 2π²r, 2v]]
                jac = np.array([[2 * v, 2 * r], [J - 2 * np.pi ** 2 * r, 2 * v]])
                ev = np.linalg.eigvals(jac)
                real = np.real(ev)
                imag = np.imag(ev)
                if all(r_ < -1e-6 for r_ in real):
                    stab = "stable_focus" if any(abs(i) > 1e-6 for i in imag) else "stable_node"
                elif all(r_ > 1e-6 for r_ in real):
                    stab = "unstable_focus" if any(abs(i) > 1e-6 for i in imag) else "unstable_node"
                else:
                    stab = "saddle"
                fps.append((float(r), float(v), stab))
    return fps


def classify_regime(delta, eta_bar, J, I=0.0):
    """Classify dynamical regime at given parameters."""
    fps = mpr_fixed_points(delta, eta_bar, J, I)
    stable = [fp for fp in fps if fp[2] in ("stable_node", "stable_focus")]
    if len(stable) >= 2:
        return "bistable"
    elif len(stable) == 1:
        return stable[0][2]
    return "other"


# %% [markdown]
# ---
# ## Figure 1(a) — Phase Diagram in (η̄, J) Space
#
# Cyan region = bistability (two stable fixed points).
# Solid curve = saddle-node bifurcation boundary (exact).
# Dashed line = boundary of stable-focus region.

# %%
delta = 1.0
n_eta, n_J = 80, 80
eta_grid = np.linspace(-8, 3, n_eta)
J_grid = np.linspace(4, 25, n_J)

regime_map = np.zeros((n_J, n_eta))
# 0=other, 1=stable_node, 2=bistable, 3=stable_focus

for i, J in enumerate(J_grid):
    for j, eta in enumerate(eta_grid):
        reg = classify_regime(delta, eta, J)
        regime_map[i, j] = {"other": 0, "stable_node": 1, "bistable": 2, "stable_focus": 3}.get(reg, 0)

# Exact saddle-node boundary (parametric in r)
r_sn = np.linspace(0.05, 1.5, 300)
eta_sn = -np.pi ** 2 * r_sn ** 2 - 3 * delta ** 2 / (4 * np.pi ** 2 * r_sn ** 2)
J_sn = 2 * np.pi ** 2 * r_sn + delta ** 2 / (2 * np.pi ** 2 * r_sn ** 3)

# Stable focus boundary
J_f = np.linspace(5, 25, 200)
eta_f = -(J_f / (2 * np.pi)) ** 2 - (np.pi / J_f) ** 2

fig, ax = plt.subplots(figsize=(7, 6))
# Custom colormap: white→green→yellow→blue
colors = ["#f5f5f5", "#a5d6a7", "#81d4fa", "#fff176"]
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors)
im = ax.imshow(
    regime_map,
    origin="lower",
    aspect="auto",
    extent=[eta_grid[0], eta_grid[-1], J_grid[0], J_grid[-1]],
    cmap=cmap,
    vmin=-0.5,
    vmax=3.5,
)

# Overlay exact boundaries
ax.plot(eta_sn, J_sn, "k-", lw=1.5, label="Saddle-node (exact)")
ax.plot(eta_f, J_f, "k--", lw=1.5, label="Focus boundary")

# Mark the bistable point from Fig 1(d)
ax.scatter([-5], [15], marker="^", s=100, c="red", edgecolors="black", zorder=5, label="Fig 1(d) point")

ax.set_xlabel(r"$\bar{\eta}$", fontsize=13)
ax.set_ylabel(r"$J$", fontsize=13)
ax.set_title("Fig 1(a): Phase Diagram (Δ = 1)", fontsize=13)
ax.legend(loc="upper left", fontsize=9)

# Custom colorbar
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(["Other", "Stable node", "Bistable", "Stable focus"])
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Figure 1(d) — Interactive Phase Portrait
# Bistable region: **η̄ = −5, J = 15, Δ = 1**
# Three fixed points: stable node (low activity), saddle, stable focus (high activity).

# %%
widget = PhasePlaneWidget(model_name="mpr")
# Already defaults to η̄ = -5, J = 15, Δ = 1
widget

# %% [markdown]
# > **Interactive tip:** Click near the high-activity focus (r ≈ 1, v ≈ −0.15) to see
# > trajectories converge to the synchronous oscillation. Click near the low-activity
# > node (r ≈ 0.08, v ≈ −1.96) to see quiescent convergence.

# %% [markdown]
# ---
# ## Figure 1(b,c) — Bifurcation Diagrams
# r and v versus **η̄** for J = 15, Δ = 1.

# %%
eta_sweep = np.linspace(-12, 3, 300)
J_fixed = 15.0

r_all, v_all, eta_all, stab_all = [], [], [], []
for eta in eta_sweep:
    fps = mpr_fixed_points(delta=1.0, eta_bar=eta, J=J_fixed)
    for r, v, stab in fps:
        r_all.append(r)
        v_all.append(v)
        eta_all.append(eta)
        stab_all.append(stab)

r_all = np.array(r_all)
v_all = np.array(v_all)
eta_all = np.array(eta_all)

# Colors by stability
stab_colors = {
    "stable_node": "#4CAF50",
    "stable_focus": "#2196F3",
    "saddle": "#9C27B0",
    "unstable_node": "#f44336",
    "unstable_focus": "#FF9800",
}

fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(12, 5))

for stab in set(stab_all):
    mask = np.array(stab_all) == stab
    ax_r.scatter(eta_all[mask], r_all[mask], c=stab_colors.get(stab, "#666"), s=12, label=stab.replace("_", " "), zorder=2)
    ax_v.scatter(eta_all[mask], v_all[mask], c=stab_colors.get(stab, "#666"), s=12, zorder=2)

# Sort and draw connecting lines for each branch (approximate)
for stab in ["stable_node", "stable_focus", "saddle"]:
    mask = np.array(stab_all) == stab
    if mask.sum() > 1:
        idx = np.argsort(eta_all[mask])
        ax_r.plot(eta_all[mask][idx], r_all[mask][idx], c=stab_colors[stab], lw=1, alpha=0.5, zorder=1)
        ax_v.plot(eta_all[mask][idx], v_all[mask][idx], c=stab_colors[stab], lw=1, alpha=0.5, zorder=1)

ax_r.set_xlabel(r"$\bar{\eta}$", fontsize=12)
ax_r.set_ylabel(r"Firing rate $r$", fontsize=12)
ax_r.set_title(r"Fig 1(b): $r$ vs $\bar{\eta}$  (J = 15, Δ = 1)", fontsize=12)
ax_r.legend(loc="upper right", fontsize=8)

ax_v.set_xlabel(r"$\bar{\eta}$", fontsize=12)
ax_v.set_ylabel(r"Mean potential $v$", fontsize=12)
ax_v.set_title(r"Fig 1(c): $v$ vs $\bar{\eta}$  (J = 15, Δ = 1)", fontsize=12)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Figure 2 — Transient Dynamics
# Exact correspondence between the FREs and the microscopic QIF ensemble.
# Parameters: **J = 15, η̄ = −5, Δ = 1**.

# %%
def simulate_mpr_time_varying(delta, eta_bar, J, I_func, t_span, ic, dt=0.005):
    """Simulate MPR equations with time-varying input I(t)."""
    def f(t, y):
        r, v = y
        r_eff = max(r, 1e-10)
        I_t = I_func(t)
        dr = delta / np.pi + 2 * r_eff * v
        dv = v ** 2 + eta_bar + J * r_eff + I_t - (np.pi * r_eff) ** 2
        return [dr, dv]

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(f, t_span, ic, t_eval=t_eval, max_step=dt * 5, dense_output=True, method="RK45")
    return sol.t, sol.y


# Initial condition: low-activity stable fixed point for baseline params
fps_base = mpr_fixed_points(delta=1.0, eta_bar=-5.0, J=15.0)
ic_low = [fp[0] for fp in fps_base if fp[2] == "stable_node"][0]
ic_low_v = [fp[1] for fp in fps_base if fp[2] == "stable_node"][0]
ic = [ic_low, ic_low_v]
print(f"Initial condition (low-activity FP): r={ic[0]:.4f}, v={ic[1]:.4f}")


# %% [markdown]
# ### Figure 2(a,c,e) — Step Input
# I(t) = 3 for 0 ≤ t < 30, then I(t) = 0.

# %%
def I_step(t, I_on=3.0, t_on=0.0, t_off=30.0):
    return I_on if t_on <= t < t_off else 0.0


t_step, y_step = simulate_mpr_time_varying(
    delta=1.0, eta_bar=-5.0, J=15.0,
    I_func=lambda t: I_step(t),
    t_span=[0, 50], ic=ic, dt=0.01
)

fig, (ax_r, ax_v, ax_I) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

ax_r.plot(t_step, y_step[0], "k-", lw=1.5)
ax_r.set_ylabel(r"$r(t)$", fontsize=12)
ax_r.set_title("Fig 2(a): Step Input — Firing Rate", fontsize=12)
ax_r.axvspan(0, 30, alpha=0.1, color="blue", label="I = 3")
ax_r.legend(loc="upper right", fontsize=9)

ax_v.plot(t_step, y_step[1], "k-", lw=1.5)
ax_v.set_ylabel(r"$v(t)$", fontsize=12)
ax_v.set_title("Fig 2(c): Step Input — Mean Potential", fontsize=12)
ax_v.axvspan(0, 30, alpha=0.1, color="blue")

I_vals = [I_step(t) for t in t_step]
ax_I.plot(t_step, I_vals, "b-", lw=1.5)
ax_I.set_ylabel(r"$I(t)$", fontsize=12)
ax_I.set_xlabel("Time", fontsize=12)
ax_I.set_title("Fig 2(g): External Input", fontsize=12)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figure 2(b,d,f) — Sinusoidal Input
# I(t) = 3 sin(πt/20)

# %%
omega = np.pi / 20.0
I0 = 3.0


def I_sine(t):
    return I0 * np.sin(omega * t)


t_sin, y_sin = simulate_mpr_time_varying(
    delta=1.0, eta_bar=-5.0, J=15.0,
    I_func=I_sine,
    t_span=[0, 80], ic=ic, dt=0.01
)

fig, (ax_r2, ax_v2, ax_I2) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

ax_r2.plot(t_sin, y_sin[0], "k-", lw=1.5)
ax_r2.set_ylabel(r"$r(t)$", fontsize=12)
ax_r2.set_title(r"Fig 2(b): Sinusoidal Input — Firing Rate  ($I_0=3, \omega=\pi/20$)", fontsize=12)

ax_v2.plot(t_sin, y_sin[1], "k-", lw=1.5)
ax_v2.set_ylabel(r"$v(t)$", fontsize=12)
ax_v2.set_title("Fig 2(d): Sinusoidal Input — Mean Potential", fontsize=12)

I_vals_sin = [I_sine(t) for t in t_sin]
ax_I2.plot(t_sin, I_vals_sin, "b-", lw=1.5)
ax_I2.set_ylabel(r"$I(t)$", fontsize=12)
ax_I2.set_xlabel("Time", fontsize=12)
ax_I2.set_title("Fig 2(h): External Input", fontsize=12)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Widget Sweep: Interactive Bifurcation Exploration
# Use the built-in sweep to explore how dynamics change with η̄.

# %%
widget2 = PhasePlaneWidget(model_name="mpr")
widget2.params = {"delta": 1.0, "eta_bar": -5.0, "J": 15.0, "I": 0.0}
widget2.xlim = [0.0, 2.0]
widget2.ylim = [-4.0, 2.0]
widget2

# %% [markdown]
# Run the sweep from the widget UI (set Sweep parameter = `eta_bar`, Min = −12, Max = 3, N = 80)
# or programmatically:

# %%
# Programmatic sweep for the bifurcation diagram
widget2.run_sweep("eta_bar", np.linspace(-12, 3, 80).tolist())

# %% [markdown]
# ---
# ## Summary
# The MPR model exactly reduces a population of QIF neurons to a 2D system in (r, v).
# - **Bistability** (Fig 1d): Two stable fixed points coexist — a quiescent low-rate state
#   and a high-rate synchronous oscillation (stable focus).
# - **Bifurcations** (Fig 1b,c): Saddle-node and Hopf bifurcations as η̄ varies.
# - **Transient dynamics** (Fig 2): The macroscopic FREs track the population response
#   to time-varying inputs (step and sinusoidal) exactly.
