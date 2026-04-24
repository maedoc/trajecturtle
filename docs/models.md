# Model Reference

## Wilson-Cowan

The Wilson-Cowan model describes coupled excitatory (`E`) and inhibitory (`I`) neural populations:

$$
\begin{aligned}
\tau_E \dot{E} &= -E + S_e(a_{EE} E - a_{EI} I + P_E) \\
\tau_I \dot{I} &= -I + S_i(a_{IE} E - a_{II} I + P_I)
\end{aligned}
$$

where $S_e$, $S_i$ are sigmoid activation functions:

$$
S(x) = \frac{1}{1 + e^{-k(x - \theta)}}
$$

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `aee` | 10.0 | 0–20 | E→E coupling strength |
| `aei` | 10.0 | 0–20 | I→E coupling strength |
| `aie` | 10.0 | 0–20 | E→I coupling strength |
| `aii` | 2.0 | 0–20 | I→I coupling strength |
| `Pe` | -2.0 | -10–10 | External input to E |
| `Pi` | -8.0 | -10–10 | External input to I |
| `ke` | 1.0 | 0.1–5 | E sigmoid gain |
| `ki` | 1.0 | 0.1–5 | I sigmoid gain |
| `thetae` | 4.0 | 0–10 | E sigmoid threshold |
| `thetai` | 4.0 | 0–10 | I sigmoid threshold |

### Typical Dynamics

- Low external drive → stable fixed point (low activity)
- Increased drive → limit cycle (oscillations)
- Strong E→E, weak I→E → epileptic-like runaway activity

---

## FitzHugh-Nagumo

A simplified model of neuronal excitability, capturing the essence of Hodgkin-Huxley dynamics with two variables:

$$
\begin{aligned}
\dot{v} &= v - \frac{v^3}{3} - w + I \\
\dot{w} &= \varepsilon (v + a - b w)
\end{aligned}
$$

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `a` | 0.7 | -1–2 | Recovery offset |
| `b` | 0.8 | 0–2 | Recovery gain |
| `epsilon` | 0.08 | 0.001–1 | Time scale separation |
| `I` | 0.5 | -2–2 | External current |

### Typical Dynamics

- $I < I_{crit}$: stable fixed point (resting state)
- $I > I_{crit}$: limit cycle (repetitive firing)
- The cubic nullcline ($\dot{v}=0$) and linear nullcline ($\dot{w}=0$) intersect to produce these transitions

---

## MPR (Quadratic Integrate-and-Fire)

The Montbrió-Pazó-Roxin model derives a firing-rate description for a population of quadratic integrate-and-fire neurons with Lorentzian-distributed excitability:

$$
\begin{aligned}
\dot{r} &= \frac{\Delta}{\pi} + 2 r v \\
\dot{v} &= v^2 + \bar{\eta} + J r + I(t) - (\pi r)^2
\end{aligned}
$$

where $r$ is the population firing rate and $v$ is the mean membrane potential.

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `delta` | 1.0 | 0.01–5 | Lorentzian half-width Δ |
| `eta_bar` | -5.0 | -20–10 | Mean excitability η̄ |
| `J` | 15.0 | -20–30 | Synaptic coupling J |
| `I` | 0.0 | -10–10 | External input I(t) |

### Key Phenomena

- **Saddle-node bifurcation**: Increasing η̄ or J creates/annihilates pairs of fixed points
- **Bistability**: For Δ=1, η̄≈-5, J≈15 there are **three coexisting fixed points**: a stable low-activity node, a saddle, and a stable high-activity focus
- **Limit cycles**: Beyond the bistable region, oscillatory firing-rate dynamics emerge

### Paper Reference

Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for networks of spiking neurons. *Physical Review X*, 5(2), 021028. [DOI: 10.1103/PhysRevX.5.021028](https://doi.org/10.1103/PhysRevX.5.021028)

### Reproducing the Paper

See the [MPR Demo](demos/index.md#mpr-bistable-regime) for interactive reproductions of Figures 1 and 2 from the paper.
