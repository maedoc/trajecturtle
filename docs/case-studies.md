# Case Study: Wilson–Cowan Saddle Point

This page walks through a concrete example where a fixed point is correctly classified as a **saddle** by the widget, yet can *look* like an unstable node in an interactive phase plane. Understanding why this happens sharpens both the math and the visualization.

## Model & Default Parameters

The Wilson–Cowan equations at the widget's default settings are

\[
\begin{aligned}
\dot E &= -E + S\bigl(a_{EE}E - a_{EI}I + P_E\bigr) \\
\dot I &= -I + S\bigl(a_{IE}E - a_{II}I + P_I\bigr)
\end{aligned}
\]

with sigmoid \(S(x)=1/(1+e^{-k(x-\theta)})\) and

| Parameter | Value |
|-----------|-------|
| \(a_{EE}\) | 10.0 |
| \(a_{EI}\) | 10.0 |
| \(a_{IE}\) | 10.0 |
| \(a_{II}\) | 2.0 |
| \(P_E\) | –2.0 |
| \(P_I\) | –8.0 |
| \(k_E, k_I\) | 1.0 |
| \(\theta_E, \theta_I\) | 4.0 |

## Fixed Points Found

Running the widget's auto-detection (or the equivalent Python routine) yields **three** equilibria:

| # | \(E^*\) | \(I^*\) | Classification | Color |
|---|---------|---------|----------------|-------|
| 1 | 0.0025 | 0.0000 | stable node | ● green |
| 2 | **0.6817** | **0.0055** | **saddle** | ◆ purple |
| 3 | 0.9424 | 0.0629 | stable focus | ◉ green |

The middle one is the focus of this study.

## Eigenvalue/Eigenvector Decomposition at the Saddle

At \((E^*, I^*)\approx(0.682, 0.006)\) the Jacobian is

\[
J = \begin{bmatrix}
 1.1700 & -2.1700 \\
 0.0549 & -1.0110
\end{bmatrix}
\]

with eigenvalues

\[
\lambda_1 \approx +1.114, \qquad \lambda_2 \approx -0.955
\]

Opposite real parts and no imaginary component. That is the *definition* of a saddle.

The corresponding eigenvectors (unnormalised) are

| Eigenvalue | Type | Eigenvector | Direction |
|------------|------|-------------|-----------|
| \(\lambda_1 \approx +1.114\) | **unstable** | \((1.00,\; 0.03)\) | almost horizontal, pointing right |
| \(\lambda_2 \approx -0.955\) | **stable** | \((0.71,\; 0.70)\) | ≈ 45° up-right |

So the **unstable manifold** is nearly flat along the \(E\)-axis, while the **stable manifold** cuts diagonally across the phase plane.

## Why It Looks "Just Unstable" in the UI

In a textbook phase-plane diagram the saddle looks obvious because the author **draws the invariant manifolds explicitly** (dashed separatrices). The interactive widget does not — it only shows:

1. Nullclines
2. A sparse vector-field grid
3. Trajectories the user clicks

Here's the crucial observation: the stable manifold is the **basin boundary**. Points just below it flow to the origin stable node; points just above it flow to the high-activity stable focus. Unless the user clicks **exactly** on the separatrix (measure-zero event with a mouse), the forward-time trajectory is repelled from the saddle toward one of the two attractors.

```
        I ↑
          │      stable focus ●
          │           ↗
          │   stable manif.  /  (basin boundary)
          │         ↗   /
    saddle◆───────↗───/
          │      ↗   /
          │    ↗   /
          │  ↗   /          ← trajectories from here
          │↗   /              all diverge from saddle
  stable node ●
                  ─────→ E
```

Because every click-able initial condition is generically *off* the stable manifold, **all visible trajectories diverge**. The user therefore sees an equilibrium that repels everything — precisely the *feeling* of an unstable node — even though mathematically it is a saddle.

## Numerical Verification: Trajectory Probes

Starting at \((E^*+\varepsilon, I^*+\varepsilon)\) with \(\varepsilon=0.02\):

| Initial condition | Final state (t = 20 s) |
|-------------------|--------------------------|
| \((0.702,\; 0.026)\) | → stable node \((0.003, 0)\) |
| \((0.662,\; -0.014)\) | → stable focus \((0.942, 0.063)\) |

The two sides of the diagonal perturbation land in **different basins**, confirming the stable manifold runs between them. A trajectory started *exactly* on the stable manifold would converge to the saddle, but in practice the manifold is a 1-D curve inside a 2-D plane — the user almost never hits it.

## Vector-Field Snapshot Around the Saddle

| \(E\) | \(I\) | \(\dot E\) | \(\dot I\) | Interpretation |
|-------|-------|------------|------------|----------------|
| 0.632 | 0.006 | –0.067 | –0.002 | left/down — toward origin |
| 0.732 | 0.006 | +0.048 | +0.003 | right/up — toward stable focus |
| 0.682 | –0.019 | +0.052 | +0.025 | right/up |
| 0.682 | 0.031 | –0.057 | –0.025 | left/down |

The flow reverses across both axes of the saddle, which is textbook saddle behaviour. The problem is not the math — it is that a **mouse click does not sample the measure-zero set** required to see convergence.

## Take-aways

1. **Classification is correct.** Both Python (`numpy.linalg.eigvals`) and the JS frontend (`eigenvalues2x2`) agree: opposite real signs &rarr; saddle.
2. **Visual ambiguity is expected.** A saddle in a generic 2-D flow separates basins. Unless the separatrix is drawn explicitly, most trajectories look like they are simply repelled.
3. **Future improvement:** explicitly computing and drawing the stable/unstable manifolds (or separatrices) would make saddles visually unmistakable.

## References

1. Scholarpedia — *Equilibrium*: http://www.scholarpedia.org/article/Equilibrium  
   Figure 3 in particular shows the two-dimensional eigenvalue-based classification diagram.
2. Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos* (2nd ed.). Sections 6.1–6.3 on saddles, manifolds, and phase portraits.
