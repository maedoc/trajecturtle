# Parallel Implementation Plan (Phase A Follow-ups)

This document records the parallel execution plan for implementing the
remaining Phase A features: 1-D line-plot mode, stochastic Heun integration,
and numerical safety guards.

## Problem

Three independent features need to land. Sequential implementation would take
~3× the wall-clock time. All three touch **disjoint regions** of `widget.js`:

| Feature | Code Region | Conflict Risk |
|---------|------------|---------------|
| 1-D line plot | `renderPhasePlane`, `computeNullclines`, `computeVectorField`, CSS | Low |
| Stochastic Heun | `heun()`, `makeProjectedRHS` (noise-free), UI noise controls | Low |
| Safety guards | `safeRK4()`, `safeHeun()`, budget wrappers around numerical functions | Low |

## Branch + Worktree Strategy

```bash
# Create feature branches from main
git branch p1-1d-mode
git branch p2-heun-noise
git branch p3-safety-guards

# Create isolated worktrees (each agent gets its own checkout)
git worktree add ../trajecturtle-1d   p1-1d-mode
git worktree add ../trajecturtle-heun  p2-heun-noise
git worktree add ../trajecturtle-safe  p3-safety-guards
```

## Task Specifications

### P1: 1-D Line-Plot Mode

**Branch:** `p1-1d-mode`  
**Worktree:** `../trajecturtle-1d`

When `display.length === 1`, the phase plane renders a 1-D line plot:
- Horizontal axis: state variable `x` (or whichever is displayed)
- Vertical axis: `dx/dt` (derivative of the displayed variable)
- Vertical range: `[-2, 2]` (arbitrary but fixed for all 1-D plots)
- Nullcline: zero-crossings of `f(x)` plotted as points on the x-axis
- Vector field: 1-D arrows along the x-axis
- Fixed points: where `f(x) = 0`, shown as colored dots
- Trajectory: time-series of `x(t)` (same as time-series panel, not shown)
- **No limit-cycle detection** in `detectRegime` for 1-D mode

Files to edit:
- `src/phase_plane_widget/static/widget.js`
  - `renderPhasePlane()`: add `is1D` branch
  - `computeNullclines()`: handle `is1D` (only one nullcline)
  - `computeVectorField()`: handle `is1D` (arrows on x-axis)
  - `detectRegime()`: already updated in main, verify 1-D branch works
- `src/phase_plane_widget/static/widget.css`: add `.ppw-1d-mode` styles

Estimated: ~80 lines of JS + 20 lines of CSS.

**Success criteria:**
- `display = [0]` on Wilson-Cowan renders a line plot
- Clicking on the x-axis sets initial condition
- Fixed points appear as colored dots at zero-crossings

### P2: Stochastic Heun Integration

**Branch:** `p2-heun-noise`  
**Worktree:** `../trajecturtle-heun`

Add Stratonovich Heun integration with per-variable noise:

```javascript
function heun(modelName, state0, t0, tMax, dt, params, noiseSigma) {
  // noiseSigma: array of length n, one entry per state variable
  // Wiener increments: dW ~ N(0, sqrt(dt))
  // Stratonovich: predictor-corrector with noise evaluated at both steps
}
```

Key constraints:
- Nullclines and vector field are **noise-free** (deterministic)
- `makeProjectedRHS()` should accept a `deterministic` flag (default true)
- Trajectories use Heun when noise is enabled, RK4 when disabled
- Per-variable noise strength sliders (0–1 range)
- Noise toggle checkbox

Files to edit:
- `src/phase_plane_widget/static/widget.js`
  - Add `heun()` function
  - Modify `makeProjectedRHS()` to accept `deterministic` flag
  - Modify `computeAll()`: use Heun when noise enabled
  - Add noise UI controls (checkbox + per-var sliders)
- `src/phase_plane_widget/widget.py`: add `noise_enable`, `noise_sigma` traits

Estimated: ~120 lines of JS + 20 lines of Python.

**Success criteria:**
- Heun with σ=0 matches RK4 within tolerance
- Noise increases trajectory dispersion
- UI toggles between deterministic and stochastic modes
- Nullclines unchanged when noise is on/off

### P3: Numerical Safety Guards

**Branch:** `p3-safety-guards`  
**Worktree:** `../trajecturtle-safe`

Wrap all numerical entry points with safety guards:

1. **NaN/Inf bailout**: Any NaN or Inf in state terminates integration early
2. **Computation budget**: max steps = 50,000 for integration
3. **Exponential clamp**: `Math.exp(x)` → `Math.exp(Math.max(-709, Math.min(x, 709)))`
4. **Newton-Raphson budget**: max 625 iterations (25×25 grid)
5. **Nullcline budget**: max 3,600 evaluations (60×60 grid)
6. **Vector field budget**: max 144 evaluations (12×12 grid)

```javascript
function safeRK4(modelName, state0, t0, tMax, dt, params, budget = 50000) {
  // Wrap RK4 with NaN/Inf detection and step counter
}

function hasBudget(counter, limit) {
  if (counter > limit) return false;
  return true;
}
```

Files to edit:
- `src/phase_plane_widget/static/widget.js`
  - Wrap `rk4()` with `safeRK4()`
  - Add budget checks to `findFixedPoints()`, `computeNullclines()`, `computeVectorField()`
  - Add `expClamp()` helper used everywhere
  - Return empty/null on failure + `console.warn()`

Estimated: ~60 lines of JS.

**Success criteria:**
- MPR model at unstable regime (where NaN could occur) degrades gracefully
- Console warning on budget exhaustion, not crash
- All three built-in models still work normally

## Integration Order

After all three branches pass review and tests, merge in this order:

1. **P3 first** — safety guards are transparent API wrappers, no functional change
2. **P2 second** — Heun builds on stable (guarded) integrator base
3. **P1 last** — rendering is independent but depends on stable base

```bash
cd /home/duke/src/phase-plane-widget

# Review each branch
for b in p3-safety-guards p2-heun-noise p1-1d-mode; do
  git log --oneline $b..origin/main
done

# Merge in order
git merge p3-safety-guards
git merge p2-heun-noise
git merge p1-1d-mode

# Clean up worktrees
git worktree remove ../trajecturtle-1d
git worktree remove ../trajecturtle-heun
git worktree remove ../trajecturtle-safe
git branch -d p1-1d-mode p2-heun-noise p3-safety-guards

# Build & deploy
uv run --extra docs python scripts/generate_demos.py
uv run --extra docs mkdocs build --strict
git commit -am "feat: merge parallel branches (1D mode, Heun, safety guards)"
git push
```

## Launch Commands

Run these three in parallel (each takes ~20–40 min):

```bash
# Agent 1: 1D mode
cd ../trajecturtle-1d
pi -p "Implement 1D line-plot phase plane mode in widget.js. When display.length===1, renderPhasePlane should draw dx/dt vs x (xlim on horizontal, derivative on vertical [-2,2]). computeNullclines and computeVectorField must handle 1D: computeNullclines returns nullcline_y as zero-crossings of f(x), computeVectorField returns 1D arrows on the x-axis. Remove limit cycle detection from detectRegime when is1D. Update CSS for 1D mode. Test by checking that Wilson-Cowan with display=[0] renders a line plot. Write a quick test script to verify." --no-sandbox --model ollama/deepseek-v4-flash

# Agent 2: Heun noise
cd ../trajecturtle-heun
pi -p "Implement stochastic Heun integration in widget.js. Add a heun() function that supports per-variable noise with noise_shape=[nvar]. Add 'noise_enable' checkbox and per-variable noise strength sliders (0–1 range). The projected RHS for nullclines/vector field must be noise-free (deterministic). Trajectories use Heun when noise is enabled, RK4 when disabled. Add Wiener increments with dt^(1/2) scaling. Stratonovich interpretation. Fix: makeProjectedRHS should use a deterministic flag. Add a test that Heun with sigma=0 agrees with RK4. Write a test script." --no-sandbox --model ollama/deepseek-v4-flash

# Agent 3: Safety guards
cd ../trajecturtle-safe
pi -p "Add numerical safety guards to widget.js. Wrap RK4/Heun with safe integrators that: (1) detect NaN/Inf and short-circuit, (2) enforce a computation budget (max steps 50000), (3) clamp exponential arguments to [-709,709], (4) add hasBudget() checks to findFixedPoints (max 625 newton iterations), computeNullclines (max 3600 evals), computeVectorField (max 144 evals). Return empty/null on failure with console warning rather than crashing. Create a test with the MPR model at an unstable regime where NaN could occur, verifying graceful degradation." --no-sandbox --model ollama/deepseek-v4-flash
```

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Agent crashes / stalls | Medium | Set timeouts; poll every 5 min; kill and retry |
| Merge conflicts | Low | Branches touch disjoint code regions (render vs integrator vs guards) |
| Tests timeout on slow machine | Medium | Reduce `t_max` in demos; use smaller grids |
| Heun ≠ RK4 at σ=0 | Medium | Increase tolerance; check floating-point differences |

## Status

- [ ] Branches created
- [ ] Worktrees created
- [ ] Agent 1 (1D) launched
- [ ] Agent 2 (Heun) launched
- [ ] Agent 3 (Safety) launched
- [ ] Agent 1 completed
- [ ] Agent 2 completed
- [ ] Agent 3 completed
- [ ] P3 merged into main
- [ ] P2 merged into main
- [ ] P1 merged into main
- [ ] Tests pass
- [ ] Docs built
- [ ] Pushed to GitHub

---

*Generated: 2026-04-24*
