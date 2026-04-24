# Phase Plane Widget — Implementation Plan (v2, post-DeepSeek review)

## Constraints

1. **Nerdamer inlined** — bundle core (~100KB) into widget.js; standalone HTML stays self-contained
2. **Custom functions supported** — sigmoid, heaviside, etc. registered in both SymPy and Nerdamer
3. **No TVB bridge** — skip entirely
4. **Stochastic Heun integrator** — additive noise, Stratonovich interpretation, noise strength slider
5. **Simple visualization** — 2D projection + time series of all N variables

---

## DeepSeek Review: Critical Findings Incorporated

### 🟢 Adopted Changes

| Finding | Action |
|---------|--------|
| **Transpiler coverage gap** | Added comprehensive `FUNCTION_MAP` table (§Function Coverage). Added property-based testing requirement: generate random SymPy expressions, transpile, evaluate at random points, assert \|error\| < 1e-12. |
| **Projection wrapper allocates per step** | Changed strategy: generate a *projected* RHS directly from Nerdamer — only the 2 displayed equations, hidden vars baked as closure constants. No allocation, no wasted FLOPs. |
| **Live editing: custom function deps** | Added topological sort requirement for `custom_functions` registration. Functions must be registered in dependency order before equations are parsed. |
| **Live editing: race conditions** | Added editor state machine (IDLE → PARSING → COMPILING → VALID → ERROR). Python sync uses debouncing + request IDs + stale-result rejection. |
| **Heun test claim wrong** | Fixed: RK4 (4th order) ≠ Heun (2nd order) even at σ=0. Test verifies convergence *rate*, not trajectory equality. |
| **noise_shape spec wrong** | Fixed: diagonal noise uses `noise_per_var: [σ_x, σ_y, ...]` (array of scalars), not a matrix. Full covariance matrix deferred to future. |
| **Missing: Web Worker** | Added Web Worker strategy for expensive RHS (N≥6 or complex custom functions) to prevent UI thread blocking. |
| **Missing: memory budget** | Time series capped at 50K steps, downsampled for display. Circular buffer for long stochastic runs. |
| **Missing: NaN/Inf** | Post-step detection in every integrator. Trajectory truncated with red indicator. Nullcline/sweep skip NaN grid points. |
| **Missing: display validation** | Guard against 0/1/duplicate/nonexistent display vars. Fallback: first 2 state vars (or first 1 for 1D). |
| **Missing: sympy.solve() budget** | 5-second timeout with `sympy.solve(..., dict=True)` in a thread. Fallback to numerical Newton-Raphson on timeout. |

---

## Function Coverage Table (SymPy → JS/Nerdamer)

Every function in this table is **tested** in the property-based suite. Anything not listed is rejected with a clear error.

| SymPy | JS / Nerdamer | Edge Case | Notes |
|-------|---------------|-----------|-------|
| `x + y` | `x + y` | — | |
| `x - y` | `x - y` | — | |
| `x * y` | `x * y` | — | |
| `x / y` | `x / y` | y=0 → Infinity | NaN caught by safety layer |
| `x ** y` | `Math.pow(x, y)` | x<0, y∉ℤ → NaN | |
| `sp.exp(x)` | `Math.exp(x)` | x>709 → Infinity | Clamped to [-709, 709] in Wilson-Cowan |
| `sp.log(x)` | `Math.log(x)` | x≤0 → NaN | |
| `sp.sqrt(x)` | `Math.sqrt(x)` | x<0 → NaN | |
| `sp.sin(x)` | `Math.sin(x)` | — | |
| `sp.cos(x)` | `Math.cos(x)` | — | |
| `sp.tan(x)` | `Math.tan(x)` | x=π/2+kπ → Infinity | |
| `sp.Abs(x)` | `Math.abs(x)` | — | |
| `sp.sign(x)` | `Math.sign(x)` | x=0 → 0 | SymPy: sign(0)=0 ✓ |
| `sp.pi` | `Math.PI` | — | |
| `sp.Heaviside(x)` | `nerdamer.sign(x)` or custom: `(x>0)?1:(x<0)?0:0.5` | x=0 → 0.5 | **Critical:** matches SymPy's H(0)=1/2 |
| `sp.tanh(x)` | `Math.tanh(x)` | — | |
| `sp.Min(x,y)` | `Math.min(x,y)` | — | |
| `sp.Max(x,y)` | `Math.max(x,y)` | — | |
| `Piecewise` | **NOT SUPPORTED** | — | Rejected with clear error; use `Heaviside` or custom function |
| `Custom: sigmoid(x,k,θ)` | Custom registered function | — | Registered via `nerdamer.setFunction()` |

**Property-based test:** `hypothesis`-style random expression generator → SymPy `lambdify` → our transpiler → evaluate at 100 random points → max error < 1e-12.

---

## Model Specification Format

```python
model_spec = {
    'name': 'Wilson-Cowan',
    'state_vars': {
        'E': {'default': 0.1, 'range': [-1, 2]},
        'I': {'default': 0.1, 'range': [-1, 2]},
    },
    'parameters': {
        'tau_E': {'default': 1.0, 'range': [0.1, 5.0], 'step': 0.1},
    },
    'equations': {
        'E': '-E/tau_E + (1 - r_E*E) * sigmoid(...)',
        'I': '-I/tau_I + (1 - r_I*I) * sigmoid(...)',
    },
    'display': ['E', 'I'],
    'custom_functions': {
        'sigmoid': {'vars': ['x', 'k', 'theta'], 'expr': '1/(1+exp(-k*(x-theta)))'},
    },
    'integrator': 'rk4',          # 'rk4' | 'heun'
    'noise_per_var': None,        # None (deterministic) or [σ_1, σ_2, ...]
}
```

---

## N-Dimensional Projection (Optimized)

The phase plane is a **2D slice through N-dimensional state space**. Non-displayed vars are clamped.

**Performance fix (from DeepSeek):** Instead of allocating a full N-dim array each step and computing N equations:
1. At model compile time, substitute clamped values into the Nerdamer expressions for the 2 displayed equations.
2. `buildFunction()` produces a 2-output function directly: `f_displayed(x, y, params)`.
3. For trajectory integration, the full N-dim system is still integrated (correct dynamics), but the phase-plane vector field uses the projected function (fast).

### N = 1
- `dx/dt` vs `x` line plot
- Zero crossings = fixed points
- Slope = stability

### N = 2
- Standard phase plane

### N > 2
- Dropdowns for X/Y axes
- Clamped sliders for hidden vars
- **Full N-dim integration** for trajectories
- **Projected 2-dim function** for vector field / nullclines
- Time series shows all N vars (capped at 50K points, downsampled)

---

## Integrators

### RK4 (deterministic)
Existing. 4th order, fixed step.

### Heun (stochastic)
Stratonovich interpretation. Additive diagonal noise.

```
Ỹ   = Yₙ + f(Yₙ)Δt + g·ΔW
Yₙ₊₁ = Yₙ + ½[f(Yₙ) + f(Ỹ)]Δt + ½g·ΔW     # g is constant (additive)
```

`g = noise_per_var[i]`, `ΔW ~ N(0, √(dt))` per variable.

- **Phase field / nullclines:** Deterministic (noise averages to 0)
- **Trajectories:** Stochastic
- **Regime detection / sweeps:** RK4 only (noise makes classification unreliable)
- **Step-size guard:** Warning if `max(g²)·dt > 0.1` (heuristic instability threshold)

---

## JS Runtime Flow

```
ModelSpec arrives
    │
    ▼
┌──────────────────────────────────────────┐
│ 1. Topological sort custom_functions     │
│ 2. Register each with nerdamer           │
│ 3. Parse equations → nerdamer exprs    │
│ 4. buildFunction() → native JS func    │
│ 5. Build PROJECTED function:             │
│    - substitute clamped vars as numbers  │
│    - buildFunction([x,y,params])         │
│    → fast 2-output function, no alloc    │
│ 6. Numerical Jacobian (2×2) from projected │
│ 7. Newton-Raphson, nullclines, RK4/Heun │
│ 8. NaN/Inf detection every step         │
│ 9. Web Worker fallback for N≥6 or       │
│    complex RHS (heavy compute)          │
└──────────────────────────────────────────┘
```

---

## Live Editor State Machine

```
         ┌─────────────┐
         ▼             │
  [IDLE] ──user types──▶ [PARSING]
                            │
                   parse OK │ parse fail
                            ▼
                   ┌──────────────┐
                   │ [COMPILING]  │
                   │ buildFunction│
                   └──────┬───────┘
              compile OK   │   compile fail
                   ┌───────┴───────┐
                   ▼               ▼
              [VALID]          [ERROR]
                   │               │
      save_changes()         show error
      debounced 500ms        keep old model
      with request ID        active
                   │
                   ▼
              [IDLE]
```

**Python sync:** 500ms debounce. Each request carries `request_id`. Python responds with same ID. JS discards responses whose ID ≠ current ID (stale-result rejection).

---

## Implementation Phases (Revised)

### Phase A: Foundation + Safety + Stochastic

**Goal:** N-dim refactor, Heun integrator, numerical safety layer, Web Worker strategy.

| Task | File | Details |
|------|------|---------|
| N-dim state vector | `widget.js` | `state=[v0,v1,…,vN]`; projected RHS with baked clamped vars |
| State var selector | `widget.js` | `<select>` dropdowns for X/Y (hidden when nvar≤2) |
| Clamped sliders | `widget.js` | Sliders for non-displayed vars; trigger phase plane recompute |
| 1D mode | `widget.js` | `dx/dt` vs `x` line plot; zero-crossing fixed points |
| Time series (all N) | `widget.js` | Overlaid lines, different colors; 50K cap, downsampling |
| Heun integrator | `widget.js` | Box-Muller for `N(0,√dt)`; additive noise per var |
| NaN/Inf safety | `widget.js` | Post-step detection; trajectory truncation; red indicator |
| Display validation | `widget.js` | Guard 0/1/duplicate/nonexistent display vars |
| Web Worker | `widget.js` | Offload heavy RHS eval (N≥6) to Worker; fallback to main thread |
| Built-in models | `models.py`, `widget.js` | Expose full state var metadata |

**Deliverable:** All 3 built-in models work unchanged. 3D Rössler model loads and projects correctly. Stochastic trajectories run. No UI blocking.

### Phase B: Symbolic Infrastructure

**Goal:** ModelSpec, SymPy, Nerdamer inline, transpiler, API.

| Task | File | Details |
|------|------|---------|
| Inline Nerdamer | `widget.js` | Minified core (~100KB) as module prefix |
| Custom function registry | `widget.js` | `registerCustomFunction()` with dep sort |
| `ModelSpec` class | `model_spec.py` | Validation, SymPy parse, `to_widget_state()` |
| `sympy_to_js` | `sympy_js.py` | Full FUNCTION_MAP coverage; property-based tests |
| `phase_plane()` | `__init__.py` | String-based API; auto-detect SymPy input |
| Widget traitlet | `widget.py` | `model_spec` Dict synced to JS |
| JS model_spec handler | `widget.js` | Parse → compile → register → switch |
| Standalone HTML | `widget.py` | Embed model_spec in initialState |

**Deliverable:** `phase_plane(equations=['...'], ...)` works in Jupyter. Standalone HTML contains custom model. Transpiler tests pass with <1e-12 error.

### Phase C: Live Editor + Polish

**Goal:** The magical editor, Python sync, examples.

| Task | File | Details |
|------|------|---------|
| Editor UI | `widget.js` | Collapsible panel; textarea per equation; param table |
| State machine | `widget.js` | IDLE→PARSING→COMPILING→VALID/ERROR |
| Error UX | `widget.js` | Per-equation error highlighting; graceful degradation |
| Python sync | `widget.js`, `widget.py` | 500ms debounce, request IDs, stale-reject |
| SymPy re-analysis | `model_spec.py` | 5s timeout `sympy.solve()`; fallback to numerical |
| Copy Spec | `widget.js` | JSON export of current model |
| Examples | `examples/` | van der Pol, Lotka-Volterra, Brusselator, Rössler, stochastic resonance |

**Deliverable:** Full feature set. Edit equations live, instant redraw. Jupyter sync. Offline standalone.

---

## File Layout

```
src/phase_plane_widget/
  __init__.py          # phase_plane() API
  model_spec.py        # ModelSpec, SymPy parsing, timeout handling
  sympy_js.py          # Transpiler + FUNCTION_MAP + property tests
  widget.py            # Traitlet wrapper + standalone HTML export
  static/
    widget.js          # N-dim + Heun + Nerdamer + editor
    widget.css         # Editor panel styles
    nerdamer.min.js    # (or inlined at top of widget.js)
examples/
  custom_model_demo.py
  stochastic_demo.py
PLAN.md
```

---

## Testing

| Phase | Test |
|-------|------|
| A | Built-in models unchanged; 3D Rössler projection; stochastic variance ∝ σ²t; NaN detection |
| B | Property-based transpiler (1000 random expressions, 100 points each, error < 1e-12); `phase_plane()` round-trip |
| C | Live edit: change `x**3→x**5`, instant redraw; rapid edits (no race); JSON export/import |

---

*Revised 2026-04-24 — incorporating DeepSeek v4 review*
