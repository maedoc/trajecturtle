# Deployment Scenarios

The phase plane widget supports three deployment modes, each optimized for a different use case.

## 1. Jupyter Notebook / JupyterLab

The native environment. All interactivity is client-side after the initial cell execution.

```python
from tvb_phaseplane import PhasePlaneWidget

widget = PhasePlaneWidget(model_name="mpr")
widget.params["J"] = 20.0
widget
```

### How It Works

1. Python creates the widget and syncs initial state via `traitlets`
2. `anywidget` loads the JS front-end in the notebook
3. **All subsequent interaction** (sliders, clicks, sweeps, noise toggles) is handled by JavaScript
4. Python can still read back computed data (nullclines, fixed points, trajectories) via synced traitlets

### Custom Models in Jupyter {#custom-models}

```python
from tvb_phaseplane import phase_plane

pp = phase_plane(
    equations=["a*x - x**3 - y", "x - b*y"],
    state_vars={"x": (-3, 3), "y": (-3, 3)},
    params={"a": (0.7, 0, 2), "b": (0.8, 0, 2)},
)
pp
```

The SymPy expressions are transpiled to JavaScript via an inlined Nerdamer CAS (~100 KB) that compiles and runs entirely in the browser.

### Passing a Model Instance {#model-instance}

You can also instantiate a model class first, configure its defaults, and then pass it to the widget. This is useful when you want to programatically set initial parameter values or read back tuned values after the user interacts with the sliders.

```python
from tvb_phaseplane import PhasePlaneWidget, MPRModel

# 1. Create a model instance
model = MPRModel()

# 2. Optionally override default parameter values
model.default_params.update({"J": 15.0, "eta_bar": -5.0})

# 3. Pass the instance to the widget
widget = PhasePlaneWidget(model=model)
widget
```

After the user adjusts sliders in the widget, read back the tuned parameter values:

```python
print("Current parameter values:")
for name, value in widget.params.items():
    print(f"  {name:12s} = {value:.4f}")
```

You can also read back computed data:

```python
print(f"Fixed points: {len(widget.fixed_points)}")
for fp in widget.fixed_points:
    print(f"  x={fp[0]:.4f}, y={fp[1]:.4f}, type={fp[2]}")
```

The ``model=`` argument accepts any ``BaseModel`` subclass whose ``name`` is registered in ``MODEL_REGISTRY`` (so the JavaScript front-end knows how to evaluate it). For arbitrary ODE systems that are *not* built in, use :func:`phase_plane` instead.

### Exporting Notebooks

Use [jupytext](https://jupytext.readthedocs.io/) to keep notebooks in plain `.py` percent-format:

```bash
jupytext --to notebook demo.py -o demo.ipynb
jupytext --sync demo.ipynb   # keep .py and .ipynb in sync
```

---

## 2. VS Code (Jupyter Extension)

VS Code's built-in Jupyter extension supports `anywidget` out of the box.

```python
from tvb_phaseplane import PhasePlaneWidget
widget = PhasePlaneWidget()
widget  # Renders in the VS Code output panel
```

No extra configuration needed.

---

## 3. Standalone HTML (No Kernel)

For blogs, documentation, course materials, or any static site where you cannot run a Python kernel:

```python
from tvb_phaseplane import PhasePlaneWidget

widget = PhasePlaneWidget(model_name="mpr")
widget.to_standalone_html("mpr_demo.html", title="MPR Phase Plane")
```

### Custom Models in Standalone HTML

```python
from tvb_phaseplane import phase_plane

pp = phase_plane(
    equations=["a*x - y", "x - b*y"],
    state_vars={"x": (-3, 3), "y": (-3, 3)},
    params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
)
pp.to_standalone_html("custom_model.html", title="Custom Model")
```

### What You Get

A **single self-contained `.html` file** (~700 KB) containing:

- All built-in model definitions
- Inlined Nerdamer CAS for custom model compilation
- RK4 and stochastic Heun integrators
- Newton-Raphson fixed-point finder with budget guards
- Nullcline and vector field computation
- HTML5 Canvas rendering engine
- Parameter sweep engine

No external dependencies. No CDN. No Python runtime. Works offline.

### Embedding in MkDocs / GitHub Pages

The generated HTML can be embedded in an `<iframe>`:

```html
<iframe src="demos/mpr_bistable.html"
        width="100%" height="750"
        frameborder="0"
        style="border:1px solid #ddd; border-radius:8px;">
</iframe>
```

This entire documentation site uses this technique — every [demo](demos/index.md) is a live, interactive widget running in your browser.

---

## 4. ipywidgets `embed_minimal_html` (Not Recommended)

```python
from ipywidgets.embed import embed_minimal_html
from tvb_phaseplane import PhasePlaneWidget

widget = PhasePlaneWidget()
embed_minimal_html("export.html", views=[widget], title="Phase Plane",
                   drop_defaults=False)
```

!!! warning "Requires widget JS runtime"

    `embed_minimal_html` produces a file that references the `anywidget` CDN bundle via RequireJS. This **does not work** when opened as a local file (`file://`). Use `to_standalone_html()` instead for true offline/self-contained deployment.

---

## Comparison

| Scenario | Python Kernel | Internet | File Size | Best For |
|----------|--------------|----------|-----------|----------|
| Jupyter Notebook | Required at runtime | Optional | ~700 KB JS | Research, exploration |
| VS Code | Required at runtime | Optional | ~700 KB JS | IDE-based workflow |
| **Standalone HTML** | **None** | **None** | **~700 KB** | **Docs, blogs, courses** |
| `embed_minimal_html` | Required to generate | Required to view | ~2 KB + CDN | Sharing with Jupyter users |

## Fixed-Point Detection & Classification

The widget automatically locates fixed points (equilibria) by intersecting nullclines and refining with a Newton–Raphson solver. Each detected point is **validated** with a short trajectory starting from a perturbed initial condition near the equilibrium. The trajectory is **not** displayed; it serves only as a cross-check of the eigenvalue-based classification.

### Classification Method

1. **Nullcline intersection** → candidate location  
2. **Newton–Raphson refinement** → precise `(x*, y*)`  
3. **Jacobian eigenvalues** at the fixed point → preliminary type  
4. **Dynamic validation**: a 3-second RK4 trajectory is launched from ` (x*+0.02, y*+0.02)` and compared against the eigenvalue prediction. If the two methods disagree (possible near bifurcations where numerical noise matters), the eigenvalue result is recomputed at the refined point and used as the final label.

### Visual Legend

| Symbol | Type | Meaning |
|--------|------|---------|
| ● filled circle | **Stable node** | Both eigenvalues real & negative; trajectories approach monotonically |
| ◉ target ring | **Stable focus** | Complex-conjugate eigenvalues with negative real part; spirals inward |
| ○ open circle | **Unstable node** | Both eigenvalues real & positive; trajectories diverge monotonically |
| ⊕ circled cross | **Unstable focus** | Complex-conjugate eigenvalues with positive real part; spirals outward |
| ◆ diamond | **Saddle** | Eigenvalues of opposite sign; stable manifold & unstable manifold |

The color key (green for stable, red/orange for unstable, purple for saddle) is preserved alongside the new shapes so that colour-blind users can still distinguish categories.

### References

- Scholarpedia, *Equilibrium*: http://www.scholarpedia.org/article/Equilibrium — in particular the two-dimensional analysis and Figure 3 showing the eigenvalue-based classification diagram.

This site is built automatically by a GitHub Actions workflow that:

1. Installs the package
2. Generates standalone HTML demos for each model
3. Builds mkdocs with mkdocstrings API docs
4. Deploys to GitHub Pages

The demos are live widgets — you can interact with them directly in the documentation.
