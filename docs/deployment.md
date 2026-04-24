# Deployment Scenarios

The phase plane widget supports three deployment modes, each optimized for a different use case.

## 1. Jupyter Notebook / JupyterLab

The native environment. All interactivity is client-side after the initial cell execution.

```python
from phase_plane_widget import PhasePlaneWidget

widget = PhasePlaneWidget(model_name="mpr")
widget.params["J"] = 20.0
widget
```

### How It Works

1. Python creates the widget and syncs initial state via `traitlets`
2. `anywidget` loads the JS front-end in the notebook
3. **All subsequent interaction** (sliders, clicks, sweeps) is handled by JavaScript — no Python kernel round-trips
4. Python can still read back computed data (nullclines, fixed points, trajectories) via the synced traitlets

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
from phase_plane_widget import PhasePlaneWidget
widget = PhasePlaneWidget()
widget  # Renders in the VS Code output panel
```

No extra configuration needed — the same `anywidget` bridge works transparently.

---

## 3. Standalone HTML (No Kernel)

For blogs, documentation, course materials, or any static site where you cannot run a Python kernel:

```python
widget = PhasePlaneWidget(model_name="mpr")
widget.params["delta"] = 1.0
widget.params["eta_bar"] = -5.0
widget.params["J"] = 15.0
widget.to_standalone_html("mpr_demo.html", title="MPR Phase Plane")
```

### What You Get

A **single self-contained `.html` file** (~40–50 KB) containing:

- All three model definitions (Wilson-Cowan, FHN, MPR)
- RK4 ODE solver
- Newton-Raphson fixed-point finder
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

## 4. ipywidgets `embed_minimal_html`

For a traditional ipywidgets-based export (requires a web server with widget support):

```python
from ipywidgets.embed import embed_minimal_html
from phase_plane_widget import PhasePlaneWidget

widget = PhasePlaneWidget()
embed_minimal_html("export.html", views=[widget], title="Phase Plane",
                   drop_defaults=False)
```

!!! warning "Requires widget JS runtime"

    `embed_minimal_html` produces a file that references the `anywidget` CDN bundle via RequireJS. This **does not work** when opened as a local file (`file://`) because RequireJS cannot load ES modules. Use `to_standalone_html()` instead for true offline/self-contained deployment.

---

## Comparison

| Scenario | Python Kernel | Internet | File Size | Best For |
|----------|--------------|----------|-----------|----------|
| Jupyter Notebook | Required at runtime | Optional | ~40KB JS | Research, exploration |
| VS Code | Required at runtime | Optional | ~40KB JS | IDE-based workflow |
| **Standalone HTML** | **None** | **None** | **~45KB** | **Docs, blogs, courses** |
| `embed_minimal_html` | Required to generate | Required to view | ~2KB + CDN | Sharing with Jupyter users |

---

## CI/CD: Auto-Generated Documentation

This site is built automatically by a [GitHub Actions workflow](https://github.com/YOUR_USERNAME/phase-plane-widget/blob/main/.github/workflows/docs.yml) that:

1. Installs the package
2. Generates standalone HTML demos for each model
3. Builds mkdocs with mkdocstrings API docs
4. Deploys to GitHub Pages

The demos are live widgets — you can interact with them directly in the documentation.
