import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Phase Plane Widget — Marimo Demo

    This notebook demonstrates the **trajecturtle** phase-plane widget
    inside a marimo notebook.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from phase_plane_widget import PhasePlaneWidget

    return PhasePlaneWidget, mo


@app.cell
def _(mo):
    c_slider = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="c (excitatory gain)")
    mu_slider = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="μ (inhibitory gain)")
    tmax_slider = mo.ui.slider(10, 500, step=10, value=100, label="t_max")

    mo.hstack([c_slider, mu_slider, tmax_slider])
    return c_slider, mu_slider, tmax_slider


@app.cell
def _(PhasePlaneWidget, c_slider, mo, mu_slider, tmax_slider):
    ppw = PhasePlaneWidget(
        model="wilson_cowan",
        display_mode="phase_plane",
        params={
            "c": float(c_slider.value),
            "mu": float(mu_slider.value),
        },
        t_max=float(tmax_slider.value),
        x0=0.5,
        y0=-1.0,
    )

    mo.ui.anywidget(ppw)
    return


if __name__ == "__main__":
    app.run()
