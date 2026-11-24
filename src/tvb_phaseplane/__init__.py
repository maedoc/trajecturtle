"""Interactive phase plane widget for neural mass models."""

from .models import BaseModel, FitzHughNagumo, MPRModel, WilsonCowan, MODEL_REGISTRY
from .model_spec import ModelSpec
from .widget import PhasePlaneWidget

__all__ = [
    "PhasePlaneWidget",
    "explore",
    "phase_plane",
    "BaseModel",
    "WilsonCowan",
    "FitzHughNagumo",
    "MPRModel",
    "MODEL_REGISTRY",
    "ModelSpec",
]
__version__ = "0.1.0"


def explore(model_name="fitzhugh_nagumo", *, parameters=None, presentation="figure", **kwargs):
    """Create a notebook widget with a picker for five experimental layouts.

    Parameters
    ----------
    model_name : str
        Built-in model name; defaults to FitzHugh–Nagumo.
    parameters : list[str] or None
        Primary slider names, in display order. None selects model defaults.
    presentation : str
        Starting layout: figure, bench, drawer, comparison or lesson.
    **kwargs
        Experiment settings passed to PhasePlaneWidget, e.g. params, t_max,
        x0, y0 or model_spec. The returned widget can also export static HTML.

    Examples
    --------
    >>> from tvb_phaseplane import explore
    >>> lab = explore(params={"I": 0.5}, parameters=["I", "epsilon"])
    >>> # Display lab as the last expression of a notebook cell.

    Layout switches preserve parameters, references and completed sweep results.
    Leaving a running sweep cancels it. No feedback or telemetry is sent.
    """
    return PhasePlaneWidget(model_name=model_name, presentation=presentation,
                            visible_parameters=parameters, view_switcher=True, **kwargs)


def phase_plane(
    equations,
    state_vars=None,
    params=None,
    display=None,
    custom_functions=None,
    integrator="rk4",
    noise_per_var=None,
    **kwargs,
):
    """Create a PhasePlaneWidget from user-supplied ODE equations.

    Parameters
    ----------
    equations : list[str] or dict[str, str]
        The right-hand sides of the ODE system.  When a list is passed
        the order must match ``state_vars``.  When a dict is passed
        keys are the state-variable names.
    state_vars : dict[str, tuple[float, float]]
        ``{name: (min, max)}`` for each state variable.
    params : dict[str, tuple[float, float, float]] or dict[str, Any]
        ``{name: (default, min, max)}`` or
        ``{name: (default, min, max, step)}`` for each parameter.
    display : list[str] | None
        Which state variables to show on the X / Y axes.  Defaults to
        the first two (or the first one for a 1-D system).
    custom_functions : dict[str, str] | None
        Optional helper functions expressed as strings, e.g.
        ``{"sigmoid": "1/(1+exp(-x))"}``.
    integrator : str
        ``"rk4"`` (deterministic) or ``"heun"`` (stochastic).
    noise_per_var : list[float] | None
        Per-variable noise strength for the Heun integrator.
    **kwargs
        Passed to ``ModelSpec`` (e.g. ``name=...``).

    Returns
    -------
    PhasePlaneWidget
        A fully configured interactive widget.

    Examples
    --------
    >>> from tvb_phaseplane import phase_plane
    >>> pp = phase_plane(
    ...     equations=['a*x - x**3 - y', 'x - b*y', 'c*(x - z)'],
    ...     state_vars={'x': (-3, 3), 'y': (-3, 3), 'z': (0, 5)},
    ...     params={'a': (0.7, 0, 2), 'b': (0.8, 0, 2), 'c': (10, 0, 20)},
    ...     display=['x', 'y'],
    ... )
    """
    spec = ModelSpec.from_strings(
        equations=equations,
        state_vars=state_vars or {},
        params=params or {},
        display=display,
        custom_functions=custom_functions,
        integrator=integrator,
        noise_per_var=noise_per_var,
        **kwargs,
    )
    widget = PhasePlaneWidget()
    widget.set_model_spec(spec.to_widget_state())
    return widget
