"""One lesson configuration reused in static HTML and separate notebook outputs.

Run with ``uv run python examples/composed_lesson.py`` to write lesson.html.
In a notebook, import experiment/notebook_views, then display the returned views
in separate cells with explanatory prose between them. Both modes use the same
panel factories, experiment controller and numerical cache.
"""
from tvb_phaseplane import PhasePlaneWidget

experiment = PhasePlaneWidget(model_name="fitzhugh_nagumo", params={"I": 0.5}, t_max=80)

LESSON_VIEWS = [
    {"panels": ["parameters"], "visible_parameters": ["I"]},
    {"panels": ["phase_plane", "time_series"]},
    {"panels": ["comparison"]},
]


def notebook_views():
    """Return linked outputs without changing the experiment's presentation."""
    return [
        experiment.view(config["panels"], parameters=config.get("visible_parameters"))
        for config in LESSON_VIEWS
    ]


if __name__ == "__main__":
    experiment.to_standalone_html(
        "lesson.html", title="FitzHugh–Nagumo: input current", views=LESSON_VIEWS,
    )
