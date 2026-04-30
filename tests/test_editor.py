"""Test the live editor UI components."""

from tvb_phaseplane import PhasePlaneWidget, phase_plane


def test_standalone_html_has_editor_elements():
    """The standalone HTML must contain editor DOM elements."""
    w = PhasePlaneWidget()
    html = w._esm
    assert 'ppw-editor-toggle' in html
    assert 'ppw-editor-panel' in html
    assert 'ppw-editor-equation' in html
    assert 'ppw-editor-apply' in html
    assert 'ppw-editor-copy' in html
    assert 'ppw-ed-expr' in html
    assert 'ppw-params' in html, "params container missing from DOM template"


def test_custom_model_sets_editor_spec():
    """phase_plane creates a widget with model_spec that the editor can use."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )
    assert pp.model_name == "custom"
    assert pp.model_spec is not None
    assert "x" in pp.model_spec["equations"]
    assert "y" in pp.model_spec["equations"]
