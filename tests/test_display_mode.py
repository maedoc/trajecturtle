"""Test display_mode trait (full / phase_plane) in PhasePlaneWidget."""

import os
import re
import tempfile

import pytest
import traitlets

import tvb_phaseplane
from tvb_phaseplane import PhasePlaneWidget


def test_display_mode_default_is_full():
    """Widget should default to display_mode='full'."""
    w = PhasePlaneWidget()
    assert w.display_mode == "full"


def test_display_mode_phase_plane_trait():
    """display_mode trait should accept 'phase_plane'."""
    w = PhasePlaneWidget(display_mode="phase_plane")
    assert w.display_mode == "phase_plane"


def test_display_mode_invalid_raises():
    """Invalid display_mode should raise TraitError."""
    with pytest.raises(traitlets.TraitError):
        PhasePlaneWidget(display_mode="banana")


def test_standalone_html_includes_display_mode():
    """Generated standalone HTML must embed display_mode in the initial state."""
    w = PhasePlaneWidget(display_mode="phase_plane")
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        out = f.name
        w.to_standalone_html(out, title="Test")

    with open(out, "r") as f:
        html = f.read()

    assert '"display_mode": "phase_plane"' in html


def test_standalone_default_is_full():
    """Default standalone HTML should have display_mode='full'."""
    w = PhasePlaneWidget()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        out = f.name
        w.to_standalone_html(out, title="Test")

    with open(out, "r") as f:
        html = f.read()

    assert '"display_mode": "full"' in html


def test_js_template_contains_phase_only_class():
    """The inline JS source must conditionally add ppw-phase-only class."""
    from tvb_phaseplane.widget import PhasePlaneWidget as PW

    js = PW._esm
    assert "ppw-phase-only" in js
    assert "displayMode === 'phase_plane'" in js


def test_css_contains_phase_only_rules():
    """The inline CSS must contain rules for ppw-phase-only."""
    from tvb_phaseplane.widget import PhasePlaneWidget as PW

    css = PW._css
    assert ".ppw-widget.ppw-phase-only" in css
    assert ".ppw-controls" in css
