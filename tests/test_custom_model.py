"""Tests for custom model compilation via nerdamer in the JS widget.

These tests verify the Python-to-JS pipeline for custom ODE models:
- ModelSpec construction builds correctly
- set_model_spec populates widget traits correctly
- The model_spec dict has the right structure for JS compilation
"""

import json

import numpy as np

from phase_plane_widget import ModelSpec, PhasePlaneWidget, phase_plane


def test_custom_linear_2d_model_spec():
    """Test that a simple 2D linear system produces a valid ModelSpec."""
    # dx/dt = a*x - y
    # dy/dt = x - b*y
    spec = ModelSpec.from_strings(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
        params={"a": (1.0, 0.0, 2.0), "b": (1.0, 0.0, 2.0)},
        name="Linear2D",
    )

    assert spec.name == "Linear2D"
    assert list(spec.state_vars.keys()) == ["x", "y"]
    assert list(spec.parameters.keys()) == ["a", "b"]
    assert spec.display == ["x", "y"]

    # Convert to widget state
    state = spec.to_widget_state()
    assert state["display"] == [0, 1]
    assert "x" in state["equations"]
    assert "y" in state["equations"]
    assert state["equations"]["x"] == "a*x - y"
    assert state["equations"]["y"] == "x - b*y"
    assert state["parameters"]["a"]["default"] == 1.0
    assert state["parameters"]["a"]["range"] == [0.0, 2.0]
    assert state["parameters"]["b"]["default"] == 1.0


def test_phase_plane_creates_widget_with_model_spec():
    """Test that phase_plane() creates a widget with compiled model_spec."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )

    assert isinstance(pp, PhasePlaneWidget)
    assert pp.model_name == "custom"
    assert pp.model_spec is not None
    assert "x" in pp.model_spec["equations"]
    assert "y" in pp.model_spec["equations"]
    assert pp.model_spec["display"] == [0, 1]


def test_model_spec_has_correct_dimensions():
    """Test that the model_spec produces a widget with correct dimensionality."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )

    n_state_vars = len(pp.model_spec["state_vars"])
    assert n_state_vars == 2, f"Expected 2 state vars, got {n_state_vars}"

    n_params = len(pp.model_spec["parameters"])
    assert n_params == 2, f"Expected 2 params, got {n_params}"

    # The spec has equations for both vars
    assert len(pp.model_spec["equations"]) == 2


def test_default_params_are_correct():
    """Test that default parameter values are set correctly from model_spec."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )

    assert "a" in pp.params
    assert "b" in pp.params
    assert pp.params["a"] == 1.0
    assert pp.params["b"] == 1.0

    # param_info should have the correct structure
    assert "a" in pp.param_info
    assert "b" in pp.param_info
    assert pp.param_info["a"][:3] == [0.0, 2.0, 1.0]  # [min, max, default]
    assert pp.param_info["b"][:3] == [0.0, 2.0, 1.0]


def test_state_names_are_correct():
    """Test that state_names trait contains the variable names in order."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )

    assert pp.state_names == ["x", "y"]


def test_limits_are_set_from_state_var_ranges():
    """Test that xlim/ylim are derived from state variable ranges."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )

    # xlim from first var range, ylim from second
    assert pp.xlim == [-3.0, 3.0], f"Expected xlim [-3,3], got {pp.xlim}"
    assert pp.ylim == [-3.0, 3.0], f"Expected ylim [-3,3], got {pp.ylim}"


def test_model_spec_to_json_serializable():
    """Test that model_spec is JSON-serializable (required for widget sync)."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )

    # Should not raise
    json_str = json.dumps(pp.model_spec)
    parsed = json.loads(json_str)
    assert parsed["display"] == [0, 1]
    assert parsed["equations"]["x"] == "a*x - y"


def test_display_is_set():
    """Test that the display trait is initialized correctly."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"a": (1.0, 0, 2), "b": (1.0, 0, 2)},
    )

    assert pp.display == [0, 1]


def test_custom_model_with_custom_functions():
    """Test model_spec with custom functions in the spec."""
    spec = ModelSpec.from_strings(
        equations=["f(x) + y", "x - b*y"],
        state_vars={"x": (-3, 3), "y": (-3, 3)},
        params={"b": (1.0, 0, 2)},
        custom_functions={"f": "x^2"},
    )

    state = spec.to_widget_state()
    assert "custom_functions" in state
    assert "f" in state["custom_functions"]


def test_3d_model_projection():
    """Test that a 3D model produces correct display and state_names."""
    pp = phase_plane(
        equations=["a*x - y", "x - b*y", "c*(x - z)"],
        state_vars={"x": (-3, 3), "y": (-3, 3), "z": (0, 5)},
        params={"a": (0.7, 0, 2), "b": (0.8, 0, 2), "c": (10, 0, 20)},
        display=["x", "y"],
    )

    assert pp.state_names == ["x", "y", "z"]
    assert pp.model_spec["display"] == [0, 1]
    assert len(pp.model_spec["equations"]) == 3
    assert pp.model_spec["equations"]["z"] is not None
