"""Test safety guards in the JavaScript computation engine.

This test extracts the JS source from PhasePlaneWidget and runs it
through Node.js to verify:
  - rk4 doesn't blow up with large t_max
  - NaN returns truncated trajectory
  - Budget guards in findFixedPoints, computeNullclines, computeVectorField
  - 1D Newton-Raphson in findFixedPoints
"""

import json
import os
import subprocess
import sys
import tempfile

import phase_plane_widget


def _get_js_source():
    """Get the full JS source from the widget (same as _esm)."""
    # The widget stores its JS inline; grab it from the class.
    return phase_plane_widget.widget.PhasePlaneWidget._esm


def _run_node_script(js_code: str) -> dict:
    """Evaluate JS code in Node.js and return the JSON result."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False
    ) as f:
        f.write(js_code)
        tmpfile = f.name

    try:
        result = subprocess.run(
            ["node", tmpfile],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Node.js exited with code {result.returncode}:\n"
                f"STDERR: {result.stderr}\n"
                f"STDOUT: {result.stdout}"
            )
        return json.loads(result.stdout.strip())
    finally:
        os.unlink(tmpfile)


# ═══════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════

def test_rk4_large_tmax_does_not_blow_up():
    """rk4 with large t_max should not blow up (step budget stops it)."""
    js_src = _get_js_source()

    # We use the Wilson-Cowan model which has a stable fixed point.
    # With t_max=1e8 and dt=0.01, the unsafe rk4 would try 1e10 steps.
    # The safe rk4 caps at 50_000 steps.
    test_code = f"""
{js_src}

const modelName = "wilson_cowan";
const params = {{
  aee: 10.0, aei: 10.0, aie: 10.0, aii: 2.0,
  Pe: -2.0, Pi: -8.0,
  ke: 1.0, ki: 1.0,
  thetae: 4.0, thetai: 4.0,
}};
const y0 = [0.5, 0.3];

// rk4 with huge t_max — safe wrapper should cap steps
const traj = rk4(modelName, y0, 0, 1e8, 0.01, params);

console.log(JSON.stringify({{
  length: traj.length,
  last_t: traj[traj.length - 1][0],
  has_nan: traj.some(row => row.some(v => typeof v !== 'number' || !isFinite(v))),
  first: traj[0],
  last: traj[traj.length - 1],
}}));
"""
    result = _run_node_script(test_code)
    assert result["length"] <= 50001, (
        f"Expected ≤50001 steps, got {result['length']}"
    )
    assert result["has_nan"] is False, "Trajectory should not contain NaN/Inf"
    assert result["last_t"] < 1e8, (
        f"Expected truncated t (budget), got {result['last_t']}"
    )


def test_rk4_nan_returns_truncated():
    """rk4 should return truncated trajectory when NaN is encountered."""
    js_src = _get_js_source()

    # For a NaN test we use the MPR model with an extreme IC that
    # triggers sqrt/divergence.
    test_code = f"""
{js_src}

const modelName = "mpr";
const params = {{ delta: 1.0, eta_bar: -5.0, J: 15.0, I: 0.0 }};
const y0 = [1000, 1000];  // extreme IC sure to blow up

const traj = rk4(modelName, y0, 0, 100, 0.01, params);

console.log(JSON.stringify({{
  length: traj.length,
  last_t: traj[traj.length - 1][0],
  truncated: traj.length < 10000,
  has_nan: traj.some(row => row.some(v => typeof v !== 'number' || !isFinite(v))),
}}));
"""
    result = _run_node_script(test_code)
    assert result["truncated"], "Expected truncated trajectory on NaN"
    assert result["has_nan"] is False, "Output should not contain NaN/Inf"
    assert result["length"] < 10000, (
        f"Expected short trajectory on blow-up, got {result['length']}"
    )


def test_find_fixed_points_budget_guard():
    """findFixedPoints returns [] when budget exhausted."""
    js_src = _get_js_source()

    # Use a tiny grid with many seeds to hit budget.
    # The RHS will be slow to converge so iterations pile up.
    test_code = f"""
{js_src}

const fProxy = (_t, [x, y], _p) => [Math.sin(x * 20), Math.cos(y * 20)];
const params = {{}};
const result = findFixedPoints(fProxy, params, [-10, 10], [-10, 10], 25);

console.log(JSON.stringify({{
  length: result.length,
  is_array: Array.isArray(result),
}}));
"""
    result = _run_node_script(test_code)
    assert result["is_array"], "Should return an array"
    # With many oscillations, budget gets exhausted so result may be []
    # or very short, but never crashes.
    assert result["length"] == 0, (
        f"Expected empty array on budget exhaustion, got length {result['length']}"
    )


def test_find_fixed_points_1d():
    """findFixedPoints works for 1D (single-variable) systems."""
    js_src = _get_js_source()

    # f(x) = x - 2  → fixed point at x=2
    test_code = f"""
{js_src}

const f1D = (_t, [x], _p) => [x - 2.0];
const params = {{}};
const fps = findFixedPoints(f1D, params, [-5, 5], [-5, 5], 25);

console.log(JSON.stringify({{
  length: fps.length,
  fps: fps,
}}));
"""
    result = _run_node_script(test_code)
    assert result["length"] >= 1, "Should find at least one fixed point"
    x_fp = result["fps"][0][0]
    assert abs(x_fp - 2.0) < 0.1, f"Expected x≈2, got {x_fp}"


def test_compute_nullclines_1d():
    """computeNullclines returns single array for 1D."""
    js_src = _get_js_source()

    # f(x) = x - 1 → nullcline crossing at x=1
    test_code = f"""
{js_src}

const f1D = (_t, [x], _p) => [x - 1.0];
const params = {{}};
const [ncX, ncY] = computeNullclines(f1D, params, [-5, 5], [-5, 5], 60);

console.log(JSON.stringify({{
  ncX_length: ncX.length,
  ncY_length: ncY.length,
  first_ncX: ncX.length > 0 ? ncX[0] : null,
}}));
"""
    result = _run_node_script(test_code)
    assert result["ncX_length"] > 0, "1D nullcline should have crossings"
    assert result["ncY_length"] == 0, "1D should have no Y nullcline"
    # ncX[0] should be near [1, 0]
    x_val = result["first_ncX"][0]
    assert abs(x_val - 1.0) < 0.1, f"Expected nullcline at x≈1, got {x_val}"


def test_compute_vector_field_1d():
    """computeVectorField returns 1D arrows for 1D."""
    js_src = _get_js_source()

    test_code = f"""
{js_src}

const f1D = (_t, [x], _p) => [-0.5 * x];
const params = {{}};
const vf = computeVectorField(f1D, params, [-5, 5], [-5, 5], 12);

console.log(JSON.stringify({{
  length: vf.length,
  all_y_zero: vf.every(v => v[1] === 0),
  all_dy_zero: vf.every(v => v[3] === 0),
  sample: vf.slice(0, 3),
}}));
"""
    result = _run_node_script(test_code)
    assert result["length"] == 12, f"Expected 12 vectors, got {result['length']}"
    assert result["all_y_zero"], "All vectors should have y=0 in 1D"
    assert result["all_dy_zero"], "All vectors should have dy=0 in 1D"


def test_widget_instantiation():
    """Widget can be created with Wilson-Cowan defaults."""
    widget = phase_plane_widget.PhasePlaneWidget(model_name="wilson_cowan")
    assert widget.model_name == "wilson_cowan"
    assert "aee" in widget.params
    assert widget.t_max == 100.0
    # Traits should be synced
    assert len(widget.state_names) == 2


if __name__ == "__main__":
    import inspect

    tests = [
        (test_widget_instantiation, "Widget instantiation"),
        (test_rk4_large_tmax_does_not_blow_up, "rk4 large t_max"),
        (test_rk4_nan_returns_truncated, "rk4 NaN truncation"),
        (test_find_fixed_points_budget_guard, "findFixedPoints budget guard"),
        (test_find_fixed_points_1d, "findFixedPoints 1D support"),
        (test_compute_nullclines_1d, "computeNullclines 1D support"),
        (test_compute_vector_field_1d, "computeVectorField 1D support"),
    ]

    passed = 0
    failed = 0
    for fn, name in tests:
        print(f"  {name} ... ", end="")
        try:
            fn()
            print("OK")
            passed += 1
        except Exception as e:
            print(f"FAIL\n    {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
