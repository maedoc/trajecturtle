"""Test the parameter sweep functionality in JS."""

import json
import os
import subprocess
import tempfile

WIDGET_JS = (os.path.join(os.path.dirname(__file__), '..', 'src', 'tvb_phaseplane', 'static', 'widget.js'))


def run_sweep_test_js(js_code: str):
    """Run JS code that appends to widget.js and prints JSON results."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write("if (typeof window === 'undefined') globalThis.window = globalThis;\n")
        f.write(open(WIDGET_JS).read().replace('export function render', 'function render'))
        f.write('\n')
        f.write(js_code)
        f.flush()
        result = subprocess.run(
            ['node', f.name],
            capture_output=True,
            text=True,
        )
        os.unlink(f.name)
    return result


def test_wilson_cowan_sweep():
    """Sweep aee in Wilson-Cowan and verify results are found."""
    js = r"""
    const model = {
        get(k) { return this['_' + k]; },
        set(k, v) { this['_' + k] = v; },
    };
    model._model_name = 'wilson_cowan';
    model._model_spec = null;
    model._params = {
        aee: 10, aei: 10, aie: 10, aii: 2,
        Pe: -2, Pi: -8,
        ke: 1, ki: 1,
        thetae: 4, thetai: 4,
    };
    model._xlim = [-0.2, 1.2];
    model._ylim = [-0.2, 1.2];
    model._display = [0, 1];
    model._clamped = [0.5, 0.5];

    const def = MODELS.wilson_cowan;
    const fProj = makeProjectedRHS('wilson_cowan', [0, 1], [0.5, 0.5]);

    const fps = findFixedPoints(fProj, model._params, model._xlim, model._ylim, 15);
    const regime = detectRegime('wilson_cowan', model._params, model._xlim, model._ylim, [0, 1], 60, 0.05);
    const sweepResult = runParameterSweep(fProj, 'wilson_cowan', 'aee', model._params, 0, 20, 10, model._xlim, model._ylim, [0, 1]);

    console.log(JSON.stringify({
        fpsCount: fps.length,
        fps: fps.slice(0, 3),
        regime: regime,
        sweepResultsCount: sweepResult.results.length,
        sweepFpsCount: sweepResult.allFps.length,
        firstFew: sweepResult.results.slice(0, 5),
    }));
    """
    result = run_sweep_test_js(js)
    assert result.returncode == 0, f"JS error: {result.stderr[:500]}\nstdout: {result.stdout[:500]}"
    data = json.loads(result.stdout.strip().split('\n')[-1])
    assert data['fpsCount'] >= 1, f"Expected at least 1 fixed point, got {data['fpsCount']}"
    assert data['sweepResultsCount'] == 10, f"Expected 10 sweep points, got {data['sweepResultsCount']}"
    assert data['sweepFpsCount'] >= 1, f"Expected at least 1 total sweep fixed point, got {data['sweepFpsCount']}"
    print(json.dumps(data, indent=2))


def test_sweep_regime_changes():
    """Sweep across a bifurcation point and verify regime changes."""
    js = r"""
    const model = {
        get(k) { return this['_' + k]; },
        set(k, v) { this['_' + k] = v; },
    };
    model._params = {
        aee: 10, aei: 10, aie: 10, aii: 2,
        Pe: -2, Pi: -8,
        ke: 1, ki: 1,
        thetae: 4, thetai: 4,
    };
    model._xlim = [-0.2, 1.2];
    model._ylim = [-0.2, 1.2];
    model._display = [0, 1];

    const fProj = makeProjectedRHS('wilson_cowan', [0, 1], [0.5, 0.5]);
    const result = runParameterSweep(fProj, 'wilson_cowan', 'aee', model._params, 0, 20, 20, model._xlim, model._ylim, [0, 1]);

    const regimes = result.results.map(r => r.regime);
    const uniqueRegimes = [...new Set(regimes)];

    console.log(JSON.stringify({
        regimes,
        uniqueRegimes,
        hasFixedPoint: uniqueRegimes.includes('fixed_point'),
        hasLimitCycle: uniqueRegimes.includes('limit_cycle'),
    }));
    """
    result = run_sweep_test_js(js)
    assert result.returncode == 0, f"JS error: {result.stderr[:500]}\nstdout: {result.stdout[:500]}"
    data = json.loads(result.stdout.strip().split('\n')[-1])
    assert data['hasFixedPoint'] or data['hasLimitCycle'], f"Expected dynamics other than just 'other': {data['uniqueRegimes']}"
    print(json.dumps(data, indent=2))


def test_find_fixed_points_matches_equilibrium():
    """For a simple linear system with known fixed point, verify finder finds it."""
    js = r"""
    const f = (t, [x, y], p) => [p.a * x - y, x - p.b * y];
    const params = { a: 1.0, b: 1.0 };
    const xlim = [-3, 3];
    const ylim = [-3, 3];
    const fps = findFixedPoints(f, params, xlim, ylim, 25);
    console.log(JSON.stringify({
        fpsCount: fps.length,
        fps: fps,
    }));
    """
    result = run_sweep_test_js(js)
    assert result.returncode == 0, f"JS error: {result.stderr[:500]}\nstdout: {result.stdout[:500]}"
    data = json.loads(result.stdout.strip().split('\n')[-1])
    assert data['fpsCount'] >= 1, f"Expected at least 1 fixed point for linear system, got {data['fpsCount']}"
    origin_found = any(abs(fp[0]) < 0.1 and abs(fp[1]) < 0.1 for fp in data['fps'])
    assert origin_found, f"Expected to find origin (0,0), got {data['fps']}"
