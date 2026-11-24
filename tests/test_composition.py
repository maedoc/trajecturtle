import json
import re
import subprocess
import shutil
from pathlib import Path

import pytest
from traitlets import TraitError

from tvb_phaseplane import ModelSpec, PhasePlaneWidget, phase_plane, explore


def test_explore_is_notebook_ready_and_exports_picker(tmp_path):
    lab = explore(params={'I': .5}, parameters=['I', 'epsilon'], t_max=80)
    assert isinstance(lab, PhasePlaneWidget)
    assert lab.model_name == 'fitzhugh_nagumo'
    assert lab.presentation == 'figure'
    assert lab.view_switcher is True
    assert lab.visible_parameters == ['I', 'epsilon']
    assert lab.params['I'] == .5 and lab.t_max == 80
    sibling = lab.explore('comparison', parameters=['I'])
    assert sibling.experiment is lab and sibling.view_switcher
    assert sibling.presentation == 'comparison'
    assert lab.presentation == 'figure'
    assert not lab.present().view_switcher
    path = tmp_path / 'lab.html'
    lab.to_standalone_html(path)
    assert '"view_switcher": true' in path.read_text()


def test_views_reference_single_experiment():
    w = PhasePlaneWidget(model_name="fitzhugh_nagumo", params={"I": 0.7})
    other = PhasePlaneWidget()
    controls = w.view(["parameters"], parameters=["I"])
    plot = w.view(["phase_plane"])
    assert controls.experiment is plot.experiment is w
    assert controls.get_state()["experiment"] == "IPY_MODEL_" + w.model_id
    assert w.experiment_id != other.experiment_id
    assert w.params["I"] == 0.7 and "epsilon" in w.params
    assert controls.visible_parameters == ["I"]
    assert plot.panels == ["phase_plane"]


@pytest.mark.parametrize('presentation', ['figure', 'bench', 'drawer', 'comparison', 'lesson'])
def test_experimental_presentations_share_state_and_export(tmp_path, presentation):
    w = PhasePlaneWidget(model_name='fitzhugh_nagumo', params={'I': .5})
    view = w.present(presentation, parameters=['I'])
    assert view.experiment is w
    assert view.presentation == presentation
    assert view.visible_parameters == ['I']
    path = tmp_path / 'presentation.html'
    w.to_standalone_html(path, views=[{'presentation': presentation, 'visible_parameters': ['I']}])
    assert f'"presentation": "{presentation}"' in path.read_text()
    assert 'name="viewport"' in path.read_text()
    assert 'reference_scene' in w.get_state()
    assert w.show_reference is True
    with pytest.raises(TraitError):
        w.present('invalid')
    with pytest.raises(ValueError, match='Invalid presentation'):
        w.to_standalone_html(path, views=[{'presentation': 'invalid'}])


def test_custom_spec_constructor_matches_setter_and_preserves_overrides():
    spec = ModelSpec.from_strings(
        equations=['a*x - y', 'x - b*y'],
        state_vars={'x': (-3, 3), 'y': (-2, 2)},
        params={'a': (1., 0., 2.), 'b': (.5, 0., 1.)},
        integrator='heun', noise_per_var=[.1, .2],
    ).to_widget_state()
    expected = PhasePlaneWidget()
    expected.set_model_spec(spec)
    actual = PhasePlaneWidget(model_spec=spec)
    for key in ('model_name', 'model_spec', 'params', 'param_info', 'state_names',
                'display', 'clamped', 'xlim', 'ylim', 'x0', 'y0',
                'integrator', 'noise_enable', 'noise_sigma'):
        assert getattr(actual, key) == getattr(expected, key), key
    overridden = PhasePlaneWidget(
        model_name='custom', model_spec=spec, params={'a': 1.7},
        xlim=[-1, 1], x0=.3, y0=.4, integrator='rk4',
        noise_enable=False, noise_sigma=[0., 0.],
    )
    assert overridden.params == {'a': 1.7, 'b': .5}
    assert overridden.xlim == [-1, 1]
    assert overridden.x0 == .3 and overridden.y0 == .4
    assert overridden.integrator == 'rk4'
    assert not overridden.noise_enable
    assert overridden.noise_sigma == [0., 0.]


def test_custom_spec_uses_projected_x_variable_limits():
    spec = ModelSpec.from_strings(
        equations=['-x', '-y', '-z'],
        state_vars={'x': (-1, 1), 'y': (-2, 2), 'z': (3, 5)},
        display=['z', 'x'], params={},
    ).to_widget_state()
    w = PhasePlaneWidget(model_spec=spec)
    assert w.display == [2, 0]
    assert w.xlim == [3, 5] and w.x0 == 4
    assert w.ylim == [-1, 1]


def test_invalid_panel():
    with pytest.raises(TraitError):
        PhasePlaneWidget(panels=["bad"])
    with pytest.raises(TraitError):
        PhasePlaneWidget().view([])


def test_compose_export(tmp_path):
    w = PhasePlaneWidget(panels=["parameters", "time_series"], visible_parameters=["Pe"], panel_layout="vertical")
    path = tmp_path / "figure.html"
    w.to_standalone_html(path)
    state = json.loads(re.search(r"const initialState = (\{.*?\});", path.read_text(), re.S).group(1))
    assert state["panels"] == ["parameters", "time_series"]
    assert state["visible_parameters"] == ["Pe"]
    assert state["panel_layout"] == "vertical"


def test_custom_noise_settings():
    w = phase_plane(["-x", "-y"], state_vars={"x": (-1, 1), "y": (-1, 1)}, integrator="heun", noise_per_var=[0.2, 0.1])
    assert w.integrator == "heun"
    assert w.noise_enable
    assert w.noise_sigma == [0.2, 0.1]


def test_linked_html_export(tmp_path):
    w = PhasePlaneWidget()
    views = [{"panels": ["parameters"], "visible_parameters": ["Pe"]}, {"panels": ["phase_plane"]}]
    output = tmp_path / "linked.html"
    w.to_standalone_html(output, views=views)
    content = output.read_text()
    assert json.loads(re.search(r"const viewConfigs = (.*?);", content).group(1)) == views
    with pytest.raises(ValueError):
        w.to_standalone_html(output, views=[{"panels": ["invalid"]}])


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js required")
def test_shared_engine_caches_geometry_and_trajectory():
    asset = Path(__file__).resolve().parents[1] / "src/tvb_phaseplane/static/widget.js"
    script = f"""
globalThis.window = globalThis;
const {{computeExperiment, experimentSession, resetExperiment, referenceIdentity, sweepIdentity}} = await import({json.dumps(asset.as_uri())});
const state = {{model_name: 'fitzhugh_nagumo', params: {{a: .7, b: .8, epsilon: .08, I: .5}},
  xlim: [-3, 3], ylim: [-1.5, 2], x0: .1, y0: .1, t_max: 2, display: [0, 1], integrator: 'rk4'}};
const model = {{get: key => state[key], set: (key, value) => state[key] = value}};
computeExperiment(model);
const geometry = state.fixed_points, trajectory = state.trajectory;
computeExperiment(model);
if (geometry !== state.fixed_points || trajectory !== state.trajectory) throw Error('cache miss');
state.x0 = .2;
computeExperiment(model);
if (geometry !== state.fixed_points || trajectory === state.trajectory) throw Error('wrong invalidation');
if (state.trajectory[0][1] !== .2) throw Error('stale initial condition');
state.params = {{...state.params, I: -.5}};
computeExperiment(model);
if (geometry === state.fixed_points) throw Error('stale geometry');
state.integrator = 'heun'; state.noise_enable = true; state.noise_sigma = [.1, .2]; state.noise_seed = 42;
computeExperiment(model);
const stochastic = JSON.stringify(state.trajectory);
const second = {{get: key => state[key], set: (key, value) => state[key] = value}};
computeExperiment(second);
if (JSON.stringify(state.trajectory) !== stochastic) throw Error('seed is not reproducible');
state.noise_seed = 43;
computeExperiment(second);
if (JSON.stringify(state.trajectory) === stochastic) throw Error('seed has no effect');
const baseline = experimentSession(model).baseline;
experimentSession(model);
if (baseline.params.I !== .5) throw Error('baseline changed with experiment');
state.reference_metadata = {{identity: referenceIdentity(model), params: {{...state.params}}}};
state.reference_trajectory = state.trajectory;
state.params = {{...state.params, I: 1}};
const oldSweep = sweepIdentity(model);
computeExperiment(model);
if (!state.reference_trajectory.length) throw Error('comparison reference lost');
state.state_names = ['different', 'names'];
computeExperiment(model);
if (state.reference_trajectory.length) throw Error('incompatible reference retained');
state.x0 = .7;
if (oldSweep === sweepIdentity(model)) throw Error('sweep identity ignored changes');
resetExperiment(model);
if (state.params.I !== .5 || state.x0 !== .1 || state.reference_trajectory.length) throw Error('reset failed');
"""
    subprocess.run(["node", "--input-type=module", "-e", script], check=True, capture_output=True, text=True)
