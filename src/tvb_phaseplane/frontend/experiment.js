const EXPERIMENT_CACHE = Symbol.for('trajecturtle.experiment-cache');
const EXPERIMENT_SESSION = Symbol.for('trajecturtle.experiment-session');
const RESET_KEYS = ['model_spec','model_name','param_info','state_names','display','clamped','params','x0','y0','xlim','ylim','t_max','noise_seed','integrator','noise_enable','noise_sigma'];

export function experimentSession(model) {
  return model[EXPERIMENT_SESSION] ||= {
    baseline: Object.fromEntries(RESET_KEYS.map(key => [key, structuredClone(model.get(key))])),
  };
}

export function resetExperiment(model) {
  for (const [key, value] of Object.entries(experimentSession(model).baseline)) model.set(key, structuredClone(value));
  model.set('reference_trajectory', []);
  model.set('reference_scene', {});
  model.set('reference_metadata', {});
  model.set('trajectory_fraction', 1);
}

export function referenceIdentity(model) {
  return JSON.stringify([model.get('model_name'), model.get('model_spec'), model.get('state_names')]);
}

export function sweepIdentity(model) {
  return JSON.stringify(RESET_KEYS.map(key => model.get(key)));
}

export function computeExperiment(model) {
  try {
    return computeExperimentState(model);
  } catch (error) {
    delete model[EXPERIMENT_CACHE];
    for (const key of ['nullcline_x', 'nullcline_y', 'vector_field', 'fixed_points', 'trajectory']) model.set(key, []);
    throw error;
  }
}

function computeExperimentState(model) {
  experimentSession(model);
  const metadata = model.get('reference_metadata');
  const scene = model.get('reference_scene');
  if (((model.get('reference_trajectory') || []).length || scene?.display) &&
      (metadata?.identity !== referenceIdentity(model) || (scene?.display && JSON.stringify(scene.display) !== JSON.stringify(model.get('display'))))) {
    model.set('reference_trajectory', []);
    model.set('reference_scene', {});
    model.set('reference_metadata', {});
  }
  const modelName = model.get("model_name");
  const params = model.get("params");
  const xlim = model.get("xlim");
  const ylim = model.get("ylim");
  const tMax = model.get("t_max");
  const x0 = model.get("x0");
  const y0 = model.get("y0");

  const def = resolveModel(modelName, model.get('model_spec'));
  const stateVars = def.stateVars || def.stateNames || [];
  const n = stateVars.length;

  // Display indices and clamped (non-displayed) state variable values
  const display = model.get("display") || def.display || [0, 1];
  let clamped = model.get("clamped");
  if (!clamped || clamped.length !== n) {
    clamped = stateVars.map((_, i) => {
      const lims = (def.defaultLims && def.defaultLims[i]) || [-1, 1];
      return (lims[0] + lims[1]) / 2;
    });
    model.set("clamped", clamped);
  }

  // Projected RHS for phase-plane computations
  const _spec4proj = modelName === 'custom' ? model.get('model_spec') : null;
  const cache = model[EXPERIMENT_CACHE] ||= {};
  const geometryKey = JSON.stringify([modelName, _spec4proj, params, display, clamped, xlim, ylim]);
  if (cache.geometry !== geometryKey) {
    const fProj = makeProjectedRHS(modelName, display, clamped, _spec4proj, def);
    const [ncX, ncY] = computeNullclines(fProj, params, xlim, ylim, 60);
    model.set("nullcline_x", ncX);
    model.set("nullcline_y", ncY);
    model.set("vector_field", computeVectorField(fProj, params, xlim, ylim, 12));
    model.set("fixed_points", findFixedPoints(fProj, params, xlim, ylim, 25));
    cache.geometry = geometryKey;
  }

  // Trajectory: integrate the full N-dimensional system
  const fullState0 = [...clamped];
  fullState0[display[0]] = x0;
  if (display.length > 1) fullState0[display[1]] = y0;
  const integrator = model.get('integrator') || 'rk4';
  const noiseSigma = (integrator === 'heun' && model.get('noise_enable'))
                     ? model.get('noise_sigma') : null;
  const settings = { integrator, noiseSigma, seed: model.get('noise_seed') || 0 };
  const trajectoryKey = JSON.stringify([modelName, _spec4proj, params, fullState0, tMax, settings]);
  if (cache.trajectory !== trajectoryKey) {
    const trajDef = (modelName === 'custom' && def && def.f) ? def : modelName;
    const traj = simulate(trajDef, fullState0, 0, tMax, 0.01, params, settings);
    const step = Math.max(1, Math.ceil(traj.length / 2000));
    const trajDisplay = traj.filter((_, i) => i % step === 0 || i === traj.length - 1);
    model.set("trajectory", trajDisplay);
    cache.trajectory = trajectoryKey;
  }
  return def;
}
