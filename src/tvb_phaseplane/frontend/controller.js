// One controller per canonical experiment model, independent of mounted panels.
const EXPERIMENT_CONTROLLER = Symbol.for('trajecturtle.experiment-controller');
const COMPUTATION_INPUTS = ['model_spec', 'model_name', 'params', 'state_names', 'display', 'clamped', 'x0', 'y0', 'xlim', 'ylim', 't_max', 'dt', 'integrator', 'noise_enable', 'noise_sigma', 'noise_seed'];
const PRESENTATION_INPUTS = ['param_info', 'show_nullclines', 'show_vector_field', 'show_trajectory', 'show_fixed_points', 'trajectory_fraction', 'reference_trajectory', 'reference_scene', 'show_reference', 'reference_metadata', 'sweep_results', 'sweep_fixed_points', 'sweep_param', 'initial_conditions', 'observation_cards'];

export function acquireExperiment(model) {
  const controller = model[EXPERIMENT_CONTROLLER] ||= createExperimentController(model);
  controller.retain();
  return controller;
}

function createExperimentController(model) {
  const subscribers = new Set();
  const listeners = [];
  let references = 0, pending = null, computing = false, disposed = false;
  let sweepJob = null, playback = null;
  const controller = {
    model, error: null, sweep: { progress: 0, status: '' },
    get playing() { return !!playback; },
    retain() { references++; },
    subscribe(callback) { subscribers.add(callback); return () => subscribers.delete(callback); },
    release() {
      if (--references > 0) return;
      clearTimeout(pending);
      controller.cancelSweep();
      controller.pause();
      disposed = true;
      for (const [event, callback] of listeners) model.off(event, callback);
      subscribers.clear();
      delete model[EXPERIMENT_CONTROLLER];
    },
    commit(values) {
      if ('trajectory_fraction' in values) controller.pause();
      for (const [key, value] of Object.entries(values)) model.set(key, value);
      if (Object.keys(values).some(key => COMPUTATION_INPUTS.includes(key))) controller.refresh();
      else emit();
      sync();
    },
    refresh() {
      clearTimeout(pending); pending = null;
      if (disposed || computing) return;
      computing = true;
      try {
        controller.definition = computeExperiment(model);
        controller.error = null;
        if (model.get('sweep_metadata')?.identity && model.get('sweep_metadata').identity !== sweepIdentity(model)) {
          controller.cancelSweep();
          model.set('sweep_results', []); model.set('sweep_fixed_points', []);
          model.set('sweep_metadata', {});
          controller.sweep.status = 'Configuration changed — run sweep again';
        }
      } catch (error) {
        controller.definition = null;
        controller.error = error;
        controller.cancelSweep();
        model.set('sweep_results', []); model.set('sweep_fixed_points', []);
      } finally {
        computing = false;
      }
      emit(); sync();
    },
    reset() {
      controller.pause(); controller.cancelSweep();
      controller.sweep = { progress: 0, status: '' };
      resetExperiment(model);
      controller.refresh();
    },
    captureReference() {
      controller.commit({
        reference_metadata: { identity: referenceIdentity(model), params: structuredClone(model.get('params')), seed: model.get('noise_seed'), captured_at: new Date().toISOString() },
        reference_scene: Object.fromEntries(['nullcline_x', 'nullcline_y', 'fixed_points', 'display', 'clamped', 'xlim', 'ylim', 'state_names'].map(key => [key, structuredClone(model.get(key) || [])])),
        reference_trajectory: (model.get('trajectory') || []).map(row => [...row]),
        show_reference: true,
      });
    },
    selectModel(name) {
      try {
        const def = resolveModel(name, model.get('model_spec'));
        const display = def.display || [0, 1];
        const clamped = def.stateVars.map((_, i) => {
          const [lo, hi] = def.defaultLims?.[i] || [-1, 1]; return (lo + hi) / 2;
        });
        const xlim = def.defaultLims?.[display[0]] || def.defaultXlim;
        const ylim = def.defaultLims?.[display[1]] || def.defaultYlim;
        controller.commit({ model_name: name, param_info: def.paramInfo, state_names: def.stateVars,
          params: { ...def.defaultParams }, display, clamped, xlim: [...xlim], ylim: [...ylim],
          x0: xlim[0] + .1 * (xlim[1] - xlim[0]), y0: ylim[0] + .1 * (ylim[1] - ylim[0]),
          reference_trajectory: [], reference_scene: {}, reference_metadata: {},
        });
      } catch (error) { controller.error = error; emit(); }
    },
    applySpec(spec) {
      const def = resolveModel('custom', spec);
      const display = spec.display || [0, Math.min(1, def.dim - 1)];
      const params = Object.fromEntries(Object.entries(def.defaultParams).map(([key, value]) => [key, model.get('params')?.[key] ?? value]));
      const clamped = def.stateVars.map((name, i) => spec.state_vars[name].default ?? ((def.defaultLims[i][0] + def.defaultLims[i][1]) / 2));
      controller.commit({ model_spec: spec, model_name: 'custom', param_info: def.paramInfo,
        state_names: def.stateVars, params, display, clamped,
        xlim: [...def.defaultXlim], ylim: [...def.defaultYlim],
      });
    },
    pause(owner) {
      if (!playback || (owner && playback.owner !== owner)) return;
      clearTimeout(playback.timer); playback = null; emit(); sync();
    },
    play(owner) {
      if (playback) { controller.pause(); return; }
      const offset = model.get('trajectory_fraction') >= 1 ? 0 : (model.get('trajectory_fraction') || 0);
      const job = playback = { owner, start: performance.now(), timer: null };
      function tick() {
        if (playback !== job || disposed) return;
        const fraction = Math.min(1, offset + (performance.now() - job.start) / 8000);
        model.set('trajectory_fraction', fraction);
        emit();
        if (fraction < 1) job.timer = setTimeout(tick, 32);
        else controller.pause();
      }
      tick();
    },
    cancelSweep(owner) {
      if (!sweepJob || (owner && sweepJob.owner !== owner)) return;
      clearTimeout(sweepJob.timer); sweepJob = null;
      model.set('sweep_running', false);
      controller.sweep.status = 'Sweep cancelled';
      emit(); sync();
    },
    startSweep(config, owner) {
      if (sweepJob || model.get('sweep_running')) return;
      const { parameter, min, max, count } = config;
      if (!parameter || ![min, max, count].every(Number.isFinite) || min >= max || !Number.isInteger(count) || count < 2 || count > 200) {
        controller.sweep.status = 'Choose a parameter, increasing bounds, and 2–200 samples'; emit(); return;
      }
      const job = sweepJob = { owner, timer: null };
      controller.sweep = { progress: 0, status: 'Running…' };
      model.set('sweep_running', true);
      function fail(error) {
        if (sweepJob !== job) return;
        sweepJob = null;
        model.set('sweep_running', false);
        model.set('sweep_results', []); model.set('sweep_fixed_points', []); model.set('sweep_metadata', {});
        controller.error = error;
        controller.sweep.status = `Sweep failed: ${error.message}`;
        emit(); sync();
      }
      try {
        const identity = sweepIdentity(model);
        model.set('sweep_metadata', { identity, parameter, min, max, count });
        const name = model.get('model_name'), spec = model.get('model_spec');
        const def = resolveModel(name, spec), params = model.get('params');
        const display = model.get('display') || def.display || [0, 1];
        const clamped = model.get('clamped');
        const xlim = model.get('xlim'), ylim = model.get('ylim');
        const projected = makeProjectedRHS(name, display, clamped, spec, def);
        const values = linspace(min, max, count), results = [], fixedPoints = [];
        let index = 0;
        function step() {
          if (sweepJob !== job || disposed) return;
          try {
            if (identity !== sweepIdentity(model)) { controller.cancelSweep(); return; }
            const end = Math.min(index + 2, values.length);
            for (; index < end; index++) {
              const value = values[index], p = { ...params, [parameter]: value };
              const regime = detectRegime(name === 'custom' ? def : name, p, xlim, ylim, display, 60, .05);
              const fps = findFixedPoints(projected, p, xlim, ylim, 15);
              results.push({ param_value: value, regime, num_fixed_points: fps.length });
              for (const fp of fps) fixedPoints.push([value, fp[0], fp[1], fp[2]]);
            }
            model.set('sweep_results', [...results]); model.set('sweep_fixed_points', [...fixedPoints]); model.set('sweep_param', parameter);
            controller.sweep.progress = Math.round(index / values.length * 100);
            if (index < values.length) job.timer = setTimeout(step, 0);
            else {
              sweepJob = null; model.set('sweep_running', false);
              controller.sweep.status = `${count} samples complete`;
            }
            emit(); sync();
          } catch (error) { fail(error); }
        }
        controller.error = null;
        step();
      } catch (error) { fail(error); }
    },
  };
  function sync() { if (!disposed) model.save_changes?.(); }
  function emit() { if (!disposed && !computing) for (const callback of [...subscribers]) callback(); }
  function listen(key, callback) { const event = 'change:' + key; model.on(event, callback); listeners.push([event, callback]); }
  function schedule() {
    if (computing || disposed || pending !== null) return;
    pending = setTimeout(() => controller.refresh(), 0);
  }
  for (const key of COMPUTATION_INPUTS) listen(key, schedule);
  for (const key of PRESENTATION_INPUTS) listen(key, emit);
  controller.refresh();
  return controller;
}
