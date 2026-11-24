// One controller per canonical experiment model, independent of mounted panels.
const EXPERIMENT_CONTROLLER = Symbol.for('trajecturtle.experiment-controller');
const COMPUTATION_INPUTS = ['model_spec', 'model_name', 'params', 'state_names', 'display', 'clamped', 'x0', 'y0', 'xlim', 'ylim', 't_max', 'dt', 'integrator', 'noise_enable', 'noise_sigma', 'noise_seed'];
const PRESENTATION_INPUTS = ['param_info', 'show_nullclines', 'show_vector_field', 'show_trajectory', 'show_fixed_points', 'trajectory_fraction', 'reference_trajectory', 'reference_scene', 'show_reference', 'reference_metadata', 'sweep_results', 'sweep_fixed_points', 'sweep_param', 'initial_conditions', 'observation_cards'];
const BIFURCATION_INPUTS = ['bifurcation_branches', 'bifurcation_special_points', 'bifurcation_config', 'bifurcation_running', 'bifurcation_metadata', 'bifurcation_presets'];

function _bifurcationModelIdentity(model) {
  return JSON.stringify([
    model.get('model_name'),
    model.get('model_spec') || null,
    model.get('state_names') || [],
  ]);
}

function _bifurcationFixedParamsIdentity(model, activeParams) {
  const params = model.get('params') || {};
  const active = new Set(activeParams || []);
  const entries = Object.keys(params)
    .filter((k) => !active.has(k))
    .sort()
    .map((k) => [k, params[k]]);
  return JSON.stringify(entries);
}

function _resolveBifurcationRange(def, model, paramName, explicitRange, rl0, rl1, currentVal) {
  if (Array.isArray(explicitRange) && explicitRange.length === 2 &&
      Number.isFinite(explicitRange[0]) && Number.isFinite(explicitRange[1])) {
    return explicitRange[0] <= explicitRange[1]
      ? [explicitRange[0], explicitRange[1]]
      : [explicitRange[1], explicitRange[0]];
  }
  if (Number.isFinite(rl0) && Number.isFinite(rl1)) {
    return rl0 <= rl1 ? [rl0, rl1] : [rl1, rl0];
  }
  const info = def.paramInfo?.[paramName] ?? model.get('param_info')?.[paramName];
  if (Array.isArray(info) && info.length >= 2 && Number.isFinite(info[0]) && Number.isFinite(info[1])) {
    return info[0] <= info[1] ? [info[0], info[1]] : [info[1], info[0]];
  }
  const v = Number.isFinite(currentVal) ? currentVal : 0;
  return [v - 1, v + 1];
}

function _buildSeedCandidates(def, model) {
  const dim = def.dim || (model.get('state_names') || []).length || 2;
  const display = model.get('display') || def.display || [0, 1];
  const clamped = model.get('clamped') || [];
  const base = Array.from({ length: dim }, (_, i) => {
    if (Number.isFinite(clamped[i])) return clamped[i];
    const lim = def.defaultLims?.[i];
    return Array.isArray(lim) && lim.length === 2 ? (lim[0] + lim[1]) / 2 : 0;
  });

  const primary = [...base];
  const x0 = model.get('x0');
  const y0 = model.get('y0');
  if (Number.isFinite(x0) && display[0] < dim) primary[display[0]] = x0;
  if (display.length > 1 && Number.isFinite(y0) && display[1] < dim) primary[display[1]] = y0;

  const fallbacks = [];
  const traj = model.get('trajectory') || [];
  if (traj.length > 0 && Array.isArray(traj[traj.length - 1]) && traj[traj.length - 1].length >= dim + 1) {
    fallbacks.push(traj[traj.length - 1].slice(1, 1 + dim));
  }
  const fps = model.get('fixed_points') || [];
  for (const fp of fps) {
    if (!Array.isArray(fp) || fp.length < 2) continue;
    const fb = [...base];
    if (display[0] < dim) fb[display[0]] = fp[0];
    if (display.length > 1 && display[1] < dim) fb[display[1]] = fp[1];
    fallbacks.push(fb);
  }
  return { primary, fallbacks };
}

function _stitchBidirectionalBranch(backwardPts, forwardPts) {
  const back = [...(backwardPts || [])].reverse();
  const fwd = [...(forwardPts || [])];
  if (back.length > 0 && fwd.length > 0) {
    const last = back[back.length - 1];
    const first = fwd[0];
    const sameP = Math.abs((last.p ?? 0) - (first.p ?? 0)) < 1e-8 &&
      Math.abs((last.p2 ?? 0) - (first.p2 ?? 0)) < 1e-8;
    const uA = last.u || [];
    const uB = first.u || [];
    const sameU = uA.length === uB.length && uA.every((v, i) => Math.abs(v - uB[i]) < 1e-6);
    if (sameP && sameU) fwd.shift();
  }
  return [...back, ...fwd];
}

function _nextBranchId(prefix, existingBranches) {
  let maxNum = 0;
  const re = new RegExp(`^${prefix}_(\\d+)$`);
  for (const br of existingBranches || []) {
    const m = re.exec(br?.id || '');
    if (m) maxNum = Math.max(maxNum, Number(m[1]));
  }
  return `${prefix}_${maxNum + 1}`;
}

function _clipRayByParam2(points, rl2Min, rl2Max) {
  if (!Number.isFinite(rl2Min) && !Number.isFinite(rl2Max)) return points || [];
  const lo = Number.isFinite(rl2Min) ? rl2Min : -Infinity;
  const hi = Number.isFinite(rl2Max) ? rl2Max : Infinity;
  const out = [];
  for (const pt of points || []) {
    if (Number.isFinite(pt?.p2) && (pt.p2 < lo || pt.p2 > hi)) break;
    out.push(pt);
  }
  return out;
}

function _registerSpecialPoints(rawSpecialPoints, branchId, param1, param2, existingSps, restartCache) {
  const counts = {};
  for (const sp of existingSps || []) {
    if (sp?.ty) counts[sp.ty] = (counts[sp.ty] || 0) + 1;
  }
  const added = [];
  for (const sp of rawSpecialPoints || []) {
    if (!sp || !sp.ty || ((sp.ty === 'EP' || sp.ty === 'UZ') && !sp.can_trace_homoclinic)) continue;
    counts[sp.ty] = (counts[sp.ty] || 0) + 1;
    const id = `${sp.ty}${counts[sp.ty]}`;
    const canTrace2D = Boolean(
      sp.can_trace_2d || sp.ty === 'LP' || sp.ty === 'HB' || sp.ty === 'LPC' || sp.ty === 'PD' || sp.ty === 'TR',
    );
    const canTraceHomoclinic = Boolean(sp.can_trace_homoclinic);
    const entry = {
      id,
      label: id,
      ty: sp.ty,
      lab: sp.lab,
      p: sp.p,
      u: Array.isArray(sp.u) ? [...sp.u] : [],
      branch_id: branchId,
      param1,
      param2: param2 ?? null,
      can_switch_lc: sp.ty === 'HB',
      can_trace_2d: canTrace2D,
      ...(canTraceHomoclinic ? { can_trace_homoclinic: true } : {}),
      ...(sp.period !== undefined ? { period: sp.period } : {}),
      ...(sp.p2 !== undefined ? { p2: sp.p2 } : {}),
    };
    added.push(entry);
    restartCache.set(id, {
      ...(sp.restartSp || {}),
      ty: sp.ty,
      lab: sp.lab,
      p: sp.p,
      p2: sp.p2,
      u: entry.u,
      param1,
      param2: param2 ?? null,
      branch_id: branchId,
      can_trace_homoclinic: canTraceHomoclinic,
      restartSp: sp.restartSp || null,
    });
  }
  return added;
}

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
  let autoClient = null, bifurcationJob = null, bifurcationGeneration = 0;
  let lastBifModelIdentity = _bifurcationModelIdentity(model);
  function _resetBifurcationOnModelChange() {
    controller.clearBifurcation();
    if ((model.get('bifurcation_presets') || []).length > 0) {
      model.set('bifurcation_presets', []);
    }
    const prevCfg = model.get('bifurcation_config') || {};
    const nextCfg = { ...prevCfg };
    let changed = false;
    for (const k of ['param1', 'param2', 'range', 'range2', 'rl0', 'rl1', 'rl2_0', 'rl2_1', 'xlim1d', 'ylim1d', 'xlim2d', 'ylim2d', 'plane2d', 'viewMode', 'stopPeriod', 'observable']) {
      if (k in nextCfg) { delete nextCfg[k]; changed = true; }
    }
    if (changed) model.set('bifurcation_config', nextCfg);
  }
  function _getAutoClient() {
    if (!autoClient) autoClient = createAutoClient(model);
    return autoClient;
  }
  function _isBifurcationJobActive(job) {
    return !disposed && job && !job.cancelled && bifurcationJob === job && bifurcationGeneration === job.id;
  }
  function _beginBifurcationJob(owner, modelIdentity) {
    if (bifurcationJob) controller.cancelBifurcation();
    bifurcationGeneration++;
    const job = { id: bifurcationGeneration, owner, modelIdentity, cancelled: false };
    bifurcationJob = job;
    model.set('bifurcation_running', true);
    const meta = model.get('bifurcation_metadata') || {};
    model.set('bifurcation_metadata', { ...meta, status: 'running', error: null });
    emit(); sync();
    return job;
  }
  function _failBifurcationJob(job, message) {
    if (!_isBifurcationJobActive(job)) return null;
    bifurcationJob = null;
    model.set('bifurcation_running', false);
    const meta = model.get('bifurcation_metadata') || {};
    model.set('bifurcation_metadata', { ...meta, status: 'error', error: message });
    emit(); sync();
    return null;
  }
  const controller = {
    model, error: null, sweep: { progress: 0, status: '' },
    restartCache: new Map(),
    get playing() { return !!playback; },
    retain() { references++; },
    subscribe(callback) { subscribers.add(callback); return () => subscribers.delete(callback); },
    release() {
      if (--references > 0) return;
      clearTimeout(pending);
      controller.cancelSweep();
      controller.cancelBifurcation();
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
        const modelIdentity = _bifurcationModelIdentity(model);
        const bifMeta = model.get('bifurcation_metadata') || {};
        const bifBranches = model.get('bifurcation_branches') || [];
        if (modelIdentity !== lastBifModelIdentity) {
          lastBifModelIdentity = modelIdentity;
          _resetBifurcationOnModelChange();
        } else if (bifurcationJob && bifurcationJob.modelIdentity !== modelIdentity) {
          controller.clearBifurcation();
        } else if (bifBranches.length > 0) {
          if (bifMeta.model_identity && bifMeta.model_identity !== modelIdentity) {
            controller.clearBifurcation();
          } else if (bifMeta.fixed_params_identity) {
            const fixedId = _bifurcationFixedParamsIdentity(model, bifMeta.active_params);
            if (fixedId !== bifMeta.fixed_params_identity) {
              if (!bifMeta.stale) model.set('bifurcation_metadata', { ...bifMeta, stale: true });
            } else if (bifMeta.stale) {
              model.set('bifurcation_metadata', { ...bifMeta, stale: false });
            }
          }
        }
      } catch (error) {
        controller.definition = null;
        controller.error = error;
        controller.cancelSweep();
        controller.cancelBifurcation();
        model.set('sweep_results', []); model.set('sweep_fixed_points', []);
      } finally {
        computing = false;
      }
      emit(); sync();
    },
    reset() {
      controller.pause(); controller.cancelSweep(); controller.cancelBifurcation();
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
    cancelBifurcation(owner = null) {
      if (!bifurcationJob || (owner && bifurcationJob.owner !== owner)) return;
      bifurcationGeneration++;
      bifurcationJob.cancelled = true;
      bifurcationJob = null;
      autoClient?.cancel();
      model.set('bifurcation_running', false);
      const meta = model.get('bifurcation_metadata') || {};
      model.set('bifurcation_metadata', { ...meta, status: 'cancelled' });
      emit(); sync();
    },
    clearBifurcation() {
      if (bifurcationJob) controller.cancelBifurcation();
      controller.restartCache.clear();
      model.set('bifurcation_branches', []);
      model.set('bifurcation_special_points', []);
      model.set('bifurcation_running', false);
      model.set('bifurcation_metadata', {});
      emit(); sync();
    },
    async startEquilibriumContinuation(config = {}, owner = null) {
      const modelName = model.get('model_name');
      const modelSpec = model.get('model_spec');
      const modelIdentity = _bifurcationModelIdentity(model);
      const job = _beginBifurcationJob(owner, modelIdentity);
      try {
        const def = resolveModel(modelName, modelSpec);
        const params = { ...(def.defaultParams || {}), ...(model.get('params') || {}) };
        const cfg = { ...(model.get('bifurcation_config') || {}), ...config };
        const param1 = cfg.param1 || cfg.parameter || Object.keys(def.defaultParams || {})[0];
        if (!param1) {
          return _failBifurcationJob(job, 'No continuation parameter available for this model');
        }
        const [rl0, rl1] = _resolveBifurcationRange(def, model, param1, cfg.range, cfg.rl0, cfg.rl1, params[param1]);
        const { primary, fallbacks } = _buildSeedCandidates(def, model);
        const seed = findSteadyStateND(def, params, primary, { fallbackStates: fallbacks });
        if (!seed.converged) {
          return _failBifurcationJob(
            job,
            `Failed to locate a steady state for continuation (residual ${seed.residual.toExponential(2)})`,
          );
        }

        const nonSmoothTokens = modelName === 'custom' ? detectNonSmoothExpressions(modelSpec) : [];
        const absDs = Math.abs(cfg.ds ?? 0.05) || 0.05;
        const p0Val = Number(params[param1] ?? 0);
        const contBase = {
          ips: 1,
          isw: 1,
          activeParams: [param1],
          rl0,
          rl1,
          dsmin: cfg.dsmin ?? 1e-4,
          dsmax: cfg.dsmax ?? 0.2,
          nmx: cfg.nmx ?? 200,
          npr: cfg.npr ?? 200,
          ilp: cfg.ilp ?? 1,
          isp: cfg.isp ?? 2,
          stpnt: { u: seed.u, par: [p0Val] },
        };
        const modelPayload = { model_name: modelName, model_spec: modelSpec, params };
        const client = _getAutoClient();

        const fwdRes = await client.run({
          model: modelPayload,
          continuation: { ...contBase, ds: absDs },
        });
        if (!_isBifurcationJobActive(job)) return null;

        const bwdRes = await client.run({
          model: modelPayload,
          continuation: { ...contBase, ds: -absDs },
        });
        if (!_isBifurcationJobActive(job)) return null;

        const existingBranches = model.get('bifurcation_branches') || [];
        const existingSps = model.get('bifurcation_special_points') || [];
        const branchId = _nextBranchId('ep', existingBranches);
        const stitchedPoints = _stitchBidirectionalBranch(bwdRes.branch, fwdRes.branch);
        const branch = {
          id: branchId,
          type: 'EP',
          param1,
          param2: null,
          points: stitchedPoints,
        };

        const rawSps = [...(bwdRes.specialPoints || []), ...(fwdRes.specialPoints || [])];
        const addedSps = _registerSpecialPoints(rawSps, branchId, param1, null, existingSps, controller.restartCache);

        const prevMeta = model.get('bifurcation_metadata') || {};
        const activeParams = [...new Set([...(prevMeta.active_params || []), param1])];
        model.set('bifurcation_branches', [...existingBranches, branch]);
        model.set('bifurcation_special_points', [...existingSps, ...addedSps]);
        model.set('bifurcation_metadata', {
          model_identity: modelIdentity,
          active_params: activeParams,
          fixed_params_identity: _bifurcationFixedParamsIdentity(model, activeParams),
          stale: false,
          status: 'complete',
          error: null,
          seed: seed.u,
          ...(nonSmoothTokens.length > 0 ? { non_smooth_tokens: nonSmoothTokens } : {}),
        });
        bifurcationJob = null;
        model.set('bifurcation_running', false);
        emit(); sync();
        return { branch, specialPoints: addedSps };
      } catch (error) {
        return _failBifurcationJob(job, error?.message || String(error));
      }
    },
    async switchToLimitCycle(specialPointId, config = {}, owner = null) {
      const entry = controller.restartCache.get(specialPointId);
      const modelIdentity = _bifurcationModelIdentity(model);
      const job = _beginBifurcationJob(owner, modelIdentity);
      try {
        if (!entry || entry.ty !== 'HB') {
          return _failBifurcationJob(job, `Limit-cycle switching requires a Hopf (HB) point (got ${entry?.ty || 'none'})`);
        }
        const modelName = model.get('model_name');
        const modelSpec = model.get('model_spec');
        const def = resolveModel(modelName, modelSpec);
        const params = { ...(def.defaultParams || {}), ...(model.get('params') || {}) };
        const cfg = { ...(model.get('bifurcation_config') || {}), ...config };
        const param1 = entry.param1 || cfg.param1 || Object.keys(def.defaultParams || {})[0];
        const [rl0, rl1] = _resolveBifurcationRange(def, model, param1, cfg.range, cfg.rl0, cfg.rl1, params[param1]);

        const client = _getAutoClient();
        const res = await client.run({
          model: { model_name: modelName, model_spec: modelSpec, params },
          continuation: {
            ips: 2,
            isw: 1,
            irs: entry.lab,
            restartSp: entry.restartSp || entry,
            activeParams: [param1],
            rl0,
            rl1,
            ds: cfg.ds ?? 0.05,
            dsmin: cfg.dsmin ?? 1e-4,
            dsmax: cfg.dsmax ?? 0.2,
            nmx: cfg.nmx ?? 200,
            npr: cfg.npr ?? 200,
            ntst: cfg.ntst ?? 50,
            ncol: cfg.ncol ?? 4,
            ilp: cfg.ilp ?? 1,
            isp: cfg.isp ?? 2,
            ...(cfg.stopPeriod !== undefined && cfg.stopPeriod !== null && Number.isFinite(Number(cfg.stopPeriod))
              ? { stopPeriod: Number(cfg.stopPeriod) }
              : {}),
          },
        });
        if (!_isBifurcationJobActive(job)) return null;

        const existingBranches = model.get('bifurcation_branches') || [];
        const existingSps = model.get('bifurcation_special_points') || [];
        const branchId = _nextBranchId('lc', existingBranches);
        const branch = {
          id: branchId,
          type: 'LC',
          param1,
          param2: null,
          source_sp_id: specialPointId,
          points: res.branch || [],
        };
        const addedSps = _registerSpecialPoints(res.specialPoints || [], branchId, param1, null, existingSps, controller.restartCache);

        const prevMeta = model.get('bifurcation_metadata') || {};
        const activeParams = [...new Set([...(prevMeta.active_params || []), param1])];
        model.set('bifurcation_branches', [...existingBranches, branch]);
        model.set('bifurcation_special_points', [...existingSps, ...addedSps]);
        model.set('bifurcation_metadata', {
          ...prevMeta,
          model_identity: modelIdentity,
          active_params: activeParams,
          fixed_params_identity: _bifurcationFixedParamsIdentity(model, activeParams),
          stale: false,
          status: 'complete',
          error: null,
        });
        bifurcationJob = null;
        model.set('bifurcation_running', false);
        emit(); sync();
        return { branch, specialPoints: addedSps };
      } catch (error) {
        return _failBifurcationJob(job, error?.message || String(error));
      }
    },
    async startTwoParamContinuation(specialPointId, param2, config = {}, owner = null) {
      const entry = controller.restartCache.get(specialPointId);
      const modelIdentity = _bifurcationModelIdentity(model);
      const job = _beginBifurcationJob(owner, modelIdentity);
      try {
        const isHomoclinic = Boolean(config.homoclinic || entry?.can_trace_homoclinic) && Boolean(entry?.restartSp?.ups);
        const isLcBif = entry && (entry.ty === 'LPC' || entry.ty === 'PD' || entry.ty === 'TR');
        const isEpBif = entry && (entry.ty === 'LP' || entry.ty === 'HB');
        if (!entry || (!isEpBif && !isLcBif && !isHomoclinic)) {
          return _failBifurcationJob(
            job,
            `Two-parameter continuation requires an LP, HB, LPC, PD, TR, or homoclinic LC point (got ${entry?.ty || 'none'})`,
          );
        }
        const modelName = model.get('model_name');
        const modelSpec = model.get('model_spec');
        const def = resolveModel(modelName, modelSpec);
        const params = { ...(def.defaultParams || {}), ...(model.get('params') || {}) };
        const cfg = { ...(model.get('bifurcation_config') || {}), ...config };
        const param1 = entry.param1 || cfg.param1 || Object.keys(def.defaultParams || {})[0];
        if (!param2 || param2 === param1) {
          return _failBifurcationJob(job, 'Choose a second parameter distinct from the primary parameter');
        }
        const [rl0, rl1] = _resolveBifurcationRange(def, model, param1, cfg.range, cfg.rl0, cfg.rl1, params[param1]);
        let rl2Min = NaN, rl2Max = NaN;
        if (Array.isArray(cfg.range2) && cfg.range2.length === 2 && Number.isFinite(cfg.range2[0]) && Number.isFinite(cfg.range2[1])) {
          rl2Min = Math.min(cfg.range2[0], cfg.range2[1]);
          rl2Max = Math.max(cfg.range2[0], cfg.range2[1]);
        } else {
          if (cfg.rl2_0 !== undefined && cfg.rl2_0 !== null && Number.isFinite(Number(cfg.rl2_0))) rl2Min = Number(cfg.rl2_0);
          if (cfg.rl2_1 !== undefined && cfg.rl2_1 !== null && Number.isFinite(Number(cfg.rl2_1))) rl2Max = Number(cfg.rl2_1);
          if (Number.isFinite(rl2Min) && Number.isFinite(rl2Max) && rl2Min > rl2Max) [rl2Min, rl2Max] = [rl2Max, rl2Min];
        }
        const absDs = Math.abs(cfg.ds ?? 0.05) || 0.05;
        const useLC = isLcBif || isHomoclinic;
        const contBase = {
          ips: useLC ? 2 : 1,
          isw: 2,
          irs: entry.lab,
          restartSp: entry.restartSp || entry,
          activeParams: [param1, param2],
          ...(useLC ? { icp: [0, 1] } : {}),
          rl0,
          rl1,
          dsmin: cfg.dsmin ?? 1e-4,
          dsmax: cfg.dsmax ?? 0.2,
          nmx: cfg.nmx ?? 200,
          npr: cfg.npr ?? 200,
          ...(useLC ? { ntst: cfg.ntst ?? entry.restartSp?.ntst ?? 50, ncol: cfg.ncol ?? entry.restartSp?.ncol ?? 4 } : {}),
          ilp: isHomoclinic ? 0 : (cfg.ilp ?? 1),
          isp: isHomoclinic ? 0 : (cfg.isp ?? 2),
        };
        const modelPayload = { model_name: modelName, model_spec: modelSpec, params };
        const client = _getAutoClient();

        const fwdRes = await client.run({
          model: modelPayload,
          continuation: { ...contBase, ds: absDs },
        });
        if (!_isBifurcationJobActive(job)) return null;

        const bwdRes = await client.run({
          model: modelPayload,
          continuation: { ...contBase, ds: -absDs },
        });
        if (!_isBifurcationJobActive(job)) return null;

        const existingBranches = model.get('bifurcation_branches') || [];
        const existingSps = model.get('bifurcation_special_points') || [];
        const branchType = isHomoclinic
          ? 'HOM2'
          : ({ LP: 'LP2', HB: 'HB2', LPC: 'LPC2', PD: 'PD2', TR: 'TR2' }[entry.ty] || 'LP2');
        const branchId = _nextBranchId(branchType.toLowerCase(), existingBranches);
        const fwdPts = _clipRayByParam2(fwdRes.branch, rl2Min, rl2Max);
        const bwdPts = _clipRayByParam2(bwdRes.branch, rl2Min, rl2Max);
        const stitchedPoints = _stitchBidirectionalBranch(bwdPts, fwdPts);
        const branch = {
          id: branchId,
          type: branchType,
          param1,
          param2,
          source_sp_id: specialPointId,
          points: stitchedPoints,
        };
        const rawSps = [...(bwdRes.specialPoints || []), ...(fwdRes.specialPoints || [])].filter((sp) => {
          if (!Number.isFinite(sp?.p2)) return true;
          if (Number.isFinite(rl2Min) && sp.p2 < rl2Min) return false;
          if (Number.isFinite(rl2Max) && sp.p2 > rl2Max) return false;
          return true;
        });
        const addedSps = _registerSpecialPoints(rawSps, branchId, param1, param2, existingSps, controller.restartCache);

        const prevMeta = model.get('bifurcation_metadata') || {};
        const activeParams = [...new Set([...(prevMeta.active_params || []), param1, param2])];
        model.set('bifurcation_branches', [...existingBranches, branch]);
        model.set('bifurcation_special_points', [...existingSps, ...addedSps]);
        model.set('bifurcation_metadata', {
          ...prevMeta,
          model_identity: modelIdentity,
          active_params: activeParams,
          fixed_params_identity: _bifurcationFixedParamsIdentity(model, activeParams),
          stale: false,
          status: 'complete',
          error: null,
        });
        bifurcationJob = null;
        model.set('bifurcation_running', false);
        emit(); sync();
        return { branch, specialPoints: addedSps };
      } catch (error) {
        return _failBifurcationJob(job, error?.message || String(error));
      }
    },
    async runBifurcationPreset(presetIndex, owner = null) {
      const presets = model.get('bifurcation_presets') || [];
      const preset = presets[presetIndex];
      if (!preset) return null;

      const prevCfg = model.get('bifurcation_config') || {};
      model.set('bifurcation_config', {
        ...prevCfg,
        param1: preset.param1,
        ...(preset.range ? { range: preset.range } : {}),
        ...(preset.param2 ? { param2: preset.param2 } : {}),
        ...(preset.range2 ? { range2: preset.range2 } : {}),
        ...(preset.xlim2d ? { xlim2d: preset.xlim2d } : {}),
        ...(preset.ylim2d ? { ylim2d: preset.ylim2d } : {}),
        ...(preset.stop_period !== undefined ? { stopPeriod: preset.stop_period } : {}),
      });
      emit(); sync();

      const epRes = await controller.startEquilibriumContinuation(
        { param1: preset.param1, range: preset.range },
        owner,
      );
      if (!epRes || disposed) return null;

      let lcRes = null;
      if (preset.auto_switch_lc || preset.auto_trace_homoclinic) {
        const hb = (epRes.specialPoints || []).find((sp) => sp.ty === 'HB');
        if (hb) {
          lcRes = await controller.switchToLimitCycle(
            hb.id,
            {
              param1: preset.param1,
              range: preset.range,
              ...(preset.stop_period !== undefined ? { stopPeriod: preset.stop_period } : {}),
            },
            owner,
          );
          if (!lcRes || disposed) return { ep: epRes, lc: lcRes, twoParam: null };
        }
      }

      let twoParamRes = null;
      if (preset.auto_trace_2d && preset.param2) {
        const twoDConfig = { range: preset.range, range2: preset.range2 };
        if (preset.auto_trace_2d === 'all') {
          const targets = [];
          const firstLP = (epRes.specialPoints || []).find((sp) => sp.ty === 'LP');
          const firstHB = (epRes.specialPoints || []).find((sp) => sp.ty === 'HB');
          if (firstLP) targets.push(firstLP);
          if (firstHB) targets.push(firstHB);
          for (const sp of targets) {
            twoParamRes = await controller.startTwoParamContinuation(sp.id, preset.param2, twoDConfig, owner);
            if (!twoParamRes || disposed) break;
          }
          if (preset.auto_trace_homoclinic && lcRes && !disposed) {
            const homSeed = [...(lcRes.specialPoints || [])]
              .filter((sp) => sp.can_trace_homoclinic)
              .sort((a, b) => (b.period ?? 0) - (a.period ?? 0))[0];
            if (homSeed) {
              twoParamRes = await controller.startTwoParamContinuation(
                homSeed.id,
                preset.param2,
                { ...twoDConfig, homoclinic: true },
                owner,
              );
            }
          }
        } else {
          const sp2d = (epRes.specialPoints || []).find((sp) => sp.ty === 'LP' || sp.ty === 'HB');
          if (sp2d) {
            twoParamRes = await controller.startTwoParamContinuation(sp2d.id, preset.param2, twoDConfig, owner);
          }
        }
        if (twoParamRes && !disposed && preset.view_2d) {
          const nextCfg = model.get('bifurcation_config') || {};
          model.set('bifurcation_config', { ...nextCfg, viewMode: '2d' });
          emit(); sync();
        }
      }
      return { ep: epRes, lc: lcRes, twoParam: twoParamRes };
    },
    async traceAllTwoParam(param2, config = {}, owner = null) {
      const def = resolveModel(model.get('model_name'), model.get('model_spec'));
      const cfg = { ...(model.get('bifurcation_config') || {}), ...config };
      const param1 = cfg.param1 || Object.keys(def.defaultParams || {})[0];
      if (!param1 || !param2 || param2 === param1) {
        const job = _beginBifurcationJob(owner, _bifurcationModelIdentity(model));
        return _failBifurcationJob(job, 'Choose two distinct continuation parameters for 2D continuation');
      }
      model.set('bifurcation_config', { ...cfg, param1, param2 });
      emit(); sync();

      const existingEp = (model.get('bifurcation_branches') || []).some(
        (b) => b && b.type === 'EP' && b.param1 === param1,
      );
      let epRes = null;
      if (!existingEp) {
        epRes = await controller.startEquilibriumContinuation({ ...config, param1 }, owner);
        if (!epRes || disposed) return null;
      }

      const sps = model.get('bifurcation_special_points') || [];
      const firstLP = sps.find((sp) => sp && sp.ty === 'LP' && sp.param1 === param1);
      const firstHB = sps.find((sp) => sp && sp.ty === 'HB' && sp.param1 === param1);
      const twoParamBranches = [];
      for (const sp of [firstLP, firstHB].filter(Boolean)) {
        const res2d = await controller.startTwoParamContinuation(sp.id, param2, { ...config }, owner);
        if (!res2d || disposed) break;
        twoParamBranches.push(res2d);
      }
      if (twoParamBranches.length > 0 && !disposed) {
        const nextCfg = model.get('bifurcation_config') || {};
        model.set('bifurcation_config', { ...nextCfg, viewMode: '2d' });
        emit(); sync();
      }
      return { ep: epRes, twoParamBranches };
    },
    jumpToBifurcationPoint(point, branch, opts = {}) {
      if (!point) return;
      const meta = model.get('bifurcation_metadata') || {};
      const p1 = branch?.param1 || point.param1 || meta.active_params?.[0];
      const p2 = branch?.param2 || point.param2 || null;
      const nextParams = { ...(model.get('params') || {}) };
      if (p1 && Number.isFinite(point.p)) nextParams[p1] = point.p;
      if (p2 && Number.isFinite(point.p2)) nextParams[p2] = point.p2;

      const updates = { params: nextParams };
      const isEp = !branch || branch.type === 'EP';
      const display = model.get('display') || [0, 1];
      if (isEp && Array.isArray(point.u) && point.u.length > 2) {
        const prevClamped = model.get('clamped') || [];
        const nextClamped = point.u.map((val, i) => (
          display.includes(i) ? (prevClamped[i] ?? val) : val
        ));
        updates.clamped = nextClamped;
      }
      if ((opts.updateInitialCondition || opts.setInitialCondition) && Array.isArray(point.u)) {
        if (Number.isFinite(point.u[display[0]])) updates.x0 = point.u[display[0]];
        if (display.length > 1 && Number.isFinite(point.u[display[1]])) updates.y0 = point.u[display[1]];
      }
      controller.commit(updates);
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
  for (const key of BIFURCATION_INPUTS) listen(key, emit);
  controller.refresh();
  return controller;
}
