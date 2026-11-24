// AUTO2000 Web Worker message handler.

let _cachedWasmModule = null;

function _buildStpntCallback(stpnt, ndim) {
  if (!stpnt) return null;
  if (typeof stpnt === 'function') return stpnt;
  const uInit = stpnt.u || [];
  const pInit = stpnt.par ?? stpnt.p ?? [];
  return (uOut, parOut) => {
    for (let i = 0; i < ndim; i++) {
      uOut[i] = Number(uInit[i] ?? 0);
    }
    if (pInit.length > 0) parOut[0] = Number(pInit[0]);
    if (pInit.length > 1) parOut[1] = Number(pInit[1]);
  };
}

function _normalizeBranchPoints(points, ips, is2D, fixedPeriod = null) {
  const isLC = ips === 2;
  return points.map((pt) => {
    const out = {
      p: pt.par[0],
      u: pt.u,
      stable: pt.stable,
      ty: null,
      lab: 0,
    };
    if (is2D && pt.par.length > 1) {
      out.p2 = pt.par[1];
    }
    if (isLC) {
      out.u_max = pt.u;
      out.u_min = pt.uMin;
      out.period = is2D ? (pt.par.length > 2 ? pt.par[2] : fixedPeriod) : pt.par[1];
    }
    return out;
  });
}

function _normalizeSpecialPoints(specialPoints, ndim, ips, is2D) {
  const isLC = ips === 2;
  const normalized = [];
  const restartHandles = {};
  for (const sp of specialPoints) {
    const restartSp = {
      itp: sp.itp,
      lab: sp.lab,
      par: sp.par,
      u1Based: sp.u1Based,
      ...(sp.ups ? {
        ntst: sp.ntst,
        ncol: sp.ncol,
        ntpl: sp.ntpl,
        nar: sp.nar,
        nfpr: sp.nfpr,
        icp: sp.icp,
        rldot: sp.rldot,
        t: sp.t,
        ups: sp.ups,
        udotps: sp.udotps,
      } : {}),
    };
    restartHandles[sp.lab] = restartSp;
    const spPeriod = sp.par?.[10] ?? sp.par2 ?? 0;
    const isNonBifLcPoint = sp.ty === 'UZ' || sp.ty === 'EP' || sp.ty === 'RG';
    const item = {
      lab: sp.lab,
      ty: sp.ty,
      p: sp.par1,
      u: sp.u1Based.length > ndim ? sp.u1Based.slice(1, ndim + 1) : sp.u1Based.slice(0, ndim),
      can_switch_lc: sp.ty === 'HB',
      can_trace_2d: sp.ty === 'LP' || sp.ty === 'HB' || sp.ty === 'LPC' || sp.ty === 'PD' || sp.ty === 'TR',
      can_trace_homoclinic: Boolean(sp.ups && isNonBifLcPoint && (sp.ty === 'UZ' || spPeriod >= 15)),
      restartSp,
    };
    if (is2D) {
      item.p2 = sp.par2;
    }
    if (isLC) {
      item.period = sp.par?.[10] ?? sp.par2;
    }
    normalized.push(item);
  }
  return { specialPoints: normalized, restartHandles };
}

async function _handleWorkerMessage(data) {
  const msg = data || {};
  if (msg.type === 'init') {
    const source = msg.wasmModule ?? msg.wasmBytes ?? msg.wasmUrl ?? null;
    _cachedWasmModule = await compileAuto2000Module(source);
    return { type: 'ready' };
  }

  if (msg.type === 'run') {
    const { jobId, model = {}, continuation = {} } = msg;
    const def = resolveModel(model.model_name, model.model_spec);
    const ndim = def.dim;
    const activeParams = Array.isArray(continuation.activeParams) ? continuation.activeParams : [];
    const ips = continuation.ips ?? 1;
    const isw = continuation.isw ?? 1;
    const is2D = isw === 2 || activeParams.length > 1;

    const mergedParams = { ...(def.defaultParams || {}), ...(model.params || {}) };
    const callParams = { ...mergedParams };
    const p0Name = activeParams[0];
    const p1Name = activeParams.length > 1 ? activeParams[1] : null;

    const func = (u, par, f) => {
      if (p0Name) callParams[p0Name] = par[0];
      if (p1Name) callParams[p1Name] = par[1];
      const rhs = def.f(0, u, callParams);
      for (let i = 0; i < ndim; i++) {
        const val = rhs[i];
        if (!Number.isFinite(val)) {
          throw new Error(`Non-finite f[${i}]=${val} in model ${model.model_name}`);
        }
        f[i] = val;
      }
    };

    let effectiveRestartSp = continuation.restartSp ?? null;
    if (effectiveRestartSp && is2D && p1Name && mergedParams[p1Name] !== undefined) {
      const parCopy = [...(effectiveRestartSp.par || [])];
      if (parCopy.length < 2 || (parCopy[1] === 0 && mergedParams[p1Name] !== 0)) {
        parCopy[0] = parCopy[0] ?? 0;
        parCopy[1] = Number(mergedParams[p1Name]);
        effectiveRestartSp = { ...effectiveRestartSp, par: parCopy };
      }
    }

    const iuz = [...(continuation.iuz ?? [])];
    const vuz = [...(continuation.vuz ?? [])];
    if (continuation.stopPeriod !== undefined && continuation.stopPeriod !== null) {
      iuz.push(-10);
      vuz.push(Number(continuation.stopPeriod));
    }

    const defaultIcp = continuation.icp ?? (ips === 2 && is2D ? [0, 1] : null);
    const fixedPeriod = (ips === 2 && is2D && effectiveRestartSp?.par?.[10] !== undefined)
      ? effectiveRestartSp.par[10]
      : null;

    const stpntCb = _buildStpntCallback(continuation.stpnt, ndim);
    const t0 = performance.now();
    const res = await runAuto2000({
      ndim,
      ips,
      irs: continuation.irs ?? 0,
      ilp: continuation.ilp ?? 1,
      isw,
      isp: continuation.isp ?? 2,
      nmx: continuation.nmx ?? 200,
      npr: continuation.npr ?? 200,
      ntst: continuation.ntst ?? 50,
      ncol: continuation.ncol ?? 4,
      ds: continuation.ds ?? 0.05,
      dsmin: continuation.dsmin ?? 1e-3,
      dsmax: continuation.dsmax ?? 0.2,
      rl0: continuation.rl0 ?? -1e30,
      rl1: continuation.rl1 ?? 1e30,
      icp: defaultIcp,
      iuz,
      vuz,
      restartSp: effectiveRestartSp,
      wasmModule: _cachedWasmModule,
      func,
      stpnt: stpntCb,
    });
    const elapsedMs = performance.now() - t0;

    if (res.error) {
      throw res.error;
    }

    const branch = _normalizeBranchPoints(res.points, ips, is2D, fixedPeriod);
    const { specialPoints, restartHandles } = _normalizeSpecialPoints(res.specialPoints, ndim, ips, is2D);
    return { type: 'result', jobId, branch, specialPoints, restartHandles, elapsedMs };
  }

  throw new Error(`Unknown worker message type: ${msg.type}`);
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = async (event) => {
    const jobId = event.data?.jobId;
    try {
      const reply = await _handleWorkerMessage(event.data);
      self.postMessage(reply);
    } catch (err) {
      self.postMessage({
        type: 'error',
        ...(jobId !== undefined ? { jobId } : {}),
        message: err?.message || String(err),
      });
    }
  };
}
