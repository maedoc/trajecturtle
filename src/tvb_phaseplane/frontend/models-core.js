/** Safe exponential that prevents overflow. */
function expClamp(x) {
  return Math.exp(Math.max(-709, Math.min(709, x)));
}

function _sigmoid(x, k, theta) {
  return 1.0 / (1.0 + expClamp(-k * (x - theta)));
}

const MODELS = {
  wilson_cowan: {
    dim: 2,
    stateNames: ["E", "I"],
    stateVars: ["E", "I"],
    display: [0, 1],
    defaultParams: {
      aee: 10.0, aei: 10.0, aie: 10.0, aii: 2.0,
      Pe: -2.0, Pi: -8.0,
      ke: 1.0, ki: 1.0,
      thetae: 4.0, thetai: 4.0,
    },
    paramInfo: {
      aee: [0.0, 20.0, 10.0, "E→E coupling"],
      aei: [0.0, 20.0, 10.0, "I→E coupling"],
      aie: [0.0, 20.0, 10.0, "E→I coupling"],
      aii: [0.0, 20.0, 2.0, "I→I coupling"],
      Pe: [-10.0, 10.0, -2.0, "External E input"],
      Pi: [-10.0, 10.0, -8.0, "External I input"],
      ke: [0.1, 5.0, 1.0, "E sigmoid gain"],
      ki: [0.1, 5.0, 1.0, "I sigmoid gain"],
      thetae: [0.0, 10.0, 4.0, "E sigmoid threshold"],
      thetai: [0.0, 10.0, 4.0, "I sigmoid threshold"],
    },
    defaultXlim: [-0.2, 1.2],
    defaultYlim: [-0.2, 1.2],
    defaultLims: [[-0.2, 1.2], [-0.2, 1.2]],
    f: (_t, [E, I], p) => {
      const Se = (x) => _sigmoid(x, p.ke, p.thetae);
      const Si = (x) => _sigmoid(x, p.ki, p.thetai);
      return [
        -E + Se(p.aee * E - p.aei * I + p.Pe),
        -I + Si(p.aie * E - p.aii * I + p.Pi),
      ];
    },
  },

  fitzhugh_nagumo: {
    dim: 2,
    stateNames: ["v", "w"],
    stateVars: ["v", "w"],
    display: [0, 1],
    defaultParams: { a: 0.7, b: 0.8, epsilon: 0.08, I: 0.5 },
    paramInfo: {
      a: [-1.0, 2.0, 0.7, "Recovery offset"],
      b: [0.0, 2.0, 0.8, "Recovery gain"],
      epsilon: [0.001, 1.0, 0.08, "Time scale (ε)"],
      I: [-2.0, 2.0, 0.5, "External current"],
    },
    defaultXlim: [-3.0, 3.0],
    defaultYlim: [-1.5, 2.0],
    defaultLims: [[-3.0, 3.0], [-1.5, 2.0]],
    f: (_t, [v, w], p) => [
      v - (v * v * v) / 3.0 - w + p.I,
      p.epsilon * (v + p.a - p.b * w),
    ],
  },

  mpr: {
    dim: 2,
    stateNames: ["r", "v"],
    stateVars: ["r", "v"],
    display: [0, 1],
    defaultParams: { delta: 1.0, eta_bar: -5.0, J: 15.0, I: 0.0 },
    paramInfo: {
      delta: [0.01, 5.0, 1.0, "Lorentzian half-width Δ"],
      eta_bar: [-20.0, 10.0, -5.0, "Mean excitability η̄"],
      J: [-20.0, 30.0, 15.0, "Synaptic coupling J"],
      I: [-10.0, 10.0, 0.0, "External input I"],
    },
    defaultXlim: [0.0, 2.0],
    defaultYlim: [-4.0, 2.0],
    defaultLims: [[0.0, 2.0], [-4.0, 2.0]],
    f: (_t, [r, v], p) => {
      const r_eff = Math.max(r, 1e-10);
      const dr = p.delta / Math.PI + 2.0 * r_eff * v;
      const dv = v * v + p.eta_bar + p.J * r_eff + p.I - (Math.PI * r_eff) ** 2;
      return [dr, dv];
    },
  },

  hindmarsh_rose: {
    dim: 3,
    stateNames: ["x", "y", "z"],
    stateVars: ["x", "y", "z"],
    display: [0, 1],
    defaultParams: { a: 1.0, b: 3.0, c: 1.0, d: 5.0, I: 3.0, r: 0.001, s: 4.0, x_r: -1.6 },
    paramInfo: {
      a: [0.5, 2.0, 1.0, "Cubic coefficient a"],
      b: [1.0, 5.0, 3.0, "Quadratic coefficient b"],
      c: [0.0, 2.0, 1.0, "Constant drive c"],
      d: [1.0, 10.0, 5.0, "Quadratic loss d"],
      I: [-2.0, 6.0, 3.0, "Applied current I"],
      r: [0.0001, 0.01, 0.001, "Slow time scale r"],
      s: [1.0, 8.0, 4.0, "Slow gain s"],
      x_r: [-3.0, 0.0, -1.6, "Resting offset x_r"],
    },
    defaultXlim: [-3.0, 3.0],
    defaultYlim: [-20.0, 10.0],
    defaultLims: [[-3.0, 3.0], [-20.0, 10.0], [-1.0, 3.5]],
    f: (_t, [x, y, z], p) => {
      const dx = y - p.a * x * x * x + p.b * x * x - z + p.I;
      const dy = p.c - p.d * x * x - y;
      const dz = p.r * (p.s * (x - p.x_r) - z);
      return [dx, dy, dz];
    },
  },
};

// ═══════════════════════════════════════════════════════════════
//  CUSTOM MODEL COMPILATION (nerdamer-powered)
// ═══════════════════════════════════════════════════════════════

/** Topological sort of custom functions based on dependency graph.
 *
 *  @param {Object} cfDict - {name: {vars: [...], expr: "..."}}
 *  @returns {Array} [[name, vars, exprStr], ...] in dependency order
 */
function sortCustomFunctions(cfDict) {
  if (!cfDict || Object.keys(cfDict).length === 0) return [];
  const names = Object.keys(cfDict);
  const visited = {};
  const ordered = [];

  function visit(name, path) {
    if (visited[name]) return;
    if (path.has(name)) {
      console.warn('Circular dependency in custom_functions involving', name);
      return;
    }
    path.add(name);
    const entry = cfDict[name];
    const exprStr = entry.expr || entry;
    for (const other of names) {
      if (other === name) continue;
      if (exprStr.includes(other)) {
        visit(other, path);
      }
    }
    visited[name] = true;
    ordered.push([name, entry.vars || [], exprStr]);
    path.delete(name);
  }

  for (const name of names) {
    if (!visited[name]) visit(name, new Set());
  }
  return ordered;
}

let _builtinMathRegistered = false;
/** Register safe_exp, step, and rww_H in nerdamer's parser and Math2 compiler scope. */
function registerBuiltinMathHelpers() {
  if (_builtinMathRegistered || typeof nerdamer === 'undefined') return;
  const core = nerdamer.getCore();
  core.Math2.safe_exp = function safe_exp(x) {
    return Math.exp(Math.max(-709, Math.min(709, x)));
  };
  core.Math2.step = function step(x) {
    return x >= 0 ? 1 : 0;
  };
  core.Math2.rww_H = function rww_H(x, d) {
    return Math.abs(d * x) < 1e-7
      ? 1 / d
      : x / (1 - Math.exp(Math.max(-709, Math.min(709, -d * x))));
  };
  const wrap = (name) => (...args) => {
    const flat = args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
    return core.PARSER.symfunction(name, flat);
  };
  core.PARSER.functions.safe_exp = [wrap('safe_exp'), 1];
  core.PARSER.functions.step = [wrap('step'), 1];
  core.PARSER.functions.rww_H = [wrap('rww_H'), [1, 2]];
  _builtinMathRegistered = true;
}

/** Register custom functions with nerdamer in topological order.
 *
 *  @param {Object} cfDict - {name: {vars: [...], expr: "..."}}
 */
function registerCustomFunctions(cfDict) {
  registerBuiltinMathHelpers();
  if (!cfDict || Object.keys(cfDict).length === 0) return;
  const ordered = sortCustomFunctions(cfDict);
  for (const [name, vars, body] of ordered) {
    try {
      nerdamer.setFunction(name, vars, body);
    } catch (e) {
      console.warn('Failed to register custom function', name, e);
    }
  }
}

/** Compile a model_spec dict into a MODELS-compatible entry.
 *
 *  @param {Object} spec - model_spec dict from Python
 *  @returns {Object} modelEntry with same shape as MODELS entries
 */
function compileModelSpec(spec) {
  if (!spec || !spec.equations || Object.keys(spec.equations).length === 0) {
    throw new Error('compileModelSpec: invalid or empty spec: ' + JSON.stringify(spec));
  }

  registerCustomFunctions(spec.custom_functions);

  const stateVars = Object.keys(spec.state_vars);
  const paramNames = Object.keys(spec.parameters || {});
  const dim = stateVars.length;
  const allArgNames = [...stateVars, ...paramNames];

  // Compile each equation
  const compiledEqs = {};
  const rawExprs = {};
  for (const name of stateVars) {
    const exprStr = spec.equations[name];
    rawExprs[name] = exprStr || '0';
    if (exprStr === undefined || exprStr === null) {
      compiledEqs[name] = () => 0;
      continue;
    }
    try {
      compiledEqs[name] = nerdamer(exprStr).buildFunction(allArgNames);
    } catch (e) {
      throw new Error(`Cannot compile equation for ${name}: ${e.message}`);
    }
  }

  // Full N-dimensional RHS
  function f(t, state, params) {
    const args = [];
    for (let i = 0; i < dim; i++) args.push(state[i]);
    for (const pn of paramNames) args.push(params[pn] !== undefined ? params[pn] : 0);
    const result = [];
    for (const name of stateVars) result.push(compiledEqs[name](...args));
    return result;
  }

  const display = spec.display || [0, Math.min(1, dim - 1)];
  const defaultParams = {};
  const paramInfo = {};
  if (spec.parameters) {
    for (const [name, info] of Object.entries(spec.parameters)) {
      defaultParams[name] = info.default;
      paramInfo[name] = [info.range[0], info.range[1], info.default, 'Parameter ' + name];
    }
  }

  const defaultLims = stateVars.map((name) => {
    const sv = spec.state_vars[name];
    return sv && sv.range ? [...sv.range] : [-5, 5];
  });
  const bounds = stateVars.map((name) => {
    const b = spec.state_vars[name]?.bounds ?? spec.bounds?.[name];
    return Array.isArray(b) && b.length === 2 ? [b[0], b[1]] : null;
  });
  const hasBounds = bounds.some((b) => b !== null);
  const defaultXlim = defaultLims[display[0]] || [-5, 5];
  const defaultYlim = display.length > 1 ? (defaultLims[display[1]] || [-5, 5]) : [-5, 5];

  return {
    dim,
    stateNames: stateVars,
    stateVars,
    display,
    defaultParams,
    paramInfo,
    defaultXlim: [...defaultXlim],
    defaultYlim: [...defaultYlim],
    defaultLims: defaultLims.map((l) => [...l]),
    bounds: hasBounds ? bounds : null,
    f,
    _compiledEqs: compiledEqs,
    _rawExprs: rawExprs,
    _allArgNames: allArgNames,
    _paramNames: paramNames,
  };
}

/** Resolve a model key or custom model spec to a model definition entry.
 *
 *  @param {string} modelName - key in MODELS (or 'custom')
 *  @param {Object} [spec] - custom model_spec dict when modelName === 'custom'
 *  @returns {Object} model definition entry
 */
function resolveModel(modelName, spec) {
  if (modelName === 'custom') return compileModelSpec(spec);
  const def = MODELS[modelName];
  if (!def) throw new Error('Unknown model: ' + modelName);
  return def;
}
