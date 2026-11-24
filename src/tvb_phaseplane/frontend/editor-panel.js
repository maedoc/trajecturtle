function escapeHtml(value) { return String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]); }

function mountEditor({ model, controller, document: doc }) {
  const scope = panelScope();
  const element = panelElement(doc, 'ppw-editor-component', `<button class="ppw-editor-toggle" aria-expanded="false">Custom model editor</button>\n      <div class="ppw-editor-panel">
        <div class="ppw-editor-section">
          <div class="ppw-editor-section-title">State Variables</div>
          <table class="ppw-editor-table ppw-editor-state-table">
            <thead><tr><th>Name</th><th>Min</th><th>Max</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
        <div class="ppw-editor-section">
          <div class="ppw-editor-section-title">Parameters</div>
          <table class="ppw-editor-table ppw-editor-param-table">
            <thead><tr><th>Name</th><th>Default</th><th>Min</th><th>Max</th><th>Step</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
        <div class="ppw-editor-section">
          <div class="ppw-editor-section-title">Equations</div>
          <div class="ppw-editor-equations"></div>
        </div>
        <div class="ppw-editor-actions">
          <button class="ppw-btn-apply ppw-editor-apply">Apply</button>
          <button class="ppw-btn-copy ppw-editor-copy">Copy Spec</button>
        </div>
        <div class="ppw-editor-error"></div>
        <div class="ppw-editor-status"></div>
      </div>
`);

  const editorToggle = element.querySelector('.ppw-editor-toggle');
  const editorPanel = element.querySelector('.ppw-editor-panel');
  const editorStateTable = editorPanel.querySelector('.ppw-editor-state-table tbody');
  const editorParamTable = editorPanel.querySelector('.ppw-editor-param-table tbody');
  const editorEquationsDiv = editorPanel.querySelector('.ppw-editor-equations');
  const editorApplyBtn = editorPanel.querySelector('.ppw-editor-apply');
  const editorCopyBtn = editorPanel.querySelector('.ppw-editor-copy');
  const editorError = editorPanel.querySelector('.ppw-editor-error');
  const editorStatus = editorPanel.querySelector('.ppw-editor-status');
  let _editorOpen = false;
  let _editorDebounceTimer = null;
  let _editorRequestId = 0;
  let _editorState = "IDLE"; // IDLE | TYPING | PARSING | VALID | ERROR
  const EDITOR_DEBOUNCE_MS = 500;

  function _setEditorStatus(text) {
    editorStatus.textContent = text;
  }

  function _setEditorState(state) {
    _editorState = state;
    if (state === "IDLE") _setEditorStatus("");
    if (state === "TYPING") _setEditorStatus("Typing...");
    if (state === "PARSING") _setEditorStatus("Parsing...");
    if (state === "VALID") _setEditorStatus("Valid ✓");
    if (state === "ERROR") _setEditorStatus("Error ✗");
  }

  function _clearPerFieldErrors() {
    editorEquationsDiv.querySelectorAll(".ppw-ed-expr").forEach((ta) => {
      ta.classList.remove("ppw-error");
    });
  }

  function _markEquationError(name) {
    const div = editorEquationsDiv.querySelector(`.ppw-editor-equation[data-name="${escapeHtml(name)}"]`);
    if (div) div.querySelector(".ppw-ed-expr").classList.add("ppw-error");
  }

  function toggleEditor() {
    _editorOpen = !_editorOpen;
    editorPanel.classList.toggle("open", _editorOpen);
    if (_editorOpen) populateEditor();
  }

  function populateEditor() {
    const modelName = model.get("model_name");
    let spec;
    if (modelName === "custom" && model.get("model_spec")) {
      spec = model.get("model_spec");
    } else {
      const def = MODELS[modelName] || MODELS.wilson_cowan;
      const stateNames = def.stateNames || [];
      const paramKeys = Object.keys(def.defaultParams || {});
      const paramInfo = def.paramInfo || {};
      const defaultLims = def.defaultLims || stateNames.map(() => [-1, 1]);
      spec = {
        name: modelName,
        state_vars: {},
        parameters: {},
        equations: {},
        display: def.display || [0, 1],
      };
      stateNames.forEach((name, i) => {
        const [lo, hi] = defaultLims[i] || [-1, 1];
        spec.state_vars[name] = { default: (lo + hi) / 2, range: [lo, hi] };
      });
      paramKeys.forEach((k) => {
        const [lo, hi, defVal] = paramInfo[k] || [-1, 1, 0];
        spec.parameters[k] = { default: defVal, range: [lo, hi], step: (hi - lo) / 500 };
      });
      stateNames.forEach((name) => {
        spec.equations[name] = "/* built-in model — equations not editable */";
      });
    }

    let svHtml = "";
    for (const [name, info] of Object.entries(spec.state_vars || {})) {
      const [lo, hi] = info.range || [-1, 1];
      svHtml += `<tr data-name="${escapeHtml(name)}"><td>${escapeHtml(name)}</td><td><input type="number" class="ppw-ed-svmin" value="${lo}"></td><td><input type="number" class="ppw-ed-svmax" value="${hi}"></td></tr>`;
    }
    editorStateTable.innerHTML = svHtml;

    let pHtml = "";
    for (const [name, info] of Object.entries(spec.parameters || {})) {
      const [lo, hi] = info.range || [-1, 1];
      const step = info.step || (hi - lo) / 500;
      pHtml += `<tr data-name="${escapeHtml(name)}"><td>${escapeHtml(name)}</td><td><input type="number" class="ppw-ed-pdef" value="${info.default}"></td><td><input type="number" class="ppw-ed-pmin" value="${lo}"></td><td><input type="number" class="ppw-ed-pmax" value="${hi}"></td><td><input type="number" class="ppw-ed-pstep" value="${step}"></td></tr>`;
    }
    editorParamTable.innerHTML = pHtml;

    let eqHtml = "";
    for (const [name, expr] of Object.entries(spec.equations || {})) {
      const readonly = expr.startsWith("/*") ? "readonly" : "";
      eqHtml += `<div class="ppw-editor-equation" data-name="${escapeHtml(name)}"><label>d${escapeHtml(name)}/dt =</label><textarea class="ppw-ed-expr" ${readonly}>${escapeHtml(expr)}</textarea></div>`;
    }
    editorEquationsDiv.innerHTML = eqHtml;

    _clearPerFieldErrors();
    _setEditorState("IDLE");

    // Input handling is delegated to this panel's lifecycle scope.
  }

  function onEditorInput() {
    _clearPerFieldErrors();
    _setEditorState("TYPING");
    editorError.classList.remove("visible");
    editorError.textContent = "";

    if (_editorDebounceTimer) scope.cancel(_editorDebounceTimer);
    _editorDebounceTimer = scope.later(() => {
      const thisRequestId = ++_editorRequestId;
      _tryAutoApply(thisRequestId);
    }, EDITOR_DEBOUNCE_MS);
  }

  function _tryAutoApply(requestId) {
    if (requestId !== _editorRequestId) return; // stale result rejection

    _setEditorState("PARSING");

    const spec = buildModelSpec();
    const { errors, equationErrors } = validateSpec(spec);

    if (requestId !== _editorRequestId) return;

    if (errors.length > 0) {
      _setEditorState("ERROR");
      equationErrors.forEach((name) => _markEquationError(name));
      editorError.textContent = errors.join("\n");
      editorError.classList.add("visible");
      return;
    }

    try {
      const entry = compileModelSpec(spec);
      if (!entry || !entry.f) throw new Error("No executable function");
    } catch (e) {
      _setEditorState("ERROR");
      editorError.textContent = `Compilation error: ${e.message || e}`;
      editorError.classList.add("visible");
      return;
    }

    if (requestId !== _editorRequestId) return;

    controller.applySpec(spec);
    _setEditorState("VALID");
  }

  function buildModelSpec() {
    const state_vars = {};
    editorStateTable.querySelectorAll("tr").forEach((tr) => {
      const name = tr.dataset.name;
      const lo = parseFloat(tr.querySelector(".ppw-ed-svmin").value);
      const hi = parseFloat(tr.querySelector(".ppw-ed-svmax").value);
      state_vars[name] = { default: (lo + hi) / 2, range: [lo, hi] };
    });

    const parameters = {};
    editorParamTable.querySelectorAll("tr").forEach((tr) => {
      const name = tr.dataset.name;
      const defVal = parseFloat(tr.querySelector(".ppw-ed-pdef").value);
      const lo = parseFloat(tr.querySelector(".ppw-ed-pmin").value);
      const hi = parseFloat(tr.querySelector(".ppw-ed-pmax").value);
      const step = parseFloat(tr.querySelector(".ppw-ed-pstep").value);
      parameters[name] = { default: defVal, range: [lo, hi], step };
    });

    const equations = {};
    editorEquationsDiv.querySelectorAll(".ppw-editor-equation").forEach((div) => {
      const name = div.dataset.name;
      const expr = div.querySelector(".ppw-ed-expr").value.trim();
      if (!expr.startsWith("/*")) equations[name] = expr;
    });

    const stateNamesList = Object.keys(state_vars);
    return {
      name: "custom",
      state_vars,
      parameters,
      equations,
      display: model.get('display') || (stateNamesList.length > 1 ? [0, 1] : [0]),
      integrator: model.get("integrator") || "rk4",
      noise_per_var: model.get("noise_enable") ? model.get("noise_sigma") : null,
    };
  }

  function validateSpec(spec) {
    const errors = [];
    const equationErrors = [];
    if (!Object.keys(spec.equations).length) errors.push('No editable equations. Load a custom model to edit its equations.');
    for (const [name, info] of Object.entries(spec.state_vars)) {
      if (!info.range.every(Number.isFinite) || info.range[0] >= info.range[1]) errors.push(`State '${name}': use increasing finite bounds`);
    }
    for (const [name, info] of Object.entries(spec.parameters)) {
      if (![...info.range, info.default, info.step].every(Number.isFinite) || info.range[0] >= info.range[1] || info.default < info.range[0] || info.default > info.range[1] || info.step <= 0) errors.push(`Parameter '${name}': check bounds, default and positive step`);
    }
    for (const [name, exprStr] of Object.entries(spec.equations)) {
      try {
        nerdamer(exprStr);
      } catch (e) {
        errors.push(`Equation '${escapeHtml(name)}': ${e.message || "parse error"}`);
        equationErrors.push(name);
      }
    }
    return { errors, equationErrors };
  }

  function applyEditor() {
    // Manual apply — force immediate application regardless of debounce
    if (_editorDebounceTimer) scope.cancel(_editorDebounceTimer);
    _editorRequestId++;
    _tryAutoApply(_editorRequestId);
  }

  function copySpec() {
    const spec = buildModelSpec();
    const json = JSON.stringify(spec, null, 2);
    navigator.clipboard.writeText(json).then(() => {
      _setEditorStatus("Spec copied to clipboard!");
      scope.later(() => { if (_editorState === "IDLE") _setEditorStatus(""); }, 2000);
    }).catch(() => {
      _setEditorStatus("Failed to copy — see browser console");
      console.log(json);
    });
  }


  scope.on(editorPanel, 'input', event => {
    if (event.target.matches('input, textarea') && !event.target.readOnly) onEditorInput();
  });
  scope.on(editorToggle, 'click', () => { toggleEditor(); editorToggle.setAttribute('aria-expanded', String(_editorOpen)); });
  scope.on(editorApplyBtn, 'click', applyEditor);
  scope.on(editorCopyBtn, 'click', copySpec);
  scope.cleanup(() => { clearTimeout(_editorDebounceTimer); _editorRequestId++; });
  return mountedPanel(element, scope);
}
