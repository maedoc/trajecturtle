// Lazy Web Worker manager & 3-host asset resolver for AUTO2000 WASM continuation.

function _toUint8Array(buf) {
  if (buf instanceof Uint8Array) return buf;
  if (buf instanceof ArrayBuffer) return new Uint8Array(buf);
  if (ArrayBuffer.isView(buf)) return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  return new Uint8Array(buf);
}

function _decodeBase64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

function _requestAssetsFromKernel(model, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      model.off?.('msg:custom', onCustomMsg);
      reject(new Error('Timed out waiting for AUTO2000 assets from Jupyter kernel.'));
    }, timeoutMs);

    const onCustomMsg = (content, buffers) => {
      if (content?.type !== 'auto_assets') return;
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      model.off?.('msg:custom', onCustomMsg);
      if (!Array.isArray(buffers) || buffers.length < 2) {
        reject(new Error('Kernel auto_assets response missing WASM or worker buffer.'));
        return;
      }
      const wasmBytes = _toUint8Array(buffers[0]);
      const workerBytes = _toUint8Array(buffers[1]);
      const workerBlob = new Blob([workerBytes], { type: 'application/javascript' });
      const workerUrl = URL.createObjectURL(workerBlob);
      resolve({ workerUrl, wasmBytes });
    };

    model.on('msg:custom', onCustomMsg);
    model.send({ type: 'request_auto_assets' });
  });
}

export async function resolveAutoAssets(model) {
  const wasmUrl = model?.get?.('auto_wasm_url') || '';
  const workerUrl = model?.get?.('auto_worker_url') || '';

  // 1. Explicit URL traits (direct HTTP/static hosting)
  if (wasmUrl || workerUrl) {
    const base = globalThis.location?.href || 'http://localhost/';
    const resolvedWorkerUrl = workerUrl
      ? new URL(workerUrl, base).href
      : new URL('auto2000-worker.js', new URL(wasmUrl, base)).href;
    const resolvedWasmUrl = wasmUrl
      ? new URL(wasmUrl, base).href
      : new URL('auto2000_core.wasm', new URL(workerUrl, base)).href;
    return { workerUrl: resolvedWorkerUrl, wasmUrl: resolvedWasmUrl };
  }

  // 2. Standalone HTML embedded assets
  const embedded = globalThis.__PPW_AUTO_ASSETS__;
  if (embedded && embedded.wasmBase64 && embedded.workerSource) {
    const wasmBytes = _decodeBase64ToBytes(embedded.wasmBase64);
    const workerBlob = new Blob([embedded.workerSource], { type: 'application/javascript' });
    const blobUrl = URL.createObjectURL(workerBlob);
    return { workerUrl: blobUrl, wasmBytes };
  }

  // 3. JupyterLab / Notebook kernel comm
  if (model && !model._isMock && typeof model.send === 'function' && typeof model.on === 'function') {
    return _requestAssetsFromKernel(model);
  }

  // 4. Lightweight standalone export without embedded assets or URLs
  throw new Error(
    'AUTO2000 WASM assets are unavailable (not bundled in this lightweight export and no auto_wasm_url or live kernel comm is present).'
  );
}

export function createAutoClient(model) {
  let worker = null;
  let readyPromise = null;
  let assetsPromise = null;
  let pendingJob = null;
  let nextJobId = 1;
  let runEpoch = 0;

  function _clearWorker() {
    if (worker) {
      worker.terminate();
      worker = null;
    }
    readyPromise = null;
  }

  function _rejectPending(err) {
    if (!pendingJob) return;
    const job = pendingJob;
    pendingJob = null;
    if (job.timer) clearTimeout(job.timer);
    job.reject(err);
  }

  async function _ensureWorkerReady() {
    if (readyPromise) return readyPromise;
    readyPromise = (async () => {
      if (!assetsPromise) {
        assetsPromise = resolveAutoAssets(model).catch((err) => {
          assetsPromise = null;
          throw err;
        });
      }
      const assets = await assetsPromise;
      const w = new Worker(assets.workerUrl);
      worker = w;

      await new Promise((resolve, reject) => {
        const onMessage = (event) => {
          const data = event.data || {};
          if (data.type === 'ready') {
            w.removeEventListener('message', onMessage);
            w.removeEventListener('error', onError);
            resolve();
          } else if (data.type === 'error' && data.jobId === undefined) {
            w.removeEventListener('message', onMessage);
            w.removeEventListener('error', onError);
            reject(new Error(data.message || 'AUTO2000 worker initialization failed'));
          }
        };
        const onError = (err) => {
          w.removeEventListener('message', onMessage);
          w.removeEventListener('error', onError);
          reject(new Error(err?.message || 'AUTO2000 worker failed to load'));
        };
        w.addEventListener('message', onMessage);
        w.addEventListener('error', onError);
        w.postMessage({
          type: 'init',
          wasmBytes: assets.wasmBytes,
          wasmUrl: assets.wasmUrl,
        });
      });

      w.addEventListener('message', (event) => {
        const data = event.data || {};
        if (!pendingJob || data.jobId !== pendingJob.jobId) return;
        const job = pendingJob;
        pendingJob = null;
        if (job.timer) clearTimeout(job.timer);
        if (data.type === 'result') {
          job.resolve({
            branch: data.branch,
            specialPoints: data.specialPoints,
            restartHandles: data.restartHandles,
            elapsedMs: data.elapsedMs,
          });
        } else if (data.type === 'error') {
          job.reject(new Error(data.message || 'AUTO2000 continuation failed'));
        }
      });

      w.addEventListener('error', (err) => {
        _clearWorker();
        _rejectPending(new Error(err?.message || 'AUTO2000 worker crashed'));
      });
    })().catch((err) => {
      _clearWorker();
      throw err;
    });

    return readyPromise;
  }

  function cancel() {
    runEpoch++;
    _clearWorker();
    _rejectPending(new Error('AUTO2000 continuation cancelled'));
  }

  async function run(payload, { timeoutMs = 15000 } = {}) {
    if (pendingJob) {
      _rejectPending(new Error('AUTO2000 continuation superseded by a new run'));
    }
    const myEpoch = runEpoch;
    await _ensureWorkerReady();
    if (myEpoch !== runEpoch || !worker) {
      throw new Error('AUTO2000 continuation cancelled');
    }
    const jobId = nextJobId++;
    return new Promise((resolve, reject) => {
      let timer = null;
      if (timeoutMs > 0) {
        timer = setTimeout(() => {
          if (pendingJob && pendingJob.jobId === jobId) {
            _clearWorker();
            _rejectPending(new Error(`AUTO2000 continuation timed out after ${timeoutMs}ms`));
          }
        }, timeoutMs);
      }
      pendingJob = { jobId, resolve, reject, timer };
      worker.postMessage({ type: 'run', jobId, ...payload });
    });
  }

  return { run, cancel };
}
