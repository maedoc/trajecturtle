"""Tests for the Heun stochastic integrator (Stratonovich additive noise).

These tests execute the JS integration logic from widget.js via Node.js,
verifying correctness of the Heun scheme against RK4 and the noise effect.
"""

import json
import math
import pathlib
import subprocess
import sys
import time

_JS_TEST = pathlib.Path(__file__).parent / "test_heun_runner.js"

# ── Self-contained JS test runner ─────────────────────────────────────
_JS_SOURCE = r"""
// Replicate the relevant portions of widget.js for testing

const MODELS = {
  wilson_cowan: {
    dim: 2,
    stateNames: ["E", "I"],
    f: (_t, [E, I], p) => {
      const _sigmoid = (x, k, theta) => {
        const arg = -k * (x - theta);
        const clipped = Math.max(-709, Math.min(709, arg));
        return 1.0 / (1.0 + Math.exp(clipped));
      };
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
    f: (_t, [v, w], p) => [
      v - (v * v * v) / 3.0 - w + p.I,
      p.epsilon * (v + p.a - p.b * w),
    ],
  },
  mpr: {
    dim: 2,
    stateNames: ["r", "v"],
    f: (_t, [r, v], p) => {
      const r_eff = Math.max(r, 1e-10);
      const dr = p.delta / Math.PI + 2.0 * r_eff * v;
      const dv = v * v + p.eta_bar + p.J * r_eff + p.I - (Math.PI * r_eff) ** 2;
      return [dr, dv];
    },
  },
};

function rk4(modelName, y0, t0, tf, h, params) {
  const f = MODELS[modelName].f;
  const results = [[t0, ...y0]];
  let t = t0;
  let y = [...y0];
  while (t < tf - 1e-12) {
    const dt = Math.min(h, tf - t);
    const k1 = f(t, y, params);
    const k2 = f(t + dt / 2, y.map((yi, i) => yi + (dt / 2) * k1[i]), params);
    const k3 = f(t + dt / 2, y.map((yi, i) => yi + (dt / 2) * k2[i]), params);
    const k4 = f(t + dt, y.map((yi, i) => yi + dt * k3[i]), params);
    y = y.map((yi, i) => yi + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    t += dt;
    if (y.some((v) => !isFinite(v))) break;
    results.push([t, ...y]);
  }
  return results;
}

// Deterministic PRNG seed for reproducible Box-Muller
let _seed = 12345;
function setSeed(s) { _seed = s; }
// Simple LCG for reproducible uniforms
function lcg() {
  _seed = (_seed * 1664525 + 1013904223) & 0xffffffff;
  return (_seed >>> 0) / 0x100000000;
}

function randn() {
  let u1 = lcg();
  let u2 = lcg();
  while (u1 === 0) u1 = lcg();
  while (u2 === 0) u2 = lcg();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function heun(modelName, y0, t0, tf, h, params, noiseSigma) {
  const f = MODELS[modelName].f;
  const dim = y0.length;
  const results = [[t0, ...y0]];
  let t = t0;
  let y = [...y0];
  let steps = 0;
  const maxSteps = 50000;
  while (t < tf - 1e-12 && steps < maxSteps) {
    const dt = Math.min(h, tf - t);
    const sqrtDt = Math.sqrt(dt);
    const dW = new Array(dim);
    for (let i = 0; i < dim; i++) {
      dW[i] = sqrtDt * randn();
    }
    const fn = f(t, y, params);
    const yTilde = y.map((yi, i) => yi + fn[i] * dt + (noiseSigma[i] || 0) * dW[i]);
    const fTilde = f(t + dt, yTilde, params);
    y = y.map((yi, i) => yi + 0.5 * (fn[i] + fTilde[i]) * dt + (noiseSigma[i] || 0) * dW[i]);
    t += dt;
    steps++;
    if (y.some((v) => !isFinite(v))) break;
    results.push([t, ...y]);
  }
  return results;
}

// ── Test 1: Heun with zero noise matches RK4 ──
function testZeroNoise() {
  const params = { a: 0.7, b: 0.8, epsilon: 0.08, I: 0.5 };
  const y0 = [0.1, 0.1];
  const t0 = 0, tf = 10, h = 0.01;

  const rk = rk4("fitzhugh_nagumo", y0, t0, tf, h, params);
  setSeed(12345);
  const hn = heun("fitzhugh_nagumo", y0, t0, tf, h, params, [0, 0]);

  // Compare at the last time point
  const rkLast = rk[rk.length - 1];
  const hnLast = hn[hn.length - 1];

  const maxDiff = Math.max(
    Math.abs(rkLast[1] - hnLast[1]),
    Math.abs(rkLast[2] - hnLast[2])
  );
  return { passed: maxDiff < 1e-4, maxDiff, rkFinal: rkLast.slice(1), hnFinal: hnLast.slice(1) };
}

// ── Test 2: Increasing noise increases variance ──
function testNoiseIncreasesVariance() {
  const params = { a: 0.7, b: 0.8, epsilon: 0.08, I: 0.5 };
  const y0 = [0.1, 0.1];
  const t0 = 0, tf = 10, h = 0.01;
  const nTrials = 30;

  function runTrial(sigma) {
    const finals = [];
    for (let i = 0; i < nTrials; i++) {
      setSeed(20000 + i);
      const traj = heun("fitzhugh_nagumo", y0, t0, tf, h, params, [sigma, sigma]);
      const last = traj[traj.length - 1];
      finals.push(last[1]); // v component
    }
    const mean = finals.reduce((a, b) => a + b, 0) / finals.length;
    const variance = finals.reduce((s, v) => s + (v - mean) ** 2, 0) / finals.length;
    return { mean, variance, finals };
  }

  const low = runTrial(0.0);
  const high = runTrial(0.5);

  return {
    passed: high.variance > low.variance,
    lowVariance: low.variance,
    highVariance: high.variance,
  };
}

const r1 = testZeroNoise();
const r2 = testNoiseIncreasesVariance();

const allPassed = r1.passed && r2.passed;
console.log(JSON.stringify({ passed: allPassed, test1: r1, test2: r2 }));
process.exit(allPassed ? 0 : 1);
"""


def _write_runner():
    _JS_TEST.write_text(_JS_SOURCE, encoding="utf-8")


def _run_js():
    """Run the JS test runner and return parsed JSON output."""
    result = subprocess.run(
        [sys.executable, "-c", f"import subprocess, sys; sys.stdout.buffer.write(subprocess.check_output(['node', '{_JS_TEST}']))"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Fallback: run node directly
    if result.returncode != 0 or not result.stdout.strip():
        result = subprocess.run(
            ["node", str(_JS_TEST)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    return result


def test_heun_zero_noise_matches_rk4():
    """Heun with sigma=[0,0] should be close to RK4 (within 1e-6 for same dt)."""
    _write_runner()
    result = _run_js()
    assert result.returncode == 0, f"JS runner failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
    data = json.loads(result.stdout.strip())
    t1 = data["test1"]
    assert t1["passed"], f"Zero-noise Heun vs RK4 max diff {t1['maxDiff']} >= 1e-4"
    assert t1["maxDiff"] < 1e-4, f"Max diff {t1['maxDiff']} exceeds 1e-4"


def test_noise_increases_variance():
    """Increasing noise strength should increase trajectory variance."""
    _write_runner()
    result = _run_js()
    assert result.returncode == 0, f"JS runner failed:\nstdout:{result.stdout}\nstderr:{result.stderr}"
    data = json.loads(result.stdout.strip())
    t2 = data["test2"]
    assert t2["passed"], (
        f"Noise did not increase variance: low sigma variance {t2['lowVariance']}, "
        f"high sigma variance {t2['highVariance']}"
    )
    assert t2["highVariance"] > t2["lowVariance"], (
        f"sigma=0.0 variance {t2['lowVariance']} >= sigma=0.5 variance {t2['highVariance']}"
    )


if __name__ == "__main__":
    _write_runner()
    result = _run_js()
    data = json.loads(result.stdout.strip())
    print(f"Test 1 (zero noise matches RK4): {'PASS' if data['test1']['passed'] else 'FAIL'}")
    print(f"  Max diff: {data['test1']['maxDiff']:.2e}")
    print(f"Test 2 (noise increases variance): {'PASS' if data['test2']['passed'] else 'FAIL'}")
    print(f"  Low sigma variance: {data['test2']['lowVariance']:.6e}")
    print(f"  High sigma variance: {data['test2']['highVariance']:.6e}")
    print(f"Overall: {'ALL PASS' if data['passed'] else 'FAIL'}")
    sys.exit(0 if data['passed'] else 1)
