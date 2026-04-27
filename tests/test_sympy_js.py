"""Property-based tests for the SymPy-to-JavaScript transpiler."""

import json
import math
import random
import subprocess
import tempfile

import pytest
import sympy as sp

from phase_plane_widget.sympy_js import FUNCTION_MAP, sympy_to_js, transpile_custom_function

# ── Symbol pool ──
SYMBOLS = [sp.Symbol(c) for c in "xyzuvw"]


def _random_expr(depth: int = 0, max_depth: int = 4) -> sp.Expr:
    """Generate a random SymPy expression tree."""
    if depth >= max_depth:
        # Terminal: symbol or number
        return random.choice(SYMBOLS) if random.random() < 0.7 else sp.Integer(random.randint(-5, 5))

    op = random.random()
    if op < 0.2:
        # Binary: +, -, *, /
        a = _random_expr(depth + 1, max_depth)
        b = _random_expr(depth + 1, max_depth)
        return random.choice([a + b, a - b, a * b, a / (b + 1)])
    elif op < 0.35:
        # Power
        a = _random_expr(depth + 1, max_depth)
        b = sp.Integer(random.randint(1, 4))
        return a**b
    elif op < 0.5:
        # Function application
        func = random.choice([
            sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt,
            sp.Abs, sp.sign, sp.tanh, sp.Heaviside,
        ])
        a = _random_expr(depth + 1, max_depth)
        return func(a)
    elif op < 0.7:
        # Terminal symbol
        return random.choice(SYMBOLS)
    else:
        # Terminal number
        return sp.Float(random.uniform(-3, 3), 5)


def _js_eval(js_expr: str, values: dict) -> float:
    """Evaluate a JS expression in Node.js with the given variable bindings."""
    args = ", ".join(values.keys())
    body = js_expr
    code = f"""
function fn({args}) {{ return {body}; }}
console.log(JSON.stringify(fn({', '.join(repr(v) for v in values.values())})));
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(code)
        fn = f.name
    try:
        result = subprocess.run(
            ["node", fn], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return float(json.loads(result.stdout.strip()))
    finally:
        import os as _os
        _os.unlink(fn)


def _sympy_eval(expr: sp.Expr, values: dict) -> float:
    """Evaluate a SymPy expression numerically."""
    subs = {sp.Symbol(k): float(v) for k, v in values.items()}
    # Skip expressions that produce complex values (e.g., sqrt of negative number)
    sympy_val = complex(expr.evalf(subs=subs))
    if not math.isfinite(sympy_val.real) or abs(sympy_val.imag) > 1e-12:
        return None  # signal skip
    return float(sympy_val.real)

def _should_skip_expr(expr: sp.Expr) -> bool:
    """Check if an expression may produce complex values."""
    # Simple heuristic: skip if expression contains sqrt of a non-literal
    for atom in expr.atoms(sp.Pow):
        if isinstance(atom.args[1], (sp.Rational, sp.Number)) and float(atom.args[1]) == 0.5:
            base = atom.args[0]
            # If base is a symbol or Add, skip to avoid negative-input complex values
            if base.free_symbols:
                return True
    return False


# ── Tests ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(100))
def test_random_expression_transpilation(seed: int):
    """Generate random SymPy expressions, transpile them, and verify
    evaluation at random points matches SymPy's numeric result."""
    random.seed(seed)
    expr = _random_expr()
    # Collect free symbols
    free = sorted({str(s) for s in expr.free_symbols}, key=lambda s: len(s))
    symbol_map = {s: s for s in free}

    try:
        js = sympy_to_js(expr, symbol_map)
    except ValueError as exc:
        if "Unsupported SymPy function" in str(exc):
            pytest.skip(f"Unsupported function: {exc}")
        raise

    # Skip expressions with sqrt of symbols (may produce complex values)
    if _should_skip_expr(expr):
        pytest.skip("Expression may produce complex values")

    values = {s: random.uniform(-1, 1) for s in free}
    sympy_val = _sympy_eval(expr, values)
    if sympy_val is None:
        pytest.skip("Complex evaluation")
    js_val = _js_eval(js, values)
    assert math.isfinite(js_val), f"JS returned non-finite: {js_val}"
    diff = abs(sympy_val - js_val)
    assert diff < 1e-10, (
        f"Mismatch for {expr}: SymPy={sympy_val}, JS={js_val}, diff={diff}\nJS={js}"
    )


def test_function_map_completeness():
    """Every function in FUNCTION_MAP should be executable in Node.js."""
    symbol_map = {"x": "x"}
    test_cases = [
        (sp.exp(sp.Symbol("x")), "Math.exp(x)"),
        (sp.log(sp.Symbol("x")), "Math.log(x)"),
        (sp.sin(sp.Symbol("x")), "Math.sin(x)"),
        (sp.cos(sp.Symbol("x")), "Math.cos(x)"),
        (sp.tan(sp.Symbol("x")), "Math.tan(x)"),
        (sp.sqrt(sp.Symbol("x")), "Math.sqrt(x)"),
        (sp.Abs(sp.Symbol("x")), "Math.abs(x)"),
        (sp.sign(sp.Symbol("x")), "Math.sign(x)"),
        (sp.tanh(sp.Symbol("x")), "Math.tanh(x)"),
        (sp.Min(sp.Symbol("x"), sp.Integer(2)), "Math.min(x, 2)"),
        (sp.Max(sp.Symbol("x"), sp.Integer(-1)), "Math.max(x, -1)"),
    ]
    for expr, expected_substr in test_cases:
        js = sympy_to_js(expr, symbol_map)
        # For commutative functions, accept either argument ordering
        if expected_substr.startswith("Math.min(") or expected_substr.startswith("Math.max("):
            parts = expected_substr.split("(", 1)[1].rstrip(")").split(", ")
            assert js.startswith("Math.min(") or js.startswith("Math.max(")
            js_args = js.split("(", 1)[1].rstrip(")").split(", ")
            assert sorted(parts) == sorted(js_args), f"Expected {parts}, got {js_args}"
        else:
            assert expected_substr in js, f"Expected {expected_substr} in {js}"

    # Evaluate Heaviside at edge cases
    h = sympy_to_js(sp.Heaviside(sp.Symbol("x")), {"x": "x"})
    assert "x > 0" in h
    assert _js_eval(h, {"x": -1}) == 0.0
    assert _js_eval(h, {"x": 1}) == 1.0
    assert _js_eval(h, {"x": 0}) == 0.5


def test_custom_function():
    """Transpile a custom helper function."""
    x = sp.Symbol("x")
    k = sp.Symbol("k")
    theta = sp.Symbol("theta")
    symbol_map = {"x": "x", "k": "k", "theta": "theta"}

    sigmoid_expr = 1 / (1 + sp.exp(-k * (x - theta)))
    js = transpile_custom_function("sigmoid", ["x", "k", "theta"], sigmoid_expr, symbol_map)
    assert "function sigmoid(x, k, theta)" in js
    assert "Math.exp" in js


def test_power_corner_cases():
    """Pow(x, n) for various n values."""
    x = sp.Symbol("x")
    tests = [
        (x**2, "Math.pow(x, 2)", 3.0, 9.0),
        (x**0.5, "Math.sqrt(x)", 4.0, 2.0),
        (x**(-1), "1/x", 2.0, 0.5),
        (x**3, "Math.pow(x, 3)", 2.0, 8.0),
    ]
    for expr, expected, val_in, expected_val in tests:
        js = sympy_to_js(expr, {"x": "x"})
        assert expected in js, f"Expected '{expected}' in {js}"
        actual = _js_eval(js, {"x": val_in})
        assert abs(actual - expected_val) < 1e-12, (
            f"Mismatch for {expr}: expected {expected_val}, got {actual}"
        )
