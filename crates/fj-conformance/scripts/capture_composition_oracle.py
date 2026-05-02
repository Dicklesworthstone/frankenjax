#!/usr/bin/env python3
"""Capture composition oracle fixtures from JAX.

Usage:
  python3 crates/fj-conformance/scripts/capture_composition_oracle.py \
      --legacy-root legacy_jax_code/jax \
      --output crates/fj-conformance/fixtures/composition_oracle.v1.json

This script records reference values for transform composition patterns:
- jit(grad), grad(grad), vmap(grad), jit(vmap)
- grad, vmap, value_and_grad, jacobian, hessian
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _try_import_jax(legacy_root: Path | None):
    venv_site = Path(__file__).resolve().parents[3] / ".venv" / "lib"
    if venv_site.exists():
        for d in venv_site.iterdir():
            sp = d / "site-packages"
            if sp.exists():
                sys.path.insert(0, str(sp))
                break

    if legacy_root:
        if not legacy_root.exists():
            raise FileNotFoundError(f"legacy JAX root does not exist: {legacy_root}")
        sys.path.insert(0, str(legacy_root))

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return jax, jnp


def _scalar_f64(value: float) -> dict[str, Any]:
    return {"kind": "scalar_f64", "value": float(value)}


def _vector_f64(arr) -> dict[str, Any]:
    shape = [int(dim) for dim in arr.shape]
    values = [float(x) for x in arr.reshape(-1).tolist()]
    return {"kind": "vector_f64", "shape": shape, "values": values}


def _matrix_f64(arr) -> dict[str, Any]:
    shape = [int(dim) for dim in arr.shape]
    values = [float(x) for x in arr.reshape(-1).tolist()]
    return {"kind": "matrix_f64", "shape": shape, "values": values}


def _to_value(x) -> dict[str, Any]:
    import jax.numpy as jnp
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return _scalar_f64(float(arr))
    if arr.ndim == 1:
        return _vector_f64(arr)
    if arr.ndim == 2:
        return _matrix_f64(arr)
    raise ValueError(f"Unsupported ndim: {arr.ndim}")


def capture_composition_cases(jax, jnp) -> list[dict[str, Any]]:
    cases = []

    def poly(x):
        return x**2 + 3*x

    for x_val in [0.0, 1.0, -2.0, 5.0]:
        result = jax.jit(jax.grad(poly))(x_val)
        cases.append({
            "case_id": f"jit_grad_poly_x{x_val}",
            "composition": "jit(grad)",
            "program": "x^2+3x",
            "args": [_scalar_f64(x_val)],
            "expected": [_to_value(result)],
        })

    def cubic(x):
        return x**3

    for x_val in [1.0, 2.0, -1.0]:
        result = jax.grad(jax.grad(cubic))(x_val)
        cases.append({
            "case_id": f"grad_grad_cubic_x{x_val}",
            "composition": "grad(grad)",
            "program": "x^3",
            "args": [_scalar_f64(x_val)],
            "expected": [_to_value(result)],
        })

    def sin_fn(x):
        return jnp.sin(x)

    x_vec = jnp.array([0.0, 1.0, 2.0, 3.0])
    result = jax.vmap(jax.grad(sin_fn))(x_vec)
    cases.append({
        "case_id": "vmap_grad_sin",
        "composition": "vmap(grad)",
        "program": "sin(x)",
        "args": [_vector_f64(x_vec)],
        "expected": [_to_value(result)],
    })

    def square(x):
        return x**2

    x_vec = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = jax.vmap(square)(x_vec)
    cases.append({
        "case_id": "vmap_square",
        "composition": "vmap",
        "program": "x^2",
        "args": [_vector_f64(x_vec)],
        "expected": [_to_value(result)],
    })

    result = jax.jit(jax.vmap(square))(x_vec)
    cases.append({
        "case_id": "jit_vmap_square",
        "composition": "jit(vmap)",
        "program": "x^2",
        "args": [_vector_f64(x_vec)],
        "expected": [_to_value(result)],
    })

    def dot_const(x):
        return jnp.dot(x, jnp.array([1.0, 2.0, 3.0]))

    x_vec = jnp.array([4.0, 5.0, 6.0])
    result = jax.grad(dot_const)(x_vec)
    cases.append({
        "case_id": "grad_dot_product",
        "composition": "grad",
        "program": "dot(x, [1,2,3])",
        "args": [_vector_f64(x_vec)],
        "expected": [_to_value(result)],
    })

    def sum_squares(x):
        return jnp.sum(x**2)

    x_vec = jnp.array([1.0, 2.0, 3.0])
    result = jax.grad(sum_squares)(x_vec)
    cases.append({
        "case_id": "grad_sum_squares",
        "composition": "grad",
        "program": "sum(x^2)",
        "args": [_vector_f64(x_vec)],
        "expected": [_to_value(result)],
    })

    x_val = 3.0
    val, grad_val = jax.value_and_grad(poly)(x_val)
    cases.append({
        "case_id": "value_and_grad_poly",
        "composition": "value_and_grad",
        "program": "x^2+3x",
        "args": [_scalar_f64(x_val)],
        "expected": [_to_value(val), _to_value(grad_val)],
    })

    def quadratic_vec(x):
        return jnp.array([x[0]**2, x[0]*x[1], x[1]**2])

    x_vec = jnp.array([2.0, 3.0])
    result = jax.jacobian(quadratic_vec)(x_vec)
    cases.append({
        "case_id": "jacobian_quadratic",
        "composition": "jacobian",
        "program": "[x0^2, x0*x1, x1^2]",
        "args": [_vector_f64(x_vec)],
        "expected": [_to_value(result)],
    })

    def quadratic_sum(x):
        return x[0]**2 + x[0]*x[1] + x[1]**2

    x_vec = jnp.array([1.0, 1.0])
    result = jax.hessian(quadratic_sum)(x_vec)
    cases.append({
        "case_id": "hessian_quadratic",
        "composition": "hessian",
        "program": "x0^2+x0*x1+x1^2",
        "args": [_vector_f64(x_vec)],
        "expected": [_to_value(result)],
    })

    return cases


def main():
    parser = argparse.ArgumentParser(description="Capture composition oracle fixtures")
    parser.add_argument("--legacy-root", type=Path, default=None, help="Legacy JAX checkout")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    jax, jnp = _try_import_jax(args.legacy_root)

    cases = capture_composition_cases(jax, jnp)

    fixture = {
        "schema_version": "frankenjax.composition-oracle.v1",
        "generated_by": "capture_composition_oracle.py",
        "generated_at_unix_ms": int(time.time() * 1000),
        "oracle_root": str(args.legacy_root) if args.legacy_root else None,
        "jax_version": jax.__version__,
        "x64_enabled": True,
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
