#!/usr/bin/env python3
"""Capture dtype promotion oracle fixtures from JAX.

Usage:
  python3 crates/fj-conformance/scripts/capture_dtype_promotion_oracle.py \
      --legacy-root legacy_jax_code/jax \
      --output crates/fj-conformance/fixtures/dtype_promotion_oracle.v1.json

This script records reference values for JAX dtype promotion rules across add/mul
operations for all standard dtype combinations.
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


DTYPE_MAP = {
    "bool": "bool_",
    "i32": "int32",
    "i64": "int64",
    "u32": "uint32",
    "u64": "uint64",
    "f32": "float32",
    "f64": "float64",
    "bf16": "bfloat16",
    "f16": "float16",
}

DTYPE_NAMES = ["bool", "i32", "i64", "u32", "u64", "f32", "f64", "bf16", "f16"]


def _get_test_values(jnp, dtype_name: str):
    jax_dtype = getattr(jnp, DTYPE_MAP[dtype_name])
    if dtype_name == "bool":
        return jnp.array(True, dtype=jax_dtype)
    if dtype_name.startswith("i") or dtype_name.startswith("u"):
        return jnp.array(7, dtype=jax_dtype)
    return jnp.array(2.5, dtype=jax_dtype)


def _dtype_to_result_name(dtype) -> str:
    name = str(dtype)
    mapping = {
        "bool": "bool",
        "int32": "int32",
        "int64": "int64",
        "uint32": "uint32",
        "uint64": "uint64",
        "float32": "float32",
        "float64": "float64",
        "bfloat16": "bfloat16",
        "float16": "float16",
    }
    return mapping.get(name, name)


def _to_python_value(x) -> bool | float:
    if x.dtype == "bool":
        return bool(x)
    return float(x)


def capture_dtype_promotion_cases(jnp) -> list[dict[str, Any]]:
    cases = []

    for op_name in ["add", "mul"]:
        op = getattr(jnp, op_name)
        for lhs_name in DTYPE_NAMES:
            for rhs_name in DTYPE_NAMES:
                lhs = _get_test_values(jnp, lhs_name)
                rhs = _get_test_values(jnp, rhs_name)
                try:
                    result = op(lhs, rhs)
                    result_dtype = _dtype_to_result_name(result.dtype)
                    result_value = _to_python_value(result)
                except Exception:
                    continue

                cases.append({
                    "case_id": f"{op_name}_{lhs_name}_{rhs_name}",
                    "operation": op_name,
                    "lhs_dtype": lhs_name,
                    "rhs_dtype": rhs_name,
                    "result_dtype": result_dtype,
                    "result_value": result_value,
                })

    return cases


def main():
    parser = argparse.ArgumentParser(description="Capture dtype promotion oracle fixtures")
    parser.add_argument("--legacy-root", type=Path, default=None, help="Legacy JAX checkout")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    jax, jnp = _try_import_jax(args.legacy_root)

    cases = capture_dtype_promotion_cases(jnp)

    fixture = {
        "schema_version": "frankenjax.dtype-promotion-oracle.v1",
        "generated_by": "capture_dtype_promotion_oracle.py",
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
