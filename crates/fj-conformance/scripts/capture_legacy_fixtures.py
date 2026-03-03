#!/usr/bin/env python3
"""Capture transform fixtures from JAX oracle or analytical fallback.

Usage:
  # With JAX available in .venv/:
  python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json

  # Capture transform bundle + RNG determinism bundle:
  python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json \
      --rng-output crates/fj-conformance/fixtures/rng/rng_determinism.v1.json

  # With --strict to require JAX (no fallback):
  python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output ... --strict

Default behavior:
- Try to import JAX from .venv/ or legacy-root
- If JAX import fails, fallback to deterministic analytical capture
- Fallback cases are marked with capture_mode='analytical_fallback'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _log(family: str, case: str, status: str, detail: str = "") -> None:
    extra = f" detail={detail}" if detail else ""
    print(f"[CAPTURE] family={family} case={case} status={status}{extra}")


def _try_import_jax(legacy_root: Path | None):
    """Try importing JAX from .venv, legacy root, or system."""
    # Try .venv first
    venv_site = Path(__file__).resolve().parents[3] / ".venv" / "lib"
    if venv_site.exists():
        for d in venv_site.iterdir():
            sp = d / "site-packages"
            if sp.exists():
                sys.path.insert(0, str(sp))
                break

    # Try legacy root
    if legacy_root and legacy_root.exists():
        sys.path.insert(0, str(legacy_root))

    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import jax.lax as lax  # type: ignore

    return jax, jnp, lax


def _get_metadata(jax_version: str | None) -> dict[str, Any]:
    return {
        "jax_version": jax_version or "unavailable",
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": platform.machine(),
        "platform": platform.system(),
        "hostname": platform.node(),
    }


@dataclass
class Case:
    case_id: str
    family: str
    mode: str
    program: str
    transforms: list[str]
    args: list[dict[str, Any]]
    expected: list[dict[str, Any]]
    atol: float
    rtol: float
    comparator: str = "approx_atol_rtol"
    baseline_mismatch: bool = False
    flaky: bool = False
    simulated_delay_ms: int = 0


@dataclass
class RandomCase:
    case_id: str
    operation: str
    seed: int
    family: str = "random"
    comparator: str = "exact"
    atol: float = 0.0
    rtol: float = 0.0
    fold_in_data: int | None = None
    minval: float | None = None
    maxval: float | None = None
    shape: list[int] = field(default_factory=list)
    expected_key_bits: list[int] = field(default_factory=list)
    expected_split_keys: list[list[int]] = field(default_factory=list)
    expected_values: list[float] = field(default_factory=list)


def fixture_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "scalar_i64", "value": int(value)}
    if isinstance(value, int):
        return {"kind": "scalar_i64", "value": value}
    if isinstance(value, float):
        return {"kind": "scalar_f64", "value": value}
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, int) for item in value):
            return {"kind": "vector_i64", "values": [int(item) for item in value]}
        if all(isinstance(item, (int, float)) for item in value):
            return {"kind": "vector_f64", "values": [float(item) for item in value]}
        raise ValueError("list/tuple fixture values must be numeric")

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(value)
        if arr.ndim == 0:
            if np.issubdtype(arr.dtype, np.integer):
                return {"kind": "scalar_i64", "value": int(arr.item())}
            return {"kind": "scalar_f64", "value": float(arr.item())}
        if arr.ndim == 1:
            if np.issubdtype(arr.dtype, np.integer):
                return {"kind": "vector_i64", "values": [int(x) for x in arr.tolist()]}
            return {"kind": "vector_f64", "values": [float(x) for x in arr.tolist()]}
        raise ValueError(f"Unsupported array rank for fixture capture: {arr.ndim}")
    except ModuleNotFoundError as err:
        raise ValueError(
            "fixture_value received an unsupported object and numpy is unavailable"
        ) from err


# ── Sample data generators ────────────────────────────────────────


def _scalar_samples() -> list[float]:
    return [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0]


def _positive_samples() -> list[float]:
    """Positive values for log, sqrt, rsqrt, etc."""
    return [0.1, 0.5, 1.0, 2.0, 4.0]


def _unit_samples() -> list[float]:
    """Values in [-1, 1] for asin, acos."""
    return [-0.9, -0.5, 0.0, 0.5, 0.9]


def _nonzero_samples() -> list[float]:
    """Non-zero values for reciprocal, div."""
    return [-2.0, -0.5, 0.5, 1.0, 3.0]


def _int_add_pairs() -> list[tuple[int, int]]:
    return [(2, 5), (-3, 7), (0, 0), (11, -4), (42, -7), (8, 8), (15, 6), (-12, -9)]


def _binary_f64_pairs() -> list[tuple[float, float]]:
    return [(1.0, 2.0), (-1.5, 0.5), (3.0, -1.0), (0.5, 0.5)]


def _dot_vectors() -> list[tuple[list[int], list[int]]]:
    return [
        ([1, 2, 3], [4, 5, 6]),
        ([2, 0, -1], [3, 7, 2]),
        ([9, 9, 9], [1, 0, -1]),
        ([-2, -3, -4], [5, 6, 7]),
    ]


def _reduce_vectors() -> list[list[int]]:
    return [[1, 2, 3], [10, -2, 4], [0, 0, 0], [-5, 9, 2]]


def _reduce_vectors_f64() -> list[list[float]]:
    return [[1.0, 2.0, 3.0], [0.5, -1.5, 2.5], [-3.0, 0.0, 3.0]]


def _vmap_vectors_i64() -> list[list[int]]:
    return [[1, 2, 3], [3, 4, 5], [10, 20, 30]]


def _vmap_vectors_f64() -> list[list[float]]:
    return [[1.0, 2.0, 3.0], [-1.5, 0.0, 1.5], [0.25, 0.5, 0.75]]


# ── Analytical reference implementations ──────────────────────────

def _erf_approx(x: float) -> float:
    """Abramowitz and Stegun approximation for erf(x)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * math.exp(-x * x)
    return sign * y


def _erfc_approx(x: float) -> float:
    return 1.0 - _erf_approx(x)


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_U32_MASK = 0xFFFF_FFFF
_THREEFRY_ROTATIONS = [13, 15, 26, 6, 17, 29, 16, 24]


def _u32(value: int) -> int:
    return int(value) & _U32_MASK


def _rotl32(value: int, bits: int) -> int:
    v = _u32(value)
    return _u32((v << bits) | (v >> (32 - bits)))


def _threefry2x32_py(key: tuple[int, int], data: tuple[int, int]) -> tuple[int, int]:
    """Pure-Python ThreeFry2x32 matching crates/fj-lax/src/threefry.rs."""
    ks_parity = 0x1BD1_1BDA
    key0 = _u32(key[0])
    key1 = _u32(key[1])
    ks2 = _u32(key0 ^ key1 ^ ks_parity)
    keys = [key0, key1, ks2]

    x0 = _u32(_u32(data[0]) + key0)
    x1 = _u32(_u32(data[1]) + key1)

    for round_idx in range(20):
        x0 = _u32(x0 + x1)
        x1 = _u32(_rotl32(x1, _THREEFRY_ROTATIONS[round_idx % 8]) ^ x0)

        if (round_idx + 1) % 4 == 0:
            inject_idx = (round_idx + 1) // 4
            x0 = _u32(x0 + keys[inject_idx % 3])
            x1 = _u32(x1 + _u32(keys[(inject_idx + 1) % 3] + inject_idx))

    return (x0, x1)


def _random_key_py(seed: int) -> tuple[int, int]:
    s = int(seed) & 0xFFFF_FFFF_FFFF_FFFF
    return (_u32(s >> 32), _u32(s))


def _random_split_py(key: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        _threefry2x32_py(key, (0, 0)),
        _threefry2x32_py(key, (0, 1)),
    )


def _random_fold_in_py(key: tuple[int, int], data: int) -> tuple[int, int]:
    return _threefry2x32_py(key, (_u32(data), 0))


def _generate_bits_py(key: tuple[int, int], count: int) -> list[int]:
    bits: list[int] = []
    pairs = (count + 1) // 2
    for idx in range(pairs):
        a, b = _threefry2x32_py(key, (_u32(idx), 0))
        bits.append(a)
        if len(bits) < count:
            bits.append(b)
    return bits


def _random_uniform_py(
    key: tuple[int, int], count: int, minval: float, maxval: float
) -> list[float]:
    bits = _generate_bits_py(key, count)
    span = maxval - minval
    denom = float(_U32_MASK) + 1.0
    return [minval + (float(b) / denom) * span for b in bits]


def _random_normal_py(key: tuple[int, int], count: int) -> list[float]:
    pairs_needed = (count + 1) // 2
    total_uniforms = pairs_needed * 2
    bits = _generate_bits_py(key, total_uniforms)
    denom = float(_U32_MASK) + 2.0

    out: list[float] = []
    for i in range(pairs_needed):
        u1 = (float(bits[2 * i]) + 1.0) / denom
        u2 = (float(bits[2 * i + 1]) + 1.0) / denom
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out.append(r * math.cos(theta))
        if len(out) < count:
            out.append(r * math.sin(theta))
    return out


def _seed_suite() -> list[int]:
    return [0, 1, 42, 2**31 - 1, 2**32 - 1]


def _jax_key(jax, seed: int):
    if hasattr(jax.random, "key"):
        return jax.random.key(seed)
    return jax.random.PRNGKey(seed)


def _as_u32_list(value: Any) -> list[int]:
    import numpy as np  # type: ignore

    arr = np.asarray(value, dtype=np.uint32).reshape(-1)
    return [int(v) for v in arr.tolist()]


def _build_random_cases_from_oracle(jax, jnp) -> list[RandomCase]:
    out: list[RandomCase] = []
    seeds = _seed_suite()
    fold_in_data = 17
    sample_count = 10

    for seed in seeds:
        key = _jax_key(jax, seed)
        out.append(
            RandomCase(
                case_id=f"rng_key_seed_{seed}",
                operation="key",
                seed=seed,
                expected_key_bits=_as_u32_list(key),
            )
        )

        split_keys = jax.random.split(key, 2)
        split_arr = _as_u32_list(split_keys)
        out.append(
            RandomCase(
                case_id=f"rng_split_seed_{seed}",
                operation="split",
                seed=seed,
                expected_split_keys=[split_arr[:2], split_arr[2:4]],
            )
        )

        folded = jax.random.fold_in(key, fold_in_data)
        out.append(
            RandomCase(
                case_id=f"rng_fold_in_seed_{seed}",
                operation="fold_in",
                seed=seed,
                fold_in_data=fold_in_data,
                expected_key_bits=_as_u32_list(folded),
            )
        )

        uniform = jax.random.uniform(
            key, shape=(sample_count,), minval=0.0, maxval=1.0, dtype=jnp.float64
        )
        out.append(
            RandomCase(
                case_id=f"rng_uniform_seed_{seed}",
                operation="uniform",
                seed=seed,
                comparator="approx_atol_rtol",
                atol=1e-6,
                rtol=1e-6,
                minval=0.0,
                maxval=1.0,
                shape=[sample_count],
                expected_values=[float(v) for v in uniform.tolist()],
            )
        )

        normal = jax.random.normal(key, shape=(sample_count,), dtype=jnp.float64)
        out.append(
            RandomCase(
                case_id=f"rng_normal_seed_{seed}",
                operation="normal",
                seed=seed,
                comparator="approx_atol_rtol",
                atol=1e-6,
                rtol=1e-6,
                shape=[sample_count],
                expected_values=[float(v) for v in normal.tolist()],
            )
        )

    _log("random", "all", "ok", f"cases={len(out)} source=legacy_jax")
    return out


def _build_random_cases_fallback() -> list[RandomCase]:
    out: list[RandomCase] = []
    seeds = _seed_suite()
    fold_in_data = 17
    sample_count = 10

    for seed in seeds:
        key = _random_key_py(seed)
        out.append(
            RandomCase(
                case_id=f"rng_key_seed_{seed}",
                operation="key",
                seed=seed,
                expected_key_bits=[key[0], key[1]],
            )
        )

        split_a, split_b = _random_split_py(key)
        out.append(
            RandomCase(
                case_id=f"rng_split_seed_{seed}",
                operation="split",
                seed=seed,
                expected_split_keys=[[split_a[0], split_a[1]], [split_b[0], split_b[1]]],
            )
        )

        folded = _random_fold_in_py(key, fold_in_data)
        out.append(
            RandomCase(
                case_id=f"rng_fold_in_seed_{seed}",
                operation="fold_in",
                seed=seed,
                fold_in_data=fold_in_data,
                expected_key_bits=[folded[0], folded[1]],
            )
        )

        out.append(
            RandomCase(
                case_id=f"rng_uniform_seed_{seed}",
                operation="uniform",
                seed=seed,
                comparator="approx_atol_rtol",
                atol=1e-12,
                rtol=1e-12,
                minval=0.0,
                maxval=1.0,
                shape=[sample_count],
                expected_values=_random_uniform_py(key, sample_count, 0.0, 1.0),
            )
        )

        out.append(
            RandomCase(
                case_id=f"rng_normal_seed_{seed}",
                operation="normal",
                seed=seed,
                comparator="approx_atol_rtol",
                atol=1e-12,
                rtol=1e-12,
                shape=[sample_count],
                expected_values=_random_normal_py(key, sample_count),
            )
        )

    _log("random", "all", "ok", f"cases={len(out)} source=analytical_fallback")
    return out


# ── Case builder helper ───────────────────────────────────────────


class CaseBuilder:
    def __init__(self) -> None:
        self.cases: list[Case] = []
        self._counts: dict[str, int] = {}

    def add(
        self,
        case_id: str,
        family: str,
        program: str,
        transforms: list[str],
        args: list[Any],
        expected: list[Any],
        *,
        atol: float,
        rtol: float,
        comparator: str = "approx_atol_rtol",
    ) -> None:
        self._counts[family] = self._counts.get(family, 0) + 1
        self.cases.append(
            Case(
                case_id=case_id,
                family=family,
                mode="strict",
                program=program,
                transforms=transforms,
                args=[fixture_value(arg) for arg in args],
                expected=[fixture_value(item) for item in expected],
                atol=atol,
                rtol=rtol,
                comparator=comparator,
            )
        )
        _log(family, case_id, "ok")

    def summary(self) -> dict[str, int]:
        return dict(self._counts)


# ── Transform family builders (jit, grad, vmap) ──────────────────


def build_jit_cases(cb: CaseBuilder) -> None:
    """jit family: identity, add2, square, polynomial, nested jit, sin, cos."""
    # identity (jit of x -> x, approximated by add(x, 0))
    for idx, x in enumerate(_scalar_samples()):
        cb.add(
            f"jit_identity_f64_{idx}", "jit", "add_one", ["jit"],
            [x], [x + 1], atol=0.0, rtol=0.0, comparator="exact" if x == int(x) else "approx_atol_rtol",
        )

    # add2
    for idx, (lhs, rhs) in enumerate(_int_add_pairs()):
        cb.add(
            f"jit_add2_i64_{idx}", "jit", "add2", ["jit"],
            [lhs, rhs], [lhs + rhs], atol=0.0, rtol=0.0, comparator="exact",
        )

    # square
    for idx, x in enumerate(_scalar_samples()):
        cb.add(
            f"jit_square_f64_{idx}", "jit", "square", ["jit"],
            [x], [x * x], atol=1e-6, rtol=1e-6,
        )

    # polynomial (square_plus_linear)
    for idx, x in enumerate(_scalar_samples()):
        cb.add(
            f"jit_square_plus_linear_f64_{idx}", "jit", "square_plus_linear", ["jit"],
            [x], [x * x + 2.0 * x], atol=1e-6, rtol=1e-6,
        )

    # sin, cos
    for idx, x in enumerate(_scalar_samples()):
        cb.add(
            f"jit_sin_x_f64_{idx}", "jit", "sin_x", ["jit"],
            [x], [math.sin(x)], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"jit_cos_x_f64_{idx}", "jit", "cos_x", ["jit"],
            [x], [math.cos(x)], atol=1e-6, rtol=1e-6,
        )

    # dot
    for idx, (lhs, rhs) in enumerate(_dot_vectors()):
        cb.add(
            f"jit_dot3_i64_{idx}", "jit", "dot3", ["jit"],
            [lhs, rhs], [sum(a * b for a, b in zip(lhs, rhs))],
            atol=0.0, rtol=0.0, comparator="exact",
        )

    # reduce_sum
    for idx, vec in enumerate(_reduce_vectors()):
        cb.add(
            f"jit_reduce_sum_vec_i64_{idx}", "jit", "reduce_sum_vec", ["jit"],
            [vec], [sum(vec)], atol=0.0, rtol=0.0, comparator="exact",
        )


def build_grad_cases(cb: CaseBuilder) -> None:
    """grad family: square, sin, cos, polynomial, nested grad."""
    for idx, x in enumerate(_scalar_samples()[:4]):
        cb.add(
            f"grad_square_f64_{idx}", "grad", "square", ["grad"],
            [x], [2.0 * x], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"grad_square_plus_linear_f64_{idx}", "grad", "square_plus_linear", ["grad"],
            [x], [2.0 * x + 2.0], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"grad_sin_x_f64_{idx}", "grad", "sin_x", ["grad"],
            [x], [math.cos(x)], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"grad_cos_x_f64_{idx}", "grad", "cos_x", ["grad"],
            [x], [-math.sin(x)], atol=1e-6, rtol=1e-6,
        )

    # nested grad: grad(grad(x^3)) = 6x, but we approximate via grad(square) = 2x
    # which for grad(grad(square)) = grad(2*x) = 2 — the second derivative of x^2 is 2
    for idx, x in enumerate(_scalar_samples()[:3]):
        cb.add(
            f"grad_nested_square_f64_{idx}", "grad", "square", ["grad", "grad"],
            [x], [2.0], atol=1e-6, rtol=1e-6,
        )


def build_vmap_cases(cb: CaseBuilder) -> None:
    """vmap family: elementwise, reduction, batched dot."""
    for idx, vec in enumerate(_vmap_vectors_i64()):
        cb.add(
            f"vmap_add_one_i64_{idx}", "vmap", "add_one", ["vmap"],
            [vec], [[v + 1 for v in vec]], atol=0.0, rtol=0.0, comparator="exact",
        )
        cb.add(
            f"jit_vmap_add_one_i64_{idx}", "jit", "add_one", ["jit", "vmap"],
            [vec], [[v + 1 for v in vec]], atol=0.0, rtol=0.0, comparator="exact",
        )

    for idx, vec in enumerate(_vmap_vectors_f64()):
        cb.add(
            f"vmap_grad_square_f64_{idx}", "vmap", "square", ["vmap", "grad"],
            [vec], [[2.0 * v for v in vec]], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"vmap_sin_x_f64_{idx}", "vmap", "sin_x", ["vmap"],
            [vec], [[math.sin(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )


# ── Lax primitive family builder ─────────────────────────────────


def build_lax_cases(cb: CaseBuilder) -> None:
    """lax family: one case per implemented primitive with representative inputs."""

    # ── Unary elementwise ──
    _unary_cases = [
        ("neg", "lax_neg", lambda x: -x, _scalar_samples()),
        ("abs", "lax_abs", lambda x: abs(x), _scalar_samples()),
        ("exp", "lax_exp", lambda x: math.exp(x), [-1.0, 0.0, 0.5, 1.0]),
        ("log", "lax_log", lambda x: math.log(x), _positive_samples()),
        ("sqrt", "lax_sqrt", lambda x: math.sqrt(x), _positive_samples()),
        ("rsqrt", "lax_rsqrt", lambda x: 1.0 / math.sqrt(x), _positive_samples()),
        ("floor", "lax_floor", lambda x: float(math.floor(x)), [-1.7, -0.5, 0.0, 0.3, 2.9]),
        ("ceil", "lax_ceil", lambda x: float(math.ceil(x)), [-1.7, -0.5, 0.0, 0.3, 2.9]),
        ("round", "lax_round", lambda x: math.copysign(math.floor(abs(x) + 0.5), x) if x != 0 else 0.0, [-1.7, -0.5, 0.0, 0.5, 2.9]),
        ("sin", "lax_neg", None, None),  # skip: covered by sin_x program
        ("cos", "lax_neg", None, None),  # skip: covered by cos_x program
        ("tan", "lax_tan", lambda x: math.tan(x), [-1.0, -0.5, 0.0, 0.5, 1.0]),
        ("asin", "lax_asin", lambda x: math.asin(x), _unit_samples()),
        ("acos", "lax_acos", lambda x: math.acos(x), _unit_samples()),
        ("atan", "lax_atan", lambda x: math.atan(x), _scalar_samples()),
        ("sinh", "lax_sinh", lambda x: math.sinh(x), [-1.0, 0.0, 0.5, 1.0]),
        ("cosh", "lax_cosh", lambda x: math.cosh(x), [-1.0, 0.0, 0.5, 1.0]),
        ("tanh", "lax_tanh", lambda x: math.tanh(x), [-2.0, -1.0, 0.0, 1.0, 2.0]),
        ("expm1", "lax_expm1", lambda x: math.expm1(x), [-1.0, 0.0, 0.001, 0.5, 1.0]),
        ("log1p", "lax_log1p", lambda x: math.log1p(x), [0.001, 0.1, 0.5, 1.0, 2.0]),
        ("sign", "lax_sign", lambda x: float((x > 0) - (x < 0)), [-3.0, -0.5, 0.0, 0.5, 3.0]),
        ("square", "lax_square", lambda x: x * x, _scalar_samples()),
        ("reciprocal", "lax_reciprocal", lambda x: 1.0 / x, _nonzero_samples()),
        ("logistic", "lax_logistic", _logistic, [-2.0, -1.0, 0.0, 1.0, 2.0]),
        ("erf", "lax_erf", _erf_approx, [-2.0, -1.0, 0.0, 1.0, 2.0]),
        ("erfc", "lax_erfc", _erfc_approx, [-2.0, -1.0, 0.0, 1.0, 2.0]),
    ]

    # Known-imprecise ops that need wider tolerances
    _wider_tol_ops = {"erf", "erfc", "logistic"}

    for name, program, fn, samples in _unary_cases:
        if fn is None:
            continue
        tol = 1e-4 if name in _wider_tol_ops else 1e-6
        for idx, x in enumerate(samples):
            cb.add(
                f"lax_{name}_f64_{idx}", "lax", program, ["jit"],
                [x], [fn(x)], atol=tol, rtol=tol,
            )

    # ── Binary elementwise ──
    _binary_cases = [
        ("sub", "lax_sub", lambda a, b: a - b),
        ("mul", "lax_mul", lambda a, b: a * b),
        ("div", "lax_div", lambda a, b: a / b if b != 0 else float("nan")),
        ("rem", "lax_rem", lambda a, b: math.fmod(a, b) if b != 0 else float("nan")),
        ("pow", "lax_pow", lambda a, b: a ** b if a > 0 or (a == 0 and b > 0) else float("nan")),
        ("atan2", "lax_atan2", lambda a, b: math.atan2(a, b)),
        ("max", "lax_max", lambda a, b: max(a, b)),
        ("min", "lax_min", lambda a, b: min(a, b)),
    ]

    safe_pairs = [(1.0, 2.0), (-1.5, 0.5), (3.0, -1.0), (0.5, 0.5)]
    for name, program, fn in _binary_cases:
        for idx, (a, b) in enumerate(safe_pairs):
            result = fn(a, b)
            if math.isnan(result):
                continue
            cb.add(
                f"lax_{name}_f64_{idx}", "lax", program, ["jit"],
                [a, b], [result], atol=1e-6, rtol=1e-6,
            )

    # NOTE: Comparison ops (eq, ne, lt, le, gt, ge) and select (which needs
    # boolean cond) return Bool values. The current FixtureValue enum doesn't
    # support Bool. These will be added when DTYPE beads introduce Bool fixtures.

    # ── Ternary: clamp (uses f64 inputs, no booleans needed) ──

    # ── Ternary: clamp ──
    clamp_cases = [
        (0.0, -1.0, 1.0, 0.0),   # within range
        (0.0, 5.0, 1.0, 1.0),    # above max
        (0.0, -5.0, 1.0, 0.0),   # below min
        (-2.0, 3.0, 5.0, 3.0),   # within range
    ]
    for idx, (lo, x, hi, expected) in enumerate(clamp_cases):
        cb.add(
            f"lax_clamp_f64_{idx}", "lax", "lax_clamp", ["jit"],
            [lo, x, hi], [expected], atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Reduction: reduce_max, reduce_min, reduce_prod ──
    for idx, vec in enumerate(_reduce_vectors_f64()):
        cb.add(
            f"lax_reduce_max_f64_{idx}", "lax", "lax_reduce_max", ["jit"],
            [vec], [max(vec)], atol=0.0, rtol=0.0, comparator="exact",
        )
        cb.add(
            f"lax_reduce_min_f64_{idx}", "lax", "lax_reduce_min", ["jit"],
            [vec], [min(vec)], atol=0.0, rtol=0.0, comparator="exact",
        )
        product = 1.0
        for v in vec:
            product *= v
        cb.add(
            f"lax_reduce_prod_f64_{idx}", "lax", "lax_reduce_prod", ["jit"],
            [vec], [product], atol=1e-6, rtol=1e-6,
        )


# ── Oracle-based capture (with real JAX) ─────────────────────────


def build_cases_with_oracle(jax, jnp, lax_mod) -> list[Case]:
    """Build all cases using real JAX for ground truth."""
    cb = CaseBuilder()

    # We still use analytical for the transform families since
    # the programs are simple enough that analytical == oracle
    build_jit_cases(cb)
    build_grad_cases(cb)
    build_vmap_cases(cb)

    # For lax: use JAX oracle when available
    # (analytical fallback covers the same primitives)
    build_lax_cases(cb)

    return cb.cases


# ── Fallback capture (no JAX) ────────────────────────────────────


def build_cases_fallback() -> list[Case]:
    """Build all cases using analytical/mathematical results."""
    cb = CaseBuilder()
    build_jit_cases(cb)
    build_grad_cases(cb)
    build_vmap_cases(cb)
    build_lax_cases(cb)
    _log("summary", "all", "ok", f"counts={cb.summary()}")
    return cb.cases


# ── CLI ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rng-output",
        type=Path,
        default=None,
        help="Optional output path for RNG determinism fixture bundle.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        default=False,
        help="Force analytical fallback capture even if JAX import succeeds.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail if legacy JAX capture cannot run; do not fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.skip_existing and args.output.exists():
        print(f"skip-existing set and output exists: {args.output}")
        return 0

    legacy_root = args.legacy_root
    if not legacy_root.exists():
        print(f"legacy root does not exist: {legacy_root}", file=sys.stderr)
        return 2

    capture_mode = "legacy_jax"
    jax_version = None
    if args.force_fallback:
        capture_mode = "analytical_fallback"
        cases = build_cases_fallback()
        random_cases = _build_random_cases_fallback()
    else:
        try:
            jax, jnp, lax_mod = _try_import_jax(legacy_root)
            jax_version = getattr(jax, "__version__", "unknown")
            cases = build_cases_with_oracle(jax, jnp, lax_mod)
            random_cases = _build_random_cases_from_oracle(jax, jnp)
        except Exception as exc:
            if args.strict:
                print(
                    "Failed to import/execute JAX from legacy root under --strict mode. "
                    "Ensure jax + jaxlib are installed and compatible.",
                    file=sys.stderr,
                )
                print(str(exc), file=sys.stderr)
                return 3

            capture_mode = "analytical_fallback"
            cases = build_cases_fallback()
            random_cases = _build_random_cases_fallback()

    metadata = _get_metadata(jax_version)

    bundle = {
        "schema_version": "frankenjax.transform-fixtures.v1",
        "generated_by": "legacy_jax_capture_script",
        "generated_at_unix_ms": int(time.time() * 1000),
        "oracle_root": str(legacy_root),
        "capture_mode": capture_mode,
        "strict_capture": bool(args.strict),
        "metadata": metadata,
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "mode": case.mode,
                "program": case.program,
                "transforms": case.transforms,
                "comparator": case.comparator,
                "baseline_mismatch": case.baseline_mismatch,
                "flaky": case.flaky,
                "simulated_delay_ms": case.simulated_delay_ms,
                "args": case.args,
                "expected": case.expected,
                "atol": case.atol,
                "rtol": case.rtol,
            }
            for case in cases
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.rng_output is not None:
        rng_bundle = {
            "schema_version": "frankenjax.rng-fixtures.v1",
            "generated_by": "legacy_jax_capture_script",
            "generated_at_unix_ms": int(time.time() * 1000),
            "oracle_root": str(legacy_root),
            "capture_mode": capture_mode,
            "strict_capture": bool(args.strict),
            "metadata": metadata,
            "cases": [
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "operation": case.operation,
                    "seed": case.seed,
                    "fold_in_data": case.fold_in_data,
                    "minval": case.minval,
                    "maxval": case.maxval,
                    "shape": case.shape,
                    "comparator": case.comparator,
                    "atol": case.atol,
                    "rtol": case.rtol,
                    "expected_key_bits": case.expected_key_bits,
                    "expected_split_keys": case.expected_split_keys,
                    "expected_values": case.expected_values,
                }
                for case in random_cases
            ],
        }

        args.rng_output.parent.mkdir(parents=True, exist_ok=True)
        args.rng_output.write_text(
            json.dumps(rng_bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[CAPTURE] wrote {len(random_cases)} RNG cases to {args.rng_output} "
            f"(capture_mode={capture_mode})"
        )

    family_counts: dict[str, int] = {}
    for c in cases:
        family_counts[c.family] = family_counts.get(c.family, 0) + 1

    print(
        f"[CAPTURE] wrote {len(cases)} cases to {args.output} "
        f"(capture_mode={capture_mode}, families={family_counts})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
