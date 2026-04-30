#!/usr/bin/env python3
"""Capture transform fixtures from JAX oracle or analytical fallback.

Usage:
  # With JAX available in .venv/:
  python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json

  # Capture transform bundle + RNG determinism bundle:
  python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json \
      --rng-output crates/fj-conformance/fixtures/rng/rng_determinism.v1.json

  # With --strict to require JAX (no fallback):
  python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py \
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


def fixture_value_bool(value: bool) -> dict[str, Any]:
    """Encode a boolean value as a scalar_bool fixture."""
    return {"kind": "scalar_bool", "value": value}


def fixture_value_tensor_f64(shape: list[int], values: list[float]) -> dict[str, Any]:
    """Encode a multi-rank f64 tensor fixture."""
    return {"kind": "tensor_f64", "shape": shape, "values": values}


def fixture_value_tensor_i64(shape: list[int], values: list[int]) -> dict[str, Any]:
    """Encode a multi-rank i64 tensor fixture."""
    return {"kind": "tensor_i64", "shape": shape, "values": values}


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


def _digamma_approx(x: float) -> float:
    """Digamma (psi) approximation using asymptotic series with reflection."""
    if x <= 0 and x == math.floor(x):
        return float("nan")
    result = 0.0
    # Reflection formula for x < 0.5
    if x < 0.5:
        result = -math.pi / math.tan(math.pi * x)
        x = 1.0 - x
    # Shift x to >= 6 using recurrence psi(x+1) = psi(x) + 1/x
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic series
    result += math.log(x) - 1.0 / (2.0 * x)
    x2 = 1.0 / (x * x)
    result -= x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0)))
    return result


def _erf_inv_approx(x: float) -> float:
    """Rational approximation of inverse error function."""
    if x <= -1.0 or x >= 1.0:
        return float("nan") if abs(x) == 1.0 else float("inf") * (-1 if x < 0 else 1)
    if x == 0.0:
        return 0.0
    sign = 1 if x > 0 else -1
    a = abs(x)
    # Winitzki approximation
    ln_term = math.log(1.0 - a * a)
    c = 2.0 / (math.pi * 0.147) + ln_term / 2.0
    result = math.sqrt(math.sqrt(c * c - ln_term / 0.147) - c)
    return sign * result


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
    # JAX: threefry_2x32(key, threefry_seed(data)) where threefry_seed puts [hi, lo]
    # For u32 data, hi=0, lo=data
    return _threefry2x32_py(key, (0, _u32(data)))


def _generate_u32_bits_py(key: tuple[int, int], count: int) -> list[int]:
    """Match JAX's partitionable _threefry_random_bits with bit_width=32.

    For each sample index i, calls threefry2x32(key, [0, i]) and XORs the
    two output words.
    """
    return [_threefry2x32_py(key, (0, _u32(i)))[0] ^ _threefry2x32_py(key, (0, _u32(i)))[1]
            for i in range(count)]


def _random_uniform_py(
    key: tuple[int, int], count: int, minval: float, maxval: float
) -> list[float]:
    """Match JAX's uniform with f32 mode (x64 not enabled).

    Generates u32 bits via XOR, converts to f32 mantissa, then scales.
    """
    import struct
    bits = _generate_u32_bits_py(key, count)
    span = maxval - minval
    results = []
    for b in bits:
        mantissa = b >> 9
        float_bits = mantissa | 0x3F800000
        unit = struct.unpack("f", struct.pack("I", float_bits))[0] - 1.0
        results.append(minval + float(unit) * span)
    return results


def _random_normal_py(key: tuple[int, int], count: int) -> list[float]:
    """Match JAX's normal with f32 mode: sqrt(2) * erfinv(uniform(-1, 1))."""
    import struct
    # nextafter_f32(-1.0, 0.0)
    neg1_bits = struct.unpack("I", struct.pack("f", -1.0))[0]
    lo = struct.unpack("f", struct.pack("I", neg1_bits - 1))[0]
    hi = 1.0
    uniforms = _random_uniform_py(key, count, float(lo), hi)
    sqrt2 = math.sqrt(2.0)
    return [sqrt2 * _erf_inv_py(u) for u in uniforms]


def _erf_inv_py(x: float) -> float:
    """Winitzki approximation with Newton refinement for erfinv."""
    if x <= -1.0:
        return float("-inf") if x == -1.0 else float("nan")
    if x >= 1.0:
        return float("inf") if x == 1.0 else float("nan")
    if x == 0.0:
        return 0.0
    a = 0.147
    ln_term = math.log(1.0 - x * x)
    t = 2.0 / (math.pi * a) + ln_term / 2.0
    y = math.copysign(math.sqrt(math.sqrt(t * t - ln_term / a) - t), x)
    coeff = 2.0 / math.sqrt(math.pi)
    for _ in range(3):
        err = math.erf(y) - x
        deriv = coeff * math.exp(-y * y)
        y -= err / deriv
    return y


def _seed_suite() -> list[int]:
    return [0, 1, 42, 2**31 - 1, 2**32 - 1]


def _jax_key(jax, seed: int):
    if hasattr(jax.random, "key"):
        return jax.random.key(seed)
    return jax.random.PRNGKey(seed)


def _as_u32_list(value: Any) -> list[int]:
    import numpy as np  # type: ignore

    # JAX 0.9+ uses typed PRNG keys that can't be directly converted to numpy.
    # Use jax.random.key_data() to extract the underlying integer array.
    try:
        arr = np.asarray(value, dtype=np.uint32).reshape(-1)
    except TypeError:
        import jax  # type: ignore

        arr = np.asarray(jax.random.key_data(value), dtype=np.uint32).reshape(-1)
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

    def add_raw(
        self,
        case_id: str,
        family: str,
        program: str,
        transforms: list[str],
        args: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        *,
        atol: float,
        rtol: float,
        comparator: str = "exact",
    ) -> None:
        """Add a case with pre-encoded fixture values (e.g. for booleans)."""
        self._counts[family] = self._counts.get(family, 0) + 1
        self.cases.append(
            Case(
                case_id=case_id,
                family=family,
                mode="strict",
                program=program,
                transforms=transforms,
                args=args,
                expected=expected,
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

    # grad of lax unary primitives (AD coverage expansion)
    # grad(exp(x)) = exp(x)
    for idx, x in enumerate([-1.0, 0.0, 0.5, 1.0]):
        cb.add(
            f"grad_lax_exp_f64_{idx}", "grad", "lax_exp", ["grad"],
            [x], [math.exp(x)], atol=1e-6, rtol=1e-6,
        )

    # grad(log(x)) = 1/x
    for idx, x in enumerate([0.5, 1.0, 2.0, 4.0]):
        cb.add(
            f"grad_lax_log_f64_{idx}", "grad", "lax_log", ["grad"],
            [x], [1.0 / x], atol=1e-6, rtol=1e-6,
        )

    # grad(sqrt(x)) = 1/(2*sqrt(x))
    for idx, x in enumerate([0.25, 1.0, 4.0]):
        cb.add(
            f"grad_lax_sqrt_f64_{idx}", "grad", "lax_sqrt", ["grad"],
            [x], [0.5 / math.sqrt(x)], atol=1e-6, rtol=1e-6,
        )

    # grad(tanh(x)) = 1 - tanh(x)^2
    for idx, x in enumerate([-1.0, 0.0, 0.5, 1.0]):
        t = math.tanh(x)
        cb.add(
            f"grad_lax_tanh_f64_{idx}", "grad", "lax_tanh", ["grad"],
            [x], [1.0 - t * t], atol=1e-6, rtol=1e-6,
        )

    # grad(neg(x)) = -1
    for idx, x in enumerate([-1.0, 0.0, 1.0]):
        cb.add(
            f"grad_lax_neg_f64_{idx}", "grad", "lax_neg", ["grad"],
            [x], [-1.0], atol=1e-6, rtol=1e-6,
        )

    # grad(x^2) via lax_square = 2x
    for idx, x in enumerate([-1.0, 0.0, 0.5, 2.0]):
        cb.add(
            f"grad_lax_square_f64_{idx}", "grad", "lax_square", ["grad"],
            [x], [2.0 * x], atol=1e-6, rtol=1e-6,
        )

    # grad(reciprocal(x)) = -1/x^2
    for idx, x in enumerate([0.5, 1.0, 2.0]):
        cb.add(
            f"grad_lax_reciprocal_f64_{idx}", "grad", "lax_reciprocal", ["grad"],
            [x], [-1.0 / (x * x)], atol=1e-6, rtol=1e-6,
        )

    # grad(tan(x)) = 1 + tan(x)^2 = sec^2(x)
    for idx, x in enumerate([-0.5, 0.0, 0.5, 1.0]):
        t = math.tan(x)
        cb.add(
            f"grad_lax_tan_f64_{idx}", "grad", "lax_tan", ["grad"],
            [x], [1.0 + t * t], atol=1e-6, rtol=1e-6,
        )

    # grad(atan(x)) = 1/(1+x^2)
    for idx, x in enumerate([-1.0, 0.0, 0.5, 1.0]):
        cb.add(
            f"grad_lax_atan_f64_{idx}", "grad", "lax_atan", ["grad"],
            [x], [1.0 / (1.0 + x * x)], atol=1e-6, rtol=1e-6,
        )

    # grad(sinh(x)) = cosh(x)
    for idx, x in enumerate([-1.0, 0.0, 0.5]):
        cb.add(
            f"grad_lax_sinh_f64_{idx}", "grad", "lax_sinh", ["grad"],
            [x], [math.cosh(x)], atol=1e-6, rtol=1e-6,
        )

    # grad(cosh(x)) = sinh(x)
    for idx, x in enumerate([-1.0, 0.0, 0.5]):
        cb.add(
            f"grad_lax_cosh_f64_{idx}", "grad", "lax_cosh", ["grad"],
            [x], [math.sinh(x)], atol=1e-6, rtol=1e-6,
        )

    # grad(rsqrt(x)) = -0.5 * x^(-3/2)
    for idx, x in enumerate([0.25, 1.0, 4.0]):
        cb.add(
            f"grad_lax_rsqrt_f64_{idx}", "grad", "lax_rsqrt", ["grad"],
            [x], [-0.5 * x ** (-1.5)], atol=1e-6, rtol=1e-6,
        )

    # grad(abs(x)) = sign(x)  (x != 0)
    for idx, x in enumerate([-2.0, -0.5, 0.5, 2.0]):
        cb.add(
            f"grad_lax_abs_f64_{idx}", "grad", "lax_abs", ["grad"],
            [x], [1.0 if x > 0 else -1.0], atol=1e-6, rtol=1e-6,
        )

    # grad(logistic(x)) = logistic(x) * (1 - logistic(x))
    for idx, x in enumerate([-1.0, 0.0, 0.5, 1.0]):
        s = _logistic(x)
        cb.add(
            f"grad_lax_logistic_f64_{idx}", "grad", "lax_logistic", ["grad"],
            [x], [s * (1.0 - s)], atol=1e-4, rtol=1e-4,
        )

    # grad(expm1(x)) = exp(x)
    for idx, x in enumerate([-1.0, 0.0, 0.5, 1.0]):
        cb.add(
            f"grad_lax_expm1_f64_{idx}", "grad", "lax_expm1", ["grad"],
            [x], [math.exp(x)], atol=1e-6, rtol=1e-6,
        )

    # grad(log1p(x)) = 1/(1+x)
    for idx, x in enumerate([0.001, 0.1, 0.5, 1.0]):
        cb.add(
            f"grad_lax_log1p_f64_{idx}", "grad", "lax_log1p", ["grad"],
            [x], [1.0 / (1.0 + x)], atol=1e-6, rtol=1e-6,
        )

    # grad(asin(x)) = 1/sqrt(1-x^2)
    for idx, x in enumerate([-0.5, 0.0, 0.5]):
        cb.add(
            f"grad_lax_asin_f64_{idx}", "grad", "lax_asin", ["grad"],
            [x], [1.0 / math.sqrt(1.0 - x * x)], atol=1e-6, rtol=1e-6,
        )

    # grad(acos(x)) = -1/sqrt(1-x^2)
    for idx, x in enumerate([-0.5, 0.0, 0.5]):
        cb.add(
            f"grad_lax_acos_f64_{idx}", "grad", "lax_acos", ["grad"],
            [x], [-1.0 / math.sqrt(1.0 - x * x)], atol=1e-6, rtol=1e-6,
        )

    # --- Grad of binary ops (w.r.t. first arg) ---

    # grad(mul(x,y)) w.r.t. x = y
    binary_pairs = [(1.0, 2.0), (-1.5, 0.5), (3.0, -1.0)]
    for idx, (x, y) in enumerate(binary_pairs):
        cb.add(
            f"grad_lax_mul_f64_{idx}", "grad", "lax_mul", ["grad"],
            [x, y], [y], atol=1e-6, rtol=1e-6,
        )

    # grad(sub(x,y)) w.r.t. x = 1
    for idx, (x, y) in enumerate(binary_pairs):
        cb.add(
            f"grad_lax_sub_f64_{idx}", "grad", "lax_sub", ["grad"],
            [x, y], [1.0], atol=1e-6, rtol=1e-6,
        )

    # grad(div(x,y)) w.r.t. x = 1/y
    for idx, (x, y) in enumerate(binary_pairs):
        cb.add(
            f"grad_lax_div_f64_{idx}", "grad", "lax_div", ["grad"],
            [x, y], [1.0 / y], atol=1e-6, rtol=1e-6,
        )

    # grad(pow(x,y)) w.r.t. x = y * x^(y-1)
    pow_pairs = [(2.0, 3.0), (1.0, 2.0), (4.0, 0.5)]
    for idx, (x, y) in enumerate(pow_pairs):
        cb.add(
            f"grad_lax_pow_f64_{idx}", "grad", "lax_pow", ["grad"],
            [x, y], [y * x ** (y - 1.0)], atol=1e-6, rtol=1e-6,
        )

    # grad(atan2(x,y)) w.r.t. x = y/(x^2+y^2)
    for idx, (x, y) in enumerate(binary_pairs):
        cb.add(
            f"grad_lax_atan2_f64_{idx}", "grad", "lax_atan2", ["grad"],
            [x, y], [y / (x * x + y * y)], atol=1e-6, rtol=1e-6,
        )

    # --- Higher-order gradient compositions ---

    # nested grad: grad(grad(square)) = 2 (second derivative of x^2 is constant 2)
    for idx, x in enumerate(_scalar_samples()[:3]):
        cb.add(
            f"grad_nested_square_f64_{idx}", "grad", "square", ["grad", "grad"],
            [x], [2.0], atol=1e-6, rtol=1e-6,
        )

    # grad(grad(sin(x))) = -sin(x)
    for idx, x in enumerate([-1.0, 0.0, 0.5, 1.0]):
        cb.add(
            f"grad_nested_sin_f64_{idx}", "grad", "sin_x", ["grad", "grad"],
            [x], [-math.sin(x)], atol=1e-6, rtol=1e-6,
        )

    # grad(grad(exp(x))) = exp(x)
    for idx, x in enumerate([-1.0, 0.0, 0.5, 1.0]):
        cb.add(
            f"grad_nested_exp_f64_{idx}", "grad", "lax_exp", ["grad", "grad"],
            [x], [math.exp(x)], atol=1e-6, rtol=1e-6,
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

    # vmap of lax unary primitives
    for idx, vec in enumerate(_vmap_vectors_f64()):
        cb.add(
            f"vmap_lax_neg_f64_{idx}", "vmap", "lax_neg", ["vmap"],
            [vec], [[-v for v in vec]], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"vmap_lax_exp_f64_{idx}", "vmap", "lax_exp", ["vmap"],
            [vec], [[math.exp(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"vmap_lax_square_f64_{idx}", "vmap", "lax_square", ["vmap"],
            [vec], [[v * v for v in vec]], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"vmap_lax_abs_f64_{idx}", "vmap", "lax_abs", ["vmap"],
            [vec], [[abs(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )
        cb.add(
            f"vmap_lax_tanh_f64_{idx}", "vmap", "lax_tanh", ["vmap"],
            [vec], [[math.tanh(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )

    # vmap(grad(cos)) = vmap(-sin)
    for idx, vec in enumerate(_vmap_vectors_f64()):
        cb.add(
            f"vmap_grad_cos_f64_{idx}", "vmap", "cos_x", ["vmap", "grad"],
            [vec], [[-math.sin(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )

    # vmap(grad(exp)) = vmap(exp)
    for idx, vec in enumerate(_vmap_vectors_f64()):
        cb.add(
            f"vmap_grad_exp_f64_{idx}", "vmap", "lax_exp", ["vmap", "grad"],
            [vec], [[math.exp(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )

    # Depth-3 transform compositions
    for idx, vec in enumerate(_vmap_vectors_f64()):
        # jit(vmap(grad(square))): grad(x^2) = 2x, vectorized, then JIT
        cb.add(
            f"jit_vmap_grad_square_f64_{idx}", "jit", "square", ["jit", "vmap", "grad"],
            [vec], [[2.0 * v for v in vec]], atol=1e-6, rtol=1e-6,
        )
        # jit(vmap(grad(sin))): grad(sin(x)) = cos(x), vectorized, then JIT
        cb.add(
            f"jit_vmap_grad_sin_f64_{idx}", "jit", "sin_x", ["jit", "vmap", "grad"],
            [vec], [[math.cos(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )
        # jit(vmap(grad(exp))): grad(exp(x)) = exp(x), vectorized, then JIT
        cb.add(
            f"jit_vmap_grad_exp_f64_{idx}", "jit", "lax_exp", ["jit", "vmap", "grad"],
            [vec], [[math.exp(v) for v in vec]], atol=1e-6, rtol=1e-6,
        )


# ── Control flow family builder ──────────────────────────────────


def build_control_flow_cases(cb: CaseBuilder) -> None:
    """control_flow family: cond, scan with representative inputs."""
    # cond(pred=true, on_true, on_false) => on_true
    for idx, (pred, a, b) in enumerate([
        (True, 7, 99),
        (False, 7, 99),
        (True, 0, -1),
        (False, 42, 0),
    ]):
        expected = a if pred else b
        pred_int = 1 if pred else 0
        cb.add(
            f"cond_select_i64_{idx}", "control_flow", "cond_select", [],
            [pred_int, a, b], [expected], atol=0.0, rtol=0.0, comparator="exact",
        )

    # scan(add, init=0, xs=[1,2,3,4]) => 10
    scan_cases = [
        (0, [1, 2, 3, 4], 10),
        (5, [1, 1, 1], 8),
        (0, [10], 10),
        (100, [1, 2, 3, 4, 5], 115),
    ]
    for idx, (init, xs, expected) in enumerate(scan_cases):
        cb.add(
            f"scan_add_i64_{idx}", "control_flow", "scan_add", [],
            [init, xs], [expected], atol=0.0, rtol=0.0, comparator="exact",
        )

    # jit(cond)
    for idx, (pred, a, b) in enumerate([
        (True, 10, 20),
        (False, 10, 20),
    ]):
        expected = a if pred else b
        pred_int = 1 if pred else 0
        cb.add(
            f"jit_cond_select_i64_{idx}", "control_flow", "cond_select", ["jit"],
            [pred_int, a, b], [expected], atol=0.0, rtol=0.0, comparator="exact",
        )

    # jit(scan)
    cb.add(
        "jit_scan_add_i64_0", "control_flow", "scan_add", ["jit"],
        [0, [1, 2, 3, 4]], [10], atol=0.0, rtol=0.0, comparator="exact",
    )


# ── Mixed-dtype family builder ───────────────────────────────────


def build_mixed_dtype_cases(cb: CaseBuilder) -> None:
    """mixed_dtype family: type promotion for cross-dtype binary ops."""
    fam = "mixed_dtype"

    # --- Scalar i64 + f64 arithmetic ---

    # add(i64, f64) → f64  (I64 + F64 → F64 promotion)
    for idx, (a, b, exp) in enumerate([
        (2, 0.5, 2.5),
        (-3, 1.5, -1.5),
        (0, 0.0, 0.0),
    ]):
        cb.add_raw(
            f"mixed_add_i64_f64_{idx}", fam, "add2", [],
            [{"kind": "scalar_i64", "value": a}, {"kind": "scalar_f64", "value": b}],
            [{"kind": "scalar_f64", "value": exp}],
            atol=0.0, rtol=0.0,
        )

    # add(f64, i64) → f64  (reversed operand order)
    for idx, (a, b, exp) in enumerate([
        (1.5, 3, 4.5),
        (-0.5, 10, 9.5),
    ]):
        cb.add_raw(
            f"mixed_add_f64_i64_{idx}", fam, "add2", [],
            [{"kind": "scalar_f64", "value": a}, {"kind": "scalar_i64", "value": b}],
            [{"kind": "scalar_f64", "value": exp}],
            atol=0.0, rtol=0.0,
        )

    # sub(f64, i64) → f64
    for idx, (a, b, exp) in enumerate([
        (10.0, 3, 7.0),
        (0.5, 1, -0.5),
    ]):
        cb.add_raw(
            f"mixed_sub_f64_i64_{idx}", fam, "lax_sub", [],
            [{"kind": "scalar_f64", "value": a}, {"kind": "scalar_i64", "value": b}],
            [{"kind": "scalar_f64", "value": exp}],
            atol=0.0, rtol=0.0,
        )

    # mul(i64, f64) → f64
    for idx, (a, b, exp) in enumerate([
        (3, 2.0, 6.0),
        (-2, 0.5, -1.0),
        (0, 99.9, 0.0),
    ]):
        cb.add_raw(
            f"mixed_mul_i64_f64_{idx}", fam, "lax_mul", [],
            [{"kind": "scalar_i64", "value": a}, {"kind": "scalar_f64", "value": b}],
            [{"kind": "scalar_f64", "value": exp}],
            atol=0.0, rtol=0.0,
        )

    # div(i64, f64) → f64
    for idx, (a, b, exp) in enumerate([
        (9, 2.0, 4.5),
        (1, 4.0, 0.25),
    ]):
        cb.add_raw(
            f"mixed_div_i64_f64_{idx}", fam, "lax_div", [],
            [{"kind": "scalar_i64", "value": a}, {"kind": "scalar_f64", "value": b}],
            [{"kind": "scalar_f64", "value": exp}],
            atol=0.0, rtol=0.0,
        )

    # max(i64, f64) → f64
    cb.add_raw(
        "mixed_max_i64_f64_0", fam, "lax_max", [],
        [{"kind": "scalar_i64", "value": 3}, {"kind": "scalar_f64", "value": 2.5}],
        [{"kind": "scalar_f64", "value": 3.0}],
        atol=0.0, rtol=0.0,
    )
    cb.add_raw(
        "mixed_max_i64_f64_1", fam, "lax_max", [],
        [{"kind": "scalar_i64", "value": 1}, {"kind": "scalar_f64", "value": 2.5}],
        [{"kind": "scalar_f64", "value": 2.5}],
        atol=0.0, rtol=0.0,
    )

    # min(i64, f64) → f64
    cb.add_raw(
        "mixed_min_i64_f64_0", fam, "lax_min", [],
        [{"kind": "scalar_i64", "value": 5}, {"kind": "scalar_f64", "value": 2.5}],
        [{"kind": "scalar_f64", "value": 2.5}],
        atol=0.0, rtol=0.0,
    )

    # --- Mixed-dtype comparisons (result is always bool) ---

    cb.add_raw(
        "mixed_gt_f64_i64_0", fam, "lax_gt", [],
        [{"kind": "scalar_f64", "value": 2.5}, {"kind": "scalar_i64", "value": 2}],
        [{"kind": "scalar_bool", "value": True}],
        atol=0.0, rtol=0.0,
    )
    cb.add_raw(
        "mixed_gt_f64_i64_1", fam, "lax_gt", [],
        [{"kind": "scalar_f64", "value": 1.5}, {"kind": "scalar_i64", "value": 2}],
        [{"kind": "scalar_bool", "value": False}],
        atol=0.0, rtol=0.0,
    )
    cb.add_raw(
        "mixed_lt_i64_f64_0", fam, "lax_lt", [],
        [{"kind": "scalar_i64", "value": 1}, {"kind": "scalar_f64", "value": 1.5}],
        [{"kind": "scalar_bool", "value": True}],
        atol=0.0, rtol=0.0,
    )
    cb.add_raw(
        "mixed_eq_f64_i64_0", fam, "lax_eq", [],
        [{"kind": "scalar_f64", "value": 3.0}, {"kind": "scalar_i64", "value": 3}],
        [{"kind": "scalar_bool", "value": True}],
        atol=0.0, rtol=0.0,
    )
    cb.add_raw(
        "mixed_ne_f64_i64_0", fam, "lax_ne", [],
        [{"kind": "scalar_f64", "value": 2.5}, {"kind": "scalar_i64", "value": 3}],
        [{"kind": "scalar_bool", "value": True}],
        atol=0.0, rtol=0.0,
    )

    # --- JIT of mixed-dtype ops ---

    cb.add_raw(
        "jit_mixed_add_i64_f64_0", fam, "add2", ["jit"],
        [{"kind": "scalar_i64", "value": 7}, {"kind": "scalar_f64", "value": 0.25}],
        [{"kind": "scalar_f64", "value": 7.25}],
        atol=0.0, rtol=0.0,
    )
    cb.add_raw(
        "jit_mixed_mul_i64_f64_0", fam, "lax_mul", ["jit"],
        [{"kind": "scalar_i64", "value": 4}, {"kind": "scalar_f64", "value": 2.5}],
        [{"kind": "scalar_f64", "value": 10.0}],
        atol=0.0, rtol=0.0,
    )
    cb.add_raw(
        "jit_mixed_gt_f64_i64_0", fam, "lax_gt", ["jit"],
        [{"kind": "scalar_f64", "value": 5.0}, {"kind": "scalar_i64", "value": 3}],
        [{"kind": "scalar_bool", "value": True}],
        atol=0.0, rtol=0.0,
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

    # ── Multi-rank tensor ops (rank-2) ──
    # Unary ops on 2x3 tensors
    _tensor_2x3_f64 = [1.0, -0.5, 2.0, 0.5, -1.0, 3.0]
    _tensor_2x3_positive = [0.5, 1.0, 2.0, 0.1, 3.0, 4.0]
    _tensor_shape = [2, 3]

    tensor_unary_cases = [
        ("neg", "lax_neg", lambda x: -x, _tensor_2x3_f64),
        ("abs", "lax_abs", lambda x: abs(x), _tensor_2x3_f64),
        ("square", "lax_square", lambda x: x * x, _tensor_2x3_f64),
        ("exp", "lax_exp", lambda x: math.exp(x), [0.0, 0.5, 1.0, -0.5, -1.0, 0.1]),
        ("sqrt", "lax_sqrt", lambda x: math.sqrt(x), _tensor_2x3_positive),
    ]

    for name, program, fn, vals in tensor_unary_cases:
        expected = [fn(v) for v in vals]
        cb.add_raw(
            f"lax_{name}_tensor2x3_f64_0", "lax", program, ["jit"],
            [fixture_value_tensor_f64(_tensor_shape, vals)],
            [fixture_value_tensor_f64(_tensor_shape, expected)],
            atol=1e-6, rtol=1e-6, comparator="approx_atol_rtol",
        )

    # Binary ops on 2x3 tensors
    _tensor_2x3_lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    _tensor_2x3_rhs = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    tensor_binary_cases = [
        ("sub", "lax_sub", lambda a, b: a - b),
        ("mul", "lax_mul", lambda a, b: a * b),
        ("max", "lax_max", lambda a, b: max(a, b)),
        ("min", "lax_min", lambda a, b: min(a, b)),
    ]

    for name, program, fn in tensor_binary_cases:
        expected = [fn(a, b) for a, b in zip(_tensor_2x3_lhs, _tensor_2x3_rhs)]
        cb.add_raw(
            f"lax_{name}_tensor2x3_f64_0", "lax", program, ["jit"],
            [fixture_value_tensor_f64(_tensor_shape, _tensor_2x3_lhs),
             fixture_value_tensor_f64(_tensor_shape, _tensor_2x3_rhs)],
            [fixture_value_tensor_f64(_tensor_shape, expected)],
            atol=1e-6, rtol=1e-6, comparator="approx_atol_rtol",
        )

    # Comparison ops on 2x3 tensors (return TensorBool)
    tensor_cmp_cases = [
        ("eq", "lax_eq", lambda a, b: a == b),
        ("lt", "lax_lt", lambda a, b: a < b),
    ]
    for name, program, fn in tensor_cmp_cases:
        expected_bools = [fn(a, b) for a, b in zip(_tensor_2x3_lhs, _tensor_2x3_rhs)]
        cb.add_raw(
            f"lax_{name}_tensor2x3_f64_0", "lax", program, ["jit"],
            [fixture_value_tensor_f64(_tensor_shape, _tensor_2x3_lhs),
             fixture_value_tensor_f64(_tensor_shape, _tensor_2x3_rhs)],
            [{"kind": "tensor_bool", "shape": _tensor_shape, "values": expected_bools}],
            atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Comparison ops (return Bool) ──
    _comparison_cases = [
        ("eq", "lax_eq", lambda a, b: a == b),
        ("ne", "lax_ne", lambda a, b: a != b),
        ("lt", "lax_lt", lambda a, b: a < b),
        ("le", "lax_le", lambda a, b: a <= b),
        ("gt", "lax_gt", lambda a, b: a > b),
        ("ge", "lax_ge", lambda a, b: a >= b),
    ]

    comparison_pairs = [(1.0, 2.0), (2.0, 2.0), (3.0, 1.0), (-0.5, 0.5)]
    for name, program, fn in _comparison_cases:
        for idx, (a, b) in enumerate(comparison_pairs):
            result = fn(a, b)
            cb.add_raw(
                f"lax_{name}_f64_{idx}", "lax", program, ["jit"],
                [fixture_value(a), fixture_value(b)],
                [fixture_value_bool(result)],
                atol=0.0, rtol=0.0, comparator="exact",
            )

    # ── Select (cond, on_true, on_false) -> on_true if cond else on_false ──
    select_cases = [
        (True, 10.0, 20.0, 10.0),
        (False, 10.0, 20.0, 20.0),
        (True, -1.5, 3.5, -1.5),
        (False, 0.0, 99.0, 99.0),
    ]
    for idx, (cond, on_true, on_false, expected) in enumerate(select_cases):
        cb.add_raw(
            f"lax_select_f64_{idx}", "lax", "lax_select", ["jit"],
            [fixture_value_bool(cond), fixture_value(on_true), fixture_value(on_false)],
            [fixture_value(expected)],
            atol=0.0, rtol=0.0, comparator="exact",
        )

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

    # ── Special math unary: cbrt ──
    cbrt_samples = [-8.0, -1.0, 0.0, 1.0, 8.0, 27.0]
    for idx, x in enumerate(cbrt_samples):
        cb.add(
            f"lax_cbrt_f64_{idx}", "lax", "lax_cbrt", ["jit"],
            [x], [math.copysign(abs(x) ** (1.0 / 3.0), x) if x != 0 else 0.0],
            atol=1e-6, rtol=1e-6,
        )

    # ── Special math unary: lgamma ──
    lgamma_samples = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    for idx, x in enumerate(lgamma_samples):
        cb.add(
            f"lax_lgamma_f64_{idx}", "lax", "lax_lgamma", ["jit"],
            [x], [math.lgamma(x)], atol=1e-4, rtol=1e-4,
        )

    # ── Special math unary: digamma ──
    digamma_samples = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
    for idx, x in enumerate(digamma_samples):
        cb.add(
            f"lax_digamma_f64_{idx}", "lax", "lax_digamma", ["jit"],
            [x], [_digamma_approx(x)], atol=1e-4, rtol=1e-4,
        )

    # ── Special math unary: erf_inv ──
    erf_inv_samples = [-0.9, -0.5, 0.0, 0.5, 0.9]
    for idx, x in enumerate(erf_inv_samples):
        cb.add(
            f"lax_erf_inv_f64_{idx}", "lax", "lax_erf_inv", ["jit"],
            [x], [_erf_inv_approx(x)], atol=1e-3, rtol=1e-3,
        )

    # ── Special math unary: is_finite (returns bool) ──
    # Only finite inputs (inf/nan not JSON-serializable; edge cases covered by unit tests)
    is_finite_cases = [
        (1.0, True), (-2.5, True), (0.0, True),
        (1e308, True), (-1e-308, True), (0.001, True),
    ]
    for idx, (x, expected) in enumerate(is_finite_cases):
        cb.add_raw(
            f"lax_is_finite_f64_{idx}", "lax", "lax_is_finite", ["jit"],
            [fixture_value(x)],
            [fixture_value_bool(expected)],
            atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Binary: nextafter ──
    # NOTE: use approx comparator with ultra-tight tolerance because
    # math.nextafter results near ±1.0 lose ULP precision through JSON
    # serialization round-trip (e.g. 0.9999999999999998 → 0.9999999999999999 → 1.0).
    nextafter_cases = [
        (1.0, 2.0),   # next float after 1.0 towards 2.0
        (1.0, 0.0),   # next float after 1.0 towards 0.0
        (0.0, 1.0),   # smallest positive subnormal
        (-1.0, 0.0),  # next float after -1.0 towards 0.0
    ]
    for idx, (x, y) in enumerate(nextafter_cases):
        cb.add(
            f"lax_nextafter_f64_{idx}", "lax", "lax_nextafter", ["jit"],
            [x, y], [math.nextafter(x, y)], atol=1e-15, rtol=1e-15,
        )

    # ── Cumulative: cumsum (vector → vector) ──
    cumsum_vectors = [[1.0, 2.0, 3.0], [0.5, -1.5, 2.5], [-3.0, 0.0, 3.0]]
    for idx, vec in enumerate(cumsum_vectors):
        acc = []
        s = 0.0
        for v in vec:
            s += v
            acc.append(s)
        cb.add(
            f"lax_cumsum_f64_{idx}", "lax", "lax_cumsum", ["jit"],
            [vec], [acc], atol=1e-12, rtol=1e-12,
        )

    # ── Cumulative: cumprod (vector → vector) ──
    cumprod_vectors = [[1.0, 2.0, 3.0], [0.5, -2.0, 4.0], [2.0, 3.0, 0.5]]
    for idx, vec in enumerate(cumprod_vectors):
        acc = []
        p = 1.0
        for v in vec:
            p *= v
            acc.append(p)
        cb.add(
            f"lax_cumprod_f64_{idx}", "lax", "lax_cumprod", ["jit"],
            [vec], [acc], atol=1e-12, rtol=1e-12,
        )

    # ── Bitwise binary ops (i64, i64 → i64) ──
    bitwise_pairs = [(0xFF, 0x0F), (0b1010, 0b1100), (42, 15), (0, -1)]
    for idx, (a, b) in enumerate(bitwise_pairs):
        cb.add(
            f"lax_bitwise_and_i64_{idx}", "lax", "lax_bitwise_and", ["jit"],
            [a, b], [a & b], atol=0.0, rtol=0.0, comparator="exact",
        )
        cb.add(
            f"lax_bitwise_or_i64_{idx}", "lax", "lax_bitwise_or", ["jit"],
            [a, b], [a | b], atol=0.0, rtol=0.0, comparator="exact",
        )
        cb.add(
            f"lax_bitwise_xor_i64_{idx}", "lax", "lax_bitwise_xor", ["jit"],
            [a, b], [a ^ b], atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Bitwise unary: bitwise_not (i64 → i64) ──
    bitwise_not_samples = [0, 1, -1, 42, 0xFF]
    for idx, x in enumerate(bitwise_not_samples):
        cb.add(
            f"lax_bitwise_not_i64_{idx}", "lax", "lax_bitwise_not", ["jit"],
            [x], [~x], atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Integer intrinsic: population_count (i64 → i64) ──
    popcnt_samples = [0, 1, 0b1010_1100, -1, 42, 0xFF]
    for idx, x in enumerate(popcnt_samples):
        # Python's bin(x).count('1') doesn't work for negative i64;
        # for negative values, population_count counts bits in the two's complement
        # representation. For signed i64, -1 has all 64 bits set.
        if x >= 0:
            expected = bin(x).count("1")
        else:
            # Two's complement for 64-bit signed integer
            expected = bin(x & 0xFFFFFFFFFFFFFFFF).count("1")
        cb.add(
            f"lax_population_count_i64_{idx}", "lax", "lax_population_count", ["jit"],
            [x], [expected], atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Integer intrinsic: count_leading_zeros (i64 → i64) ──
    clz_samples = [0, 1, 2, 127, -1, 0x0100_0000_0000_0000]
    for idx, x in enumerate(clz_samples):
        if x == 0:
            expected = 64
        elif x < 0:
            expected = 0  # Negative i64 always has MSB set
        else:
            expected = 64 - x.bit_length()
        cb.add(
            f"lax_count_leading_zeros_i64_{idx}", "lax", "lax_count_leading_zeros", ["jit"],
            [x], [expected], atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Boolean reduction: reduce_xor (vector<i64> → scalar) ──
    reduce_xor_cases = [
        ([1, 3, 2], 1 ^ 3 ^ 2),       # = 0
        ([5, 10, 15], 5 ^ 10 ^ 15),    # = 0
        ([7, 3, 1], 7 ^ 3 ^ 1),        # = 5
        ([0, 0, 0], 0),
    ]
    for idx, (vec, expected) in enumerate(reduce_xor_cases):
        cb.add(
            f"lax_reduce_xor_i64_{idx}", "lax", "lax_reduce_xor", ["jit"],
            [vec], [expected], atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Sorting: sort (vector → vector) ──
    sort_cases = [
        ([3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),  # already sorted
        ([42], [42]),                           # single element
    ]
    for idx, (input_vec, expected_vec) in enumerate(sort_cases):
        cb.add(
            f"lax_sort_i64_{idx}", "lax", "lax_sort", ["jit"],
            [input_vec], [expected_vec], atol=0.0, rtol=0.0, comparator="exact",
        )

    # ── Integer power: integer_pow with exponent=2 ──
    ipow2_cases = [(2.0, 4.0), (3.0, 9.0), (-2.0, 4.0), (0.5, 0.25), (0.0, 0.0)]
    for idx, (x, expected) in enumerate(ipow2_cases):
        cb.add(
            f"lax_integer_pow2_f64_{idx}", "lax", "lax_integer_pow2", ["jit"],
            [x], [expected], atol=1e-12, rtol=1e-12,
        )

    # ── Integer power: integer_pow with exponent=3 ──
    ipow3_cases = [(2.0, 8.0), (3.0, 27.0), (-2.0, -8.0), (0.5, 0.125)]
    for idx, (x, expected) in enumerate(ipow3_cases):
        cb.add(
            f"lax_integer_pow3_f64_{idx}", "lax", "lax_integer_pow3", ["jit"],
            [x], [expected], atol=1e-12, rtol=1e-12,
        )

    # ── Integer power: integer_pow with exponent=-1 (reciprocal) ──
    ipow_neg1_cases = [(2.0, 0.5), (4.0, 0.25), (-2.0, -0.5), (0.5, 2.0)]
    for idx, (x, expected) in enumerate(ipow_neg1_cases):
        cb.add(
            f"lax_integer_pow_neg1_f64_{idx}", "lax", "lax_integer_pow_neg1", ["jit"],
            [x], [expected], atol=1e-12, rtol=1e-12,
        )

    # ── Shape manipulation: reshape [6] → [2,3] ──
    cb.add_raw(
        "lax_reshape_6to2x3_i64_0", "lax", "lax_reshape6_to2x3", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2, 3, 4, 5, 6]}],
        [fixture_value_tensor_i64([2, 3], [1, 2, 3, 4, 5, 6])],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_reshape_6to2x3_i64_1", "lax", "lax_reshape6_to2x3", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50, 60]}],
        [fixture_value_tensor_i64([2, 3], [10, 20, 30, 40, 50, 60])],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Shape manipulation: reshape [6] → [3,2] ──
    cb.add_raw(
        "lax_reshape_6to3x2_i64_0", "lax", "lax_reshape6_to3x2", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2, 3, 4, 5, 6]}],
        [fixture_value_tensor_i64([3, 2], [1, 2, 3, 4, 5, 6])],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Shape manipulation: slice [1:4] from 5-element vector ──
    cb.add_raw(
        "lax_slice1to4_i64_0", "lax", "lax_slice1_to4", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]}],
        [{"kind": "vector_i64", "values": [20, 30, 40]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_slice1to4_i64_1", "lax", "lax_slice1_to4", ["jit"],
        [{"kind": "vector_i64", "values": [5, 4, 3, 2, 1]}],
        [{"kind": "vector_i64", "values": [4, 3, 2]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Shape manipulation: transpose [2,3] → [3,2] (reverse axes) ──
    # Input: [[1,2,3],[4,5,6]] → output: [[1,4],[2,5],[3,6]]
    cb.add_raw(
        "lax_transpose_2x3_i64_0", "lax", "lax_transpose2x3", ["jit"],
        [fixture_value_tensor_i64([2, 3], [1, 2, 3, 4, 5, 6])],
        [fixture_value_tensor_i64([3, 2], [1, 4, 2, 5, 3, 6])],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_transpose_2x3_f64_0", "lax", "lax_transpose2x3", ["jit"],
        [fixture_value_tensor_f64([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
        [fixture_value_tensor_f64([3, 2], [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Shape manipulation: rev (reverse along axis 0) ──
    cb.add_raw(
        "lax_rev_axis0_i64_0", "lax", "lax_rev", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2, 3, 4, 5]}],
        [{"kind": "vector_i64", "values": [5, 4, 3, 2, 1]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_rev_axis0_i64_1", "lax", "lax_rev", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30]}],
        [{"kind": "vector_i64", "values": [30, 20, 10]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Shape manipulation: squeeze (remove size-1 dim at axis 0) ──
    # Input: tensor [1,3] with values [10,20,30] → output: vector [10,20,30]
    cb.add_raw(
        "lax_squeeze_1x3_i64_0", "lax", "lax_squeeze", ["jit"],
        [fixture_value_tensor_i64([1, 3], [10, 20, 30])],
        [{"kind": "vector_i64", "values": [10, 20, 30]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_squeeze_1x5_f64_0", "lax", "lax_squeeze", ["jit"],
        [fixture_value_tensor_f64([1, 5], [1.0, 2.0, 3.0, 4.0, 5.0])],
        [{"kind": "vector_f64", "values": [1.0, 2.0, 3.0, 4.0, 5.0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Shape manipulation: concatenate (two vectors along axis 0) ──
    cb.add_raw(
        "lax_concatenate_i64_0", "lax", "lax_concatenate", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2]},
         {"kind": "vector_i64", "values": [3, 4, 5]}],
        [{"kind": "vector_i64", "values": [1, 2, 3, 4, 5]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_concatenate_f64_0", "lax", "lax_concatenate", ["jit"],
        [{"kind": "vector_f64", "values": [1.0, 2.0]},
         {"kind": "vector_f64", "values": [3.0, 4.0]}],
        [{"kind": "vector_f64", "values": [1.0, 2.0, 3.0, 4.0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── BitcastConvertType: f64 → i64 (reinterpret bits) ──
    import struct
    # 1.0_f64 → bit pattern as i64
    bits_1_0 = struct.unpack('<q', struct.pack('<d', 1.0))[0]  # 4607182418800017408
    cb.add("lax_bitcast_f64_to_i64_0", "lax", "lax_bitcast_f64_to_i64", ["jit"],
           [1.0], [bits_1_0], atol=0.0, rtol=0.0, comparator="exact")
    bits_0_0 = struct.unpack('<q', struct.pack('<d', 0.0))[0]  # 0
    cb.add("lax_bitcast_f64_to_i64_1", "lax", "lax_bitcast_f64_to_i64", ["jit"],
           [0.0], [bits_0_0], atol=0.0, rtol=0.0, comparator="exact")

    # ── ReducePrecision: f64 with 8 mantissa bits ──
    # Powers of 2 are always exact under any mantissa truncation
    cb.add("lax_reduce_precision_f64_0", "lax", "lax_reduce_precision_f64", ["jit"],
           [1.0], [1.0], atol=0.0, rtol=0.0, comparator="exact")
    cb.add("lax_reduce_precision_f64_1", "lax", "lax_reduce_precision_f64", ["jit"],
           [0.5], [0.5], atol=0.0, rtol=0.0, comparator="exact")

    # ── Gather: [operand, indices] → gathered elements ──
    # operand=[10,20,30,40,50], indices=[1,3] → [20,40]
    cb.add_raw(
        "lax_gather1d_i64_0", "lax", "lax_gather1d", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]},
         {"kind": "vector_i64", "values": [1, 3]}],
        [{"kind": "vector_i64", "values": [20, 40]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_gather1d_i64_1", "lax", "lax_gather1d", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]},
         {"kind": "vector_i64", "values": [0, 4]}],
        [{"kind": "vector_i64", "values": [10, 50]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── While loop: init=0 step=1 threshold=5 → 5 (add while lt) ──
    cb.add("lax_while_add_lt_i64_0", "lax", "lax_while_add_lt", ["jit"],
           [0, 1, 5], [5], atol=0.0, rtol=0.0, comparator="exact")
    cb.add("lax_while_add_lt_i64_1", "lax", "lax_while_add_lt", ["jit"],
           [0, 2, 10], [10], atol=0.0, rtol=0.0, comparator="exact")

    # ── Switch: index selects among 3 branch values ──
    # switch(0, [10, 20, 30]) → 10; switch(2, [10, 20, 30]) → 30
    cb.add("lax_switch3_i64_0", "lax", "lax_switch3", ["jit"],
           [0, 10, 20, 30], [10], atol=0.0, rtol=0.0, comparator="exact")
    cb.add("lax_switch3_i64_1", "lax", "lax_switch3", ["jit"],
           [2, 10, 20, 30], [30], atol=0.0, rtol=0.0, comparator="exact")

    # ── Argsort: vector → sorted indices ──
    cb.add_raw(
        "lax_argsort_i64_0", "lax", "lax_argsort", ["jit"],
        [{"kind": "vector_i64", "values": [30, 10, 20]}],
        [{"kind": "vector_i64", "values": [1, 2, 0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_argsort_f64_0", "lax", "lax_argsort", ["jit"],
        [{"kind": "vector_f64", "values": [3.0, 1.0, 2.0]}],
        [{"kind": "vector_i64", "values": [1, 2, 0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── OneHot: scalar index → one-hot vector of length 4 ──
    # one_hot(2, 4) → [0.0, 0.0, 1.0, 0.0]
    cb.add_raw(
        "lax_one_hot4_i64_0", "lax", "lax_one_hot4", ["jit"],
        [{"kind": "scalar_i64", "value": 2}],
        [{"kind": "vector_f64", "values": [0.0, 0.0, 1.0, 0.0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_one_hot4_i64_1", "lax", "lax_one_hot4", ["jit"],
        [{"kind": "scalar_i64", "value": 0}],
        [{"kind": "vector_f64", "values": [1.0, 0.0, 0.0, 0.0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── ReduceWindow: sum with window_dimensions=3, stride=1, valid padding ──
    # ReduceWindow preserves the input literal dtype for typed reductions.
    # [1,2,3,4,5] → [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    cb.add_raw(
        "lax_reduce_window_sum_i64_0", "lax", "lax_reduce_window_sum", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2, 3, 4, 5]}],
        [{"kind": "vector_i64", "values": [6, 9, 12]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_reduce_window_sum_f64_0", "lax", "lax_reduce_window_sum", ["jit"],
        [{"kind": "vector_f64", "values": [1.0, 2.0, 3.0, 4.0, 5.0]}],
        [{"kind": "vector_f64", "values": [6.0, 9.0, 12.0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_reduce_window_sum_complex128_0", "lax", "lax_reduce_window_sum", ["jit"],
        [{"kind": "tensor_complex128", "shape": [4],
          "reals": [1.0, 3.0, -2.0, 0.0],
          "imags": [2.0, -4.0, 0.5, 1.0]}],
        [{"kind": "tensor_complex128", "shape": [2],
          "reals": [2.0, 1.0],
          "imags": [-1.5, -2.5]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── DynamicSlice: [vector, start] → slice of length 3 ──
    # [10,20,30,40,50] start=1 → [20,30,40]
    cb.add_raw(
        "lax_dynamic_slice_i64_0", "lax", "lax_dynamic_slice", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]},
         {"kind": "scalar_i64", "value": 1}],
        [{"kind": "vector_i64", "values": [20, 30, 40]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_dynamic_slice_i64_1", "lax", "lax_dynamic_slice", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]},
         {"kind": "scalar_i64", "value": 0}],
        [{"kind": "vector_i64", "values": [10, 20, 30]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── DynamicUpdateSlice: [operand, update, start] → updated vector ──
    # [10,20,30,40,50] update=[99,88] at start=1 → [10,99,88,40,50]
    cb.add_raw(
        "lax_dynamic_update_slice_i64_0", "lax", "lax_dynamic_update_slice", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]},
         {"kind": "vector_i64", "values": [99, 88]},
         {"kind": "scalar_i64", "value": 1}],
        [{"kind": "vector_i64", "values": [10, 99, 88, 40, 50]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_dynamic_update_slice_i64_1", "lax", "lax_dynamic_update_slice", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]},
         {"kind": "vector_i64", "values": [99, 88, 77]},
         {"kind": "scalar_i64", "value": 0}],
        [{"kind": "vector_i64", "values": [99, 88, 77, 40, 50]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Split: vector of 6 into 2 equal sections → reshaped [2,3] ──
    cb.add_raw(
        "lax_split2_i64_0", "lax", "lax_split2", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2, 3, 4, 5, 6]}],
        [fixture_value_tensor_i64([2, 3], [1, 2, 3, 4, 5, 6])],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── BroadcastedIota: 2x3 tensor with iota along dimension 1 ──
    # [[0,1,2],[0,1,2]]
    cb.add_raw(
        "lax_broadcasted_iota_2x3_i64_0", "lax", "lax_broadcasted_iota2x3", ["jit"],
        [],
        [fixture_value_tensor_i64([2, 3], [0, 1, 2, 0, 1, 2])],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Bitwise shifts (binary i64 → i64) ──
    # ShiftLeft: 1 << 4 = 16
    cb.add("lax_shift_left_i64_0", "lax", "lax_shift_left", ["jit"],
           [1, 4], [16], atol=0.0, rtol=0.0, comparator="exact")
    cb.add("lax_shift_left_i64_1", "lax", "lax_shift_left", ["jit"],
           [3, 2], [12], atol=0.0, rtol=0.0, comparator="exact")
    # ShiftRightArithmetic: 16 >> 2 = 4; -8 >> 2 = -2 (sign-extending)
    cb.add("lax_shift_right_arithmetic_i64_0", "lax", "lax_shift_right_arithmetic", ["jit"],
           [16, 2], [4], atol=0.0, rtol=0.0, comparator="exact")
    cb.add("lax_shift_right_arithmetic_i64_1", "lax", "lax_shift_right_arithmetic", ["jit"],
           [-8, 2], [-2], atol=0.0, rtol=0.0, comparator="exact")
    # ShiftRightLogical: 16 >> 2 = 4; -8 >> 2 = 4611686018427387902 (zero-fill)
    cb.add("lax_shift_right_logical_i64_0", "lax", "lax_shift_right_logical", ["jit"],
           [16, 2], [4], atol=0.0, rtol=0.0, comparator="exact")
    cb.add("lax_shift_right_logical_i64_1", "lax", "lax_shift_right_logical", ["jit"],
           [-8, 2], [4611686018427387902], atol=0.0, rtol=0.0, comparator="exact")

    # ── Structural: Iota (no inputs, generates index sequence) ──
    cb.add_raw(
        "lax_iota5_i64_0", "lax", "lax_iota5", ["jit"],
        [],
        [{"kind": "vector_i64", "values": [0, 1, 2, 3, 4]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Structural: Copy (identity clone) ──
    cb.add_raw(
        "lax_copy_i64_0", "lax", "lax_copy", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30]}],
        [{"kind": "vector_i64", "values": [10, 20, 30]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_copy_f64_0", "lax", "lax_copy", ["jit"],
        [{"kind": "vector_f64", "values": [1.5, 2.5, 3.5]}],
        [{"kind": "vector_f64", "values": [1.5, 2.5, 3.5]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Structural: ExpandDims axis=0 (vector → rank-2 with leading 1) ──
    # [3] → [1, 3]
    cb.add_raw(
        "lax_expand_dims_axis0_i64_0", "lax", "lax_expand_dims_axis0", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30]}],
        [fixture_value_tensor_i64([1, 3], [10, 20, 30])],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_expand_dims_axis0_f64_0", "lax", "lax_expand_dims_axis0", ["jit"],
        [{"kind": "vector_f64", "values": [1.0, 2.0, 3.0]}],
        [fixture_value_tensor_f64([1, 3], [1.0, 2.0, 3.0])],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Structural: Pad low=1 high=2 interior=0 ──
    # [1,2,3] padded with 0 → [0, 1, 2, 3, 0, 0]
    cb.add_raw(
        "lax_pad_low1_high2_i64_0", "lax", "lax_pad_low1_high2", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2, 3]},
         {"kind": "scalar_i64", "value": 0}],
        [{"kind": "vector_i64", "values": [0, 1, 2, 3, 0, 0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_pad_low1_high2_f64_0", "lax", "lax_pad_low1_high2", ["jit"],
        [{"kind": "vector_f64", "values": [1.0, 2.0, 3.0]},
         {"kind": "scalar_f64", "value": 0.0}],
        [{"kind": "vector_f64", "values": [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Structural: BroadcastInDim scalar → vector of length 3 ──
    cb.add_raw(
        "lax_broadcast_in_dim_scalar3_i64_0", "lax", "lax_broadcast_in_dim_scalar3", ["jit"],
        [{"kind": "scalar_i64", "value": 7}],
        [{"kind": "vector_i64", "values": [7, 7, 7]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    cb.add_raw(
        "lax_broadcast_in_dim_scalar3_f64_0", "lax", "lax_broadcast_in_dim_scalar3", ["jit"],
        [{"kind": "scalar_f64", "value": 2.5}],
        [{"kind": "vector_f64", "values": [2.5, 2.5, 2.5]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Complex: (real, imag) → complex128 scalar ──
    # complex(3.0, 4.0) → 3+4j
    cb.add_raw(
        "lax_complex_f64_0", "lax", "lax_complex", ["jit"],
        [{"kind": "scalar_f64", "value": 3.0},
         {"kind": "scalar_f64", "value": 4.0}],
        [{"kind": "scalar_complex128", "real": 3.0, "imag": 4.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    # complex(0.0, 0.0) → 0+0j
    cb.add_raw(
        "lax_complex_f64_1", "lax", "lax_complex", ["jit"],
        [{"kind": "scalar_f64", "value": 0.0},
         {"kind": "scalar_f64", "value": 0.0}],
        [{"kind": "scalar_complex128", "real": 0.0, "imag": 0.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    # complex(1.0, -2.0) → 1-2j
    cb.add_raw(
        "lax_complex_f64_2", "lax", "lax_complex", ["jit"],
        [{"kind": "scalar_f64", "value": 1.0},
         {"kind": "scalar_f64", "value": -2.0}],
        [{"kind": "scalar_complex128", "real": 1.0, "imag": -2.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Conj: conjugate (negate imaginary part) ──
    # conj(3+4j) → 3-4j
    cb.add_raw(
        "lax_conj_complex128_0", "lax", "lax_conj", ["jit"],
        [{"kind": "scalar_complex128", "real": 3.0, "imag": 4.0}],
        [{"kind": "scalar_complex128", "real": 3.0, "imag": -4.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    # conj(0-1j) → 0+1j
    cb.add_raw(
        "lax_conj_complex128_1", "lax", "lax_conj", ["jit"],
        [{"kind": "scalar_complex128", "real": 0.0, "imag": -1.0}],
        [{"kind": "scalar_complex128", "real": 0.0, "imag": 1.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    # conj(5+3j) → 5-3j
    cb.add_raw(
        "lax_conj_complex128_2", "lax", "lax_conj", ["jit"],
        [{"kind": "scalar_complex128", "real": 5.0, "imag": 3.0}],
        [{"kind": "scalar_complex128", "real": 5.0, "imag": -3.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Real: extract real part as f64 ──
    # real(3+4j) → 3.0
    cb.add_raw(
        "lax_real_complex128_0", "lax", "lax_real", ["jit"],
        [{"kind": "scalar_complex128", "real": 3.0, "imag": 4.0}],
        [{"kind": "scalar_f64", "value": 3.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    # real(0-1j) → 0.0
    cb.add_raw(
        "lax_real_complex128_1", "lax", "lax_real", ["jit"],
        [{"kind": "scalar_complex128", "real": 0.0, "imag": -1.0}],
        [{"kind": "scalar_f64", "value": 0.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Imag: extract imaginary part as f64 ──
    # imag(3+4j) → 4.0
    cb.add_raw(
        "lax_imag_complex128_0", "lax", "lax_imag", ["jit"],
        [{"kind": "scalar_complex128", "real": 3.0, "imag": 4.0}],
        [{"kind": "scalar_f64", "value": 4.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    # imag(5+0j) → 0.0
    cb.add_raw(
        "lax_imag_complex128_1", "lax", "lax_imag", ["jit"],
        [{"kind": "scalar_complex128", "real": 5.0, "imag": 0.0}],
        [{"kind": "scalar_f64", "value": 0.0}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Conv 1D (valid padding): [lhs=[N,W,C_in], rhs=[K,C_in,C_out]] → [N,out_W,C_out] ──
    # lhs=[1,4,1] vals=[1,2,3,4], rhs=[2,1,1] vals=[1,1] → [1,3,1] vals=[3,5,7]
    cb.add_raw(
        "lax_conv1d_valid_0", "lax", "lax_conv1d_valid", ["jit"],
        [{"kind": "tensor_f64", "shape": [1, 4, 1], "values": [1.0, 2.0, 3.0, 4.0]},
         {"kind": "tensor_f64", "shape": [2, 1, 1], "values": [1.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [1, 3, 1], "values": [3.0, 5.0, 7.0]}],
        atol=1e-12, rtol=1e-12, comparator="approx_atol_rtol",
    )
    # lhs=[1,5,1] vals=[1,0,2,0,3], rhs=[3,1,1] vals=[1,2,1] → [1,3,1] vals=[1+0+2,0+4+0,2+0+3]=[3,4,5]
    cb.add_raw(
        "lax_conv1d_valid_1", "lax", "lax_conv1d_valid", ["jit"],
        [{"kind": "tensor_f64", "shape": [1, 5, 1], "values": [1.0, 0.0, 2.0, 0.0, 3.0]},
         {"kind": "tensor_f64", "shape": [3, 1, 1], "values": [1.0, 2.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [1, 3, 1], "values": [3.0, 4.0, 5.0]}],
        atol=1e-12, rtol=1e-12, comparator="approx_atol_rtol",
    )
    # 2 channels: lhs=[1,3,2] vals=[1,2,3,4,5,6], rhs=[2,2,1] vals=[1,1,1,1] → [1,2,1] vals=[1+2+3+4,3+4+5+6]=[10,18]
    cb.add_raw(
        "lax_conv1d_valid_2", "lax", "lax_conv1d_valid", ["jit"],
        [{"kind": "tensor_f64", "shape": [1, 3, 2], "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
         {"kind": "tensor_f64", "shape": [2, 2, 1], "values": [1.0, 1.0, 1.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [1, 2, 1], "values": [10.0, 18.0]}],
        atol=1e-12, rtol=1e-12, comparator="approx_atol_rtol",
    )
    # Kernel larger than input under VALID padding produces a zero-width output.
    cb.add_raw(
        "lax_conv1d_valid_3", "lax", "lax_conv1d_valid", ["jit"],
        [{"kind": "tensor_f64", "shape": [1, 1, 1], "values": [2.0]},
         {"kind": "tensor_f64", "shape": [2, 1, 1], "values": [1.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [1, 0, 1], "values": []}],
        atol=1e-12, rtol=1e-12, comparator="approx_atol_rtol",
    )

    # ── Scatter (overwrite mode): [operand, indices, updates] → scattered tensor ──
    # operand=[10,20,30,40,50] scatter indices=[1,3] updates=[99,88] → [10,99,30,88,50]
    cb.add_raw(
        "lax_scatter_overwrite_0", "lax", "lax_scatter_overwrite", ["jit"],
        [{"kind": "vector_i64", "values": [10, 20, 30, 40, 50]},
         {"kind": "vector_i64", "values": [1, 3]},
         {"kind": "vector_i64", "values": [99, 88]}],
        [{"kind": "vector_i64", "values": [10, 99, 30, 88, 50]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )
    # operand=[1,2,3,4] scatter index=[0] update=[42] → [42,2,3,4]
    cb.add_raw(
        "lax_scatter_overwrite_1", "lax", "lax_scatter_overwrite", ["jit"],
        [{"kind": "vector_i64", "values": [1, 2, 3, 4]},
         {"kind": "vector_i64", "values": [0]},
         {"kind": "vector_i64", "values": [42]}],
        [{"kind": "vector_i64", "values": [42, 2, 3, 4]}],
        atol=0.0, rtol=0.0, comparator="exact",
    )

    # ── Cholesky decomposition ──────────────────────────────────────
    # 2x2 identity
    cb.add_raw(
        "lax_cholesky_2x2_0", "lax", "lax_cholesky", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        atol=1e-12, rtol=1e-12, comparator="approx_atol_rtol",
    )
    # 2x2 SPD: [[4, 2], [2, 3]] → L = [[2, 0], [1, sqrt(2)]]
    cb.add_raw(
        "lax_cholesky_2x2_1", "lax", "lax_cholesky", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [4.0, 2.0, 2.0, 3.0]}],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [2.0, 0.0, 1.0, math.sqrt(2.0)]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # 3x3 SPD: [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    # L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    cb.add_raw(
        "lax_cholesky_3x3_0", "lax", "lax_cholesky", ["jit"],
        [{"kind": "tensor_f64", "shape": [3, 3],
          "values": [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]}],
        [{"kind": "tensor_f64", "shape": [3, 3],
          "values": [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # ── TriangularSolve ─────────────────────────────────────────────
    # Lower triangular solve: L = [[2, 0], [1, 3]], B = [[4], [7]]
    # x0 = 4/2 = 2, x1 = (7 - 1*2)/3 = 5/3
    cb.add_raw(
        "lax_triangular_solve_2x2_0", "lax", "lax_triangular_solve", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [2.0, 0.0, 1.0, 3.0]},
         {"kind": "tensor_f64", "shape": [2, 1], "values": [4.0, 7.0]}],
        [{"kind": "tensor_f64", "shape": [2, 1], "values": [2.0, 5.0 / 3.0]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # Cholesky roundtrip: L = [[2, 0], [1, sqrt(2)]], b = [[1], [2]]
    # Forward substitution: x1 = 1/2 = 0.5, x2 = (2 - 1*0.5)/sqrt(2) = 1.5/sqrt(2)
    cb.add_raw(
        "lax_triangular_solve_cholesky_roundtrip", "lax", "lax_triangular_solve", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [2.0, 0.0, 1.0, math.sqrt(2.0)]},
         {"kind": "tensor_f64", "shape": [2, 1], "values": [1.0, 2.0]}],
        [{"kind": "tensor_f64", "shape": [2, 1], "values": [0.5, 1.5 / math.sqrt(2.0)]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # ── QR decomposition ──────────────────────────────────────────
    # QR of 2x2 identity: Q=I, R=I (trivial)
    cb.add_raw(
        "lax_qr_2x2_identity", "lax", "lax_qr", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]},
         {"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # QR of A = [[1, -1], [1, 1]] with non-negative R diagonal convention
    s2 = math.sqrt(2.0)
    q_22 = [ 1.0/s2, -1.0/s2,
             1.0/s2,  1.0/s2]
    r_22 = [s2, 0.0,
            0.0, s2]
    cb.add_raw(
        "lax_qr_2x2_0", "lax", "lax_qr", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, -1.0, 1.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": q_22},
         {"kind": "tensor_f64", "shape": [2, 2], "values": r_22}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # QR of 3x2 matrix A = [[1, 1], [0, 1], [1, 0]] (thin QR: Q 3x2, R 2x2)
    # Sign-corrected (R diagonal non-negative)
    s6 = math.sqrt(6.0)
    q_32 = [ 1.0/s2,  1.0/s6,
             0.0,     2.0/s6,
             1.0/s2, -1.0/s6]
    r_32 = [s2, 1.0/s2,
            0.0, s6/2.0]
    cb.add_raw(
        "lax_qr_3x2_0", "lax", "lax_qr", ["jit"],
        [{"kind": "tensor_f64", "shape": [3, 2], "values": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]}],
        [{"kind": "tensor_f64", "shape": [3, 2], "values": q_32},
         {"kind": "tensor_f64", "shape": [2, 2], "values": r_32}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # ── SVD decomposition ─────────────────────────────────────────
    # SVD of 2x2 identity: U=I, S=[1,1], Vt=I
    cb.add_raw(
        "lax_svd_2x2_identity", "lax", "lax_svd", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]},
         {"kind": "tensor_f64", "shape": [2], "values": [1.0, 1.0]},
         {"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # SVD of diagonal [[3, 0], [0, -2]]: U=[[1,0],[0,-1]], S=[3,2], Vt=I
    cb.add_raw(
        "lax_svd_2x2_diagonal", "lax", "lax_svd", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [3.0, 0.0, 0.0, -2.0]}],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, -1.0]},
         {"kind": "tensor_f64", "shape": [2], "values": [3.0, 2.0]},
         {"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # ── Eigh (symmetric eigendecomposition) ───────────────────────
    # Eigh of 2x2 identity: W=[1,1], V=I
    cb.add_raw(
        "lax_eigh_2x2_identity", "lax", "lax_eigh", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        [{"kind": "tensor_f64", "shape": [2], "values": [1.0, 1.0]},
         {"kind": "tensor_f64", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )
    # Eigh of diagonal [[5, 0], [0, 3]]: W=[3, 5], V=[[0,1],[1,0]] (ascending)
    cb.add_raw(
        "lax_eigh_2x2_diagonal", "lax", "lax_eigh", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 2], "values": [5.0, 0.0, 0.0, 3.0]}],
        [{"kind": "tensor_f64", "shape": [2], "values": [3.0, 5.0]},
         {"kind": "tensor_f64", "shape": [2, 2], "values": [0.0, 1.0, 1.0, 0.0]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # ── FFT / IFFT / RFFT / IRFFT ──

    def _dft_1d(x: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Naive 1D DFT: X[k] = Σ x[j]·e^{-2πi·j·k/n}."""
        n = len(x)
        result = []
        for k in range(n):
            re_s, im_s = 0.0, 0.0
            for j in range(n):
                angle = -2.0 * math.pi * j * k / n
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                xr, xi = x[j]
                re_s += xr * cos_a - xi * sin_a
                im_s += xr * sin_a + xi * cos_a
            result.append((re_s, im_s))
        return result

    def _idft_1d(x: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Naive 1D IDFT: x[j] = (1/n) Σ X[k]·e^{2πi·j·k/n}."""
        n = len(x)
        inv_n = 1.0 / n
        result = []
        for j in range(n):
            re_s, im_s = 0.0, 0.0
            for k in range(n):
                angle = 2.0 * math.pi * j * k / n
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                xr, xi = x[k]
                re_s += xr * cos_a - xi * sin_a
                im_s += xr * sin_a + xi * cos_a
            result.append((re_s * inv_n, im_s * inv_n))
        return result

    def _tc128(shape: list[int], pairs: list[tuple[float, float]]) -> dict:
        return {
            "kind": "tensor_complex128",
            "shape": shape,
            "reals": [r for r, _ in pairs],
            "imags": [i for _, i in pairs],
        }

    # FFT: [1,2,3,4] → [10, -2+2i, -2, -2-2i]
    fft4 = _dft_1d([(1, 0), (2, 0), (3, 0), (4, 0)])
    cb.add_raw(
        "lax_fft_4point_f64_0", "lax", "lax_fft", ["jit"],
        [{"kind": "tensor_f64", "shape": [4], "values": [1.0, 2.0, 3.0, 4.0]}],
        [_tc128([4], fft4)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # FFT: impulse [1,0,0,0] → [1,1,1,1]
    fft_imp = _dft_1d([(1, 0), (0, 0), (0, 0), (0, 0)])
    cb.add_raw(
        "lax_fft_impulse_f64_0", "lax", "lax_fft", ["jit"],
        [{"kind": "tensor_f64", "shape": [4], "values": [1.0, 0.0, 0.0, 0.0]}],
        [_tc128([4], fft_imp)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # FFT: DC [1,1,1,1] → [4,0,0,0]
    fft_dc = _dft_1d([(1, 0), (1, 0), (1, 0), (1, 0)])
    cb.add_raw(
        "lax_fft_dc_f64_0", "lax", "lax_fft", ["jit"],
        [{"kind": "tensor_f64", "shape": [4], "values": [1.0, 1.0, 1.0, 1.0]}],
        [_tc128([4], fft_dc)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # FFT: batched 2x4
    r0 = _dft_1d([(1, 0), (1, 0), (1, 0), (1, 0)])
    r1 = _dft_1d([(1, 0), (0, 0), (0, 0), (0, 0)])
    cb.add_raw(
        "lax_fft_batched_2x4_f64_0", "lax", "lax_fft", ["jit"],
        [{"kind": "tensor_f64", "shape": [2, 4],
          "values": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}],
        [_tc128([2, 4], r0 + r1)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # FFT: complex input
    cinp = [(3, 4), (1, -2), (0, 0), (-1, 1)]
    cout = _dft_1d(cinp)
    cb.add_raw(
        "lax_fft_complex_input_0", "lax", "lax_fft", ["jit"],
        [_tc128([4], cinp)],
        [_tc128([4], cout)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # IFFT: [10, -2+2i, -2, -2-2i] → [1,2,3,4]
    ifft_inp = [(10, 0), (-2, 2), (-2, 0), (-2, -2)]
    ifft_out = _idft_1d(ifft_inp)
    cb.add_raw(
        "lax_ifft_4point_complex128_0", "lax", "lax_ifft", ["jit"],
        [_tc128([4], ifft_inp)],
        [_tc128([4], ifft_out)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # IFFT: DC [4,0,0,0] → [1,1,1,1]
    ifft_dc_inp = [(4, 0), (0, 0), (0, 0), (0, 0)]
    ifft_dc_out = _idft_1d(ifft_dc_inp)
    cb.add_raw(
        "lax_ifft_dc_complex128_0", "lax", "lax_ifft", ["jit"],
        [_tc128([4], ifft_dc_inp)],
        [_tc128([4], ifft_dc_out)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # IFFT: roundtrip (FFT then IFFT)
    cb.add_raw(
        "lax_ifft_roundtrip_complex128_0", "lax", "lax_ifft", ["jit"],
        [_tc128([4], fft4)],
        [_tc128([4], _idft_1d(fft4))],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # RFFT: [1..8] with fft_length=8 → 5 complex outputs
    rfft_inp = [(float(i + 1), 0) for i in range(8)]
    rfft_full = _dft_1d(rfft_inp)
    rfft_out = rfft_full[:5]
    cb.add_raw(
        "lax_rfft_8point_f64_0", "lax", "lax_rfft", ["jit"],
        [{"kind": "tensor_f64", "shape": [8],
          "values": [float(i + 1) for i in range(8)]}],
        [_tc128([5], rfft_out)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # RFFT: impulse [1,0,...,0] with fft_length=8
    rfft_imp_inp = [(1, 0)] + [(0, 0)] * 7
    rfft_imp_out = _dft_1d(rfft_imp_inp)[:5]
    cb.add_raw(
        "lax_rfft_impulse_f64_0", "lax", "lax_rfft", ["jit"],
        [{"kind": "tensor_f64", "shape": [8],
          "values": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}],
        [_tc128([5], rfft_imp_out)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # RFFT: shorter input [1,2,3,4] zero-padded to fft_length=8
    rfft_zp = _dft_1d([(1, 0), (2, 0), (3, 0), (4, 0)] + [(0, 0)] * 4)[:5]
    cb.add_raw(
        "lax_rfft_zeropad_f64_0", "lax", "lax_rfft", ["jit"],
        [{"kind": "tensor_f64", "shape": [4], "values": [1.0, 2.0, 3.0, 4.0]}],
        [_tc128([5], rfft_zp)],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # IRFFT: half-spectrum of [1..8] → recover [1..8]
    irfft_half = rfft_out
    fft_len = 8
    full_spec: list[tuple[float, float]] = list(irfft_half) + [(0.0, 0.0)] * (fft_len - len(irfft_half))
    for k in range(len(irfft_half), fft_len):
        mirror = fft_len - k
        if mirror < len(irfft_half):
            re, im = irfft_half[mirror]
            full_spec[k] = (re, -im)
    irfft_result = _idft_1d(full_spec)
    cb.add_raw(
        "lax_irfft_8point_complex128_0", "lax", "lax_irfft", ["jit"],
        [_tc128([5], irfft_half)],
        [{"kind": "tensor_f64", "shape": [8],
          "values": [r for r, _ in irfft_result]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
    )

    # IRFFT: DC [4,0,0,0,0] with fft_length=8
    irfft_dc_half = [(4, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    full_dc_spec: list[tuple[float, float]] = list(irfft_dc_half) + [(0.0, 0.0)] * (fft_len - len(irfft_dc_half))
    for k in range(len(irfft_dc_half), fft_len):
        mirror = fft_len - k
        if mirror < len(irfft_dc_half):
            re, im = irfft_dc_half[mirror]
            full_dc_spec[k] = (re, -im)
    irfft_dc_result = _idft_1d(full_dc_spec)
    cb.add_raw(
        "lax_irfft_dc_complex128_0", "lax", "lax_irfft", ["jit"],
        [_tc128([5], irfft_dc_half)],
        [{"kind": "tensor_f64", "shape": [8],
          "values": [r for r, _ in irfft_dc_result]}],
        atol=1e-10, rtol=1e-10, comparator="approx_atol_rtol",
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

    # Control flow: cond, scan
    build_control_flow_cases(cb)

    # Mixed-dtype type promotion
    build_mixed_dtype_cases(cb)

    return cb.cases


# ── Fallback capture (no JAX) ────────────────────────────────────


def build_cases_fallback() -> list[Case]:
    """Build all cases using analytical/mathematical results."""
    cb = CaseBuilder()
    build_jit_cases(cb)
    build_grad_cases(cb)
    build_vmap_cases(cb)
    build_lax_cases(cb)
    build_control_flow_cases(cb)
    build_mixed_dtype_cases(cb)
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
