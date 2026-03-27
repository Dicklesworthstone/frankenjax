#!/usr/bin/env python3
"""Capture linalg/FFT oracle fixtures from JAX.

Usage:
  python3 crates/fj-conformance/scripts/capture_linalg_fft_oracle.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output crates/fj-conformance/fixtures/linalg_fft_oracle.v1.json

This script records reference values for:
- Cholesky / QR / SVD / Eigh / TriangularSolve
- FFT / IFFT / RFFT / IRFFT

Unlike the main transform fixture capture script, this emits the narrow schema
consumed by `tests/linalg_fft_oracle_parity.rs`.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def _try_import_jax(legacy_root: Path):
    venv_site = Path(__file__).resolve().parents[3] / ".venv" / "lib"
    if venv_site.exists():
        for d in venv_site.iterdir():
            sp = d / "site-packages"
            if sp.exists():
                sys.path.insert(0, str(sp))
                break

    if legacy_root.exists():
        sys.path.insert(0, str(legacy_root))

    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import jax.lax as lax  # type: ignore

    return jax, jnp, lax


def _metadata(jax_version: str) -> dict[str, Any]:
    return {
        "jax_version": jax_version,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "hardware": platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _shape_of(arr) -> list[int]:
    return [int(dim) for dim in arr.shape]


def _real_value(arr) -> dict[str, Any]:
    shape = _shape_of(arr)
    flat = [float(x) for x in arr.reshape(-1).tolist()]
    if len(shape) == 1:
        return {"kind": "vector_f64", "shape": shape, "values": flat}
    if len(shape) == 2:
        return {"kind": "matrix_f64", "shape": shape, "values": flat}
    return {"kind": "tensor_f64", "shape": shape, "values": flat}


def _complex_value(arr) -> dict[str, Any]:
    shape = _shape_of(arr)
    flat = arr.reshape(-1)
    reals = [float(x.real) for x in flat.tolist()]
    imags = [float(x.imag) for x in flat.tolist()]
    if len(shape) == 1:
        return {
            "kind": "complex_vector",
            "shape": shape,
            "values": list(zip(reals, imags)),
        }
    return {
        "kind": "tensor_complex128",
        "shape": shape,
        "reals": reals,
        "imags": imags,
    }


def _value(arr) -> dict[str, Any]:
    if getattr(arr.dtype, "kind", "") == "c":
        return _complex_value(arr)
    return _real_value(arr)


def _case(
    case_id: str,
    operation: str,
    inputs: list[dict[str, Any]],
    expected_outputs: list[dict[str, Any]],
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    case = {
        "case_id": case_id,
        "operation": operation,
        "inputs": inputs,
        "expected_outputs": expected_outputs,
    }
    if params:
        case["params"] = params
    return case


def build_cases(jnp, lax) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    chol_2x2_i = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    cases.append(
        _case(
            "cholesky_2x2_identity",
            "cholesky",
            [_real_value(chol_2x2_i)],
            [_real_value(jnp.linalg.cholesky(chol_2x2_i))],
        )
    )

    chol_2x2_spd = jnp.array([[4.0, 2.0], [2.0, 3.0]], dtype=jnp.float64)
    cases.append(
        _case(
            "cholesky_2x2_spd",
            "cholesky",
            [_real_value(chol_2x2_spd)],
            [_real_value(jnp.linalg.cholesky(chol_2x2_spd))],
        )
    )

    chol_3x3_spd = jnp.array(
        [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]],
        dtype=jnp.float64,
    )
    cases.append(
        _case(
            "cholesky_3x3_spd",
            "cholesky",
            [_real_value(chol_3x3_spd)],
            [_real_value(jnp.linalg.cholesky(chol_3x3_spd))],
        )
    )

    chol_4x4 = jnp.array(
        [
            [10.0, 3.0, 1.0, 0.5],
            [3.0, 8.0, 2.0, 1.0],
            [1.0, 2.0, 6.0, 1.5],
            [0.5, 1.0, 1.5, 5.0],
        ],
        dtype=jnp.float64,
    )
    cases.append(
        _case(
            "cholesky_4x4_spd",
            "cholesky",
            [_real_value(chol_4x4)],
            [_real_value(jnp.linalg.cholesky(chol_4x4))],
        )
    )

    qr_2x2_i = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    q, r = jnp.linalg.qr(qr_2x2_i, mode="reduced")
    cases.append(_case("qr_2x2_identity", "qr", [_real_value(qr_2x2_i)], [_real_value(q), _real_value(r)]))

    qr_2x2 = jnp.array([[1.0, -1.0], [1.0, 1.0]], dtype=jnp.float64)
    q, r = jnp.linalg.qr(qr_2x2, mode="reduced")
    cases.append(_case("qr_2x2_general", "qr", [_real_value(qr_2x2)], [_real_value(q), _real_value(r)]))

    qr_3x2 = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64)
    q, r = jnp.linalg.qr(qr_3x2, mode="reduced")
    cases.append(_case("qr_3x2_tall", "qr", [_real_value(qr_3x2)], [_real_value(q), _real_value(r)]))

    qr_4x3 = jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0], [1.0, 3.0, 5.0]],
        dtype=jnp.float64,
    )
    q, r = jnp.linalg.qr(qr_4x3, mode="reduced")
    cases.append(_case("qr_4x3_tall", "qr", [_real_value(qr_4x3)], [_real_value(q), _real_value(r)]))

    svd_i = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    u, s, vt = jnp.linalg.svd(svd_i, full_matrices=False)
    cases.append(_case("svd_2x2_identity", "svd", [_real_value(svd_i)], [_real_value(u), _real_value(s), _real_value(vt)]))

    svd_diag = jnp.array([[3.0, 0.0], [0.0, -2.0]], dtype=jnp.float64)
    u, s, vt = jnp.linalg.svd(svd_diag, full_matrices=False)
    cases.append(_case("svd_2x2_diag", "svd", [_real_value(svd_diag)], [_real_value(u), _real_value(s), _real_value(vt)]))

    svd_sym = jnp.array([[2.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    u, s, vt = jnp.linalg.svd(svd_sym, full_matrices=False)
    cases.append(_case("svd_2x2_symmetric", "svd", [_real_value(svd_sym)], [_real_value(u), _real_value(s), _real_value(vt)]))

    svd_3x2 = jnp.array([[3.0, 1.0], [0.0, 2.0], [0.0, 0.0]], dtype=jnp.float64)
    u, s, vt = jnp.linalg.svd(svd_3x2, full_matrices=False)
    cases.append(_case("svd_3x2_rectangular", "svd", [_real_value(svd_3x2)], [_real_value(u), _real_value(s), _real_value(vt)]))

    eigh_i = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    w, v = jnp.linalg.eigh(eigh_i)
    cases.append(_case("eigh_2x2_identity", "eigh", [_real_value(eigh_i)], [_real_value(w), _real_value(v)]))

    eigh_2x2 = jnp.array([[2.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    w, v = jnp.linalg.eigh(eigh_2x2)
    cases.append(_case("eigh_2x2_symmetric", "eigh", [_real_value(eigh_2x2)], [_real_value(w), _real_value(v)]))

    eigh_3x3 = jnp.array(
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.0], [0.5, 0.0, 2.0]],
        dtype=jnp.float64,
    )
    w, v = jnp.linalg.eigh(eigh_3x3)
    cases.append(_case("eigh_3x3_symmetric", "eigh", [_real_value(eigh_3x3)], [_real_value(w), _real_value(v)]))

    eigh_rep = jnp.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]],
        dtype=jnp.float64,
    )
    w, v = jnp.linalg.eigh(eigh_rep)
    cases.append(_case("eigh_3x3_repeated", "eigh", [_real_value(eigh_rep)], [_real_value(w), _real_value(v)]))

    tsl_i = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    rhs_i = jnp.array([[3.0, 4.0], [5.0, 6.0]], dtype=jnp.float64)
    cases.append(
        _case(
            "tsolve_lower_identity",
            "triangular_solve",
            [_real_value(tsl_i), _real_value(rhs_i)],
            [_real_value(lax.linalg.triangular_solve(tsl_i, rhs_i, left_side=True, lower=True))],
            {"lower": "true"},
        )
    )

    tsl_l = jnp.array([[2.0, 0.0], [1.0, 3.0]], dtype=jnp.float64)
    rhs_l = jnp.array([[4.0], [7.0]], dtype=jnp.float64)
    cases.append(
        _case(
            "tsolve_lower_2x2",
            "triangular_solve",
            [_real_value(tsl_l), _real_value(rhs_l)],
            [_real_value(lax.linalg.triangular_solve(tsl_l, rhs_l, left_side=True, lower=True))],
            {"lower": "true"},
        )
    )

    tsl_t = jnp.array(
        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 3.0, 1.0]],
        dtype=jnp.float64,
    )
    rhs_t = jnp.array([[5.0, 6.0], [19.0, -3.0], [5.0, -2.0]], dtype=jnp.float64)
    cases.append(
        _case(
            "tsolve_lower_transpose_unitdiag_3x2",
            "triangular_solve",
            [_real_value(tsl_t), _real_value(rhs_t)],
            [
                _real_value(
                    lax.linalg.triangular_solve(
                        tsl_t,
                        rhs_t,
                        left_side=True,
                        lower=True,
                        transpose_a=True,
                        unit_diagonal=True,
                    )
                )
            ],
            {"lower": "true", "transpose_a": "true", "unit_diagonal": "true"},
        )
    )

    fft_dc = jnp.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex128)
    cases.append(_case("fft_dc", "fft", [_complex_value(fft_dc)], [_complex_value(jnp.fft.fft(fft_dc))]))

    fft_imp = jnp.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128)
    cases.append(_case("fft_impulse", "fft", [_complex_value(fft_imp)], [_complex_value(jnp.fft.fft(fft_imp))]))

    fft_general = jnp.array([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j], dtype=jnp.complex128)
    cases.append(_case("fft_general", "fft", [_complex_value(fft_general)], [_complex_value(jnp.fft.fft(fft_general))]))

    fft_batched = jnp.array(
        [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
        dtype=jnp.complex128,
    )
    cases.append(_case("fft_batched_2x4", "fft", [_complex_value(fft_batched)], [_complex_value(jnp.fft.fft(fft_batched, axis=-1))]))

    ifft_dc = jnp.array([4.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex128)
    cases.append(_case("ifft_dc", "ifft", [_complex_value(ifft_dc)], [_complex_value(jnp.fft.ifft(ifft_dc))]))

    rfft_dc = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
    cases.append(_case("rfft_dc", "rfft", [_real_value(rfft_dc)], [_complex_value(jnp.fft.rfft(rfft_dc, n=4))], {"fft_length": "4"}))

    rfft_ramp = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    cases.append(_case("rfft_ramp", "rfft", [_real_value(rfft_ramp)], [_complex_value(jnp.fft.rfft(rfft_ramp, n=4))], {"fft_length": "4"}))

    rfft_rank2 = jnp.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float64)
    cases.append(_case("rfft_rank2_zeropad", "rfft", [_real_value(rfft_rank2)], [_complex_value(jnp.fft.rfft(rfft_rank2, n=4, axis=-1))], {"fft_length": "4"}))

    irfft_known = jnp.array([10.0 + 0.0j, -2.0 + 2.0j, -2.0 + 0.0j], dtype=jnp.complex128)
    cases.append(_case("irfft_known", "irfft", [_complex_value(irfft_known)], [_real_value(jnp.fft.irfft(irfft_known, n=4))], {"fft_length": "4"}))

    irfft_odd = jnp.fft.rfft(jnp.array([1.0, -2.0, 0.5, 3.0, -1.5], dtype=jnp.float64), n=5)
    cases.append(_case("irfft_odd_roundtrip", "irfft", [_complex_value(irfft_odd)], [_real_value(jnp.fft.irfft(irfft_odd, n=5))], {"fft_length": "5"}))

    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.legacy_root.exists():
        print(f"legacy root does not exist: {args.legacy_root}", file=sys.stderr)
        return 2

    try:
        jax, jnp, lax = _try_import_jax(args.legacy_root)
    except Exception as exc:
        print(
            "Failed to import JAX for linalg/FFT oracle capture. "
            "Ensure jax + jaxlib are installed in .venv or reachable via --legacy-root.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 3

    cases = build_cases(jnp, lax)
    bundle = {
        "schema_version": "frankenjax.linalg-fft-oracle.v2",
        "generated_by": "capture_linalg_fft_oracle.py",
        "generated_at_unix_ms": int(time.time() * 1000),
        "oracle_root": str(args.legacy_root),
        "metadata": _metadata(getattr(jax, "__version__", "unknown")),
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[CAPTURE] wrote {len(cases)} linalg/fft oracle cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
