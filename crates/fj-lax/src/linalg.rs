#![forbid(unsafe_code)]

//! Linear algebra primitives: Cholesky, triangular solve, and QR decomposition.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;
use crate::tensor_contraction::matmul_2d_into;
use crate::tensor_contraction::rank2_complex_matmul;
use crate::type_promotion::promote_dtype;

type ComplexScalar = (f64, f64);
type ComplexVector = Vec<ComplexScalar>;
type ComplexMatrix = Vec<ComplexScalar>;
type EigQrResult = (ComplexVector, ComplexVector);

// ── Complex Arithmetic Helpers ──────────────────────────────────────

fn complex_abs(z: (f64, f64)) -> f64 {
    (z.0 * z.0 + z.1 * z.1).sqrt()
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

// NOTE: this naive `b.0² + b.1²` denominator overflows for |b| ≳ 1.3e154 (the
// quotient collapses to (0,0)), unlike the Div PRIMITIVE's `complex_div` in
// arithmetic.rs which uses Smith's algorithm. Do NOT "fix" this one the same way:
// its exact bits are pinned by `lu_real_fast_path_bit_identical_to_complex_zero_imag_path`
// and `cholesky_real_fast_path_bit_identical_to_complex_path`, which require the
// complex path to be bit-for-bit equal to the real fast path. Smith's changes the
// op order (1 ULP) and breaks that invariant. The overflow only bites at
// pathological (≥1e154) matrix entries; making this overflow-safe would require
// updating the real fast paths in lockstep.
fn complex_div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let denom = b.0 * b.0 + b.1 * b.1;
    (
        (a.0 * b.0 + a.1 * b.1) / denom,
        (a.1 * b.0 - a.0 * b.1) / denom,
    )
}

fn complex_sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}

fn complex_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}

fn complex_conj(a: (f64, f64)) -> (f64, f64) {
    (a.0, -a.1)
}

#[allow(dead_code)]
fn complex_sqrt_real(x: f64) -> (f64, f64) {
    if x >= 0.0 {
        (x.sqrt(), 0.0)
    } else {
        (0.0, (-x).sqrt())
    }
}

// ── Matrix Helpers ──────────────────────────────────────────────────────────

/// Extract a rank-2 (matrix) tensor from Value, returning its dimensions and
/// complex (re, im) elements in row-major order. Real numbers become (x, 0.0).
#[allow(clippy::type_complexity)]
fn extract_complex_matrix(
    primitive: Primitive,
    value: &Value,
) -> Result<(usize, usize, Vec<(f64, f64)>, DType), EvalError> {
    let tensor = match value {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "expected matrix (rank-2 tensor), got scalar".to_owned(),
            });
        }
    };

    if tensor.rank() != 2 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "expected rank-2 tensor (matrix), got rank-{}",
                tensor.rank()
            ),
        });
    }

    let m = tensor.shape.dims[0] as usize;
    let n = tensor.shape.dims[1] as usize;

    let elements: Vec<(f64, f64)> = tensor
        .elements
        .iter()
        .map(|lit| {
            if let Some((re, im)) = lit.as_complex128() {
                Ok((re, im))
            } else if let Some((re, im)) = lit.as_complex64() {
                Ok((re as f64, im as f64))
            } else if let Some(v) = lit.as_f64() {
                Ok((v, 0.0))
            } else {
                Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "expected numeric elements",
                })
            }
        })
        .collect::<Result<_, _>>()?;

    Ok((m, n, elements, tensor.dtype))
}

/// Build a Value::Tensor from row-major complex data and shape.
fn complex_matrix_to_value(
    m: usize,
    n: usize,
    data: &[(f64, f64)],
    dtype: DType,
) -> Result<Value, EvalError> {
    let elements: Vec<Literal> = data
        .iter()
        .map(|&(re, im)| match dtype {
            DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
            DType::Complex128 => Literal::from_complex128(re, im),
            DType::BF16 => Literal::from_bf16_f64(re),
            DType::F16 => Literal::from_f16_f64(re),
            DType::F32 => Literal::from_f32(re as f32),
            _ => Literal::from_f64(re),
        })
        .collect();
    let shape = Shape {
        dims: vec![m as u32, n as u32],
    };
    let tensor = TensorValue::new(dtype, shape, elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

/// Extract a rank-2 (matrix) tensor from Value, returning its dimensions and
/// f64 elements in row-major order.
fn extract_matrix(
    primitive: Primitive,
    value: &Value,
) -> Result<(usize, usize, Vec<f64>, DType), EvalError> {
    let tensor = match value {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: "expected matrix (rank-2 tensor), got scalar".to_owned(),
            });
        }
    };

    if tensor.rank() != 2 {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!(
                "expected rank-2 tensor (matrix), got rank-{}",
                tensor.rank()
            ),
        });
    }

    let m = tensor.shape.dims[0] as usize;
    let n = tensor.shape.dims[1] as usize;

    let elements: Vec<f64> = tensor
        .elements
        .iter()
        .map(|lit| {
            lit.as_f64().ok_or(EvalError::TypeMismatch {
                primitive,
                detail: "expected numeric elements",
            })
        })
        .collect::<Result<_, _>>()?;

    Ok((m, n, elements, tensor.dtype))
}

/// Build a Value::Tensor from row-major f64 data and shape, emitting
/// element literals that match the declared `dtype`. Before this, the
/// helper unconditionally used `Literal::from_f64`, leaving F32/BF16/F16
/// linalg outputs (Cholesky, QR, SVD, Eigh, TriangularSolve) declaring
/// their input dtype but storing F64Bits — a dtype/element invariant
/// violation.
fn matrix_to_value(m: usize, n: usize, data: &[f64], dtype: DType) -> Result<Value, EvalError> {
    let shape = Shape {
        dims: vec![m as u32, n as u32],
    };
    if dtype == DType::F64 {
        let tensor =
            TensorValue::new_f64_values(shape, data.to_vec()).map_err(EvalError::InvalidTensor)?;
        return Ok(Value::Tensor(tensor));
    }

    let elements: Vec<Literal> = data
        .iter()
        .map(|&v| linalg_literal_from_f64(dtype, v))
        .collect();
    let tensor = TensorValue::new(dtype, shape, elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

fn linalg_literal_from_f64(dtype: DType, value: f64) -> Literal {
    match dtype {
        DType::BF16 => Literal::from_bf16_f64(value),
        DType::F16 => Literal::from_f16_f64(value),
        DType::F32 => Literal::from_f32(value as f32),
        _ => Literal::from_f64(value),
    }
}

// ── Cholesky decomposition ──────────────────────────────────────────

/// Compute the lower-triangular Cholesky factor L such that A = L * L^H.
///
/// Uses the standard row-by-row algorithm (Cholesky–Banachiewicz).
/// Input must be a Hermitian positive-definite matrix (symmetric for real).
pub(crate) fn eval_cholesky(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Cholesky;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    if !matches!(inputs[0].dtype(), DType::Complex64 | DType::Complex128) {
        let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
        return eval_cholesky_real_matrix(m, n, a, dtype);
    }

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("Cholesky requires a square matrix, got {m}x{n}"),
        });
    }

    let mut l = vec![(0.0_f64, 0.0_f64); m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut sum = (0.0_f64, 0.0_f64);
            for k in 0..j {
                sum = complex_add(sum, complex_mul(l[i * m + k], complex_conj(l[j * m + k])));
            }

            if i == j {
                let diag_val = complex_sub(a[i * m + i], sum);
                l[i * m + j] = (diag_val.0.sqrt(), 0.0);
            } else {
                let numer = complex_sub(a[i * m + j], sum);
                l[i * m + j] = complex_div(numer, l[j * m + j]);
            }
        }
    }

    complex_matrix_to_value(m, m, &l, dtype)
}

fn eval_cholesky_real_matrix(
    m: usize,
    n: usize,
    a: Vec<f64>,
    dtype: DType,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Cholesky;

    if m != n {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("Cholesky requires a square matrix, got {m}x{n}"),
        });
    }

    // Small matrices keep the scalar Cholesky–Banachiewicz kernel (bit-identical
    // to the complex path — see cholesky_real_fast_path_bit_identical_*). Large
    // matrices use the cache-blocked right-looking factorization, which is
    // GEMM-bound and far faster but reorders the FP accumulation (verified by
    // reconstruction + scalar-parity within tolerance, not bit-identity).
    let l = if m >= CHOLESKY_BLOCK_THRESHOLD {
        cholesky_real_blocked(m, &a)
    } else {
        cholesky_real_scalar(m, &a)
    };

    cholesky_real_matrix_to_value(m, m, l, dtype)
}

fn cholesky_real_matrix_to_value(
    m: usize,
    n: usize,
    data: Vec<f64>,
    dtype: DType,
) -> Result<Value, EvalError> {
    if dtype == DType::F64 {
        let shape = Shape {
            dims: vec![m as u32, n as u32],
        };
        let tensor = TensorValue::new_f64_values(shape, data).map_err(EvalError::InvalidTensor)?;
        return Ok(Value::Tensor(tensor));
    }

    matrix_to_value(m, n, &data, dtype)
}

/// Scalar lower-triangular Cholesky (Cholesky–Banachiewicz), row by row.
/// Returns `L` (m×m, strict upper triangle zero) with `A = L Lᵀ`.
fn cholesky_real_scalar(m: usize, a: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut acc = 0.0_f64;
            for k in 0..j {
                acc += l[i * m + k] * l[j * m + k];
            }
            if i == j {
                l[i * m + j] = (a[i * m + i] - acc).sqrt();
            } else {
                let numer = a[i * m + j] - acc;
                let djj = l[j * m + j];
                l[i * m + j] = (numer * djj) / (djj * djj);
            }
        }
    }
    l
}

/// Minimum scalar products before the Cholesky lower-triangle Schur update pays
/// for scoped-thread fan-out.
const CHOLESKY_SCHUR_PARALLEL_MIN_OPS: usize = 1 << 20;
/// Keep enough rows per worker that thread spawn overhead stays amortized.
const CHOLESKY_SCHUR_MIN_ROWS_PER_THREAD: usize = 96;

/// Minimum scalar products before the Cholesky panel solve pays for scoped
/// thread fan-out. Each row below the diagonal panel is independent once L11 has
/// been factored, but rows are still solved left-to-right to preserve per-row FP
/// order.
const CHOLESKY_PANEL_PARALLEL_MIN_OPS: usize = 1 << 20;
/// Keep enough panel rows per worker to amortize scoped-thread setup.
const CHOLESKY_PANEL_MIN_ROWS_PER_THREAD: usize = 96;

fn cholesky_panel_solve(
    l11_source: &[f64],
    panel_rows: &mut [f64],
    n: usize,
    j: usize,
    jb: usize,
    rem: usize,
) {
    let products = rem.saturating_mul(jb.saturating_mul(jb.saturating_sub(1)) / 2);
    let max_threads = rem.div_ceil(CHOLESKY_PANEL_MIN_ROWS_PER_THREAD);
    let available = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = if products >= CHOLESKY_PANEL_PARALLEL_MIN_OPS {
        available.min(max_threads).max(1)
    } else {
        1
    };

    if threads <= 1 {
        cholesky_panel_solve_rows(l11_source, &mut panel_rows[..rem * n], n, j, jb, rem);
        return;
    }

    let rows_per = rem.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest = &mut panel_rows[..rem * n];
        let mut row_start = 0usize;
        while row_start < rem {
            let chunk_rows = rows_per.min(rem - row_start);
            let (chunk, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            scope.spawn(move || {
                cholesky_panel_solve_rows(l11_source, chunk, n, j, jb, chunk_rows);
            });
            row_start += chunk_rows;
        }
    });
}

fn cholesky_panel_solve_rows(
    l11_source: &[f64],
    rows: &mut [f64],
    n: usize,
    j: usize,
    jb: usize,
    row_count: usize,
) {
    for local_r in 0..row_count {
        let row = &mut rows[local_r * n..local_r * n + n];
        for c in 0..jb {
            let mut s = row[j + c];
            for t in 0..c {
                s -= row[j + t] * l11_source[(j + c) * n + (j + t)];
            }
            row[j + c] = s / l11_source[(j + c) * n + (j + c)];
        }
    }
}

/// Apply the Cholesky trailing update `A22 -= L21 * L21^T` only to the lower
/// triangle. Later panels read only those lower-triangle entries, and the strict
/// upper triangle is zeroed before returning.
fn cholesky_schur_update_lower(
    a: &mut [f64],
    n: usize,
    base: usize,
    l21: &[f64],
    rem: usize,
    jb: usize,
) {
    let products = rem.saturating_mul(rem + 1) / 2;
    let ops = products.saturating_mul(jb);
    let max_threads = rem.div_ceil(CHOLESKY_SCHUR_MIN_ROWS_PER_THREAD);
    let available = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = if ops >= CHOLESKY_SCHUR_PARALLEL_MIN_OPS {
        available.min(max_threads).max(1)
    } else {
        1
    };

    if threads <= 1 {
        cholesky_schur_update_lower_rows(&mut a[base * n..], n, base, l21, jb, 0, rem);
        return;
    }

    let rows_per = rem.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest = &mut a[base * n..];
        let mut p_start = 0usize;
        while p_start < rem {
            let chunk_rows = rows_per.min(rem - p_start);
            let (chunk, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            scope.spawn(move || {
                cholesky_schur_update_lower_rows(chunk, n, base, l21, jb, p_start, chunk_rows);
            });
            p_start += chunk_rows;
        }
    });
}

fn cholesky_schur_update_lower_rows(
    rows: &mut [f64],
    n: usize,
    base: usize,
    l21: &[f64],
    jb: usize,
    p_start: usize,
    row_count: usize,
) {
    for local_p in 0..row_count {
        let p = p_start + local_p;
        let p_row = &l21[p * jb..p * jb + jb];
        let a_row = &mut rows[local_p * n..local_p * n + n];
        let mut q = 0usize;
        while q + 4 <= p + 1 {
            let q0 = &l21[q * jb..q * jb + jb];
            let q1 = &l21[(q + 1) * jb..(q + 1) * jb + jb];
            let q2 = &l21[(q + 2) * jb..(q + 2) * jb + jb];
            let q3 = &l21[(q + 3) * jb..(q + 3) * jb + jb];
            let dots = cholesky_schur_dot4(p_row, q0, q1, q2, q3);
            a_row[base + q] -= dots[0];
            a_row[base + q + 1] -= dots[1];
            a_row[base + q + 2] -= dots[2];
            a_row[base + q + 3] -= dots[3];
            q += 4;
        }
        while q <= p {
            let q_row = &l21[q * jb..q * jb + jb];
            let dot = cholesky_schur_dot(p_row, q_row);
            a_row[base + q] -= dot;
            q += 1;
        }
    }
}

const CHOLESKY_SCHUR_DOT_LANES: usize = 8;
type CholeskySchurF64xN = std::simd::Simd<f64, CHOLESKY_SCHUR_DOT_LANES>;

#[inline]
fn cholesky_schur_f64x8(values: &[f64]) -> CholeskySchurF64xN {
    CholeskySchurF64xN::from_array([
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
    ])
}

#[inline]
fn cholesky_schur_dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());

    let mut c = 0usize;
    let mut acc = CholeskySchurF64xN::splat(0.0);
    while c + CHOLESKY_SCHUR_DOT_LANES <= lhs.len() {
        acc += cholesky_schur_f64x8(&lhs[c..]) * cholesky_schur_f64x8(&rhs[c..]);
        c += CHOLESKY_SCHUR_DOT_LANES;
    }

    let lanes = acc.as_array();
    let mut dot =
        lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    while c < lhs.len() {
        dot += lhs[c] * rhs[c];
        c += 1;
    }
    dot
}

#[inline]
fn cholesky_schur_dot4(
    lhs: &[f64],
    rhs0: &[f64],
    rhs1: &[f64],
    rhs2: &[f64],
    rhs3: &[f64],
) -> [f64; 4] {
    debug_assert_eq!(lhs.len(), rhs0.len());
    debug_assert_eq!(lhs.len(), rhs1.len());
    debug_assert_eq!(lhs.len(), rhs2.len());
    debug_assert_eq!(lhs.len(), rhs3.len());

    let mut c = 0usize;
    let mut acc0 = CholeskySchurF64xN::splat(0.0);
    let mut acc1 = CholeskySchurF64xN::splat(0.0);
    let mut acc2 = CholeskySchurF64xN::splat(0.0);
    let mut acc3 = CholeskySchurF64xN::splat(0.0);
    while c + CHOLESKY_SCHUR_DOT_LANES <= lhs.len() {
        let lv = cholesky_schur_f64x8(&lhs[c..]);
        acc0 += lv * cholesky_schur_f64x8(&rhs0[c..]);
        acc1 += lv * cholesky_schur_f64x8(&rhs1[c..]);
        acc2 += lv * cholesky_schur_f64x8(&rhs2[c..]);
        acc3 += lv * cholesky_schur_f64x8(&rhs3[c..]);
        c += CHOLESKY_SCHUR_DOT_LANES;
    }

    let lanes0 = acc0.as_array();
    let lanes1 = acc1.as_array();
    let lanes2 = acc2.as_array();
    let lanes3 = acc3.as_array();
    let mut dots = [
        lanes0[0]
            + lanes0[1]
            + lanes0[2]
            + lanes0[3]
            + lanes0[4]
            + lanes0[5]
            + lanes0[6]
            + lanes0[7],
        lanes1[0]
            + lanes1[1]
            + lanes1[2]
            + lanes1[3]
            + lanes1[4]
            + lanes1[5]
            + lanes1[6]
            + lanes1[7],
        lanes2[0]
            + lanes2[1]
            + lanes2[2]
            + lanes2[3]
            + lanes2[4]
            + lanes2[5]
            + lanes2[6]
            + lanes2[7],
        lanes3[0]
            + lanes3[1]
            + lanes3[2]
            + lanes3[3]
            + lanes3[4]
            + lanes3[5]
            + lanes3[6]
            + lanes3[7],
    ];
    while c < lhs.len() {
        let lv = lhs[c];
        dots[0] += lv * rhs0[c];
        dots[1] += lv * rhs1[c];
        dots[2] += lv * rhs2[c];
        dots[3] += lv * rhs3[c];
        c += 1;
    }
    dots
}

/// Block size for the right-looking Cholesky panel.
const CHOLESKY_BLOCK_SIZE: usize = 128;
/// Smaller panel for mid-size matrices where the 128-wide trailing update spills
/// enough cache to lose, while 512+ still prefers the wider panel.
const CHOLESKY_MIDSIZE_BLOCK_SIZE: usize = 64;
/// Matrices at least this large use the blocked Cholesky; below it the scalar
/// kernel runs (keeping the small-n bit-identical invariant).
const CHOLESKY_BLOCK_THRESHOLD: usize = 256;

/// Cache-blocked right-looking Cholesky: A = L Lᵀ, returning lower-triangular L
/// (strict upper zero). For each diagonal block it factors the block (scalar),
/// solves the sub-diagonal panel against L11ᵀ, then applies the symmetric Schur
/// update A22 -= L21 L21ᵀ via the cache-blocked GEMM (`matmul_2d`). The trailing
/// update is the O(n³) cost and runs at GEMM speed instead of strided scalar
/// dot products. Not bit-identical to the scalar kernel (GEMM reorders the sum),
/// but numerically equivalent — verified by reconstruction + scalar parity.
fn cholesky_real_blocked(n: usize, a_in: &[f64]) -> Vec<f64> {
    let mut a = a_in.to_vec(); // lower triangle becomes L in place
    let nb = if n < 512 {
        CHOLESKY_MIDSIZE_BLOCK_SIZE
    } else {
        CHOLESKY_BLOCK_SIZE
    };

    let mut j = 0;
    while j < n {
        let jb = nb.min(n - j);

        // (a) Factor the jb×jb diagonal block (already-updated Schur complement).
        for c in 0..jb {
            let mut d = a[(j + c) * n + (j + c)];
            for t in 0..c {
                let v = a[(j + c) * n + (j + t)];
                d -= v * v;
            }
            let diag = d.sqrt();
            a[(j + c) * n + (j + c)] = diag;
            for r in (c + 1)..jb {
                let mut s = a[(j + r) * n + (j + c)];
                for t in 0..c {
                    s -= a[(j + r) * n + (j + t)] * a[(j + c) * n + (j + t)];
                }
                a[(j + r) * n + (j + c)] = s / diag;
            }
        }

        let rem = n - (j + jb);
        if rem > 0 {
            // (b) Panel solve: L21 = A21 · L11^{-T} (rows j+jb..n, cols j..j+jb).
            let panel_start = (j + jb) * n;
            let (l11_source, panel_rows) = a.split_at_mut(panel_start);
            cholesky_panel_solve(l11_source, panel_rows, n, j, jb, rem);

            // (c) Trailing symmetric update A22 -= L21 · L21ᵀ via blocked GEMM.
            let mut l21 = vec![0.0_f64; rem * jb];
            for p in 0..rem {
                for c in 0..jb {
                    let v = a[(j + jb + p) * n + (j + c)];
                    l21[p * jb + c] = v;
                }
            }
            cholesky_schur_update_lower(&mut a, n, j + jb, &l21, rem, jb);
        }

        j += jb;
    }

    // Zero the strict upper triangle to match the scalar kernel's L layout.
    for i in 0..n {
        for jj in (i + 1)..n {
            a[i * n + jj] = 0.0;
        }
    }
    a
}

// ── Triangular solve ────────────────────────────────────────────────

/// Solve a triangular linear system: find X such that A * X = B (or A^T * X = B).
///
/// By default, A is assumed lower-triangular and `left_side=true`.
/// Params:
///   - `lower` (default "true"): if "true", A is lower-triangular; else upper.
///   - `transpose_a` (default "false"): if "true", solve A^T X = B instead.
///   - `unit_diagonal` (default "false"): if "true", diagonal of A is assumed 1.
///
/// Batched triangular solve `op(A)·X = B` for ALL `n_b` RHS columns at once
/// (`X` starts as `B`, n×n_b row-major, solved in place). Reads each `A` entry
/// ONCE and applies it across every column (inner column loop auto-vectorizes),
/// so the triangular factor streams from cache once instead of once per column.
/// BIT-IDENTICAL to a per-column solve: each column's ascending-`k` fold is
/// unchanged — only the loop nesting differs. `forward` = substitute rows in
/// ascending order (`lower != transpose`); `transpose` reads `A[k][i]` instead of
/// `A[i][k]`; the diagonal is `A[i][i]` in every case.
/// REAL fast path for [`batched_tri_solve`]: when A and B are real, the per-element
/// complex `(ac−bd, ad+bc)` mul / `ac/c²` div degenerate to real `a·x` / `x/d` but
/// still cost ~4× the mults and read 2× the bytes. This does the substitution in
/// pure f64 across all `n_b` columns at once — ~2-4× faster, and `x/d` is also the
/// exact real division JAX uses (the complex path's `a·c/c²` double-rounds). Same
/// loop structure / ascending-`k` fold as the complex version.
fn batched_tri_solve_real(
    a: &[f64],
    x: &mut [f64],
    n: usize,
    n_b: usize,
    forward: bool,
    transpose: bool,
    unit_diagonal: bool,
) {
    let a_at = |i: usize, k: usize| -> f64 {
        if transpose {
            a[k * n + i]
        } else {
            a[i * n + k]
        }
    };
    let solve_row = |i: usize, ks: std::ops::Range<usize>, x: &mut [f64]| {
        for k in ks {
            let aik = a_at(i, k);
            if k < i {
                let (head, tail) = x.split_at_mut(i * n_b);
                let xk = &head[k * n_b..k * n_b + n_b];
                let xi = &mut tail[..n_b];
                for col in 0..n_b {
                    xi[col] -= aik * xk[col];
                }
            } else {
                let (head, tail) = x.split_at_mut(k * n_b);
                let xi = &mut head[i * n_b..i * n_b + n_b];
                let xk = &tail[..n_b];
                for col in 0..n_b {
                    xi[col] -= aik * xk[col];
                }
            }
        }
        let d = if unit_diagonal { 1.0 } else { a[i * n + i] };
        let xi = &mut x[i * n_b..i * n_b + n_b];
        for v in xi.iter_mut() {
            *v /= d;
        }
    };
    if forward {
        for i in 0..n {
            solve_row(i, 0..i, x);
        }
    } else {
        for i in (0..n).rev() {
            solve_row(i, (i + 1)..n, x);
        }
    }
}

fn batched_tri_solve(
    a: &[(f64, f64)],
    x: &mut [(f64, f64)],
    n: usize,
    n_b: usize,
    forward: bool,
    transpose: bool,
    unit_diagonal: bool,
) {
    let one = (1.0, 0.0);
    let a_at = |i: usize, k: usize| -> (f64, f64) {
        if transpose {
            a[k * n + i]
        } else {
            a[i * n + k]
        }
    };
    // Process one pivot row `i` (subtract its already-solved neighbours `k`, then
    // divide by the diagonal) across all columns.
    let solve_row = |i: usize, ks: &[usize], x: &mut [(f64, f64)]| {
        for &k in ks {
            let aik = a_at(i, k);
            // x[i][:] and x[k][:] are disjoint rows; split so both can be borrowed.
            if k < i {
                let (head, tail) = x.split_at_mut(i * n_b);
                let xk = &head[k * n_b..k * n_b + n_b];
                let xi = &mut tail[..n_b];
                for col in 0..n_b {
                    xi[col] = complex_sub(xi[col], complex_mul(aik, xk[col]));
                }
            } else {
                let (head, tail) = x.split_at_mut(k * n_b);
                let xi = &mut head[i * n_b..i * n_b + n_b];
                let xk = &tail[..n_b];
                for col in 0..n_b {
                    xi[col] = complex_sub(xi[col], complex_mul(aik, xk[col]));
                }
            }
        }
        let diag = if unit_diagonal { one } else { a[i * n + i] };
        let xi = &mut x[i * n_b..i * n_b + n_b];
        for v in xi.iter_mut() {
            *v = complex_div(*v, diag);
        }
    };
    if forward {
        for i in 0..n {
            let ks: Vec<usize> = (0..i).collect();
            solve_row(i, &ks, x);
        }
    } else {
        for i in (0..n).rev() {
            let ks: Vec<usize> = ((i + 1)..n).collect();
            solve_row(i, &ks, x);
        }
    }
}

pub(crate) fn eval_triangular_solve(
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::TriangularSolve;

    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let (m_a, n_a, a, dtype_a) = extract_complex_matrix(primitive, &inputs[0])?;
    let (m_b, n_b, b, dtype_b) = extract_complex_matrix(primitive, &inputs[1])?;

    if m_a != n_a {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("A must be square, got {m_a}x{n_a}"),
        });
    }

    if m_a != m_b {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: Shape {
                dims: vec![m_a as u32, n_a as u32],
            },
            right: Shape {
                dims: vec![m_b as u32, n_b as u32],
            },
        });
    }

    let lower = params.get("lower").is_none_or(|v| v.trim() != "false");
    let transpose_a = params
        .get("transpose_a")
        .is_some_and(|v| v.trim() == "true");
    let unit_diagonal = params
        .get("unit_diagonal")
        .is_some_and(|v| v.trim() == "true");

    let n = m_a;
    let forward = lower != transpose_a;
    let out_dtype = promote_dtype(dtype_a, dtype_b);

    // REAL fast path: when neither operand is complex, do the substitution in pure
    // f64 (≈2-4× the complex path: 1 vs 4 mults/step, half the bytes). Output reuses
    // the shared builder via (v, 0) tuples. JAX's triangular_solve does not raise for
    // a zero/near-zero diagonal — `x/d` yields inf/nan (singular) or a finite large
    // value (near-singular), matching jnp (NumPy raises).
    let is_real = !matches!(dtype_a, DType::Complex64 | DType::Complex128)
        && !matches!(dtype_b, DType::Complex64 | DType::Complex128);
    if is_real {
        let ar: Vec<f64> = a.iter().map(|t| t.0).collect();
        let mut xr: Vec<f64> = b.iter().map(|t| t.0).collect();
        batched_tri_solve_real(&ar, &mut xr, n, n_b, forward, transpose_a, unit_diagonal);
        let x: Vec<(f64, f64)> = xr.into_iter().map(|v| (v, 0.0)).collect();
        return complex_matrix_to_value(n, n_b, &x, out_dtype);
    }

    // Complex path: solve all n_b RHS columns together, streaming A once.
    let mut x = b.clone();
    batched_tri_solve(&a, &mut x, n, n_b, forward, transpose_a, unit_diagonal);
    complex_matrix_to_value(n, n_b, &x, out_dtype)
}

// ── QR decomposition ───────────────────────────────────────────────

/// Compute the thin QR decomposition A = Q R using Householder reflections.
///
/// Returns `[Q, R]` where Q is m×k unitary and R is k×n upper-triangular,
/// with k = min(m, n). When `full_matrices=true`, Q is m×m and R is m×n.
pub(crate) fn eval_qr(
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::Qr;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");

    // Real input → the (blocked) real Householder QR. Householder handles zero
    // pivots/columns fine, so this runs for ALL real matrices (the prior
    // `all elements != 0` gate needlessly sent any real matrix containing a zero
    // onto the ~3-4× slower complex path — see qr_real_path_handles_zeros_*).
    if !matches!(inputs[0].dtype(), DType::Complex64 | DType::Complex128) {
        let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
        return eval_qr_real_matrix(m, n, a, dtype, full_matrices);
    }

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    let k = m.min(n);
    let zero = (0.0, 0.0);
    let one = (1.0, 0.0);

    let mut r = a;
    let mut v_store: Vec<Vec<(f64, f64)>> = Vec::with_capacity(k);
    let mut tau_store: Vec<(f64, f64)> = Vec::with_capacity(k);

    for j in 0..k {
        let mut v: Vec<(f64, f64)> = (j..m).map(|i| r[i * n + j]).collect();
        let norm_v = v.iter().map(|x| x.0 * x.0 + x.1 * x.1).sum::<f64>().sqrt();

        let v0_abs = complex_abs(v[0]);
        let alpha = if v0_abs > 0.0 {
            let phase = (v[0].0 / v0_abs, v[0].1 / v0_abs);
            (-norm_v * phase.0, -norm_v * phase.1)
        } else {
            (-norm_v, 0.0)
        };
        v[0] = complex_sub(v[0], alpha);
        let v_norm_sq: f64 = v.iter().map(|x| x.0 * x.0 + x.1 * x.1).sum();

        if v_norm_sq > f64::EPSILON * 1e4 {
            let tau = (2.0 / v_norm_sq, 0.0);

            for col in j..n {
                let mut dot = zero;
                for (vi, row) in v.iter().zip(j..m) {
                    dot = complex_add(dot, complex_mul(complex_conj(*vi), r[row * n + col]));
                }
                let tau_dot = complex_mul(tau, dot);
                for (vi, row) in v.iter().zip(j..m) {
                    r[row * n + col] = complex_sub(r[row * n + col], complex_mul(*vi, tau_dot));
                }
            }

            v_store.push(v);
            tau_store.push(tau);
        } else {
            v_store.push(vec![zero; m - j]);
            tau_store.push(zero);
        }
    }

    let q_cols = if full_matrices { m } else { k };
    let mut q = vec![zero; m * q_cols];

    for i in 0..q_cols.min(m) {
        q[i * q_cols + i] = one;
    }

    for j in (0..k).rev() {
        let v = &v_store[j];
        let tau = tau_store[j];
        if complex_abs(tau) < f64::EPSILON {
            continue;
        }

        for col in j..q_cols {
            let mut dot = zero;
            for (vi, row) in v.iter().zip(j..m) {
                dot = complex_add(dot, complex_mul(complex_conj(*vi), q[row * q_cols + col]));
            }
            let tau_dot = complex_mul(tau, dot);
            for (vi, row) in v.iter().zip(j..m) {
                q[row * q_cols + col] =
                    complex_sub(q[row * q_cols + col], complex_mul(*vi, tau_dot));
            }
        }
    }

    let r_rows = if full_matrices { m } else { k };
    let mut r_out = vec![zero; r_rows * n];
    for i in 0..r_rows {
        for jj in i..n {
            r_out[i * n + jj] = r[i * n + jj];
        }
    }

    for i in 0..k {
        let diag = r_out[i * n + i];
        let diag_abs = complex_abs(diag);
        if diag_abs > f64::EPSILON {
            let phase = complex_div(diag, (diag_abs, 0.0));
            let phase_conj = complex_conj(phase);
            for jj in 0..n {
                r_out[i * n + jj] = complex_mul(phase_conj, r_out[i * n + jj]);
            }
            for row in 0..m {
                q[row * q_cols + i] = complex_mul(q[row * q_cols + i], phase);
            }
        }
    }

    let q_val = complex_matrix_to_value(m, q_cols, &q, dtype)?;
    let r_val = complex_matrix_to_value(r_rows, n, &r_out, dtype)?;

    Ok(vec![q_val, r_val])
}

fn eval_qr_real_matrix(
    m: usize,
    n: usize,
    a: Vec<f64>,
    dtype: DType,
    full_matrices: bool,
) -> Result<Vec<Value>, EvalError> {
    // Gate the WY-blocked path on the trailing matrix spilling L3 repeatedly under the
    // BLAS-2 rank-1 update (bead wpjbg): below QR_BLOCK_MIN the contiguous cache-friendly
    // scalar reflector loop is memory-bound but L3-resident and beats the blocked GEMMs;
    // it also preserves the small-n bit-identity goldens.
    let blocked = m.min(n) >= QR_BLOCK_MIN;
    eval_qr_real_matrix_impl(m, n, a, dtype, full_matrices, blocked)
}

/// Factor one Householder reflector for column `j` (rows `j..m`) and apply it to
/// columns `[j, col_end)` of the row-major `r` (stride `n`). Stores the reflector in
/// the flat packed `v_store` (slot at `qr_reflector_offset(m, j)`) and its scale in
/// `tau_store[j]`. `col_end == n` reproduces the original scalar factor loop exactly.
#[allow(clippy::too_many_arguments)]
fn qr_factor_col(
    r: &mut [f64],
    n: usize,
    m: usize,
    j: usize,
    col_end: usize,
    v_scratch: &mut [f64],
    v_store: &mut [f64],
    tau_store: &mut [f64],
) {
    let v_len = m - j;
    for row_offset in 0..v_len {
        v_scratch[row_offset] = r[(j + row_offset) * n + j];
    }
    let v = &mut v_scratch[..v_len];
    let norm_v = v.iter().map(|x| x * x).sum::<f64>().sqrt();

    let v0_abs = (v[0] * v[0]).sqrt();
    let alpha = if v0_abs > 0.0 {
        let phase = v[0] / v0_abs;
        -norm_v * phase
    } else {
        -norm_v
    };
    v[0] -= alpha;
    let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();

    if v_norm_sq > f64::EPSILON * 1e4 {
        let tau = 2.0 / v_norm_sq;
        apply_real_householder_columns(r, n, j, j, col_end, v, tau);
        let v_base = qr_reflector_offset(m, j);
        v_store[v_base..v_base + v_len].copy_from_slice(v);
        tau_store[j] = tau;
    } else {
        // v_store slot stays zeroed (pre-initialized) so the block reflector treats
        // this reflector as the identity, matching the scalar path's tau==0 skip.
        tau_store[j] = 0.0;
    }
}

fn eval_qr_real_matrix_impl(
    m: usize,
    n: usize,
    a: Vec<f64>,
    dtype: DType,
    full_matrices: bool,
    blocked: bool,
) -> Result<Vec<Value>, EvalError> {
    let k = m.min(n);

    let mut r = a;
    let mut v_scratch = vec![0.0_f64; m];
    let mut v_store = vec![0.0_f64; qr_reflector_packed_len(m, k)];
    let mut tau_store = vec![0.0_f64; k];
    let mut panel_ts: Vec<(usize, usize, Vec<f64>)> = Vec::new();

    if blocked {
        // R = QᵀA: factor each QR_BLOCK-column panel (reflectors applied only within the
        // panel's own columns), build the compact-WY T, then apply the panel block
        // reflector M = H_{p+b-1}…H_p = I − V Tᵀ Vᵀ to the trailing columns via two GEMMs.
        let mut p = 0;
        while p < k {
            let b = QR_BLOCK.min(k - p);
            for j in p..p + b {
                qr_factor_col(
                    &mut r,
                    n,
                    m,
                    j,
                    p + b,
                    &mut v_scratch,
                    &mut v_store,
                    &mut tau_store,
                );
            }
            let t = qr_compact_wy_t(&v_store, &tau_store, m, p, b);
            if p + b < n {
                qr_block_apply(&mut r, m, n, &v_store, &t, p, b, p + b, n, true);
            }
            panel_ts.push((p, b, t));
            p += b;
        }
    } else {
        for j in 0..k {
            qr_factor_col(
                &mut r,
                n,
                m,
                j,
                n,
                &mut v_scratch,
                &mut v_store,
                &mut tau_store,
            );
        }
    }

    let q_cols = if full_matrices { m } else { k };
    let mut q = vec![0.0; m * q_cols];

    for i in 0..q_cols.min(m) {
        q[i * q_cols + i] = 1.0;
    }

    if blocked {
        // Q = H_0…H_{k-1} applied to I: apply each panel block reflector I − V T Vᵀ
        // (note: T, not Tᵀ — the forward product, opposite of the R update) backward,
        // over columns [p, q_cols) (columns < p stay identity, so they are skipped).
        for (p, b, t) in panel_ts.iter().rev() {
            qr_block_apply(&mut q, m, q_cols, &v_store, t, *p, *b, *p, q_cols, false);
        }
    } else {
        for j in (0..k).rev() {
            let tau = tau_store[j];
            if (tau * tau).sqrt() < f64::EPSILON {
                continue;
            }

            let v_len = m - j;
            let v_base = qr_reflector_offset(m, j);
            let v = &v_store[v_base..v_base + v_len];
            apply_real_householder_columns(&mut q, q_cols, j, j, q_cols, v, tau);
        }
    }

    let r_rows = if full_matrices { m } else { k };
    let mut r_out = vec![0.0; r_rows * n];
    for i in 0..r_rows {
        for jj in i..n {
            r_out[i * n + jj] = r[i * n + jj];
        }
    }

    for i in 0..k {
        let diag = r_out[i * n + i];
        let diag_abs = (diag * diag).sqrt();
        if diag_abs > f64::EPSILON {
            let phase = (diag * diag_abs) / (diag_abs * diag_abs);
            for jj in 0..n {
                r_out[i * n + jj] *= phase;
            }
            for row in 0..m {
                q[row * q_cols + i] *= phase;
            }
        }
    }
    for i in 1..r_rows {
        for jj in 0..i.min(n) {
            r_out[i * n + jj] = 0.0;
        }
    }

    let q_val = matrix_to_value(m, q_cols, &q, dtype)?;
    let r_val = matrix_to_value(r_rows, n, &r_out, dtype)?;

    Ok(vec![q_val, r_val])
}

#[inline]
fn qr_reflector_packed_len(rows: usize, reflectors: usize) -> usize {
    reflectors * rows - (reflectors * reflectors.saturating_sub(1)) / 2
}

#[inline]
fn qr_reflector_offset(rows: usize, reflector: usize) -> usize {
    reflector * rows - (reflector * reflector.saturating_sub(1)) / 2
}

const QR_REFLECTOR_COL_TILE: usize = 16;

fn apply_real_householder_columns(
    matrix: &mut [f64],
    row_stride: usize,
    row_start: usize,
    col_start: usize,
    col_end: usize,
    v: &[f64],
    tau: f64,
) {
    debug_assert!(col_start <= col_end);
    debug_assert!(col_end <= row_stride);

    let mut col = col_start;
    while col < col_end {
        let width = QR_REFLECTOR_COL_TILE.min(col_end - col);
        let mut dots = [0.0_f64; QR_REFLECTOR_COL_TILE];

        let mut row_offset = 0;
        while row_offset < v.len() {
            let vi = v[row_offset];
            let row_base = (row_start + row_offset) * row_stride + col;
            let mut lane = 0;
            while lane < width {
                dots[lane] += vi * matrix[row_base + lane];
                lane += 1;
            }
            row_offset += 1;
        }

        let mut lane = 0;
        while lane < width {
            dots[lane] *= tau;
            lane += 1;
        }

        let mut row_offset = 0;
        while row_offset < v.len() {
            let vi = v[row_offset];
            let row_base = (row_start + row_offset) * row_stride + col;
            let mut lane = 0;
            while lane < width {
                matrix[row_base + lane] -= vi * dots[lane];
                lane += 1;
            }
            row_offset += 1;
        }

        col += width;
    }
}

/// Panel width for the WY-blocked QR (b reflectors fused into one block reflector).
const QR_BLOCK: usize = 32;
/// Minimum `min(m,n)` to take the WY-blocked path. Below it the scalar reflector loop
/// (cache-friendly contiguous rank-1 update, L3-resident) wins and the bit-identity
/// goldens are preserved (they all use small n); at/above it the trailing matrix spills
/// L3 repeatedly under the BLAS-2 rank-1 update, so the in-place rank-b block apply —
/// which sweeps the trailing matrix only twice per panel instead of `b` times — wins:
/// measured (same-binary A/B) 1.67x at n=1024, 1.05x at n=2048, and 1.25x at
/// n=4096. See bead wpjbg / project_qr_matmul2d_unsuitable and frankenjax-p1vbf.59.
const QR_BLOCK_MIN: usize = 1024;

/// Compact-WY upper-triangular `T` (b×b) for the panel of `b` Householder reflectors
/// starting at global column `p`: the block reflector `H_p…H_{p+b-1} = I − V T Vᵀ`
/// (V is the m×b trapezoidal reflector matrix; reflector `p+j` lives in the flat
/// `v_store` slot at `qr_reflector_offset(m, p+j)`, indexed from row `p+j`).
fn qr_compact_wy_t(v_store: &[f64], tau_store: &[f64], m: usize, p: usize, b: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; b * b];
    for j in 0..b {
        let tau_j = tau_store[p + j];
        t[j * b + j] = tau_j;
        if tau_j == 0.0 || j == 0 {
            continue;
        }
        let vj_base = qr_reflector_offset(m, p + j);
        let vj_len = m - (p + j);
        // w[i] = v_{p+i} · v_{p+j} over the overlapping rows ≥ p+j.
        let mut w = vec![0.0_f64; j];
        for (i, wi) in w.iter_mut().enumerate() {
            let vi_base = qr_reflector_offset(m, p + i);
            let offset = j - i; // v_{p+i}[offset+tt] aligns with v_{p+j}[tt] (row p+j+tt)
            let mut s = 0.0;
            for tt in 0..vj_len {
                s += v_store[vi_base + offset + tt] * v_store[vj_base + tt];
            }
            *wi = s;
        }
        // T[0:j, j] = −tau_j · T[0:j,0:j] · w  (T upper-triangular).
        for i in 0..j {
            let mut s = 0.0;
            for l in i..j {
                s += t[i * b + l] * w[l];
            }
            t[i * b + j] = -tau_j * s;
        }
    }
    t
}

/// Apply the panel `(p, b)` block reflector to columns `[col_start, col_end)` of the
/// row-major `c` (m rows, `c_cols` cols) as two cache-blocked GEMMs `C −= V (W₂)` with
/// `W₂ = T(ᵀ) (Vᵀ C)`. `transpose_t == true` applies `I − V Tᵀ Vᵀ` (the R update:
/// M = H_{p+b-1}…H_p = Qᵀ_panel); `false` applies `I − V T Vᵀ` (the Q update:
/// H_p…H_{p+b-1}). Reassociates vs the per-reflector loop (tolerance-equal, not bit-
/// identical — gated to the large-n path).
#[allow(clippy::too_many_arguments)]
fn qr_block_apply(
    c: &mut [f64],
    m: usize,
    c_cols: usize,
    v_store: &[f64],
    t: &[f64],
    p: usize,
    b: usize,
    col_start: usize,
    col_end: usize,
    transpose_t: bool,
) {
    let cols = col_end - col_start;
    if cols == 0 {
        return;
    }
    // W1 = Vᵀ C_sub (b×cols), accumulated IN PLACE from the strided row-major trailing
    // matrix — no contiguous copy of C, no GEMM pack/spawn (those O(n³) extra memory
    // passes are exactly why the matmul_2d block apply lost, bead wpjbg). Reflector p+lj
    // has support rows ≥ p+lj; sweeping the trailing rows once, each row is read once and
    // fanned into the active reflectors' W1 rows (a cache-resident b×cols buffer).
    let mut w1 = vec![0.0_f64; b * cols];
    for row in p..m {
        let crow = &c[row * c_cols + col_start..row * c_cols + col_start + cols];
        let lj_max = (row - p + 1).min(b);
        for lj in 0..lj_max {
            let base = p + lj;
            let vval = v_store[qr_reflector_offset(m, base) + (row - base)];
            if vval == 0.0 {
                continue;
            }
            let wrow = &mut w1[lj * cols..lj * cols + cols];
            for col in 0..cols {
                wrow[col] += vval * crow[col];
            }
        }
    }
    let mut w2 = vec![0.0_f64; b * cols];
    if transpose_t {
        // W2 = Tᵀ W1 : W2[i] = Σ_{l≤i} T[l][i] W1[l]  (Tᵀ lower-triangular).
        for i in 0..b {
            for l in 0..=i {
                let tli = t[l * b + i];
                if tli != 0.0 {
                    for col in 0..cols {
                        w2[i * cols + col] += tli * w1[l * cols + col];
                    }
                }
            }
        }
    } else {
        // W2 = T W1 : W2[i] = Σ_{l≥i} T[i][l] W1[l]  (T upper-triangular).
        for i in 0..b {
            for l in i..b {
                let til = t[i * b + l];
                if til != 0.0 {
                    for col in 0..cols {
                        w2[i * cols + col] += til * w1[l * cols + col];
                    }
                }
            }
        }
    }
    // C_sub −= V W2, again IN PLACE on the strided trailing matrix: sweep the rows once,
    // each row gets the rank-b correction Σ_lj V[row,lj]·W2[lj,:] (W2 is b×cols, cache-
    // resident). Reads/writes each C row exactly once.
    for row in p..m {
        let lj_max = (row - p + 1).min(b);
        let crow = &mut c[row * c_cols + col_start..row * c_cols + col_start + cols];
        for lj in 0..lj_max {
            let base = p + lj;
            let vval = v_store[qr_reflector_offset(m, base) + (row - base)];
            if vval == 0.0 {
                continue;
            }
            let w2row = &w2[lj * cols..lj * cols + cols];
            for col in 0..cols {
                crow[col] -= vval * w2row[col];
            }
        }
    }
}

/// Bench-only entry: run the real QR with the WY-blocked path forced on or off, so a
/// same-binary A/B can isolate the blocked GEMM passes from the scalar reflector loop
/// at a fixed size (the public `Qr` primitive auto-gates on `QR_BLOCK_MIN`).
#[doc(hidden)]
pub fn qr_real_bench(a: Vec<f64>, m: usize, n: usize, blocked: bool) -> Vec<Value> {
    eval_qr_real_matrix_impl(m, n, a, DType::F64, false, blocked).expect("qr bench")
}

// ── LU Decomposition ───────────────────────────────────────────────

/// Compute LU decomposition with partial pivoting: PA = LU
///
/// Returns `[lu, pivots, permutation]` where:
///   - lu is m×n containing both L (unit lower triangular) and U (upper triangular)
///     L is stored below the diagonal, U on and above diagonal
///   - pivots is length min(m,n) containing pivot indices
///   - permutation is length m containing the row permutation
pub(crate) fn eval_lu(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::Lu;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    if !matches!(inputs[0].dtype(), DType::Complex64 | DType::Complex128) {
        let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
        return eval_lu_real_matrix(m, n, a, dtype);
    }

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    eval_lu_complex_matrix(m, n, a, dtype)
}

fn eval_lu_complex_matrix(
    m: usize,
    n: usize,
    a: Vec<(f64, f64)>,
    dtype: DType,
) -> Result<Vec<Value>, EvalError> {
    let k = m.min(n);

    let mut lu = a;
    let mut pivots: Vec<i64> = (0..k as i64).collect();
    let mut perm: Vec<i64> = (0..m as i64).collect();

    for col in 0..k {
        let mut max_val = complex_abs(lu[col * n + col]);
        let mut max_row = col;
        for row in (col + 1)..m {
            let val = complex_abs(lu[row * n + col]);
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        pivots[col] = max_row as i64;

        if max_row != col {
            perm.swap(col, max_row);
            for j in 0..n {
                lu.swap(col * n + j, max_row * n + j);
            }
        }

        let diag = lu[col * n + col];
        if complex_abs(diag) < f64::EPSILON * 1e-10 {
            continue;
        }

        for row in (col + 1)..m {
            let factor = complex_div(lu[row * n + col], diag);
            lu[row * n + col] = factor;
            for j in (col + 1)..n {
                lu[row * n + j] =
                    complex_sub(lu[row * n + j], complex_mul(factor, lu[col * n + j]));
            }
        }
    }

    let lu_val = complex_matrix_to_value(m, n, &lu, dtype)?;

    // Upstream JAX returns pivots and permutation as int32 (see lax.linalg.lu)
    let pivots_lits: Vec<Literal> = pivots.iter().map(|&p| Literal::I64(p)).collect();
    let pivots_shape = Shape {
        dims: vec![k as u32],
    };
    let pivots_val = Value::Tensor(
        TensorValue::new(DType::I32, pivots_shape, pivots_lits)
            .map_err(EvalError::InvalidTensor)?,
    );

    let perm_lits: Vec<Literal> = perm.iter().map(|&p| Literal::I64(p)).collect();
    let perm_shape = Shape {
        dims: vec![m as u32],
    };
    let perm_val = Value::Tensor(
        TensorValue::new(DType::I32, perm_shape, perm_lits).map_err(EvalError::InvalidTensor)?,
    );

    Ok(vec![lu_val, pivots_val, perm_val])
}

/// Panel width for the cache-blocked right-looking LU.
const LU_BLOCK_SIZE: usize = 128;
/// Matrices with `min(m, n)` at least this large take the blocked LU; below it
/// the scalar kernel runs (keeping the small-n bit-identical invariant and the
/// conformance goldens, which only cover small matrices).
const LU_BLOCK_THRESHOLD: usize = 256;

/// Scalar right-looking LU with partial pivoting, factoring `lu` in place and
/// returning `(pivots, perm)`. This is the reference kernel: each output element
/// accumulates its rank-1 updates in column order, bit-for-bit matching the
/// complex-zero-imag path (see lu_real_fast_path_bit_identical_to_complex…).
fn lu_factor_real_scalar(lu: &mut [f64], m: usize, n: usize) -> (Vec<i64>, Vec<i64>) {
    let k = m.min(n);
    let mut pivots: Vec<i64> = (0..k as i64).collect();
    let mut perm: Vec<i64> = (0..m as i64).collect();

    for col in 0..k {
        let mut max_val = (lu[col * n + col] * lu[col * n + col]).sqrt();
        let mut max_row = col;
        for row in (col + 1)..m {
            let candidate = lu[row * n + col];
            let val = (candidate * candidate).sqrt();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        pivots[col] = max_row as i64;

        if max_row != col {
            perm.swap(col, max_row);
            for j in 0..n {
                lu.swap(col * n + j, max_row * n + j);
            }
        }

        let diag = lu[col * n + col];
        if (diag * diag).sqrt() < f64::EPSILON * 1e-10 {
            continue;
        }

        for row in (col + 1)..m {
            let factor = (lu[row * n + col] * diag) / (diag * diag);
            lu[row * n + col] = factor;
            for j in (col + 1)..n {
                lu[row * n + j] -= factor * lu[col * n + j];
            }
        }
    }

    (pivots, perm)
}

/// Cache-blocked right-looking LU with partial pivoting (LAPACK `getrf` shape),
/// factoring `lu` in place and returning `(pivots, perm)` in the same format as
/// `lu_factor_real_scalar`.
///
/// For each panel of `LU_BLOCK_SIZE` columns it (1) factors the panel over all
/// rows below — pivoting on the full column and swapping whole rows, with the
/// rank-1 updates confined to the panel's own columns; (2) forward-solves the
/// trailing block row `U12 = L11⁻¹·A12`; and (3) applies the Schur update
/// `A22 -= L21·U12` as one cache-blocked, auto-threaded `matmul_2d`. The trailing
/// update is the O(n³) cost, so it now runs at GEMM-microkernel speed instead of
/// strided scalar rank-1 sweeps.
///
/// Not bit-identical to the scalar kernel: the GEMM accumulates a block of panel
/// columns in one reordered sum rather than as sequential rank-1 subtractions, so
/// the trailing values (and, on ties, the pivot choice) can differ in the last
/// ulp. It is numerically equivalent — `P·A = L·U` holds to machine precision —
/// which is exactly the guarantee JAX's blocked LAPACK `getrf` provides, and is
/// verified by reconstruction parity against the scalar kernel.
fn lu_factor_real_blocked(lu: &mut [f64], m: usize, n: usize) -> (Vec<i64>, Vec<i64>) {
    let k = m.min(n);
    let mut pivots: Vec<i64> = (0..k as i64).collect();
    let mut perm: Vec<i64> = (0..m as i64).collect();
    let nb = LU_BLOCK_SIZE;
    let mut l21 = Vec::with_capacity(m.saturating_mul(nb));
    let mut u12 = Vec::with_capacity(nb.saturating_mul(n));
    let mut prod = Vec::new();

    let mut j = 0;
    while j < k {
        let jb = nb.min(k - j);
        let panel_end = j + jb;

        // (1) Factor the panel columns [j, panel_end) over rows [j, m). Pivot on
        //     the full column and swap whole rows (so the already-factored left
        //     columns and the untouched trailing columns stay row-consistent);
        //     rank-1 updates touch only the panel's own columns.
        for col in j..panel_end {
            let mut max_val = (lu[col * n + col] * lu[col * n + col]).sqrt();
            let mut max_row = col;
            for row in (col + 1)..m {
                let candidate = lu[row * n + col];
                let val = (candidate * candidate).sqrt();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            pivots[col] = max_row as i64;
            if max_row != col {
                perm.swap(col, max_row);
                for jj in 0..n {
                    lu.swap(col * n + jj, max_row * n + jj);
                }
            }

            let diag = lu[col * n + col];
            if (diag * diag).sqrt() < f64::EPSILON * 1e-10 {
                continue;
            }
            for row in (col + 1)..m {
                let factor = (lu[row * n + col] * diag) / (diag * diag);
                lu[row * n + col] = factor;
                for jj in (col + 1)..panel_end {
                    lu[row * n + jj] -= factor * lu[col * n + jj];
                }
            }
        }

        // (2) Forward-solve U12 = L11⁻¹ · A12 (panel rows × trailing columns).
        //     L11 is unit lower-triangular; A12 was row-swapped but not yet
        //     rank-1 updated by the panel.
        if panel_end < n {
            for ri in 1..jb {
                let r = j + ri;
                for t in 0..ri {
                    let l_rt = lu[r * n + (j + t)];
                    if l_rt != 0.0 {
                        let trow = j + t;
                        for jj in panel_end..n {
                            lu[r * n + jj] -= l_rt * lu[trow * n + jj];
                        }
                    }
                }
            }
        }

        // (3) Trailing Schur update A22 -= L21 · U12 via blocked GEMM.
        let rows_below = m - panel_end;
        let cols_right = n - panel_end.min(n);
        if rows_below > 0 && cols_right > 0 {
            l21.clear();
            for p in 0..rows_below {
                let src = (panel_end + p) * n + j;
                l21.extend_from_slice(&lu[src..src + jb]);
            }
            debug_assert_eq!(l21.len(), rows_below * jb);

            u12.clear();
            for t in 0..jb {
                let src = (j + t) * n + panel_end;
                u12.extend_from_slice(&lu[src..src + cols_right]);
            }
            debug_assert_eq!(u12.len(), jb * cols_right);

            prod.resize(rows_below * cols_right, 0.0);
            matmul_2d_into(&l21, rows_below, jb, &u12, cols_right, &mut prod);
            for p in 0..rows_below {
                let row = (panel_end + p) * n + panel_end;
                let pr = p * cols_right;
                for q in 0..cols_right {
                    lu[row + q] -= prod[pr + q];
                }
            }
        }

        j = panel_end;
    }

    (pivots, perm)
}

fn eval_lu_real_matrix(
    m: usize,
    n: usize,
    mut lu: Vec<f64>,
    dtype: DType,
) -> Result<Vec<Value>, EvalError> {
    let k = m.min(n);
    // Large factorizations take the cache-blocked right-looking path whose
    // O(n³) Schur update runs at GEMM speed; small ones stay on the scalar
    // kernel, preserving the bit-identical-to-complex-path invariant and the
    // small-n conformance goldens (blocked reorders the trailing sum, so it is
    // numerically equivalent but not bit-identical — see lu_factor_real_blocked).
    let (pivots, perm) = if k >= LU_BLOCK_THRESHOLD {
        lu_factor_real_blocked(&mut lu, m, n)
    } else {
        lu_factor_real_scalar(&mut lu, m, n)
    };

    let lu_val = matrix_to_value(m, n, &lu, dtype)?;

    let pivots_lits: Vec<Literal> = pivots.iter().map(|&p| Literal::I64(p)).collect();
    let pivots_shape = Shape {
        dims: vec![k as u32],
    };
    let pivots_val = Value::Tensor(
        TensorValue::new(DType::I32, pivots_shape, pivots_lits)
            .map_err(EvalError::InvalidTensor)?,
    );

    let perm_lits: Vec<Literal> = perm.iter().map(|&p| Literal::I64(p)).collect();
    let perm_shape = Shape {
        dims: vec![m as u32],
    };
    let perm_val = Value::Tensor(
        TensorValue::new(DType::I32, perm_shape, perm_lits).map_err(EvalError::InvalidTensor)?,
    );

    Ok(vec![lu_val, pivots_val, perm_val])
}

// ── SVD (Singular Value Decomposition) ─────────────────────────────

/// Compute the thin SVD: A = U diag(S) V^T using one-sided Jacobi rotations.
///
/// Returns `[U, S, Vt]` where:
///   - U is m×k (left singular vectors)
///   - S is a vector of k singular values (descending order)
///   - Vt is k×n (right singular vectors, transposed)
///
/// k = min(m, n). When `full_matrices=true`, U is m×m and Vt is n×n.
pub(crate) fn eval_svd(
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::Svd;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");

    // Real fast path (thin SVD). For real input the complex kernel below
    // touches only the real component (A^H A, the Hermitian Jacobi sweep, and
    // U = A V Σ⁻¹ all reduce to real arithmetic when every imaginary part is
    // zero), but it pays two avoidable costs: 16-byte (re,im) striding through
    // complex_mul/complex_div, and — dominating at n≈48 — an O(n²) max-pivot
    // *search* before every single rotation, making the classic Jacobi O(n⁴).
    // `eval_svd_real_thin` works in a contiguous Vec<f64> and replaces the
    // max-pivot search with a row-cyclic sweep (O(n³·sweeps), ~8 sweeps),
    // computing the same spectrum to machine precision. The result is a valid
    // SVD with identical singular values (the eigenvalues of AᵀA are unique);
    // U/Vᵀ may differ by per-column sign/rotation within the SVD's inherent
    // freedom, so parity is verified by reconstruction + spectrum match against
    // the complex path, not bit-identity (see svd_real_fast_path_*). Handles
    // both thin and full_matrices (the latter extends U via real Gram–Schmidt).
    if !matches!(inputs[0].dtype(), DType::Complex64 | DType::Complex128) {
        let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
        return eval_svd_real(m, n, &a, dtype, full_matrices);
    }

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    let k = m.min(n);
    let zero = (0.0, 0.0);

    // One-sided complex Jacobi SVD (Hari–Veselić): orthogonalize the COLUMNS of A
    // in place by right-side unitary Jacobi rotations, accumulating V. Never forms
    // AᴴA, so the condition number is not squared and small singular values keep
    // their relative accuracy (ε·‖A‖, not √ε·‖A‖) — the complex analogue of the
    // real one-sided path. At convergence W = A·V has orthogonal columns:
    // σ_i = ‖W[:,i]‖, U[:,i] = W[:,i]/σ_i. See bead frankenjax-4kx6m.
    let mut w = a.clone(); // m×n working matrix; its columns become orthogonal.
    let mut v = vec![zero; n * n];
    for i in 0..n {
        v[i * n + i] = (1.0, 0.0);
    }

    let eps = f64::EPSILON;
    let max_sweeps = 60;
    for _ in 0..max_sweeps {
        let mut converged = true;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                // 2×2 Hermitian Gram of columns p,q: [[α, γ], [γ̄, β]] with α,β real,
                // γ = col_pᴴ·col_q = Σ conj(W[i,p])·W[i,q].
                let mut alpha = 0.0_f64;
                let mut beta = 0.0_f64;
                let mut gamma = zero;
                for i in 0..m {
                    let wip = w[i * n + p];
                    let wiq = w[i * n + q];
                    alpha += wip.0 * wip.0 + wip.1 * wip.1;
                    beta += wiq.0 * wiq.0 + wiq.1 * wiq.1;
                    gamma = complex_add(gamma, complex_mul(complex_conj(wip), wiq));
                }
                let gmag = (gamma.0 * gamma.0 + gamma.1 * gamma.1).sqrt();
                // Columns already orthogonal to working precision: skip.
                if gmag <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;
                // Real symmetric Schur on [[α, |γ|], [|γ|, β]] gives c,s; the column
                // update carries the complex phase ζ = γ/|γ|. The resulting rotation
                // J = [[c, ζs], [−ζ̄s, c]] is unitary and drives the (p,q) Gram entry
                // (which equals ζ·[cs(α−β) + |γ|(c²−s²)]) to zero.
                let tau = (beta - alpha) / (2.0 * gmag);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = (1.0_f64 / (1.0 + t * t).sqrt(), 0.0);
                let s = t / (1.0 + t * t).sqrt();
                let zeta = (gamma.0 / gmag, gamma.1 / gmag); // unit phase ζ
                let neg_czs = complex_mul((-s, 0.0), complex_conj(zeta)); // −ζ̄·s
                let zeta_s = complex_mul((s, 0.0), zeta); //                 ζ·s
                // new_p = c·p − ζ̄s·q ; new_q = ζs·p + c·q.
                for i in 0..m {
                    let wip = w[i * n + p];
                    let wiq = w[i * n + q];
                    w[i * n + p] = complex_add(complex_mul(c, wip), complex_mul(neg_czs, wiq));
                    w[i * n + q] = complex_add(complex_mul(zeta_s, wip), complex_mul(c, wiq));
                }
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = complex_add(complex_mul(c, vip), complex_mul(neg_czs, viq));
                    v[i * n + q] = complex_add(complex_mul(zeta_s, vip), complex_mul(c, viq));
                }
            }
        }
        if converged {
            break;
        }
    }

    // Column norms of W are the singular values; sort descending.
    let mut col_norm = vec![0.0_f64; n];
    for j in 0..n {
        let mut s2 = 0.0_f64;
        for i in 0..m {
            let wij = w[i * n + j];
            s2 += wij.0 * wij.0 + wij.1 * wij.1;
        }
        col_norm[j] = s2.sqrt();
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| col_norm[b].total_cmp(&col_norm[a]));

    let mut sigma = vec![0.0_f64; k];
    let mut v_sorted = vec![zero; n * n];
    let mut u = vec![zero; m * k];
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
        if new_col < k {
            let sg = col_norm[old_col];
            sigma[new_col] = sg;
            // U[:,i] = W[:,i]/σ_i; a numerically-zero σ leaves the column zero
            // (full_matrices later fills a unitary basis), matching the prior guard.
            if sg > f64::EPSILON * 1e4 {
                for row in 0..m {
                    u[row * k + new_col] = complex_div(w[row * n + old_col], (sg, 0.0));
                }
            }
        }
    }

    // Build outputs
    let u_cols = if full_matrices { m } else { k };
    let vt_rows = if full_matrices { n } else { k };

    // For full_matrices, extend U to m×m unitary matrix
    let u_out = if full_matrices && u_cols > k {
        complex_extend_unitary_columns(&u, m, k, u_cols)
    } else {
        u
    };

    // Build V^H (conjugate transpose)
    let vh_out = if full_matrices {
        let mut vh = vec![zero; n * n];
        for i in 0..n {
            for j in 0..n {
                vh[i * n + j] = complex_conj(v_sorted[j * n + i]);
            }
        }
        vh
    } else {
        let mut vh = vec![zero; k * n];
        for i in 0..k {
            for j in 0..n {
                vh[i * n + j] = complex_conj(v_sorted[j * n + i]);
            }
        }
        vh
    };

    let u_val = complex_matrix_to_value(m, u_cols, &u_out, dtype)?;

    let s_elements: Vec<Literal> = sigma
        .iter()
        .map(|&v| linalg_literal_from_f64(dtype, v))
        .collect();
    let s_shape = Shape {
        dims: vec![k as u32],
    };
    let s_dtype = match dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        _ => dtype,
    };
    let s_tensor =
        TensorValue::new(s_dtype, s_shape, s_elements).map_err(EvalError::InvalidTensor)?;
    let s_val = Value::Tensor(s_tensor);

    let vh_val = complex_matrix_to_value(vt_rows, n, &vh_out, dtype)?;

    Ok(vec![u_val, s_val, vh_val])
}

const SVD_COMPRESSED_MIN_DIM: usize = 32;
const SVD_COMPRESSED_MAX_RANK: usize = 16;
const SVD_COMPRESSED_REL_RESIDUAL: f64 = 1.0e-20;

#[allow(clippy::too_many_lines)]
fn compressed_rrqr_svd_real(
    m: usize,
    n: usize,
    a: &[f64],
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if m < n || n < SVD_COMPRESSED_MIN_DIM || !a.iter().all(|value| value.is_finite()) {
        return None;
    }

    let k = m.min(n);
    let mut residual_cols = Vec::with_capacity(n);
    let mut residual_norms = Vec::with_capacity(n);
    for col in 0..n {
        let mut column = Vec::with_capacity(m);
        let mut norm_sq = 0.0;
        for row in 0..m {
            let value = a[row * n + col];
            column.push(value);
            norm_sq += value * value;
        }
        residual_cols.push(column);
        residual_norms.push(norm_sq);
    }

    let initial_trace: f64 = residual_norms.iter().sum();
    if initial_trace == 0.0 {
        let sigma = vec![0.0; k];
        let u = vec![0.0; m * k];
        let mut v = vec![0.0; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        return Some((sigma, u, v));
    }

    let rank_limit = SVD_COMPRESSED_MAX_RANK.min((n / 2).max(1));
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(rank_limit);
    let mut r_rows: Vec<Vec<f64>> = Vec::with_capacity(rank_limit);

    for rank in 0..rank_limit {
        let mut pivot = rank;
        let mut pivot_norm = residual_norms[rank];
        for (candidate, &norm_sq) in residual_norms.iter().enumerate().skip(rank + 1) {
            if norm_sq > pivot_norm {
                pivot = candidate;
                pivot_norm = norm_sq;
            }
        }

        if pivot_norm <= 0.0 {
            return None;
        }

        if pivot != rank {
            residual_cols.swap(rank, pivot);
            residual_norms.swap(rank, pivot);
            permutation.swap(rank, pivot);
            for r_row in &mut r_rows {
                r_row.swap(rank, pivot);
            }
        }

        let norm = pivot_norm.sqrt();
        if norm <= f64::EPSILON {
            return None;
        }

        let mut q = vec![0.0; m];
        for row in 0..m {
            q[row] = residual_cols[rank][row] / norm;
        }
        for previous_q in &q_cols {
            let mut projection = 0.0;
            for row in 0..m {
                projection += previous_q[row] * q[row];
            }
            for row in 0..m {
                q[row] -= projection * previous_q[row];
            }
        }
        let q_norm_sq = q.iter().map(|value| value * value).sum::<f64>();
        if q_norm_sq <= f64::EPSILON {
            return None;
        }
        let q_norm = q_norm_sq.sqrt();
        for value in &mut q {
            *value /= q_norm;
        }

        for col in rank..n {
            let mut projection = 0.0;
            for row in 0..m {
                projection += q[row] * residual_cols[col][row];
            }
            for row in 0..m {
                residual_cols[col][row] -= projection * q[row];
            }
            residual_norms[col] = residual_cols[col]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .max(0.0);
        }

        let mut r_row = vec![0.0; n];
        for col in 0..n {
            let original_col = permutation[col];
            let mut projection = 0.0;
            for row in 0..m {
                projection += q[row] * a[row * n + original_col];
            }
            r_row[col] = projection;
        }
        q_cols.push(q);
        r_rows.push(r_row);

        let active_rank = rank + 1;
        let remaining_trace: f64 = residual_norms.iter().skip(active_rank).sum();
        if active_rank < n && remaining_trace <= initial_trace * SVD_COMPRESSED_REL_RESIDUAL {
            return Some(compressed_rrqr_svd_result(
                &q_cols,
                &r_rows,
                &permutation,
                m,
                n,
                active_rank,
            ));
        }
    }

    None
}

fn compressed_rrqr_svd_result(
    q_cols: &[Vec<f64>],
    r_rows: &[Vec<f64>],
    permutation: &[usize],
    m: usize,
    n: usize,
    rank: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    let mut rrt = low_rank_rrt(r_rows, rank, n);
    let (lambdas, eigenvectors) = jacobi_eigendecomposition_cyclic(&mut rrt, rank);
    let mut order: Vec<usize> = (0..rank).collect();
    order.sort_by(|&left, &right| {
        lambdas[right]
            .total_cmp(&lambdas[left])
            .then_with(|| left.cmp(&right))
    });

    let mut sigma = vec![0.0; k];
    let active_cols = rank.min(k);
    let mut u = vec![0.0; m * k];
    let mut v_active = vec![0.0; n * active_cols];

    for (new_col, &eig_col) in order.iter().take(active_cols).enumerate() {
        let sg = lambdas[eig_col].max(0.0).sqrt();
        sigma[new_col] = sg;

        for row in 0..m {
            let mut value = 0.0;
            for basis in 0..rank {
                value += q_cols[basis][row] * eigenvectors[basis * rank + eig_col];
            }
            u[row * k + new_col] = value;
        }

        if sg > f64::EPSILON * 1e4 {
            for (permuted_col, &original_col) in permutation.iter().enumerate() {
                let mut value = 0.0;
                for basis in 0..rank {
                    value += r_rows[basis][permuted_col] * eigenvectors[basis * rank + eig_col];
                }
                v_active[original_col * active_cols + new_col] = value / sg;
            }
        }
    }

    let v = extend_orthogonal_columns(&v_active, n, active_cols, n);
    (sigma, u, v)
}

/// Core real thin SVD via one-sided Jacobi (Hestenes / Demmel–Veselić). Orthogonalize
/// the COLUMNS of A in place by right-side Jacobi rotations, accumulating V; never form
/// AᵀA, so small singular values keep high *relative* accuracy (≈ε‖A‖, not √ε‖A‖). At
/// convergence W = A·V has orthogonal columns. Returns `(sigma[k], u[m*k], v[n*n])` with
/// σ descending, `U[:,i] = W[:,i]/σ_i` (a numerically-zero σ leaves that U column zero),
/// and V the full n×n right singular vectors as columns. Shared by `eval_svd_real` and
/// the SVD-based pseudoinverse `pinv_svd` (beads frankenjax-96i7w, -4kx6m).
fn one_sided_jacobi_svd_real(m: usize, n: usize, a: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if let Some(result) = compressed_rrqr_svd_real(m, n, a) {
        return result;
    }

    let k = m.min(n);
    // Store W column-major because Jacobi sweeps stream columns p/q repeatedly.
    // The copy also records initial column norms, then applies a deterministic
    // descending-norm ordering before the first sweep. This is a bounded
    // Drmac-Veselic-style preconditioner: it changes only the internal Jacobi
    // path, while the public SVD contract remains spectrum + reconstruction
    // parity with deterministic tie-breaking.
    let mut w = vec![0.0_f64; m * n]; // n columns, each length m.
    let mut initial_norm_sq = vec![0.0_f64; n];
    for row in 0..m {
        for col in 0..n {
            let value = a[row * n + col];
            w[col * m + row] = value;
            initial_norm_sq[col] += value * value;
        }
    }
    // Store V column-major for the same reason as W: every Jacobi rotation
    // streams two V columns p/q. The public sorted V returned below is still
    // materialized row-major.
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let mut initial_order: Vec<usize> = (0..n).collect();
    initial_order.sort_by(|&left, &right| {
        initial_norm_sq[right]
            .total_cmp(&initial_norm_sq[left])
            .then_with(|| left.cmp(&right))
    });
    if initial_order
        .iter()
        .enumerate()
        .any(|(idx, &col)| idx != col)
    {
        let mut ordered_w = vec![0.0_f64; m * n];
        let mut ordered_v = vec![0.0_f64; n * n];
        for (new_col, &old_col) in initial_order.iter().enumerate() {
            let src = old_col * m;
            let dst = new_col * m;
            ordered_w[dst..dst + m].copy_from_slice(&w[src..src + m]);
            ordered_v[new_col * n + old_col] = 1.0;
        }
        w = ordered_w;
        v = ordered_v;
    }

    let eps = f64::EPSILON;
    let max_sweeps = 60;
    for _ in 0..max_sweeps {
        let mut converged = true;
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let p_base = p * m;
                let q_base = q * m;
                // 2×2 Gram of columns p,q of the current W (never the full AᵀA).
                let mut alpha = 0.0_f64; // ‖col_p‖²
                let mut beta = 0.0_f64; //  ‖col_q‖²
                let mut gamma = 0.0_f64; // col_p · col_q
                let p_col = &w[p_base..p_base + m];
                let q_col = &w[q_base..q_base + m];
                for i in 0..m {
                    let wip = p_col[i];
                    let wiq = q_col[i];
                    alpha += wip * wip;
                    beta += wiq * wiq;
                    gamma += wip * wiq;
                }
                // Columns already orthogonal to working precision: skip.
                if gamma.abs() <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;
                // Symmetric Schur on [[alpha, gamma], [gamma, beta]] (GVL 8.4.1):
                // smaller root of t² + 2τt − 1 = 0.
                let tau = (beta - alpha) / (2.0 * gamma);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Rotate columns p,q of W: new_p = c·p − s·q, new_q = s·p + c·q.
                let (w_left, w_right) = w.split_at_mut(q_base);
                let p_col = &mut w_left[p_base..p_base + m];
                let q_col = &mut w_right[..m];
                for i in 0..m {
                    let wip = p_col[i];
                    let wiq = q_col[i];
                    p_col[i] = c * wip - s * wiq;
                    q_col[i] = s * wip + c * wiq;
                }
                // Accumulate the same rotation into V.
                let p_base = p * n;
                let q_base = q * n;
                let (v_left, v_right) = v.split_at_mut(q_base);
                let p_col = &mut v_left[p_base..p_base + n];
                let q_col = &mut v_right[..n];
                for i in 0..n {
                    let vip = p_col[i];
                    let viq = q_col[i];
                    p_col[i] = c * vip - s * viq;
                    q_col[i] = s * vip + c * viq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Column norms of W are the singular values; sort descending.
    let mut col_norm = vec![0.0_f64; n];
    for j in 0..n {
        let mut s2 = 0.0_f64;
        for i in 0..m {
            let wij = w[j * m + i];
            s2 += wij * wij;
        }
        col_norm[j] = s2.sqrt();
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| col_norm[b].total_cmp(&col_norm[a]));

    let mut sigma = vec![0.0_f64; k];
    let mut v_sorted = vec![0.0_f64; n * n];
    let mut u = vec![0.0_f64; m * k];
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            v_sorted[row * n + new_col] = v[old_col * n + row];
        }
        if new_col < k {
            let sg = col_norm[old_col];
            sigma[new_col] = sg;
            if sg > f64::EPSILON * 1e4 {
                for row in 0..m {
                    u[row * k + new_col] = w[old_col * m + row] / sg;
                }
            }
        }
    }

    (sigma, u, v_sorted)
}

/// Real-SVD fast path: a contiguous-f64, cyclic-Jacobi replacement for the
/// complex `eval_svd` kernel when the input is real. Produces a valid SVD
/// (thin: U m×k, Vᵀ k×n; full_matrices: U m×m, Vᵀ n×n) whose singular values
/// match the complex path to machine precision; U/Vᵀ are equal up to the SVD's
/// intrinsic per-column sign/rotation freedom. Verified by reconstruction +
/// spectrum parity rather than bit-identity (see the `svd_real_fast_path_*`
/// tests). For full_matrices, U's k economy columns are extended to an m×m
/// orthonormal basis via real Gram–Schmidt (`extend_orthogonal_columns`).
fn eval_svd_real(
    m: usize,
    n: usize,
    a: &[f64],
    dtype: DType,
    full_matrices: bool,
) -> Result<Vec<Value>, EvalError> {
    let k = m.min(n);
    let (sigma, u, v_sorted) = one_sided_jacobi_svd_real(m, n, a);

    // For full_matrices, extend U's k economy columns to an m×m orthonormal
    // basis (real Gram–Schmidt); thin keeps the m×k economy U.
    let u_cols = if full_matrices { m } else { k };
    let u_out = if full_matrices && u_cols > k {
        extend_orthogonal_columns(&u, m, k, u_cols)
    } else {
        u
    };

    // Vᵀ (transpose of real V). full_matrices: n×n; thin: k×n.
    let vt_rows = if full_matrices { n } else { k };
    let mut vh = vec![0.0_f64; vt_rows * n];
    for i in 0..vt_rows {
        for j in 0..n {
            vh[i * n + j] = v_sorted[j * n + i];
        }
    }

    let u_val = matrix_to_value(m, u_cols, &u_out, dtype)?;

    let s_elements: Vec<Literal> = sigma
        .iter()
        .map(|&v| linalg_literal_from_f64(dtype, v))
        .collect();
    let s_tensor = TensorValue::new(
        dtype,
        Shape {
            dims: vec![k as u32],
        },
        s_elements,
    )
    .map_err(EvalError::InvalidTensor)?;
    let s_val = Value::Tensor(s_tensor);

    let vh_val = matrix_to_value(vt_rows, n, &vh, dtype)?;

    Ok(vec![u_val, s_val, vh_val])
}

/// Eigendecomposition of a real symmetric n×n matrix by tridiagonalization +
/// implicit-shift symmetric tridiagonal QL (the LAPACK/EISPACK `tred2`+`tql2`
/// approach). Returns `(eigenvalues, V)` with `A = V diag(eigenvalues) Vᵀ`,
/// eigenvectors as columns (`V[row*n + col]`), unsorted (the caller sorts).
///
/// Reuses the tested [`hessenberg_reduction`]: for a SYMMETRIC matrix the
/// upper-Hessenberg form is symmetric tridiagonal, and the accumulated `Q`
/// satisfies `A = Q·T·Qᵀ`, so the eigenvectors of `T` map to those of `A` by
/// left-multiplication by `Q`. The implicit-QL sweep is O(n²) per step and
/// converges in O(n) steps total, versus cyclic Jacobi's ~8 sweeps of O(n³)
/// rotations — `~5n³` vs `~32n³`. Accuracy is the LAPACK-grade `O(ε·‖A‖)`,
/// matching JAX's `syevd`; eigh parity is reconstruction + sorted spectrum, both
/// preserved (eigenvectors keep the decomposition's intrinsic sign freedom).
fn tridiag_ql_eigendecomposition(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    // A = Q·T·Qᵀ with T symmetric tridiagonal (Householder reduction).
    let (h, q) = hessenberg_reduction(a, n);
    let mut d = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];
    for i in 0..n {
        d[i] = h[i * n + i];
        e[i] = if i + 1 < n { h[(i + 1) * n + i] } else { 0.0 };
    }
    // `symmetric_tridiagonal_ql` accumulates the eigenvectors COLUMN-MAJOR so its hot
    // plane-rotation loop streams contiguous columns instead of striding by `n` per row
    // (the dominant eigh cache cliff, ur4h3). Transpose Q in place, run QL, transpose back
    // in place. This preserves the same QL arithmetic while avoiding the extra n²
    // allocation and two full copy loops used by the out-of-place transpose.
    let mut z = q;
    transpose_square_in_place(&mut z, n);
    symmetric_tridiagonal_ql(&mut d, &mut e, &mut z, n);
    transpose_square_in_place(&mut z, n);
    (d, z)
}

fn transpose_square_in_place(matrix: &mut [f64], n: usize) {
    for row in 0..n {
        for col in row + 1..n {
            matrix.swap(row * n + col, col * n + row);
        }
    }
}

/// Implicit-shift symmetric tridiagonal QL with eigenvector accumulation (EISPACK
/// `tql2`). `d` = diagonal, `e[i]` = subdiagonal coupling `d[i]`/`d[i+1]`
/// (`e[n-1]` unused), `z` = the accumulated transform in COLUMN-MAJOR layout
/// (`z[col*n + row]`; eigenvectors of `A` are its columns on output — the caller
/// transposes to/from row-major). Each implicit-shift sweep deflates one eigenvalue with a
/// Wilkinson shift; plane rotations chase the bulge up the tridiagonal. O(n²) per
/// sweep, O(n) sweeps. Bit-equivalent in spirit to LAPACK `dsteqr`.
fn symmetric_tridiagonal_ql(d: &mut [f64], e: &mut [f64], z: &mut [f64], n: usize) {
    if n <= 1 {
        return;
    }
    let eps = f64::EPSILON;
    for l in 0..n {
        let mut iter = 0usize;
        loop {
            // Find a negligible subdiagonal `e[m]` splitting off the leading block.
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= eps * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break; // d[l] has converged
            }
            iter += 1;
            if iter > 50 {
                break; // non-convergence safeguard (essentially never hit)
            }
            // Wilkinson shift from the trailing 2×2 of the active block.
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            let sgn = if g >= 0.0 { r.abs() } else { -r.abs() };
            g = d[m] - d[l] + e[l] / (g + sgn);
            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            let mut zeroed = false;
            let mut i = m;
            while i > l {
                i -= 1;
                let mut f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    zeroed = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                // Accumulate the plane rotation into eigenvector columns i, i+1. `z` is
                // COLUMN-MAJOR (z[col*n + row]), so columns i and i+1 are two CONTIGUOUS
                // n-runs — the rotation streams them instead of striding by `n` per row
                // (the QL cache cliff). Values are identical to the row-major form (same
                // s,c, same per-row formulas, same k order); the caller transposes in/out.
                let (lo, hi) = z.split_at_mut((i + 1) * n);
                let col_i = &mut lo[i * n..i * n + n];
                let col_i1 = &mut hi[..n];
                for k in 0..n {
                    f = col_i1[k];
                    col_i1[k] = s * col_i[k] + c * f;
                    col_i[k] = c * col_i[k] - s * f;
                }
            }
            if zeroed {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

/// Row-cyclic Jacobi eigendecomposition of a real symmetric n×n matrix.
///
/// Returns `(eigenvalues, V)` with `A = V diag(eigenvalues) Vᵀ`, eigenvectors as
/// columns (`V[row*n + col]`). Unlike the classic `jacobi_eigendecomposition`
/// (which scans all O(n²) off-diagonals to pick the largest pivot *before every
/// rotation* — O(n⁴) overall), this sweeps the upper triangle in fixed
/// (p,q) order and converges in a handful of sweeps (O(n³·sweeps)). Each
/// rotation uses the numerically stable symmetric-Schur coefficients
/// (Golub & Van Loan, Alg. 8.4.1): solve `t² + 2τt − 1 = 0` for the smaller
/// root, then `c = 1/√(1+t²)`, `s = tc`. Converges to the same spectrum as the
/// max-pivot kernel to machine precision.
fn jacobi_eigendecomposition_cyclic(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n <= 1 {
        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        return (eigenvalues, v);
    }

    let tol = f64::EPSILON * 1e2;
    let max_sweeps = 100;

    for _ in 0..max_sweeps {
        // Off-diagonal magnitude (max |a_pq|, p<q). Convergence when below tol.
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off = off.max(a[p * n + q].abs());
            }
        }
        if off < tol {
            break;
        }

        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // Symmetric Schur: t is the smaller root of t² + 2τt − 1 = 0.
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // A ← Jᵀ A J. Update columns p,q (left mult by J on rows handled
                // by the symmetric column pass below).
                for i in 0..n {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip - s * aiq;
                    a[i * n + q] = s * aip + c * aiq;
                }
                for i in 0..n {
                    let api = a[p * n + i];
                    let aqi = a[q * n + i];
                    a[p * n + i] = c * api - s * aqi;
                    a[q * n + i] = s * api + c * aqi;
                }
                // Off-diagonal (p,q) is annihilated; pin it to exact zero.
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // V ← V J.
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Complex Jacobi eigendecomposition of a Hermitian n×n matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvalues are real and eigenvectors are complex.
///
/// Reference (max-pivot) kernel: production paths use the faster
/// [`complex_jacobi_eigendecomposition_cyclic`]; this is retained for the cyclic
/// kernel's parity + A/B timing tests (and is itself the bit-for-bit correctness anchor).
#[allow(dead_code)]
fn complex_jacobi_eigendecomposition(
    a: &mut [(f64, f64)],
    n: usize,
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let zero = (0.0, 0.0);
    let one = (1.0, 0.0);

    let mut v = vec![zero; n * n];
    for i in 0..n {
        v[i * n + i] = one;
    }

    let max_iter = 100 * n * n;
    let tol = f64::EPSILON * 1e2;

    for _ in 0..max_iter {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = complex_abs(a[i * n + j]);
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // For Hermitian A, diagonal elements are real
        let app = a[p * n + p].0;
        let aqq = a[q * n + q].0;
        let apq = a[p * n + q];
        let apq_abs = complex_abs(apq);

        // Phase of off-diagonal: A[p][q] = |A[p][q]| * e^{i*phi}
        let phase = if apq_abs > f64::EPSILON {
            complex_div(apq, (apq_abs, 0.0))
        } else {
            one
        };

        // Two-step approach:
        // Step 1: Apply phase rotation D to make A[p][q] real
        //         D = diag(..., 1, ..., e^{-i*phi}, ...) with e^{-i*phi} at position q
        //         B = D * A * D^H has B[p][q] = |A[p][q]| (real)
        // Step 2: Apply real Givens rotation G to zero B[p][q]
        // Combined unitary: U = D^H * G * D

        // Standard real Jacobi angle computation (treating |apq| as the off-diagonal)
        let theta = if (app - aqq).abs() < f64::EPSILON {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq_abs / (app - aqq)).atan()
        };
        let (s, c) = theta.sin_cos();

        // Combined unitary U = D·G, where D = diag(1, e^{-iφ}) realifies the pivot
        // (Dᴴ A D has real off-diagonal |apq|) and G = [[c,-s],[s,c]] is the real
        // Givens that zeros it, so Uᴴ A U = Gᵀ (Dᴴ A D) G is diagonal:
        //   U = [[c,          -s        ],
        //        [s·e^{-iφ},   c·e^{-iφ} ]]
        // The previous U = [[c,-s·e^{-iφ}],[s·e^{iφ},c]] was NOT D·G — it preserved
        // the trace but not the determinant/spectrum (|off-diagonal| was unchanged),
        // i.e. it was not a diagonalizing similarity
        // (frankenjax-complex-jacobi-wrong-eigenvalues-15mx4).
        let phase_conj = complex_conj(phase); // e^{-iφ}
        let s_phase = complex_mul((s, 0.0), phase); // s·e^{iφ}
        let c_phase = complex_mul((c, 0.0), phase); // c·e^{iφ}
        let s_phase_conj = complex_mul((s, 0.0), phase_conj); // s·e^{-iφ}
        let c_phase_conj = complex_mul((c, 0.0), phase_conj); // c·e^{-iφ}

        // Apply Uᴴ from the left (Uᴴ = [[c, s·e^{iφ}], [−s, c·e^{iφ}]]):
        let row_p: Vec<_> = (0..n).map(|j| a[p * n + j]).collect();
        let row_q: Vec<_> = (0..n).map(|j| a[q * n + j]).collect();
        for j in 0..n {
            // (Uᴴ A)[p][j] = c·A[p][j] + s·e^{iφ}·A[q][j]
            a[p * n + j] = complex_add(
                complex_mul((c, 0.0), row_p[j]),
                complex_mul(s_phase, row_q[j]),
            );
            // (Uᴴ A)[q][j] = −s·A[p][j] + c·e^{iφ}·A[q][j]
            a[q * n + j] = complex_add(
                complex_mul((-s, 0.0), row_p[j]),
                complex_mul(c_phase, row_q[j]),
            );
        }

        // Apply U from the right (U = [[c, −s], [s·e^{-iφ}, c·e^{-iφ}]]):
        let col_p: Vec<_> = (0..n).map(|i| a[i * n + p]).collect();
        let col_q: Vec<_> = (0..n).map(|i| a[i * n + q]).collect();
        for i in 0..n {
            // (A U)[i][p] = c·A[i][p] + s·e^{-iφ}·A[i][q]
            a[i * n + p] = complex_add(
                complex_mul((c, 0.0), col_p[i]),
                complex_mul(s_phase_conj, col_q[i]),
            );
            // (A U)[i][q] = −s·A[i][p] + c·e^{-iφ}·A[i][q]
            a[i * n + q] = complex_add(
                complex_mul((-s, 0.0), col_p[i]),
                complex_mul(c_phase_conj, col_q[i]),
            );
        }

        // Diagonal elements should be real; off-diagonal (p,q) annihilated.
        a[p * n + p] = (a[p * n + p].0, 0.0);
        a[q * n + q] = (a[q * n + q].0, 0.0);
        a[p * n + q] = zero;
        a[q * n + p] = zero;

        // V ← V U (same right-multiply by U as applied to A's columns).
        let vp: Vec<_> = (0..n).map(|i| v[i * n + p]).collect();
        let vq: Vec<_> = (0..n).map(|i| v[i * n + q]).collect();
        for i in 0..n {
            // V'[i][p] = c·V[i][p] + s·e^{-iφ}·V[i][q]
            v[i * n + p] = complex_add(
                complex_mul((c, 0.0), vp[i]),
                complex_mul(s_phase_conj, vq[i]),
            );
            // V'[i][q] = −s·V[i][p] + c·e^{-iφ}·V[i][q]
            v[i * n + q] = complex_add(
                complex_mul((-s, 0.0), vp[i]),
                complex_mul(c_phase_conj, vq[i]),
            );
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].0).collect();
    (eigenvalues, v)
}

/// Row-cyclic complex (Hermitian) Jacobi eigendecomposition — the complex twin of
/// [`jacobi_eigendecomposition_cyclic`]. The classic [`complex_jacobi_eigendecomposition`]
/// scans all O(n²) off-diagonals to pick the largest pivot before every rotation
/// (O(n⁴) overall, dominating complex eigh / thin-SVD at n≈48); this sweeps the upper
/// triangle in fixed (p,q) order and converges in a handful of sweeps (O(n³·sweeps)).
/// It applies the *same* corrected per-pivot unitary `U = D·G`, so it converges to the
/// same Hermitian spectrum to machine precision (both callers re-sort, so the iteration
/// order is unobservable). Verified by `complex_jacobi_cyclic_matches_maxpivot`.
fn complex_jacobi_eigendecomposition_cyclic(
    a: &mut [(f64, f64)],
    n: usize,
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let zero = (0.0, 0.0);
    let one = (1.0, 0.0);

    let mut v = vec![zero; n * n];
    for i in 0..n {
        v[i * n + i] = one;
    }
    if n <= 1 {
        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].0).collect();
        return (eigenvalues, v);
    }

    let tol = f64::EPSILON * 1e2;
    let max_sweeps = 100;

    for _ in 0..max_sweeps {
        // Convergence: largest |a_pq|, p<q, below tol.
        let mut off_max = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off_max = off_max.max(complex_abs(a[p * n + q]));
            }
        }
        if off_max < tol {
            break;
        }

        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                let apq_abs = complex_abs(apq);
                if apq_abs == 0.0 {
                    continue; // already annihilated (pinned to exact zero below)
                }
                let app = a[p * n + p].0;
                let aqq = a[q * n + q].0;

                let phase = if apq_abs > f64::EPSILON {
                    complex_div(apq, (apq_abs, 0.0))
                } else {
                    one
                };
                let theta = if (app - aqq).abs() < f64::EPSILON {
                    std::f64::consts::FRAC_PI_4
                } else {
                    0.5 * (2.0 * apq_abs / (app - aqq)).atan()
                };
                let (s, c) = theta.sin_cos();

                // Corrected combined unitary U = D·G (see complex_jacobi_eigendecomposition):
                //   U = [[c, -s], [s·e^{-iφ}, c·e^{-iφ}]].
                let phase_conj = complex_conj(phase);
                let s_phase = complex_mul((s, 0.0), phase);
                let c_phase = complex_mul((c, 0.0), phase);
                let s_phase_conj = complex_mul((s, 0.0), phase_conj);
                let c_phase_conj = complex_mul((c, 0.0), phase_conj);

                // Uᴴ from the left.
                let row_p: Vec<_> = (0..n).map(|j| a[p * n + j]).collect();
                let row_q: Vec<_> = (0..n).map(|j| a[q * n + j]).collect();
                for j in 0..n {
                    a[p * n + j] = complex_add(
                        complex_mul((c, 0.0), row_p[j]),
                        complex_mul(s_phase, row_q[j]),
                    );
                    a[q * n + j] = complex_add(
                        complex_mul((-s, 0.0), row_p[j]),
                        complex_mul(c_phase, row_q[j]),
                    );
                }

                // U from the right.
                let col_p: Vec<_> = (0..n).map(|i| a[i * n + p]).collect();
                let col_q: Vec<_> = (0..n).map(|i| a[i * n + q]).collect();
                for i in 0..n {
                    a[i * n + p] = complex_add(
                        complex_mul((c, 0.0), col_p[i]),
                        complex_mul(s_phase_conj, col_q[i]),
                    );
                    a[i * n + q] = complex_add(
                        complex_mul((-s, 0.0), col_p[i]),
                        complex_mul(c_phase_conj, col_q[i]),
                    );
                }

                a[p * n + p] = (a[p * n + p].0, 0.0);
                a[q * n + q] = (a[q * n + q].0, 0.0);
                a[p * n + q] = zero;
                a[q * n + p] = zero;

                // V ← V U.
                let vp: Vec<_> = (0..n).map(|i| v[i * n + p]).collect();
                let vq: Vec<_> = (0..n).map(|i| v[i * n + q]).collect();
                for i in 0..n {
                    v[i * n + p] = complex_add(
                        complex_mul((c, 0.0), vp[i]),
                        complex_mul(s_phase_conj, vq[i]),
                    );
                    v[i * n + q] = complex_add(
                        complex_mul((-s, 0.0), vp[i]),
                        complex_mul(c_phase_conj, vq[i]),
                    );
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].0).collect();
    (eigenvalues, v)
}

/// Jacobi eigendecomposition of a symmetric n×n matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column-major in an n×n array.
#[cfg(test)]
fn jacobi_eigendecomposition(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Initialize V = I
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let tol = f64::EPSILON * 1e2;

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute Jacobi rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < f64::EPSILON {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let (sin_t, cos_t) = theta.sin_cos();

        // Apply rotation to A: A' = G^T A G
        // Update rows/cols p and q
        let mut new_row_p = vec![0.0; n];
        let mut new_row_q = vec![0.0; n];
        for i in 0..n {
            new_row_p[i] = cos_t * a[p * n + i] + sin_t * a[q * n + i];
            new_row_q[i] = -sin_t * a[p * n + i] + cos_t * a[q * n + i];
        }

        for i in 0..n {
            a[p * n + i] = new_row_p[i];
            a[q * n + i] = new_row_q[i];
            a[i * n + p] = new_row_p[i]; // symmetric
            a[i * n + q] = new_row_q[i]; // symmetric
        }

        // Fix the 2x2 block
        a[p * n + p] = cos_t * new_row_p[p] + sin_t * new_row_p[q];
        a[q * n + q] = -sin_t * new_row_q[p] + cos_t * new_row_q[q];
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update eigenvector matrix V
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = cos_t * vip + sin_t * viq;
            v[i * n + q] = -sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Extend a set of k orthogonal columns in an m×k matrix to m×m_full
/// by adding orthogonal complement columns via Gram-Schmidt.
fn extend_orthogonal_columns(u: &[f64], m: usize, k: usize, m_full: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; m * m_full];

    // Copy existing columns
    for i in 0..m {
        for j in 0..k {
            result[i * m_full + j] = u[i * k + j];
        }
    }

    // Add standard basis vectors and orthogonalize
    let mut added = k;
    for basis_idx in 0..m {
        if added >= m_full {
            break;
        }

        // Start with e_{basis_idx}
        let mut col = vec![0.0_f64; m];
        col[basis_idx] = 1.0;

        // Orthogonalize against all existing columns
        for j in 0..added {
            let mut dot = 0.0;
            for i in 0..m {
                dot += col[i] * result[i * m_full + j];
            }
            for i in 0..m {
                col[i] -= dot * result[i * m_full + j];
            }
        }

        // Check if linearly independent
        let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > f64::EPSILON * 1e4 {
            for i in 0..m {
                result[i * m_full + added] = col[i] / norm;
            }
            added += 1;
        }
    }

    result
}

/// Extend a set of k unitary columns in an m×k complex matrix to m×m_full
/// by adding orthogonal complement columns via Gram-Schmidt.
fn complex_extend_unitary_columns(
    u: &[(f64, f64)],
    m: usize,
    k: usize,
    m_full: usize,
) -> Vec<(f64, f64)> {
    let zero = (0.0, 0.0);
    let one = (1.0, 0.0);
    let mut result = vec![zero; m * m_full];

    for i in 0..m {
        for j in 0..k {
            result[i * m_full + j] = u[i * k + j];
        }
    }

    let mut added = k;
    for basis_idx in 0..m {
        if added >= m_full {
            break;
        }

        let mut col = vec![zero; m];
        col[basis_idx] = one;

        for j in 0..added {
            let mut dot = zero;
            for i in 0..m {
                dot = complex_add(
                    dot,
                    complex_mul(complex_conj(result[i * m_full + j]), col[i]),
                );
            }
            for i in 0..m {
                col[i] = complex_sub(col[i], complex_mul(dot, result[i * m_full + j]));
            }
        }

        let norm: f64 = col
            .iter()
            .map(|x| x.0 * x.0 + x.1 * x.1)
            .sum::<f64>()
            .sqrt();
        if norm > f64::EPSILON * 1e4 {
            for i in 0..m {
                result[i * m_full + added] = complex_div(col[i], (norm, 0.0));
            }
            added += 1;
        }
    }

    result
}

// ── Eigh (Symmetric Eigendecomposition) ────────────────────────────

/// Closed-form analytic eigendecomposition of a 3×3 real **symmetric** matrix.
///
/// `a` is row-major length 9; only the upper triangle (`a[0],a[1],a[2],a[4],
/// a[5],a[8]`) is read, matching the iterative Jacobi kernel's symmetric
/// assumption. Returns `(w, v)` with eigenvalues **ascending** and eigenvectors
/// as **columns** (`v[row*3 + col]`), orthonormal, satisfying `A = V diag(w) Vᵀ`.
///
/// Returns `None` when the analytic result fails a tight residual /
/// orthonormality check (the caller then falls back to the iterative Jacobi
/// solver, preserving parity). This is the >=2x algorithmic replacement for the
/// per-matrix Jacobi sweep on the hot 3×3 batched-eigh path: eigenvalues come
/// from the trigonometric solution of the characteristic cubic (Smith 1961 /
/// Kopp), and eigenvectors from a numerically robust "isolate the
/// best-separated eigenvalue, then solve the 2×2 problem in the orthogonal
/// plane" construction that stays well-conditioned for degenerate spectra.
pub fn analytic_eigh_3x3(a: &[f64]) -> Option<([f64; 3], [f64; 9])> {
    if a.len() != 9 {
        return None;
    }
    let a00 = a[0];
    let a11 = a[4];
    let a22 = a[8];
    let a01 = a[1];
    let a02 = a[2];
    let a12 = a[5];

    // Reject non-finite inputs (let Jacobi handle / surface them).
    if !(a00.is_finite()
        && a11.is_finite()
        && a22.is_finite()
        && a01.is_finite()
        && a02.is_finite()
        && a12.is_finite())
    {
        return None;
    }

    let p1 = a01 * a01 + a02 * a02 + a12 * a12;

    // Eigenvalues, ascending.
    let w = if p1 == 0.0 {
        // Already diagonal: eigenvalues are the diagonal entries.
        let mut d = [a00, a11, a22];
        d.sort_by(f64::total_cmp);
        d
    } else {
        let q = (a00 + a11 + a22) / 3.0;
        let p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) + (a22 - q) * (a22 - q) + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        // p = sqrt(non-negative finite) so p is finite & >= 0 here; p == 0 only
        // for an (already-handled) scalar matrix. Guard defensively.
        if p <= 0.0 {
            return None;
        }
        // B = (A - q I) / p ; r = det(B) / 2.
        let b00 = (a00 - q) / p;
        let b11 = (a11 - q) / p;
        let b22 = (a22 - q) / p;
        let b01 = a01 / p;
        let b02 = a02 / p;
        let b12 = a12 / p;
        let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02);
        let r = (det_b / 2.0).clamp(-1.0, 1.0);
        let phi = r.acos() / 3.0;
        let two_p = 2.0 * p;
        // The three roots are q + 2p·cos(phi + 2πk/3), k=0,1,2; 2π/3 = 4·(π/3)·½.
        let third = std::f64::consts::FRAC_PI_3; // π/3
        let eig_hi = q + two_p * phi.cos();
        let eig_lo = q + two_p * (phi + 2.0 * third).cos();
        let eig_mid = 3.0 * q - eig_hi - eig_lo;
        let mut d = [eig_lo, eig_mid, eig_hi];
        d.sort_by(f64::total_cmp);
        d
    };

    // Eigenvectors. Isolate the best-separated eigenvalue (an endpoint of the
    // ascending spectrum: whichever of the two outer gaps is larger), solve it
    // directly, then reduce the remaining pair to a 2×2 problem in the plane
    // orthogonal to that eigenvector — robust even when the other two
    // eigenvalues coincide.
    let gap_low = w[1] - w[0];
    let gap_high = w[2] - w[1];
    let iso = if gap_low >= gap_high { 0usize } else { 2usize };

    let v_iso = eigenvector_3x3_for(a00, a11, a22, a01, a02, a12, w[iso])?;

    // Orthonormal basis {u1, u2} of the plane ⟂ v_iso.
    let (u1, u2) = orthonormal_complement_basis(v_iso);

    // Project A onto {u1, u2}: symmetric 2×2 B.
    let au1 = symv3(a00, a11, a22, a01, a02, a12, u1);
    let au2 = symv3(a00, a11, a22, a01, a02, a12, u2);
    let b11 = dot3(u1, au1);
    let b12 = dot3(u1, au2);
    let b22 = dot3(u2, au2);

    // 2×2 symmetric eigenvectors (eigenvalues for output come from `w`, not B).
    let tr = b11 + b22;
    let diff = b11 - b22;
    let disc = (diff * diff + 4.0 * b12 * b12).max(0.0).sqrt();
    let mu_minus = 0.5 * (tr - disc);
    let mu_plus = 0.5 * (tr + disc);
    let c_minus = eig2x2_vector(b11, b12, b22, mu_minus);
    let c_plus = eig2x2_vector(b11, b12, b22, mu_plus);
    let vec_minus = combine2(u1, u2, c_minus);
    let vec_plus = combine2(u1, u2, c_plus);

    // Assemble columns in ascending order. `iso` is an endpoint; the remaining
    // two ascending slots take (mu_minus, mu_plus), themselves ascending.
    let mut cols = [[0.0_f64; 3]; 3];
    cols[iso] = v_iso;
    let (slot_lo, slot_hi) = if iso == 0 {
        (1usize, 2usize)
    } else {
        (0usize, 1usize)
    };
    cols[slot_lo] = vec_minus;
    cols[slot_hi] = vec_plus;

    let mut v = [0.0_f64; 9];
    for col in 0..3 {
        v[col] = cols[col][0];
        v[3 + col] = cols[col][1];
        v[6 + col] = cols[col][2];
    }

    // Validate: orthonormality (VᵀV = I) and reconstruction (A V = V diag(w)).
    let scale = 1.0 + a00.abs().max(a11.abs()).max(a22.abs()).max(p1.sqrt());
    let tol = 1e-9 * scale;
    for i in 0..3 {
        for j in 0..3 {
            let ci = [v[i], v[3 + i], v[6 + i]];
            let cj = [v[j], v[3 + j], v[6 + j]];
            let d = dot3(ci, cj);
            let target = if i == j { 1.0 } else { 0.0 };
            if (d - target).abs() > 1e-9 {
                return None;
            }
        }
    }
    for col in 0..3 {
        let vc = [v[col], v[3 + col], v[6 + col]];
        let avc = symv3(a00, a11, a22, a01, a02, a12, vc);
        for row in 0..3 {
            if (avc[row] - w[col] * vc[row]).abs() > tol {
                return None;
            }
        }
    }

    Some((w, v))
}

/// Eigenvector of a 3×3 symmetric matrix for eigenvalue `lambda`, via the
/// largest cross product of the rows of `(A - lambda I)` (each row is ⟂ the
/// eigenvector). Returns `None` if the rows are too parallel to recover a
/// direction (degenerate at this eigenvalue).
#[allow(clippy::too_many_arguments)]
fn eigenvector_3x3_for(
    a00: f64,
    a11: f64,
    a22: f64,
    a01: f64,
    a02: f64,
    a12: f64,
    lambda: f64,
) -> Option<[f64; 3]> {
    let r0 = [a00 - lambda, a01, a02];
    let r1 = [a01, a11 - lambda, a12];
    let r2 = [a02, a12, a22 - lambda];
    let c0 = cross3(r0, r1);
    let c1 = cross3(r1, r2);
    let c2 = cross3(r2, r0);
    let n0 = dot3(c0, c0);
    let n1 = dot3(c1, c1);
    let n2 = dot3(c2, c2);
    let (best, best_n) = if n0 >= n1 && n0 >= n2 {
        (c0, n0)
    } else if n1 >= n2 {
        (c1, n1)
    } else {
        (c2, n2)
    };
    if best_n <= 0.0 {
        return None;
    }
    let inv = 1.0 / best_n.sqrt();
    Some([best[0] * inv, best[1] * inv, best[2] * inv])
}

/// Build an orthonormal basis `{u1, u2}` of the plane orthogonal to the unit
/// vector `v`.
fn orthonormal_complement_basis(v: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    // Choose the standard basis axis least aligned with `v` for stability.
    let ax = v[0].abs();
    let ay = v[1].abs();
    let az = v[2].abs();
    let e = if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let d = dot3(e, v);
    let mut u1 = [e[0] - d * v[0], e[1] - d * v[1], e[2] - d * v[2]];
    let n = dot3(u1, u1).sqrt();
    let inv = 1.0 / n;
    u1 = [u1[0] * inv, u1[1] * inv, u1[2] * inv];
    let u2 = cross3(v, u1);
    (u1, u2)
}

/// Eigenvector (in 2D) of the symmetric 2×2 `[[b11,b12],[b12,b22]]` for
/// eigenvalue `mu`, normalized.
fn eig2x2_vector(b11: f64, b12: f64, b22: f64, mu: f64) -> [f64; 2] {
    // Null vector of (B - mu I): rows (b11-mu, b12) and (b12, b22-mu); a null
    // vector is ⟂ a row, i.e. (-row.1, row.0). Use the longer row.
    let r0 = [b11 - mu, b12];
    let r1 = [b12, b22 - mu];
    let n0 = r0[0] * r0[0] + r0[1] * r0[1];
    let n1 = r1[0] * r1[0] + r1[1] * r1[1];
    let v = if n0 >= n1 {
        [-r0[1], r0[0]]
    } else {
        [-r1[1], r1[0]]
    };
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if n == 0.0 {
        // B is (near) a multiple of I in this plane: any unit vector works.
        [1.0, 0.0]
    } else {
        [v[0] / n, v[1] / n]
    }
}

/// `c[0]*u1 + c[1]*u2` for a 2-vector `c` expressed in the `{u1,u2}` basis.
fn combine2(u1: [f64; 3], u2: [f64; 3], c: [f64; 2]) -> [f64; 3] {
    [
        c[0] * u1[0] + c[1] * u2[0],
        c[0] * u1[1] + c[1] * u2[1],
        c[0] * u1[2] + c[1] * u2[2],
    ]
}

/// Symmetric 3×3 matrix-vector product `A x` from the upper triangle.
#[allow(clippy::too_many_arguments)]
fn symv3(a00: f64, a11: f64, a22: f64, a01: f64, a02: f64, a12: f64, x: [f64; 3]) -> [f64; 3] {
    [
        a00 * x[0] + a01 * x[1] + a02 * x[2],
        a01 * x[0] + a11 * x[1] + a12 * x[2],
        a02 * x[0] + a12 * x[1] + a22 * x[2],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Compute the eigendecomposition of a Hermitian matrix: A = V diag(W) V^H.
///
/// Returns `[W, V]` where W is a vector of real eigenvalues (ascending) and V
/// contains eigenvectors as columns.
pub(crate) fn eval_eigh(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::Eigh;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    // Dispatch based on dtype: real matrices use simpler real Jacobi algorithm
    let is_complex = matches!(inputs[0], Value::Tensor(ref t) if t.dtype == DType::Complex64 || t.dtype == DType::Complex128);

    if is_complex {
        let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
        let zero = (0.0, 0.0);
        let one = (1.0, 0.0);
        if m != n {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("Eigh requires a square matrix, got {m}x{n}"),
            });
        }

        // For 2x2 Hermitian matrices, use direct quadratic formula
        let (eigenvalues, eigenvectors) = if m == 2 {
            // A = [[a, b], [b*, d]] where a, d are real
            let aa = a[0].0;
            let b = a[1];
            let dd = a[3].0;
            let b_abs_sq = b.0 * b.0 + b.1 * b.1;

            // Eigenvalues from characteristic equation: λ² - (a+d)λ + (ad - |b|²) = 0
            let trace = aa + dd;
            let det = aa * dd - b_abs_sq;
            let discrim = trace * trace - 4.0 * det;
            let sqrt_discrim = discrim.max(0.0).sqrt();

            let lambda1 = (trace - sqrt_discrim) / 2.0;
            let lambda2 = (trace + sqrt_discrim) / 2.0;

            // Eigenvector for λ1: (A - λ1*I) * v = 0
            // v is in null space of [[a-λ1, b], [b*, d-λ1]]
            // v1 = [-b, a-λ1]^T (normalized) or use [d-λ1, -b*]^T
            let mut v = vec![zero; 4];

            if b_abs_sq > f64::EPSILON * f64::EPSILON {
                // Non-trivial off-diagonal
                let v1_unnorm = (complex_conj(b), (lambda1 - aa, 0.0));
                let v1_norm = (v1_unnorm.0.0 * v1_unnorm.0.0
                    + v1_unnorm.0.1 * v1_unnorm.0.1
                    + v1_unnorm.1.0 * v1_unnorm.1.0
                    + v1_unnorm.1.1 * v1_unnorm.1.1)
                    .sqrt();
                v[0] = complex_div(v1_unnorm.0, (v1_norm, 0.0));
                v[2] = complex_div(v1_unnorm.1, (v1_norm, 0.0));

                let v2_unnorm = (complex_conj(b), (lambda2 - aa, 0.0));
                let v2_norm = (v2_unnorm.0.0 * v2_unnorm.0.0
                    + v2_unnorm.0.1 * v2_unnorm.0.1
                    + v2_unnorm.1.0 * v2_unnorm.1.0
                    + v2_unnorm.1.1 * v2_unnorm.1.1)
                    .sqrt();
                v[1] = complex_div(v2_unnorm.0, (v2_norm, 0.0));
                v[3] = complex_div(v2_unnorm.1, (v2_norm, 0.0));
            } else {
                // Diagonal matrix, eigenvectors are standard basis
                v[0] = one;
                v[3] = one;
            }

            (vec![lambda1, lambda2], v)
        } else {
            let mut a_work = a;
            // Row-cyclic sweeps (O(m³·sweeps)) instead of the max-pivot kernel's
            // O(m⁴); same Hermitian spectrum to machine precision, sorted below.
            complex_jacobi_eigendecomposition_cyclic(&mut a_work, m)
        };

        // Sort eigenvalues in ascending order (JAX convention for eigh)
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

        let mut w_sorted = vec![0.0_f64; m];
        let mut v_sorted = vec![zero; m * m];
        for (new_col, &old_col) in indices.iter().enumerate() {
            w_sorted[new_col] = eigenvalues[old_col];
            for row in 0..m {
                v_sorted[row * m + new_col] = eigenvectors[row * m + old_col];
            }
        }

        // Eigenvalues are always real for Hermitian matrices
        let w_dtype = match dtype {
            DType::Complex64 => DType::F32,
            DType::Complex128 => DType::F64,
            _ => dtype,
        };
        let w_elements: Vec<Literal> = w_sorted
            .iter()
            .map(|&v| linalg_literal_from_f64(w_dtype, v))
            .collect();
        let w_shape = Shape {
            dims: vec![m as u32],
        };
        let w_tensor =
            TensorValue::new(w_dtype, w_shape, w_elements).map_err(EvalError::InvalidTensor)?;
        let w_val = Value::Tensor(w_tensor);

        let v_val = complex_matrix_to_value(m, m, &v_sorted, dtype)?;

        return Ok(vec![w_val, v_val]);
    }

    // Real symmetric matrix path
    let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("Eigh requires a square matrix, got {m}x{n}"),
        });
    }

    // Fast path: closed-form analytic 3×3 symmetric eigensolver (ascending,
    // orthonormal columns). Falls back to iterative Jacobi when the analytic
    // residual check fails (ill-conditioned), preserving parity. The batched
    // path (fj-dispatch append_eigh_decomposition) calls the SAME
    // `analytic_eigh_3x3`, so batched and single eval stay bit-identical.
    let (w_sorted, v_sorted) = if m == 3
        && let Some((w3, v3)) = analytic_eigh_3x3(&a)
    {
        (w3.to_vec(), v3.to_vec())
    } else {
        // Tridiagonalization + implicit-shift QL (LAPACK `tred2`+`tql2`), `~5n³`
        // versus cyclic Jacobi's `~8·O(n³)` rotation sweeps. Same spectrum to
        // machine precision; eigenvectors differ only within the eigendecomposition's
        // intrinsic sign/rotation freedom, and eigh conformance is reconstruction +
        // spectrum based (V diag(w) Vᵀ = A).
        let (eigenvalues, eigenvectors) = tridiag_ql_eigendecomposition(&a, m);

        // Sort eigenvalues in ascending order (JAX convention for eigh)
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

        let mut w_sorted = vec![0.0_f64; m];
        let mut v_sorted = vec![0.0_f64; m * m];
        for (new_col, &old_col) in indices.iter().enumerate() {
            w_sorted[new_col] = eigenvalues[old_col];
            for row in 0..m {
                v_sorted[row * m + new_col] = eigenvectors[row * m + old_col];
            }
        }
        (w_sorted, v_sorted)
    };

    let w_elements: Vec<Literal> = w_sorted
        .iter()
        .map(|&v| linalg_literal_from_f64(dtype, v))
        .collect();
    let w_shape = Shape {
        dims: vec![m as u32],
    };
    let w_tensor =
        TensorValue::new(dtype, w_shape, w_elements).map_err(EvalError::InvalidTensor)?;
    let w_val = Value::Tensor(w_tensor);

    let v_val = matrix_to_value(m, m, &v_sorted, dtype)?;

    Ok(vec![w_val, v_val])
}

// ── General Linear Solve ────────────────────────────────────────────

/// Solve the complex system `A x = B` (A is n×n, B is n×`ncols`, both row-major
/// complex) by Gaussian elimination with partial pivoting on the augmented matrix
/// `[A | B]`. Returns the n×`ncols` solution row-major. A singular `A` is not an
/// error: the zero/near-zero pivot divides through to inf/NaN, matching `eval_solve`'s
/// real-path contract (JAX returns inf/NaN; NumPy raises). All `ncols` right-hand
/// sides share one O(n³) factorization.
/// Cache-blocked right-looking complex LU with partial pivoting, factoring the
/// row-major n×n complex matrix `lu` in place and returning the swap-built row
/// permutation `perm` (`perm[i]` = original row now in position `i`). Mirrors
/// [`lu_factor_real_blocked`]: each panel of `LU_BLOCK_SIZE` columns is factored
/// over all rows below (full-column complex pivot, whole-row swaps, panel-local
/// rank-1 updates), then the trailing block row `U12 = L11⁻¹·A12` is forward-solved
/// and the Schur update `A22 -= L21·U12` runs through the threaded complex GEMM
/// [`rank2_complex_matmul`]. Combined L\U is stored in `lu` (unit-diagonal L below,
/// U on/above). Numerically equivalent to the scalar elimination (P·A = L·U to
/// machine precision) — the block-reordered GEMM sum differs only at ulp level,
/// exactly JAX's blocked complex getrf guarantee. Divides through tiny pivots (no
/// skip), so a singular column propagates inf/NaN, matching `complex_solve_system`.
fn complex_lu_factor_blocked(lu: &mut [(f64, f64)], n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    let nb = LU_BLOCK_SIZE;

    let mut j = 0;
    while j < n {
        let jb = nb.min(n - j);
        let panel_end = j + jb;

        // (1) Factor panel columns [j, panel_end) over rows [j, n): full-column
        //     complex pivot + whole-row swap, rank-1 updates confined to the panel.
        for col in j..panel_end {
            let mut best = complex_abs(lu[col * n + col]);
            let mut max_row = col;
            for row in (col + 1)..n {
                let mag = complex_abs(lu[row * n + col]);
                if mag > best {
                    best = mag;
                    max_row = row;
                }
            }
            if max_row != col {
                perm.swap(col, max_row);
                for c in 0..n {
                    lu.swap(col * n + c, max_row * n + c);
                }
            }
            let pivot = lu[col * n + col];
            for row in (col + 1)..n {
                let factor = complex_div(lu[row * n + col], pivot);
                lu[row * n + col] = factor;
                for c in (col + 1)..panel_end {
                    lu[row * n + c] =
                        complex_sub(lu[row * n + c], complex_mul(factor, lu[col * n + c]));
                }
            }
        }

        // (2) Forward-solve U12 = L11⁻¹ · A12 (panel rows × trailing columns).
        if panel_end < n {
            for ri in 1..jb {
                let r = j + ri;
                for t in 0..ri {
                    let l_rt = lu[r * n + (j + t)];
                    if l_rt != (0.0, 0.0) {
                        let trow = j + t;
                        for c in panel_end..n {
                            lu[r * n + c] =
                                complex_sub(lu[r * n + c], complex_mul(l_rt, lu[trow * n + c]));
                        }
                    }
                }
            }
        }

        // (3) Trailing Schur update A22 -= L21 · U12 via the threaded complex GEMM.
        let rows_below = n - panel_end;
        let cols_right = n - panel_end;
        if rows_below > 0 && cols_right > 0 {
            let mut l21 = Vec::with_capacity(rows_below * jb);
            for p in 0..rows_below {
                let src = (panel_end + p) * n + j;
                l21.extend_from_slice(&lu[src..src + jb]);
            }
            let mut u12 = Vec::with_capacity(jb * cols_right);
            for t in 0..jb {
                let src = (j + t) * n + panel_end;
                u12.extend_from_slice(&lu[src..src + cols_right]);
            }
            let prod = rank2_complex_matmul(&l21, rows_below, jb, &u12, cols_right);
            for p in 0..rows_below {
                let row = (panel_end + p) * n + panel_end;
                let pr = p * cols_right;
                for q in 0..cols_right {
                    lu[row + q] = complex_sub(lu[row + q], prod[pr + q]);
                }
            }
        }

        j = panel_end;
    }

    perm
}

/// Forward/back substitution for the combined complex L\U factor from
/// [`complex_lu_factor_blocked`], solving `A·X = B` for the `ncols` RHS columns of
/// `b` (row-major n×ncols) under row permutation `perm`. Unit-diagonal L (forward,
/// no divide), U on/above the diagonal (back, divide by `U[i,i]`).
fn complex_lu_solve(
    lu: &[(f64, f64)],
    perm: &[usize],
    b: &[(f64, f64)],
    n: usize,
    ncols: usize,
) -> Vec<(f64, f64)> {
    // Solve all `ncols` RHS columns together (X starts as P·B, solved in place): read
    // each `lu[i][k]` ONCE and apply it across every column (inner column loop
    // vectorizes) so the shared L\U factor streams from cache once instead of once
    // per column. BIT-IDENTICAL to the per-column forward/back substitution — each
    // column's ascending-`k` complex fold is unchanged, only the loop nesting differs.
    let mut x = vec![(0.0_f64, 0.0_f64); n * ncols];
    // Forward: y[i][:] = (P·B)[i][:] − Σ_{k<i} lu[i][k]·y[k][:]  (stored in x).
    for i in 0..n {
        let xi = i * ncols;
        let src = perm[i] * ncols;
        x[xi..xi + ncols].copy_from_slice(&b[src..src + ncols]);
        for k in 0..i {
            let l = lu[i * n + k];
            let (head, tail) = x.split_at_mut(xi);
            let yk = &head[k * ncols..k * ncols + ncols];
            let yi = &mut tail[..ncols];
            for col in 0..ncols {
                yi[col] = complex_sub(yi[col], complex_mul(l, yk[col]));
            }
        }
    }
    // Back: x[i][:] = (y[i][:] − Σ_{k>i} lu[i][k]·x[k][:]) / lu[i][i].
    for i in (0..n).rev() {
        let xi = i * ncols;
        for k in (i + 1)..n {
            let u = lu[i * n + k];
            let (head, tail) = x.split_at_mut(k * ncols);
            let yi = &mut head[xi..xi + ncols];
            let xk = &tail[..ncols];
            for col in 0..ncols {
                yi[col] = complex_sub(yi[col], complex_mul(u, xk[col]));
            }
        }
        let diag = lu[i * n + i];
        let xi_slice = &mut x[xi..xi + ncols];
        for v in xi_slice.iter_mut() {
            *v = complex_div(*v, diag);
        }
    }
    x
}

/// Parity sign `(-1)^(transpositions)` of a permutation given as `perm[i]` = the
/// element now at position `i`. A cycle of length `L` is `L-1` transpositions, so
/// even-length cycles flip the sign.
fn complex_perm_sign(perm: &[usize]) -> f64 {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut sign = 1.0_f64;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut len = 0usize;
        let mut node = start;
        while !visited[node] {
            visited[node] = true;
            node = perm[node];
            len += 1;
        }
        if len.is_multiple_of(2) {
            sign = -sign;
        }
    }
    sign
}

fn complex_solve_system(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    n: usize,
    ncols: usize,
) -> Vec<(f64, f64)> {
    // Large systems route through the cache-blocked complex LU whose O(n³) Schur
    // update runs at complex-GEMM speed; small n keeps the bit-identical augmented
    // Gaussian elimination below (parity goldens, same gate as the real path).
    if n >= LU_BLOCK_THRESHOLD {
        let mut lu = a.to_vec();
        let perm = complex_lu_factor_blocked(&mut lu, n);
        return complex_lu_solve(&lu, &perm, b, n, ncols);
    }

    let w = n + ncols;
    let mut m = vec![(0.0_f64, 0.0_f64); n * w];
    for i in 0..n {
        m[i * w..i * w + n].copy_from_slice(&a[i * n..i * n + n]);
        m[i * w + n..i * w + w].copy_from_slice(&b[i * ncols..i * ncols + ncols]);
    }

    for col in 0..n {
        // Partial pivot: largest-magnitude entry at/below the diagonal (best
        // available conditioning; a zero best means a singular column → inf/NaN).
        let mut piv = col;
        let mut best = complex_abs(m[col * w + col]);
        for r in (col + 1)..n {
            let mag = complex_abs(m[r * w + col]);
            if mag > best {
                best = mag;
                piv = r;
            }
        }
        if piv != col {
            for c in 0..w {
                m.swap(col * w + c, piv * w + c);
            }
        }
        let pivot = m[col * w + col];
        for r in (col + 1)..n {
            let factor = complex_div(m[r * w + col], pivot);
            for c in col..w {
                m[r * w + c] = complex_sub(m[r * w + c], complex_mul(factor, m[col * w + c]));
            }
        }
    }

    let mut x = vec![(0.0_f64, 0.0_f64); n * ncols];
    for jcol in 0..ncols {
        for row in (0..n).rev() {
            let mut s = m[row * w + n + jcol];
            for c in (row + 1)..n {
                s = complex_sub(s, complex_mul(m[row * w + c], x[c * ncols + jcol]));
            }
            x[row * ncols + jcol] = complex_div(s, m[row * w + row]);
        }
    }
    x
}

/// Solve the linear system Ax = b using LU decomposition with partial pivoting.
///
/// Matches `jnp.linalg.solve(a, b)` / `jax.scipy.linalg.solve(a, b)`.
///
/// inputs: [A, b] where A is n×n and b is n or n×m
/// Returns x such that Ax = b
pub(crate) fn eval_solve(
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let _ = params; // unused for now
    let primitive = Primitive::Solve;

    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    let (m_a, n_a, a, dtype_a) = extract_complex_matrix(primitive, &inputs[0])?;

    if m_a != n_a {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("A must be square, got {m_a}x{n_a}"),
        });
    }

    let n = m_a;

    // Complex A or b → complex Gaussian-elimination solve (jnp.linalg.solve
    // supports complex). The result of a solve is inexact, so integer/bool inputs
    // promote to a floating dtype.
    let dtype_b = inputs[1].dtype();
    if matches!(dtype_a, DType::Complex64 | DType::Complex128)
        || matches!(dtype_b, DType::Complex64 | DType::Complex128)
    {
        return eval_solve_complex(primitive, &a, n, &inputs[1], dtype_a, dtype_b);
    }
    let output_dtype = match promote_dtype(dtype_a, dtype_b) {
        dt @ (DType::F16 | DType::BF16 | DType::F32 | DType::F64) => dt,
        _ => DType::F64,
    };

    // Handle vector or matrix b
    let (b_rows, b_cols, b_elements) = match &inputs[1] {
        Value::Tensor(t) => {
            if t.rank() == 1 {
                let len = t.shape.dims[0] as usize;
                let elems: Vec<f64> = t.elements.iter().filter_map(|l| l.as_f64()).collect();
                if elems.len() != len {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "b must have numeric elements",
                    });
                }
                (len, 1, elems)
            } else if t.rank() == 2 {
                let rows = t.shape.dims[0] as usize;
                let cols = t.shape.dims[1] as usize;
                let elems: Vec<f64> = t.elements.iter().filter_map(|l| l.as_f64()).collect();
                if elems.len() != rows * cols {
                    return Err(EvalError::TypeMismatch {
                        primitive,
                        detail: "b must have numeric elements",
                    });
                }
                (rows, cols, elems)
            } else {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("b must be rank-1 or rank-2, got rank-{}", t.rank()),
                });
            }
        }
        Value::Scalar(lit) => {
            if let Some(v) = lit.as_f64() {
                (1, 1, vec![v])
            } else {
                return Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "b must be numeric",
                });
            }
        }
    };

    if b_rows != n {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: Shape {
                dims: vec![n as u32, n as u32],
            },
            right: Shape {
                dims: vec![b_rows as u32, b_cols as u32],
            },
        });
    }

    // A is guaranteed real here (complex was rejected above), so the
    // imaginary parts are all zero.
    let a_real: Vec<f64> = a.iter().map(|(re, _im)| *re).collect();

    // Solve using existing solve function
    let result = if b_cols == 1 {
        solve(&a_real, &b_elements, n).ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "singular matrix".to_owned(),
        })?
    } else {
        solve_multi_rhs(&a_real, &b_elements, n, b_cols).ok_or_else(|| EvalError::Unsupported {
            primitive,
            detail: "singular matrix".to_owned(),
        })?
    };

    // Build output tensor at the promoted dtype. The previous code emitted
    // F64Bits literals under the declared `output_dtype`, so an F32 input
    // produced a tensor whose declared dtype disagreed with its literals.
    let elements: Vec<Literal> = result
        .iter()
        .map(|&v| linalg_literal_from_f64(output_dtype, v))
        .collect();

    let shape = if b_cols == 1 {
        Shape {
            dims: vec![n as u32],
        }
    } else {
        Shape {
            dims: vec![n as u32, b_cols as u32],
        }
    };

    let tensor =
        TensorValue::new(output_dtype, shape, elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

/// Complex `jnp.linalg.solve(A, b)`: A is the already-extracted complex n×n matrix,
/// `b_val` is the (rank-1 vector or rank-2 matrix) right-hand side. Mirrors the real
/// path — errors on singular, shape-checks `b`, returns a complex tensor of shape
/// `[n]` (vector b) or `[n, ncols]` (matrix b). Result dtype follows numpy/JAX
/// `result_type`: complex128 if any operand is complex128 or a float64 is mixed with
/// the complex operand, else complex64.
fn eval_solve_complex(
    primitive: Primitive,
    a: &[(f64, f64)],
    n: usize,
    b_val: &Value,
    dtype_a: DType,
    dtype_b: DType,
) -> Result<Value, EvalError> {
    let to_c = |lit: &Literal| -> Option<(f64, f64)> {
        lit.as_complex128()
            .or_else(|| lit.as_complex64().map(|(r, i)| (r as f64, i as f64)))
            .or_else(|| lit.as_f64().map(|r| (r, 0.0)))
    };
    let numeric_err = EvalError::TypeMismatch {
        primitive,
        detail: "b must have numeric elements",
    };
    let (b_rows, b_cols, is_vector, b_elems): (usize, usize, bool, Vec<(f64, f64)>) = match b_val {
        Value::Tensor(t) if t.rank() == 1 => {
            let len = t.shape.dims[0] as usize;
            let e: Option<Vec<_>> = t.elements.iter().map(&to_c).collect();
            (len, 1, true, e.ok_or(numeric_err)?)
        }
        Value::Tensor(t) if t.rank() == 2 => {
            let rows = t.shape.dims[0] as usize;
            let cols = t.shape.dims[1] as usize;
            let e: Option<Vec<_>> = t.elements.iter().map(&to_c).collect();
            (rows, cols, false, e.ok_or(numeric_err)?)
        }
        Value::Tensor(t) => {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("b must be rank-1 or rank-2, got rank-{}", t.rank()),
            });
        }
        Value::Scalar(lit) => (1, 1, true, vec![to_c(lit).ok_or(numeric_err)?]),
    };

    if b_rows != n {
        return Err(EvalError::ShapeMismatch {
            primitive,
            left: Shape {
                dims: vec![n as u32, n as u32],
            },
            right: Shape {
                dims: vec![b_rows as u32, b_cols as u32],
            },
        });
    }

    // Singular A divides through to inf/NaN (not an error), matching the real path.
    let x = complex_solve_system(a, &b_elems, n, b_cols);

    let wide = matches!(dtype_a, DType::Complex128 | DType::F64)
        || matches!(dtype_b, DType::Complex128 | DType::F64);
    let out_dtype = if wide {
        DType::Complex128
    } else {
        DType::Complex64
    };
    let elements: Vec<Literal> = x
        .iter()
        .map(|&(re, im)| match out_dtype {
            DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
            _ => Literal::from_complex128(re, im),
        })
        .collect();
    let shape = if is_vector {
        Shape {
            dims: vec![n as u32],
        }
    } else {
        Shape {
            dims: vec![n as u32, b_cols as u32],
        }
    };
    let tensor = TensorValue::new(out_dtype, shape, elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

/// Complex determinant via LU with partial pivoting: the product of the complex
/// pivots times `(-1)^(row swaps)`. Returns `(0,0)` for a singular matrix (a zero
/// pivot). `n == 0` gives the empty-product `1`.
fn complex_det(a: &[(f64, f64)], n: usize) -> (f64, f64) {
    // Large inputs route through the cache-blocked complex LU (det = sign(P) ·
    // Π U_ii). Numerically equivalent to the scalar elimination, gated above the
    // parity-golden sizes (small n stays bit-identical). An exactly-zero pivot on
    // the U diagonal short-circuits to (0,0) as the scalar path does.
    if n >= LU_BLOCK_THRESHOLD {
        let mut lu = a.to_vec();
        let perm = complex_lu_factor_blocked(&mut lu, n);
        let mut det = (complex_perm_sign(&perm), 0.0_f64);
        for i in 0..n {
            let u = lu[i * n + i];
            if u == (0.0, 0.0) {
                return (0.0, 0.0);
            }
            det = complex_mul(det, u);
        }
        return det;
    }

    let mut m = a.to_vec();
    let mut det = (1.0_f64, 0.0_f64);
    for col in 0..n {
        let mut piv = col;
        let mut best = complex_abs(m[col * n + col]);
        for r in (col + 1)..n {
            let mag = complex_abs(m[r * n + col]);
            if mag > best {
                best = mag;
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            det = (-det.0, -det.1);
        }
        let pivot = m[col * n + col];
        det = complex_mul(det, pivot);
        if complex_abs(pivot) == 0.0 {
            return (0.0, 0.0); // singular
        }
        for r in (col + 1)..n {
            let factor = complex_div(m[r * n + col], pivot);
            for c in (col + 1)..n {
                m[r * n + c] = complex_sub(m[r * n + c], complex_mul(factor, m[col * n + c]));
            }
        }
    }
    det
}

/// Evaluate matrix determinant.
///
/// Input: A square matrix [n, n]
/// Output: Scalar determinant
pub(crate) fn eval_det(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Value, EvalError> {
    let primitive = Primitive::Det;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "det requires square matrix",
        });
    }

    // Complex input → complex determinant (the real-part-only path below would be
    // silently wrong). Output keeps the input's complex dtype.
    if matches!(dtype, DType::Complex64 | DType::Complex128) {
        let (re, im) = complex_det(&a, n);
        let lit = if dtype == DType::Complex64 {
            Literal::from_complex64(re as f32, im as f32)
        } else {
            Literal::from_complex128(re, im)
        };
        return Ok(Value::Scalar(lit));
    }

    let a_real: Vec<f64> = a.iter().map(|(re, _im)| *re).collect();
    let d = det(&a_real, n);
    // det of a real matrix keeps the input's float dtype (JAX: f32 in -> f32 out);
    // the prior scalar_f64 widened f32/bf16/f16 determinants to f64. Integer inputs
    // fall through linalg_literal_from_f64's `_ => from_f64` arm (int linalg -> float).
    Ok(Value::Scalar(linalg_literal_from_f64(dtype, d)))
}

/// Evaluate sign and log of matrix determinant.
///
/// Input: A square matrix [n, n]
/// Output: Two scalars (sign, logabsdet)
pub(crate) fn eval_slogdet(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::Slogdet;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "slogdet requires square matrix",
        });
    }

    // Complex input → complex sign + real logabsdet. For det = r·e^{iθ},
    // sign = e^{iθ} = det/|det| (0 if singular) and logabsdet = ln|det| (−∞ if
    // singular). The real-part-only path below would be silently wrong.
    if matches!(dtype, DType::Complex64 | DType::Complex128) {
        let (re, im) = complex_det(&a, n);
        let mag = re.hypot(im);
        let logabsdet = mag.ln();
        let (sre, sim) = if mag > 0.0 {
            (re / mag, im / mag)
        } else {
            (0.0, 0.0)
        };
        let (sign_lit, log_val) = if dtype == DType::Complex64 {
            (
                Literal::from_complex64(sre as f32, sim as f32),
                Value::Scalar(Literal::from_f32(logabsdet as f32)),
            )
        } else {
            (
                Literal::from_complex128(sre, sim),
                Value::scalar_f64(logabsdet),
            )
        };
        return Ok(vec![Value::Scalar(sign_lit), log_val]);
    }

    let a_real: Vec<f64> = a.iter().map(|(re, _im)| *re).collect();
    let (sign, logabsdet) = slogdet(&a_real, n);
    // sign + logabsdet keep the input's float dtype (JAX: f32 in -> f32 out); the prior
    // scalar_f64 widened f32/bf16/f16 slogdet outputs to f64.
    Ok(vec![
        Value::Scalar(linalg_literal_from_f64(dtype, sign)),
        Value::Scalar(linalg_literal_from_f64(dtype, logabsdet)),
    ])
}

// ── General Eigendecomposition (eig) ────────────────────────────────

/// Evaluate general eigendecomposition for non-symmetric matrices.
///
/// Input: A square matrix [n, n]
/// Output: (eigenvalues [n] complex, eigenvectors [n, n] complex)
///
/// Uses QR iteration for small matrices. Returns complex eigenvalues/eigenvectors
/// since non-symmetric matrices can have complex eigenvalues.
/// Complex QR factorization by modified Gram-Schmidt: `A = Q R` with `Q` unitary
/// (columns orthonormal under the Hermitian inner product) and `R` upper-triangular,
/// both row-major n×n. Superseded in `complex_eig_qr` by the O(p²) Hessenberg Givens
/// step; retained as a `#[cfg(test)]` reference factorization.
#[cfg(test)]
fn complex_qr_mgs(a: &[ComplexScalar], n: usize) -> EigQrResult {
    let mut q = a.to_vec();
    let mut r = vec![(0.0_f64, 0.0_f64); n * n];
    for j in 0..n {
        for i in 0..j {
            // R[i][j] = <q_i, q_j> = Σ_k conj(q[k][i])·q[k][j]
            let mut dot = (0.0_f64, 0.0_f64);
            for k in 0..n {
                dot = complex_add(dot, complex_mul(complex_conj(q[k * n + i]), q[k * n + j]));
            }
            r[i * n + j] = dot;
            for k in 0..n {
                q[k * n + j] = complex_sub(q[k * n + j], complex_mul(dot, q[k * n + i]));
            }
        }
        let norm = (0..n)
            .map(|k| complex_abs(q[k * n + j]).powi(2))
            .sum::<f64>()
            .sqrt();
        r[j * n + j] = (norm, 0.0);
        if norm > 0.0 {
            for k in 0..n {
                q[k * n + j] = (q[k * n + j].0 / norm, q[k * n + j].1 / norm);
            }
        }
    }
    (q, r)
}

/// Row-major complex n×n matrix product. Only used to build test matrices and the
/// `complex_qr_mgs` oracle now that `complex_eig_qr` runs on Hessenberg form.
#[cfg(test)]
fn complex_matmul_n(x: &[(f64, f64)], y: &[(f64, f64)], n: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0_f64, 0.0_f64); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = (0.0_f64, 0.0_f64);
            for k in 0..n {
                s = complex_add(s, complex_mul(x[i * n + k], y[k * n + j]));
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Complex Givens rotation `G = [[c, s], [-conj(s), c]]` (c real ≥ 0, `|c|²+|s|²=1`)
/// zeroing the second component: `G·(f, g)ᵀ = (r, 0)ᵀ`. Returns `(c, s)`.
#[inline]
fn complex_givens(f: (f64, f64), g: (f64, f64)) -> (f64, (f64, f64)) {
    let absf = complex_abs(f);
    let absg = complex_abs(g);
    let r = (absf * absf + absg * absg).sqrt();
    if r == 0.0 {
        return (1.0, (0.0, 0.0));
    }
    if absf == 0.0 {
        // c = 0, s = conj(g)/|g| so that s·g = |g| = r.
        return (0.0, (g.0 / absg, -g.1 / absg));
    }
    let c = absf / r;
    // s = (f/|f|)·conj(g)/r.
    let phase_f = (f.0 / absf, f.1 / absf);
    let conj_g_scaled = (g.0 / r, -g.1 / r);
    (c, complex_mul(phase_f, conj_g_scaled))
}

/// Reduce a complex `n×n` matrix `a` (row-major) to upper-Hessenberg form `H` by a
/// sequence of complex Givens similarities, returning `(H, Q0)` with `A = Q0·H·Q0ᴴ`
/// and `Q0` unitary. The complex analog of [`hessenberg_reduction`]: for each column
/// `k` it zeros the sub-subdiagonal entries `h[k+2..n][k]` against the pivot row
/// `k+1`, applying each rotation `G` on the left (rows `k+1, j`) and `Gᴴ` on the
/// right (cols `k+1, j`), accumulating `Gᴴ` into `Q0`. O(n³), Givens-based so it
/// avoids complex Householder bookkeeping; each rotation is an explicit 2×2 unitary.
fn complex_hessenberg_reduction(a: &[(f64, f64)], n: usize) -> (ComplexMatrix, ComplexMatrix) {
    let mut h = a.to_vec();
    let mut q = vec![(0.0_f64, 0.0_f64); n * n];
    for i in 0..n {
        q[i * n + i] = (1.0, 0.0);
    }
    if n < 3 {
        return (h, q);
    }
    for k in 0..n - 2 {
        for j in (k + 2)..n {
            let f = h[(k + 1) * n + k];
            let g = h[j * n + k];
            if complex_abs(g) == 0.0 {
                continue;
            }
            let (c, s) = complex_givens(f, g);
            let cc = (c, 0.0);
            let s_conj = complex_conj(s);
            // Left: rows (k+1, j) ← G·rows.
            for col in 0..n {
                let r1 = h[(k + 1) * n + col];
                let r2 = h[j * n + col];
                h[(k + 1) * n + col] = complex_add(complex_mul(cc, r1), complex_mul(s, r2));
                h[j * n + col] = complex_sub(complex_mul(cc, r2), complex_mul(s_conj, r1));
            }
            // Right: cols (k+1, j) ← cols·Gᴴ  (Gᴴ = [[c, -s], [conj(s), c]]).
            for row in 0..n {
                let ca = h[row * n + (k + 1)];
                let cb = h[row * n + j];
                h[row * n + (k + 1)] = complex_add(complex_mul(cc, ca), complex_mul(s_conj, cb));
                h[row * n + j] = complex_sub(complex_mul(cc, cb), complex_mul(s, ca));
            }
            // Accumulate Gᴴ into Q (same right-multiply on cols k+1, j).
            for row in 0..n {
                let ca = q[row * n + (k + 1)];
                let cb = q[row * n + j];
                q[row * n + (k + 1)] = complex_add(complex_mul(cc, ca), complex_mul(s_conj, cb));
                q[row * n + j] = complex_sub(complex_mul(cc, cb), complex_mul(s, ca));
            }
        }
        // Force exact Hessenberg structure (rotations zero these, but be explicit).
        for j in (k + 2)..n {
            h[j * n + k] = (0.0, 0.0);
        }
    }
    (h, q)
}

/// Eigenvector of complex `A` (= `Q0·H·Q0ᴴ`, with `H` its complex upper-Hessenberg
/// form) for eigenvalue `lambda`, by two steps of inverse iteration on the
/// HESSENBERG factor `(H − λI)` (O(n²) via [`complex_hessenberg_solve`]) then the
/// unitary back-transform `v_A = Q0·v_H`. Because `H = Q0ᴴ A Q0`, a unit eigenvector
/// `y` of `H` gives the eigenvector `Q0·y` of `A`. Computing all `n` eigenvectors is
/// O(n³) versus O(n⁴) for the full inverse iteration in [`complex_eig_eigenvector`];
/// eig parity is reconstruction (`A·v = λ·v`), preserved up to scale/phase.
fn complex_eig_eigenvector_hessenberg(
    h: &[(f64, f64)],
    q0: &[(f64, f64)],
    n: usize,
    lambda: (f64, f64),
) -> Vec<(f64, f64)> {
    let mut y = vec![(1.0_f64, 0.0_f64); n];
    for _ in 0..2 {
        let mut m: Vec<(f64, f64)> = (0..n * n)
            .map(|idx| {
                let e = h[idx];
                if idx / n == idx % n {
                    complex_sub(e, lambda)
                } else {
                    e
                }
            })
            .collect();
        let mut b = y.clone();
        let x = complex_hessenberg_solve(&mut m, &mut b, n);
        let norm = x
            .iter()
            .map(|z| complex_abs(*z).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            break;
        }
        y = x.iter().map(|z| (z.0 / norm, z.1 / norm)).collect();
    }
    let mut v = vec![(0.0_f64, 0.0_f64); n];
    for (row, slot) in v.iter_mut().enumerate().take(n) {
        let mut acc = (0.0_f64, 0.0_f64);
        let qbase = row * n;
        for (col, yc) in y.iter().enumerate() {
            acc = complex_add(acc, complex_mul(q0[qbase + col], *yc));
        }
        *slot = acc;
    }
    normalize_vector(v)
}

/// Eigenvector for complex eigenvalue `lambda` of complex `a` by two steps of inverse
/// iteration on `(A − λI)`, reusing the nudging [`complex_linear_solve`] (which keeps
/// the intentionally near-singular solve finite). Returns a unit-norm complex vector.
/// Superseded in production by [`complex_eig_eigenvector_hessenberg`] (O(n²)); kept as
/// a `#[cfg(test)]` reference oracle.
#[cfg(test)]
fn complex_eig_eigenvector(a: &[(f64, f64)], n: usize, lambda: (f64, f64)) -> Vec<(f64, f64)> {
    let mut x = vec![(1.0_f64, 0.0_f64); n];
    for _ in 0..2 {
        let mut m: Vec<(f64, f64)> = (0..n * n)
            .map(|idx| {
                let e = a[idx];
                if idx / n == idx % n {
                    complex_sub(e, lambda)
                } else {
                    e
                }
            })
            .collect();
        let mut b = x.clone();
        let sol = complex_linear_solve(&mut m, &mut b, n);
        let norm = sol
            .iter()
            .map(|z| complex_abs(*z).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            break;
        }
        x = sol.iter().map(|z| (z.0 / norm, z.1 / norm)).collect();
    }
    x
}

/// Principal complex square root: `re ≥ 0`, with the imaginary part taking the
/// sign of `z`'s imaginary part. `(complex_sqrt(z))² == z`.
fn complex_sqrt(z: (f64, f64)) -> (f64, f64) {
    // Numerically stable principal square root (Kahan / numpy npy_csqrt). The naive
    // `im = sqrt((|z| - re)/2)` catastrophically cancels when re >> |im| and drops the
    // imaginary part (violating sqrt(z)^2 == z); derive the smaller component from the
    // identity 2*re_out*im_out == z.1 instead. See the matching fix in arithmetic.rs.
    let r = complex_abs(z);
    if r == 0.0 {
        return (0.0, 0.0);
    }
    if z.0 >= 0.0 {
        let re = ((r + z.0) * 0.5).sqrt();
        let im = z.1 / (2.0 * re);
        (re, im)
    } else {
        let im_mag = ((r - z.0) * 0.5).sqrt();
        let re = z.1.abs() / (2.0 * im_mag);
        (re, if z.1 < 0.0 { -im_mag } else { im_mag })
    }
}

/// One Wilkinson-shifted QR sweep on the leading `p×p` active block of the complex
/// upper-Hessenberg matrix `t` (row-major `n×n`), done with complex Givens rotations
/// in O(p²): `B = t − μI` is reduced to upper-triangular `R` by left rotations
/// (`G_i` zeros the subdiagonal `B[i+1][i]`), then `t ← R·Q + μI` with `Q = G_0ᴴ…
/// G_{p-2}ᴴ` applied as right rotations — the standard explicit single-shift
/// Hessenberg QR step. Replaces the previous O(p³) `complex_qr_mgs` + `complex_matmul_n`
/// per sweep (which copied and densely re-factored the whole block), making
/// `complex_eig_qr`'s iteration O(n³) total. Only the leading `p×p` block is touched
/// (the deflated trailing rows/cols stay frozen), exactly as the dense version did.
fn complex_hessenberg_shifted_qr_step(
    t: &mut [ComplexScalar],
    n: usize,
    p: usize,
    mu: ComplexScalar,
) {
    for i in 0..p {
        t[i * n + i] = complex_sub(t[i * n + i], mu);
    }
    // Left rotations: reduce (t − μI) to upper triangular R, recording each G_i.
    let mut rots: Vec<(f64, ComplexScalar)> = Vec::with_capacity(p.saturating_sub(1));
    for i in 0..p - 1 {
        let (c, s) = complex_givens(t[i * n + i], t[(i + 1) * n + i]);
        rots.push((c, s));
        let cc = (c, 0.0);
        let s_conj = complex_conj(s);
        for col in i..p {
            let r1 = t[i * n + col];
            let r2 = t[(i + 1) * n + col];
            t[i * n + col] = complex_add(complex_mul(cc, r1), complex_mul(s, r2));
            t[(i + 1) * n + col] = complex_sub(complex_mul(cc, r2), complex_mul(s_conj, r1));
        }
        t[(i + 1) * n + i] = (0.0, 0.0);
    }
    // Right rotations: t ← R·Q (Q = ∏ G_iᴴ). G_iᴴ acts on columns (i, i+1); it fills
    // only down to row i+1, so restrict to rows 0..=i+1.
    for (i, &(c, s)) in rots.iter().enumerate() {
        let cc = (c, 0.0);
        let s_conj = complex_conj(s);
        let rend = (i + 2).min(p);
        for row in 0..rend {
            let ca = t[row * n + i];
            let cb = t[row * n + (i + 1)];
            t[row * n + i] = complex_add(complex_mul(cc, ca), complex_mul(s_conj, cb));
            t[row * n + (i + 1)] = complex_sub(complex_mul(cc, cb), complex_mul(s, ca));
        }
    }
    for i in 0..p {
        t[i * n + i] = complex_add(t[i * n + i], mu);
    }
}

/// Complex non-Hermitian eigendecomposition: Wilkinson-shifted QR with deflation
/// drives `A` to its (fully upper-triangular) complex Schur form — every eigenvalue
/// lands on the diagonal (unlike the real case there are no 2×2 blocks) — then each
/// eigenvector is recovered by inverse iteration on the original `A`. Eigenvectors
/// are column-major (column k pairs with eigenvalue k). The shift is essential: an
/// unshifted sweep fails for equal-modulus spectra (e.g. eigenvalues on the unit
/// circle), where it never converges and returns garbage.
fn complex_eig_qr(a: &[ComplexScalar], n: usize) -> EigQrResult {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![a[0]], vec![(1.0, 0.0)]);
    }
    // Reduce A to complex upper-Hessenberg ONCE (A = Q0·H·Q0ᴴ, Q0 unitary). The QR
    // iteration runs on a copy (destroyed into Schur form); H0 + Q0 are reused for
    // the O(n³) eigenvector phase below. Working on a Hessenberg matrix lets each QR
    // sweep be an O(p²) complex-Givens step instead of a full O(p³) re-factorization.
    let (h0, q0) = complex_hessenberg_reduction(a, n);
    let mut t = h0.clone();

    // Wilkinson-shifted QR with deflation. The unshifted iteration fails to
    // converge when eigenvalues share a modulus — subdiagonals shrink at rate
    // |λ_{i+1}/λ_i| ≈ 1 (e.g. a spectrum on the unit circle) — and returns garbage;
    // the shift breaks the degeneracy so each trailing eigenvalue converges and
    // deflates. The active block is the leading p×p of `t`.
    let mut p = n;
    let mut iters = 0usize;
    let mut since_deflate = 0usize;
    let max_iters = 100 * n + 100;
    while p > 1 {
        // Deflate when the active block's bottom subdiagonal is negligible.
        let sub = complex_abs(t[(p - 1) * n + (p - 2)]);
        let dscale = complex_abs(t[(p - 2) * n + (p - 2)]) + complex_abs(t[(p - 1) * n + (p - 1)]);
        if sub <= f64::EPSILON * dscale.max(f64::MIN_POSITIVE) {
            t[(p - 1) * n + (p - 2)] = (0.0, 0.0);
            p -= 1;
            since_deflate = 0;
            continue;
        }
        if iters >= max_iters {
            break; // safety: leave whatever is on the diagonal
        }
        iters += 1;
        since_deflate += 1;

        let aa = t[(p - 2) * n + (p - 2)];
        let bb = t[(p - 2) * n + (p - 1)];
        let cc = t[(p - 1) * n + (p - 2)];
        let dd = t[(p - 1) * n + (p - 1)];
        let mu = if since_deflate.is_multiple_of(10) {
            // Exceptional shift: the Wilkinson shift can stagnate (e.g. a nilpotent
            // trailing 2×2 → shift 0 on an orthogonal/cyclic block, which is QR-
            // stationary). Inject an off-axis ad-hoc shift keyed to the active
            // subdiagonal magnitudes to break the symmetry (cf. LAPACK hqr).
            let s = complex_abs(t[(p - 1) * n + (p - 2)])
                + if p >= 3 {
                    complex_abs(t[(p - 2) * n + (p - 3)])
                } else {
                    0.0
                };
            let s = if s > 0.0 { s } else { 1.0 };
            complex_add(dd, (0.75 * s, 0.31 * s))
        } else {
            // Wilkinson shift: the trailing-2×2 eigenvalue closer to t[p-1][p-1].
            let mid = (0.5 * (aa.0 + dd.0), 0.5 * (aa.1 + dd.1));
            let half_diff = (0.5 * (aa.0 - dd.0), 0.5 * (aa.1 - dd.1));
            let disc = complex_add(complex_mul(half_diff, half_diff), complex_mul(bb, cc));
            let sq = complex_sqrt(disc);
            let mu1 = complex_add(mid, sq);
            let mu2 = complex_sub(mid, sq);
            if complex_abs(complex_sub(mu1, dd)) <= complex_abs(complex_sub(mu2, dd)) {
                mu1
            } else {
                mu2
            }
        };

        // Shifted QR step on the active p×p Hessenberg block, O(p²) via Givens.
        complex_hessenberg_shifted_qr_step(&mut t, n, p, mu);
    }
    let eigenvalues: Vec<(f64, f64)> = (0..n).map(|i| t[i * n + i]).collect();
    // Eigenvectors via O(n²) inverse iteration on the complex Hessenberg factor
    // (H − λI) + unitary back-transform — O(n³) total. Reuse the same H0/Q0 the QR
    // iteration started from (A = Q0·H0·Q0ᴴ); no second reduction needed.
    let mut eigenvectors = vec![(0.0_f64, 0.0_f64); n * n];
    for (col, &lambda) in eigenvalues.iter().enumerate() {
        let v = complex_eig_eigenvector_hessenberg(&h0, &q0, n, lambda);
        for row in 0..n {
            eigenvectors[row * n + col] = v[row];
        }
    }
    (eigenvalues, eigenvectors)
}

pub(crate) fn eval_eig(
    inputs: &[Value],
    _params: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<Value>, EvalError> {
    let primitive = Primitive::Eig;

    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "eig requires square matrix",
        });
    }

    // Compute eigenvalues and eigenvectors. A complex input uses the complex QR
    // iteration (the previous code took the real part only — silently wrong); a real
    // input keeps the real-Schur kernel.
    let (eigenvalues, eigenvectors) = if matches!(dtype, DType::Complex64 | DType::Complex128) {
        complex_eig_qr(&a, n)
    } else {
        let a_real: Vec<f64> = a.iter().map(|(re, _im)| *re).collect();
        eig_qr_iteration(&a_real, n)
    };

    // Eigenvalues as Complex128 (can be complex for non-symmetric)
    let w_elements: Vec<Literal> = eigenvalues
        .iter()
        .map(|&(re, im)| Literal::from_complex128(re, im))
        .collect();
    let w_shape = Shape {
        dims: vec![n as u32],
    };
    let w_tensor = TensorValue::new(DType::Complex128, w_shape, w_elements)
        .map_err(EvalError::InvalidTensor)?;
    let w_val = Value::Tensor(w_tensor);

    // Eigenvectors as Complex128 [n, n]
    let v_elements: Vec<Literal> = eigenvectors
        .iter()
        .map(|&(re, im)| Literal::from_complex128(re, im))
        .collect();
    let v_shape = Shape {
        dims: vec![n as u32, n as u32],
    };
    let v_tensor = TensorValue::new(DType::Complex128, v_shape, v_elements)
        .map_err(EvalError::InvalidTensor)?;
    let v_val = Value::Tensor(v_tensor);

    Ok(vec![w_val, v_val])
}

/// Solve `M x = b` for a complex n×n `M` (row-major) and complex `b` via Gaussian
/// elimination with partial pivoting (`M`/`b` are overwritten). A pivot that
/// underflows the matrix scale is nudged to `ε·scale`, so the solve never divides
/// by zero: for inverse iteration `M = A − λI` is intentionally near-singular and
/// the resulting large-norm solution is exactly the (un-normalized) eigenvector.
/// Superseded in production by [`complex_hessenberg_solve`] (O(n²)); now a
/// `#[cfg(test)]` reference used only by the eigenvector oracles.
#[cfg(test)]
fn complex_linear_solve(m: &mut [(f64, f64)], b: &mut [(f64, f64)], n: usize) -> Vec<(f64, f64)> {
    let scale = m.iter().map(|z| complex_abs(*z)).fold(0.0_f64, f64::max);
    let tiny = (f64::EPSILON * scale).max(f64::MIN_POSITIVE);
    for col in 0..n {
        // Partial pivot: largest-magnitude entry in this column at/below the diagonal.
        let mut piv = col;
        let mut best = complex_abs(m[col * n + col]);
        for r in (col + 1)..n {
            let mag = complex_abs(m[r * n + col]);
            if mag > best {
                best = mag;
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let mut pivot = m[col * n + col];
        if complex_abs(pivot) < tiny {
            pivot = (tiny, 0.0);
            m[col * n + col] = pivot;
        }
        for r in (col + 1)..n {
            let factor = complex_div(m[r * n + col], pivot);
            for c in col..n {
                m[r * n + c] = complex_sub(m[r * n + c], complex_mul(factor, m[col * n + c]));
            }
            b[r] = complex_sub(b[r], complex_mul(factor, b[col]));
        }
    }
    let mut x = vec![(0.0_f64, 0.0_f64); n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s = complex_sub(s, complex_mul(m[col * n + c], x[c]));
        }
        let mut pivot = m[col * n + col];
        if complex_abs(pivot) < tiny {
            pivot = (tiny, 0.0);
        }
        x[col] = complex_div(s, pivot);
    }
    x
}

/// Eigenvector for eigenvalue `lambda` of the real n×n matrix `a` (row-major) by
/// two steps of inverse iteration on the FULL `A − λI`. This is the O(n³)-per-vector
/// reference the O(n²) [`eig_eigenvector_hessenberg`] replaced in production; retained
/// as a `#[cfg(test)]` oracle so the Hessenberg path can be checked to produce an
/// equivalent eigenvector (same reconstruction residual / spectrum, up to scale/phase).
#[cfg(test)]
fn eig_eigenvector(a: &[f64], n: usize, lambda: (f64, f64)) -> Vec<(f64, f64)> {
    let mut vec_x = vec![(1.0, 0.0); n];
    for _ in 0..2 {
        let mut m: Vec<(f64, f64)> = (0..n * n)
            .map(|idx| {
                let e = (a[idx], 0.0);
                if idx / n == idx % n {
                    complex_sub(e, lambda)
                } else {
                    e
                }
            })
            .collect();
        let mut b = vec_x.clone();
        let x = complex_linear_solve(&mut m, &mut b, n);
        let norm = x
            .iter()
            .map(|z| complex_abs(*z).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            break;
        }
        vec_x = x.iter().map(|z| (z.0 / norm, z.1 / norm)).collect();
    }
    vec_x
}

/// Eigenvector of the real n×n matrix `a` (= `q0 · h · q0ᵀ`, with `h` its upper-
/// Hessenberg form and `q0` the orthogonal reduction matrix) for eigenvalue
/// `lambda`, by two steps of inverse iteration on the HESSENBERG factor `(H − λI)`
/// instead of the full `(A − λI)`. Because `H = Q0ᵀ A Q0`, a unit eigenvector `y` of
/// `H` maps to the eigenvector `Q0·y` of `A` (`A(Q0 y) = Q0 H y = λ Q0 y`). Solving
/// against a Hessenberg matrix is O(n²) (one subdiagonal elimination per column +
/// dense back-substitution) versus O(n³) for a full LU, so computing all `n`
/// eigenvectors drops from O(n⁴) to O(n³). Numerically equivalent to the full
/// inverse iteration up to the eigenvector's intrinsic scale/phase freedom — eig
/// parity is reconstruction (`A·v = λ·v`) + spectrum, both preserved (the public
/// contract; see `assert_eig_residual_complex`). Mirrors `eig_eigenvector`'s 2-step
/// inverse iteration and the same near-singular pivot regularization.
fn eig_eigenvector_hessenberg(
    h: &[f64],
    q0: &[f64],
    n: usize,
    lambda: (f64, f64),
) -> Vec<(f64, f64)> {
    // Complex H − λI (dense, but only the first subdiagonal is nonzero below the
    // diagonal — exact Hessenberg). Built once; refactored per inverse-iteration
    // step (the step's RHS changes, the matrix does not, so this matches the
    // full-matrix routine's per-step factor-and-solve).
    let mut y = vec![(1.0_f64, 0.0_f64); n];
    for _ in 0..2 {
        let mut m: Vec<(f64, f64)> = (0..n * n)
            .map(|idx| {
                let e = (h[idx], 0.0);
                if idx / n == idx % n {
                    complex_sub(e, lambda)
                } else {
                    e
                }
            })
            .collect();
        let mut b = y.clone();
        let x = complex_hessenberg_solve(&mut m, &mut b, n);
        let norm = x
            .iter()
            .map(|z| complex_abs(*z).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            break;
        }
        y = x.iter().map(|z| (z.0 / norm, z.1 / norm)).collect();
    }
    // Back-transform v_A = Q0 · y (real Q0 × complex y), then renormalize (Q0 is
    // orthogonal so this is already unit-norm to rounding; renormalize for safety).
    let mut v = vec![(0.0_f64, 0.0_f64); n];
    for (row, slot) in v.iter_mut().enumerate().take(n) {
        let mut acc = (0.0_f64, 0.0_f64);
        let qbase = row * n;
        for (col, yc) in y.iter().enumerate() {
            let q = q0[qbase + col];
            acc.0 += q * yc.0;
            acc.1 += q * yc.1;
        }
        *slot = acc;
    }
    normalize_vector(v)
}

/// Solve `M x = b` for a complex UPPER-HESSENBERG `M` (zero below the first
/// subdiagonal) in O(n²): partial-pivot LU where each column has at most one
/// below-diagonal entry (row `k+1`), so a single adjacent-row elimination per
/// column suffices, followed by dense back-substitution. `M`/`b` are overwritten.
/// Near-singular pivots are clamped exactly as [`complex_linear_solve`] (inverse
/// iteration deliberately drives `M = H − λI` near-singular). BIT-equivalent in
/// structure to the full solver restricted to Hessenberg fill, but never touches
/// the already-zero sub-subdiagonal — that is the whole O(n³)→O(n²) saving.
fn complex_hessenberg_solve(
    m: &mut [(f64, f64)],
    b: &mut [(f64, f64)],
    n: usize,
) -> Vec<(f64, f64)> {
    let scale = m.iter().map(|z| complex_abs(*z)).fold(0.0_f64, f64::max);
    let tiny = (f64::EPSILON * scale).max(f64::MIN_POSITIVE);
    for col in 0..n {
        // Only row `col+1` carries a subdiagonal entry in this column.
        if col + 1 < n {
            let here = complex_abs(m[col * n + col]);
            let below = complex_abs(m[(col + 1) * n + col]);
            if below > here {
                for c in col..n {
                    m.swap(col * n + c, (col + 1) * n + c);
                }
                b.swap(col, col + 1);
            }
        }
        let mut pivot = m[col * n + col];
        if complex_abs(pivot) < tiny {
            pivot = (tiny, 0.0);
            m[col * n + col] = pivot;
        }
        if col + 1 < n {
            let factor = complex_div(m[(col + 1) * n + col], pivot);
            // Eliminate the single subdiagonal entry; update row col+1 across cols≥col.
            for c in col..n {
                m[(col + 1) * n + c] =
                    complex_sub(m[(col + 1) * n + c], complex_mul(factor, m[col * n + c]));
            }
            b[col + 1] = complex_sub(b[col + 1], complex_mul(factor, b[col]));
        }
    }
    // Dense back-substitution over the upper-triangular U.
    let mut x = vec![(0.0_f64, 0.0_f64); n];
    for col in (0..n).rev() {
        let mut s = b[col];
        for c in (col + 1)..n {
            s = complex_sub(s, complex_mul(m[col * n + c], x[c]));
        }
        let mut pivot = m[col * n + col];
        if complex_abs(pivot) < tiny {
            pivot = (tiny, 0.0);
        }
        x[col] = complex_div(s, pivot);
    }
    x
}

/// Householder reflector for a length-2 or length-3 real vector `a`: returns
/// `(v, tau)` with `P = I − tau·v·vᵀ` mapping `a` to `(±‖a‖, 0[, 0])`. `v` has
/// `a.len()` meaningful leading entries; `tau == 0` signals the identity (zero `a`).
#[inline]
fn make_house(a: &[f64]) -> ([f64; 3], f64) {
    let len = a.len();
    let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut v = [0.0f64; 3];
    if norm == 0.0 {
        return (v, 0.0);
    }
    // Choose the reflection sign to avoid cancellation in `a[0] − r`.
    let r = if a[0] >= 0.0 { -norm } else { norm };
    v[0] = a[0] - r;
    v[1..len].copy_from_slice(&a[1..len]);
    let vnorm2: f64 = v[..len].iter().map(|x| x * x).sum();
    let tau = if vnorm2 == 0.0 { 0.0 } else { 2.0 / vnorm2 };
    (v, tau)
}

/// One Francis implicit double-shift QR sweep on the unreduced upper-Hessenberg
/// active block `[lo, hi]` of the row-major `n×n` matrix `h` (block size ≥ 3).
/// All Householder reflectors act only on indices within `[lo, hi]`, so the sweep
/// is an orthogonal similarity of the block — its eigenvalues are preserved while
/// the bulge is chased back to Hessenberg form. The double shift uses the trailing
/// 2×2's (trace, det) so a complex-conjugate shift pair is applied entirely in real
/// arithmetic (the whole point of the implicit double shift). Every 10 sweeps
/// without deflation an exceptional shift (LAPACK `dlahqr`) breaks stagnation.
fn francis_sweep(h: &mut [f64], n: usize, lo: usize, hi: usize, since_deflate: usize) {
    let (s, t) = if since_deflate > 0 && since_deflate.is_multiple_of(10) {
        // Exceptional shift keyed to the trailing subdiagonal magnitudes.
        let sh = h[hi * n + (hi - 1)].abs() + h[(hi - 1) * n + (hi - 2)].abs();
        let h11 = 0.75 * sh + h[hi * n + hi];
        let h21 = sh;
        let h12 = -0.4375 * sh;
        (2.0 * h11, h11 * h11 - h12 * h21)
    } else {
        let aa = h[(hi - 1) * n + (hi - 1)];
        let bb = h[(hi - 1) * n + hi];
        let cc = h[hi * n + (hi - 1)];
        let dd = h[hi * n + hi];
        (aa + dd, aa * dd - bb * cc)
    };

    // First column of (H² − sH + tI) restricted to the bulge rows lo, lo+1, lo+2.
    let h00 = h[lo * n + lo];
    let h01 = h[lo * n + (lo + 1)];
    let h10 = h[(lo + 1) * n + lo];
    let h11d = h[(lo + 1) * n + (lo + 1)];
    let h21d = h[(lo + 2) * n + (lo + 1)];
    let mut bulge = [
        h00 * h00 + h01 * h10 - s * h00 + t,
        h10 * (h00 + h11d - s),
        h10 * h21d,
    ];

    // Chase the 3-element bulge down the subdiagonal.
    for k in lo..=(hi - 2) {
        let (v, tau) = make_house(&bulge[..3]);
        if tau != 0.0 {
            let (r0, r1, r2) = (k, k + 1, k + 2);
            let cstart = if k > lo { k - 1 } else { lo };
            for col in cstart..=hi {
                let d = tau
                    * (v[0] * h[r0 * n + col] + v[1] * h[r1 * n + col] + v[2] * h[r2 * n + col]);
                h[r0 * n + col] -= d * v[0];
                h[r1 * n + col] -= d * v[1];
                h[r2 * n + col] -= d * v[2];
            }
            let rend = (k + 3).min(hi);
            for row in lo..=rend {
                let d = tau
                    * (v[0] * h[row * n + r0] + v[1] * h[row * n + r1] + v[2] * h[row * n + r2]);
                h[row * n + r0] -= d * v[0];
                h[row * n + r1] -= d * v[1];
                h[row * n + r2] -= d * v[2];
            }
        }
        if k < hi - 2 {
            bulge = [h[(k + 1) * n + k], h[(k + 2) * n + k], h[(k + 3) * n + k]];
        } else {
            bulge[0] = h[(k + 1) * n + k];
            bulge[1] = h[(k + 2) * n + k];
        }
    }

    // Final 2-element reflector zeroing the last bulge entry (row hi, col hi-2).
    let (v, tau) = make_house(&bulge[..2]);
    if tau != 0.0 {
        let (r0, r1) = (hi - 1, hi);
        for col in (hi - 2)..=hi {
            let d = tau * (v[0] * h[r0 * n + col] + v[1] * h[r1 * n + col]);
            h[r0 * n + col] -= d * v[0];
            h[r1 * n + col] -= d * v[1];
        }
        for row in lo..=hi {
            let d = tau * (v[0] * h[row * n + r0] + v[1] * h[row * n + r1]);
            h[row * n + r0] -= d * v[0];
            h[row * n + r1] -= d * v[1];
        }
    }
}

/// Drive a real upper-Hessenberg `h` (row-major `n×n`) to real quasi-triangular
/// (Schur) form via Francis implicit double-shift QR with deflation — LAPACK
/// `dlahqr`'s algorithm. Cubic convergence (≈ O(n) total sweeps) replaces the
/// unshifted iteration's linear convergence (which stalls — up to 200·n sweeps —
/// for clustered / complex-conjugate spectra). Returns `true` on convergence; a
/// `false` (cap exceeded) lets the caller keep its `complex_eig_qr` safety net.
/// Reflectors stay inside the active block, so the bottom-right eigenvalues are
/// unchanged by the (stale) coupling columns — eigenvalues are read off the
/// resulting 1×1 / 2×2 diagonal blocks.
fn francis_qr_schur(h: &mut [f64], n: usize) -> bool {
    if n <= 2 {
        return true;
    }
    let eps = f64::EPSILON;
    let mut hi = n - 1;
    let mut total = 0usize;
    let max_total = 60 * n + 100;
    let mut since_deflate = 0usize;

    loop {
        if hi < 2 {
            return true; // remaining leading 1×1 / 2×2 read directly by the caller
        }
        // Smallest `lo` such that the block [lo, hi] is unreduced (scan up from hi
        // to the first negligible subdiagonal, which splits the block and is zeroed).
        let mut lo = hi;
        while lo > 0 {
            let scale = h[(lo - 1) * n + (lo - 1)].abs() + h[lo * n + lo].abs();
            let thresh = eps * if scale > 0.0 { scale } else { 1.0 };
            if h[lo * n + (lo - 1)].abs() <= thresh {
                h[lo * n + (lo - 1)] = 0.0;
                break;
            }
            lo -= 1;
        }
        let block = hi - lo;
        if block == 0 {
            hi -= 1; // 1×1 deflated
            since_deflate = 0;
        } else if block == 1 {
            hi -= 2; // isolated 2×2 deflated (hi ≥ 2 in the body, so no underflow)
            since_deflate = 0;
        } else {
            if total >= max_total {
                return false;
            }
            total += 1;
            since_deflate += 1;
            francis_sweep(h, n, lo, hi, since_deflate);
        }
    }
}

/// QR iteration for general eigendecomposition.
///
/// Simple implementation for V1 correctness. Uses basic QR iteration without
/// shifts or deflation for small matrices.
fn eig_qr_iteration(a: &[f64], n: usize) -> EigQrResult {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![(a[0], 0.0)], vec![(1.0, 0.0)]);
    }

    // For 2x2, use direct formula
    if n == 2 {
        let a00 = a[0];
        let a01 = a[1];
        let a10 = a[2];
        let a11 = a[3];

        let trace = a00 + a11;
        let det = a00 * a11 - a01 * a10;
        let disc = trace * trace - 4.0 * det;

        // Compute eigenvalues - may be complex if disc < 0
        let eigenvalues = if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            vec![
                ((trace + sqrt_disc) / 2.0, 0.0),
                ((trace - sqrt_disc) / 2.0, 0.0),
            ]
        } else {
            let sqrt_disc = (-disc).sqrt();
            vec![
                (trace / 2.0, sqrt_disc / 2.0),
                (trace / 2.0, -sqrt_disc / 2.0),
            ]
        };

        // Simplified eigenvector computation
        // For complex eigenvalues or simple cases, return identity-like vectors
        let eigenvectors = if disc >= 0.0 && a10.abs() > 1e-10 {
            let lambda1_re = eigenvalues[0].0;
            let lambda2_re = eigenvalues[1].0;
            let v1 = normalize_vector(vec![(lambda1_re - a11, 0.0), (a10, 0.0)]);
            let v2 = normalize_vector(vec![(lambda2_re - a11, 0.0), (a10, 0.0)]);
            vec![v1[0], v2[0], v1[1], v2[1]]
        } else {
            vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
        };

        return (eigenvalues, eigenvectors);
    }

    // For larger matrices, reduce once to Hessenberg form and keep QR
    // iterations in that form. The previous dense Gram-Schmidt loop rebuilt a
    // full Q/R factorization and performed two dense matrix multiplies on every
    // step. Hessenberg + Givens QR performs the same orthogonal-similarity
    // iteration class, but each step touches only O(n^2) data.
    let (mut t, q0) = hessenberg_reduction(a, n);

    // Keep the upper-Hessenberg factor H (and Q0, with A = Q0·H·Q0ᵀ) BEFORE the QR
    // iteration overwrites `t` with the Schur form. Eigenvectors are then found by
    // O(n²) inverse iteration on (H − λI) and back-transformed by Q0 — O(n³) total
    // for all n eigenvectors, versus O(n⁴) inverse-iterating the full (A − λI).
    // Force exact Hessenberg structure: `hessenberg_reduction` leaves ≤1e-15·‖col‖
    // dust below the subdiagonal when it skips an already-reduced column, and the
    // O(n²) solver assumes those are exactly zero (the QR iteration treats H the
    // same way, so this is the consistent matrix).
    let mut h_hess = t.clone();
    for row in 2..n {
        for col in 0..row - 1 {
            h_hess[row * n + col] = 0.0;
        }
    }

    // Unshifted Hessenberg QR drives `t` to real Schur (quasi-triangular) form:
    // negligible subdiagonals separate 1×1 real blocks from isolated 2×2 blocks
    // (complex-conjugate pairs, whose subdiagonal keeps a fixed nonzero magnitude
    // because the pair shares a modulus). Iterate until quasi-triangular — no two
    // *consecutive* non-negligible subdiagonals, which would be an unconverged ≥3×3
    // coupled block — then read eigenvalues off the 1×1/2×2 blocks. Reading only the
    // diagonal (the previous behaviour) silently dropped every complex eigenvalue
    // (frankenjax-eig-nonsymmetric-broken-n3-66pmy).
    //
    // `subdiag_negligible(t, j)` tests the subdiagonal linking blocks j-1 and j
    // against the LAPACK-style relative criterion |t[j,j-1]| ≤ ε·(|t[j-1,j-1]|+|t[j,j]|).
    let subdiag_negligible = |t: &[f64], j: usize| -> bool {
        let scale = t[(j - 1) * n + j - 1].abs() + t[j * n + j].abs();
        t[j * n + j - 1].abs() <= f64::EPSILON * scale.max(f64::MIN_POSITIVE)
    };
    // Drive `t` to real quasi-triangular Schur form with the Francis implicit
    // double-shift QR (cubic convergence; stays in real arithmetic for complex-
    // conjugate eigenvalue pairs). This replaces the unshifted deflation-window
    // sweep, which converged only linearly and STALLED — hitting the 200·n cap and
    // forcing the O(n⁴) complex_eig_qr fallback — for clustered / complex-pair
    // spectra (frankenjax-omv7z). Eigenvectors still come from the separate `h_hess`
    // copy, so the Schur vectors are not accumulated.
    let converged = francis_qr_schur(&mut t, n);
    // Global gate (the definition of convergence): quasi-triangular iff no two
    // *consecutive* non-negligible subdiagonals remain anywhere (an unconverged
    // ≥3×3 coupled block). If Francis hit its cap without converging this is false →
    // fall back to complex_eig_qr (e.g. exotic equal-modulus spectra).
    let reached_quasi_triangular =
        converged && (2..n).all(|j| subdiag_negligible(&t, j) || subdiag_negligible(&t, j - 1));

    // Unshifted real QR cannot reach quasi-triangular form when ≥3 eigenvalues share
    // a modulus (e.g. cube-roots-of-unity: {1, e^{±2πi/3}}, all |λ|=1) — it would
    // otherwise return garbage. Fall back to the Wilkinson-shifted complex solver,
    // which converges per eigenvalue regardless of modulus. (Real eigenvectors for
    // the common path are produced below via inverse iteration on the original `a`.)
    if !reached_quasi_triangular {
        let a_complex: Vec<(f64, f64)> = a.iter().map(|&x| (x, 0.0)).collect();
        return complex_eig_qr(&a_complex, n);
    }

    // Walk the quasi-triangular form: 1×1 → real eigenvalue; an isolated 2×2 block
    // → its eigenvalue pair (real or complex-conjugate) via the trace/discriminant
    // formula — which is exact for the block regardless of its internal convergence,
    // so accuracy depends only on the *separating* subdiagonals being negligible.
    let mut eigenvalues: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if i + 1 < n && !subdiag_negligible(&t, i + 1) {
            let (a00, a01) = (t[i * n + i], t[i * n + i + 1]);
            let (a10, a11) = (t[(i + 1) * n + i], t[(i + 1) * n + i + 1]);
            let trace = a00 + a11;
            let det = a00 * a11 - a01 * a10;
            let disc = trace * trace - 4.0 * det;
            if disc >= 0.0 {
                let r = disc.sqrt();
                eigenvalues.push(((trace + r) / 2.0, 0.0));
                eigenvalues.push(((trace - r) / 2.0, 0.0));
            } else {
                let im = (-disc).sqrt() / 2.0;
                eigenvalues.push((trace / 2.0, im));
                eigenvalues.push((trace / 2.0, -im));
            }
            i += 2;
        } else {
            eigenvalues.push((t[i * n + i], 0.0));
            i += 1;
        }
    }

    // Eigenvectors of the ORIGINAL `a` (the QR iteration's Schur vectors are not
    // eigenvectors unless T is diagonal). One eigenvalue → one column, paired by
    // index, via inverse iteration on A − λI.
    let mut eigenvectors = vec![(0.0_f64, 0.0_f64); n * n];
    for (col, &lambda) in eigenvalues.iter().enumerate() {
        let vk = eig_eigenvector_hessenberg(&h_hess, &q0, n, lambda);
        for row in 0..n {
            eigenvectors[row * n + col] = vk[row];
        }
    }

    (eigenvalues, eigenvectors)
}

fn hessenberg_reduction(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut h = a.to_vec();
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    if n < 3 {
        return (h, q);
    }

    // Reuse the reflector buffer across panels. Each iteration writes the full
    // active prefix before use, so this preserves the Householder arithmetic
    // while avoiding one heap allocation per reduction step.
    let mut householder = vec![0.0_f64; n - 1];
    let mut left_scaled = vec![0.0_f64; n];
    for k in 0..n - 2 {
        let start = k + 1;
        let len = n - start;
        let mut norm = 0.0;
        for row in start..n {
            let value = h[row * n + k];
            norm += value * value;
        }
        norm = norm.sqrt();
        if norm <= 1e-15 {
            continue;
        }

        let v = &mut householder[..len];
        for row in start..n {
            v[row - start] = h[row * n + k];
        }
        if v[0] >= 0.0 {
            v[0] += norm;
        } else {
            v[0] -= norm;
        }

        let mut v_norm_sq = 0.0;
        for value in v.iter() {
            v_norm_sq += value * value;
        }
        if v_norm_sq <= 1e-30 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        apply_householder_left_with_scratch(&mut h, n, start, k, v, beta, &mut left_scaled);
        apply_householder_right(&mut h, n, 0, start, v, beta);
        apply_householder_right(&mut q, n, 0, start, v, beta);

        for row in start + 1..n {
            h[row * n + k] = 0.0;
        }
    }

    (h, q)
}

#[cfg(test)]
fn apply_householder_left(
    matrix: &mut [f64],
    n: usize,
    row_start: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
) {
    // H ← (I − β v vᵀ) H over rows [row_start, row_start+v.len()) × cols [col_start, n).
    // The previous form iterated columns-outer / rows-inner, reading `matrix[row*n+col]`
    // with the row (hence flat index) striding by `n` for a fixed `col` — a cache cliff
    // once the matrix exceeds L2 (the cause of eigh's super-cubic wall-clock, ur4h3).
    // Reordered to rows-outer / cols-inner so every access streams a CONTIGUOUS row slice
    // (cache-friendly + autovectorizable). BIT-IDENTICAL: `scaled[col]` accumulates over
    // offset in the SAME ascending order as the old per-column `dot`, then each element is
    // updated exactly once with the identically-grouped `(β·dot)·v_i`.
    let width = n - col_start;
    if width == 0 {
        return;
    }
    let mut scaled = vec![0.0_f64; width]; // dot products, then β·dot
    apply_householder_left_with_scratch(matrix, n, row_start, col_start, v, beta, &mut scaled);
}

fn apply_householder_left_with_scratch(
    matrix: &mut [f64],
    n: usize,
    row_start: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
    scaled_scratch: &mut [f64],
) {
    let width = n - col_start;
    if width == 0 {
        return;
    }
    let scaled = &mut scaled_scratch[..width];
    scaled.fill(0.0);
    for (offset, &v_i) in v.iter().enumerate() {
        let base = (row_start + offset) * n + col_start;
        for (s, &m) in scaled.iter_mut().zip(&matrix[base..base + width]) {
            *s += v_i * m;
        }
    }
    for s in scaled.iter_mut() {
        *s *= beta; // β·dot, matching the old `scale = beta * dot` before `* v_i`
    }
    for (offset, &v_i) in v.iter().enumerate() {
        let base = (row_start + offset) * n + col_start;
        for (m, &s) in matrix[base..base + width].iter_mut().zip(scaled.iter()) {
            *m -= s * v_i;
        }
    }
}

fn apply_householder_right(
    matrix: &mut [f64],
    n: usize,
    row_start: usize,
    col_start: usize,
    v: &[f64],
    beta: f64,
) {
    for row in row_start..n {
        let row_base = row * n;
        let mut dot = 0.0;
        for (offset, &v_i) in v.iter().enumerate() {
            dot += matrix[row_base + col_start + offset] * v_i;
        }
        let scale = beta * dot;
        for (offset, &v_i) in v.iter().enumerate() {
            matrix[row_base + col_start + offset] -= scale * v_i;
        }
    }
}

/// Unshifted Givens Hessenberg QR sweep (full `n×n`). Superseded in production by
/// `francis_qr_schur`; retained as a `#[cfg(test)]` baseline for the convergence
/// A/B bench.
#[cfg(test)]
fn hessenberg_qr_step(h: &mut [f64], mut q_total: Option<&mut [f64]>, n: usize) {
    let mut rotations = Vec::with_capacity(n.saturating_sub(1));

    for i in 0..n - 1 {
        let diagonal = h[i * n + i];
        let subdiagonal = h[(i + 1) * n + i];
        let radius = diagonal.hypot(subdiagonal);
        let (c, s) = if radius <= 1e-300 {
            (1.0, 0.0)
        } else {
            (diagonal / radius, subdiagonal / radius)
        };
        rotations.push((c, s));

        for col in i..n {
            let top_idx = i * n + col;
            let bottom_idx = (i + 1) * n + col;
            let top = h[top_idx];
            let bottom = h[bottom_idx];
            h[top_idx] = c * top + s * bottom;
            h[bottom_idx] = -s * top + c * bottom;
        }
        h[(i + 1) * n + i] = 0.0;
    }

    for (i, (c, s)) in rotations.into_iter().enumerate() {
        apply_givens_right(h, n, i, c, s);
        if let Some(q_total) = q_total.as_deref_mut() {
            apply_givens_right(q_total, n, i, c, s);
        }
    }
}

/// One unshifted Hessenberg QR sweep restricted to the ACTIVE leading `p×p` block
/// of a row-major `n×n` Hessenberg matrix (`p ≤ n`). Identical in form to
/// [`hessenberg_qr_step`] but every Givens rotation touches only columns/rows
/// `< p`, so the per-step cost is O(p²) instead of O(n²). When the trailing
/// `[p, n)` block has deflated (its coupling subdiagonal `t[p][p-1] ≈ 0`), the
/// matrix is block-upper-triangular and the active block's eigenvalues are
/// independent of the (now stale) coupling columns `[p, n)` — so skipping them
/// leaves every diagonal block's eigenvalues unchanged. The `eig_qr_iteration`
/// caller reads eigenvalues off those diagonal blocks and takes eigenvectors from
/// the separate `h_hess` copy, so the stale coupling region is never observed.
#[cfg(test)]
fn hessenberg_qr_step_leading(h: &mut [f64], n: usize, p: usize) {
    if p < 2 {
        return;
    }
    let mut rotations = Vec::with_capacity(p - 1);
    for i in 0..p - 1 {
        let diagonal = h[i * n + i];
        let subdiagonal = h[(i + 1) * n + i];
        let radius = diagonal.hypot(subdiagonal);
        let (c, s) = if radius <= 1e-300 {
            (1.0, 0.0)
        } else {
            (diagonal / radius, subdiagonal / radius)
        };
        rotations.push((c, s));

        for col in i..p {
            let top_idx = i * n + col;
            let bottom_idx = (i + 1) * n + col;
            let top = h[top_idx];
            let bottom = h[bottom_idx];
            h[top_idx] = c * top + s * bottom;
            h[bottom_idx] = -s * top + c * bottom;
        }
        h[(i + 1) * n + i] = 0.0;
    }

    for (i, (c, s)) in rotations.into_iter().enumerate() {
        // RQ accumulation, restricted to the active rows `< p`.
        for row in 0..p {
            let left_idx = row * n + i;
            let right_idx = left_idx + 1;
            let left = h[left_idx];
            let right = h[right_idx];
            h[left_idx] = c * left + s * right;
            h[right_idx] = -s * left + c * right;
        }
    }
}

#[cfg(test)]
fn apply_givens_right(matrix: &mut [f64], n: usize, col: usize, c: f64, s: f64) {
    for row in 0..n {
        let left_idx = row * n + col;
        let right_idx = left_idx + 1;
        let left = matrix[left_idx];
        let right = matrix[right_idx];
        matrix[left_idx] = c * left + s * right;
        matrix[right_idx] = -s * left + c * right;
    }
}

fn normalize_vector(v: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let norm: f64 = v
        .iter()
        .map(|(re, im)| re * re + im * im)
        .sum::<f64>()
        .sqrt();
    if norm < 1e-15 {
        v
    } else {
        v.iter().map(|(re, im)| (re / norm, im / norm)).collect()
    }
}

#[cfg(test)]
fn qr_decomposition(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Classical Gram-Schmidt QR (V1). Q is built and consumed column by column,
    // so the columns are kept in contiguous buffers (`q_cols`) and the original
    // A column j is extracted once (`col_j`). This makes the dot-product and
    // axpy inner loops stride-1 instead of striding by `n` down a column of the
    // row-major matrices (a cache miss per element). The accumulation order over
    // k is unchanged, so q and r are bit-for-bit identical to the row-major form.
    let mut r = vec![0.0; n * n];
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);

    for j in 0..n {
        let col_j: Vec<f64> = (0..n).map(|k| a[k * n + j]).collect();
        let mut v = col_j.clone();

        for (i, q_col_i) in q_cols.iter().enumerate() {
            let mut dot = 0.0;
            for k in 0..n {
                dot += q_col_i[k] * col_j[k];
            }
            r[i * n + j] = dot;
            for k in 0..n {
                v[k] -= dot * q_col_i[k];
            }
        }

        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r[j * n + j] = norm;
        let mut q_col_j = vec![0.0; n];
        if norm > 1e-15 {
            for k in 0..n {
                q_col_j[k] = v[k] / norm;
            }
        }
        q_cols.push(q_col_j);
    }

    // Reassemble row-major Q. eval_eig's QR path is purely real; the public
    // Complex128 wrapping happens only after q_total is fully accumulated.
    let mut q = vec![0.0; n * n];
    for (j, q_col_j) in q_cols.iter().enumerate() {
        for k in 0..n {
            q[k * n + j] = q_col_j[k];
        }
    }

    (q, r)
}

#[cfg(test)]
fn matrix_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // i-k-j order: the inner j-loop streams a contiguous row of B and C rather
    // than the i-j-k order's stride-n walk down a column of B. Each c[i][j]
    // still accumulates ascending-k, so the result is bit-for-bit identical.
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        let a_row = i * n;
        let c_row = i * n;
        for k in 0..n {
            let a_ik = a[a_row + k];
            let b_row = k * n;
            for j in 0..n {
                c[c_row + j] += a_ik * b[b_row + j];
            }
        }
    }
    c
}

#[cfg(test)]
fn upper_triangular_matrix_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        let a_row = i * n;
        let c_row = i * n;
        for k in i..n {
            let a_ik = a[a_row + k];
            let b_row = k * n;
            for j in 0..n {
                c[c_row + j] += a_ik * b[b_row + j];
            }
        }
    }
    c
}

/// Complex i-k-j GEMM. Retained (and unit-tested) for the future
/// complex-eigenvalue path; eval_eig's V1 real path accumulates q_total with
/// the cheaper real `matrix_mul`, so this currently has no production caller.
#[allow(dead_code)]
fn matrix_mul_complex(a: &[(f64, f64)], b: &[(f64, f64)], n: usize) -> Vec<(f64, f64)> {
    // i-k-j order (see matrix_mul). Real and imaginary parts each accumulate
    // ascending-k exactly as before, so the result is bit-for-bit identical.
    let mut c = vec![(0.0, 0.0); n * n];
    for i in 0..n {
        let a_row = i * n;
        let c_row = i * n;
        for k in 0..n {
            let (ar, ai) = a[a_row + k];
            let b_row = k * n;
            for j in 0..n {
                let (br, bi) = b[b_row + j];
                c[c_row + j].0 += ar * br - ai * bi;
                c[c_row + j].1 += ar * bi + ai * br;
            }
        }
    }
    c
}

// ── Determinant ─────────────────────────────────────────────────────

/// Compute the determinant of a square matrix via LU decomposition.
///
/// Matches `jnp.linalg.det(a)`.
pub fn det(a: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 1.0; // Empty matrix has determinant 1
    }
    if n == 1 {
        return a[0];
    }
    if n == 2 {
        return a[0] * a[3] - a[1] * a[2];
    }

    // Large square inputs route through the cache-blocked right-looking LU whose
    // O(n³) Schur update runs at GEMM speed (same kernel `eval_solve`/`eval_lu`
    // use). The block-reordered trailing sum is numerically equivalent (P·A=L·U
    // to machine precision) but not bit-identical to the scalar elimination, so
    // small n (conformance goldens) stays on the unblocked path below. JAX itself
    // factors with blocked LAPACK getrf, so the large-n det matches it to
    // tolerance. The blocked path divides through tiny-but-nonzero pivots exactly
    // as JAX/getrf does (no early zero short-circuit), so genuinely singular
    // inputs fall out as a ~0 diagonal product rather than a forced 0.0.
    if n >= LU_BLOCK_THRESHOLD {
        let mut lu = a.to_vec();
        let (pivots, _perm) = lu_factor_real_blocked(&mut lu, n, n);
        let mut result = 1.0_f64;
        for (col, &piv) in pivots.iter().enumerate() {
            if piv as usize != col {
                result = -result;
            }
        }
        for i in 0..n {
            result *= lu[i * n + i];
        }
        return result;
    }

    // LU decomposition with partial pivoting
    let mut lu = a.to_vec();
    let mut sign = 1.0_f64;

    for k in 0..n {
        // Find pivot
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Only an exactly-zero pivot column is structurally singular -> det 0
        // (and short-circuiting here avoids a 0/0 = NaN in the elimination
        // below). A merely tiny pivot is NOT zeroed: jnp.linalg.det divides
        // through it, so e.g. det(diag(1e-16, 2, 3)) is 6e-16, not 0. The old
        // `< 1e-15` threshold wrongly collapsed such small determinants to 0.
        if max_val == 0.0 {
            return 0.0; // structurally singular (zero pivot column)
        }

        // Swap rows if needed
        if max_row != k {
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            sign = -sign;
        }

        // Eliminate below pivot
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / lu[k * n + k];
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    // Determinant is product of diagonal elements times sign
    let mut result = sign;
    for i in 0..n {
        result *= lu[i * n + i];
    }
    result
}

/// Compute sign and log of absolute determinant.
///
/// Returns (sign, logabsdet) where det(a) = sign * exp(logabsdet).
/// Matches `jnp.linalg.slogdet(a)`.
///
/// Accumulates `logabsdet = Σ ln|U_ii|` directly from the LU factors rather
/// than forming `det(a)` and taking its log: the determinant of a large matrix
/// routinely exceeds f64 range (e.g. a 200×200 matrix with diagonal ≈100 has
/// det ≈ 1e400 → +inf, or ≈1e-400 → 0), and `log(det)` would then yield ±inf
/// where JAX/LAPACK return a finite `logabsdet`. Avoiding the product is the
/// entire reason slogdet exists. Singular detection keeps `det`'s `< 1e-15`
/// pivot threshold so that behavior is unchanged.
pub fn slogdet(a: &[f64], n: usize) -> (f64, f64) {
    if n == 0 {
        // det of the empty matrix is 1; log|1| = 0.
        return (1.0, 0.0);
    }

    // Large inputs route through the cache-blocked GEMM LU (see `det`): the
    // factorization is numerically equivalent to the scalar elimination below
    // and matches JAX's blocked getrf to tolerance, so it is gated above the
    // conformance-golden sizes (n < LU_BLOCK_THRESHOLD stays bit-identical).
    if n >= LU_BLOCK_THRESHOLD {
        let mut lu = a.to_vec();
        let (pivots, _perm) = lu_factor_real_blocked(&mut lu, n, n);
        let mut sign = 1.0_f64;
        for (col, &piv) in pivots.iter().enumerate() {
            if piv as usize != col {
                sign = -sign;
            }
        }
        let mut logabsdet = 0.0_f64;
        for i in 0..n {
            let pivot = lu[i * n + i];
            if pivot.abs() < 1e-15 {
                return (0.0, f64::NEG_INFINITY); // singular
            }
            if pivot < 0.0 {
                sign = -sign;
            }
            logabsdet += pivot.abs().ln();
        }
        return (sign, logabsdet);
    }

    let mut lu = a.to_vec();
    let mut sign = 1.0_f64;
    let mut logabsdet = 0.0_f64;

    for k in 0..n {
        // Partial pivot: largest-magnitude entry in column k at/below the diagonal.
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return (0.0, f64::NEG_INFINITY); // singular
        }

        if max_row != k {
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            sign = -sign;
        }

        let pivot = lu[k * n + k];
        if pivot < 0.0 {
            sign = -sign;
        }
        logabsdet += pivot.abs().ln();

        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            for j in (k + 1)..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    (sign, logabsdet)
}

// ── Matrix Inverse ──────────────────────────────────────────────────

/// Compute the inverse of a square matrix via Gauss-Jordan elimination.
///
/// Matches `jnp.linalg.inv(a)`.
pub fn inv(a: &[f64], n: usize) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(vec![]);
    }

    // Create augmented matrix [A | I]
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Gauss-Jordan elimination
    for k in 0..n {
        // Find pivot
        let mut max_val = aug[k * 2 * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = aug[i * 2 * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular matrix
        }

        // Swap rows
        if max_row != k {
            for j in 0..(2 * n) {
                aug.swap(k * 2 * n + j, max_row * 2 * n + j);
            }
        }

        // Scale pivot row
        let pivot = aug[k * 2 * n + k];
        for j in 0..(2 * n) {
            aug[k * 2 * n + j] /= pivot;
        }

        // Eliminate column
        for i in 0..n {
            if i != k {
                let factor = aug[i * 2 * n + k];
                for j in 0..(2 * n) {
                    aug[i * 2 * n + j] -= factor * aug[k * 2 * n + j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    Some(result)
}

/// Compute the Moore-Penrose pseudoinverse, matching `jnp.linalg.pinv(a, rcond)`.
///
/// Two paths: a low-rank QR shortcut ([`pinv_m_ge_n_low_rank_qr`]) for tall, genuinely
/// low-rank inputs, else the general SVD form ([`pinv_svd`]) `A⁺ = V Σ⁺ Uᵀ` with the
/// rcond cutoff `σ_i > rcond·σ_max` applied to the true singular values — exactly JAX's
/// definition. The SVD is the high-relative-accuracy one-sided Jacobi kernel, so small
/// singular values (and their `1/σ` weights) stay accurate for ill-conditioned A. The
/// earlier Gram form (`AᵀA` / `AAᵀ` eigendecomposition, `1/λ = 1/σ²`) squared the
/// condition number, giving small σ only √ε relative accuracy; results agree for
/// well-conditioned inputs but the SVD form is correct near the cutoff.
pub fn pinv(a: &[f64], m: usize, n: usize, rcond: f64) -> Vec<f64> {
    if m == 0 || n == 0 {
        return vec![0.0; n * m];
    }

    if m >= n
        && n >= PINV_LOW_RANK_MIN_DIM
        && let Some(result) = pinv_m_ge_n_low_rank_qr(a, m, n, rcond)
    {
        return result;
    }

    pinv_svd(a, m, n, rcond)
}

const PINV_LOW_RANK_MIN_DIM: usize = 16;
const PINV_LOW_RANK_MAX_RANK: usize = 32;
/// Accept the rank-revealing QR certificate when unresolved residual energy is
/// at or below the same squared singular-value cutoff used for Σ⁺.
const PINV_LOW_RANK_RESIDUAL_MARGIN: f64 = 1.0;

fn pinv_m_ge_n_low_rank_qr(a: &[f64], m: usize, n: usize, rcond: f64) -> Option<Vec<f64>> {
    if !rcond.is_finite() || rcond == 0.0 || !a.iter().all(|value| value.is_finite()) {
        return None;
    }

    let mut residual_cols = Vec::with_capacity(n);
    let mut residual_norms = Vec::with_capacity(n);
    for col in 0..n {
        let mut column = Vec::with_capacity(m);
        let mut norm_sq = 0.0;
        for row in 0..m {
            let value = a[row * n + col];
            column.push(value);
            norm_sq += value * value;
        }
        residual_cols.push(column);
        residual_norms.push(norm_sq);
    }

    let initial_trace: f64 = residual_norms.iter().sum();
    if initial_trace == 0.0 {
        return Some(vec![0.0; n * m]);
    }

    let rank_limit = PINV_LOW_RANK_MAX_RANK.min((n / 4).max(1));
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(rank_limit);
    let mut r_rows: Vec<Vec<f64>> = Vec::with_capacity(rank_limit);
    let rcond_sq = rcond.abs() * rcond.abs();

    for rank in 0..rank_limit {
        let mut pivot = rank;
        let mut pivot_norm = residual_norms[rank];
        for (candidate, &norm_sq) in residual_norms
            .iter()
            .enumerate()
            .skip(rank + 1)
            .take(n - rank - 1)
        {
            if norm_sq > pivot_norm {
                pivot = candidate;
                pivot_norm = norm_sq;
            }
        }

        if pivot_norm <= 0.0 {
            return None;
        }

        if pivot != rank {
            residual_cols.swap(rank, pivot);
            residual_norms.swap(rank, pivot);
            permutation.swap(rank, pivot);
            for r_row in &mut r_rows {
                r_row.swap(rank, pivot);
            }
        }

        let norm = pivot_norm.sqrt();
        if norm <= f64::EPSILON {
            return None;
        }

        let mut q = vec![0.0; m];
        for row in 0..m {
            q[row] = residual_cols[rank][row] / norm;
        }
        for previous_q in &q_cols {
            let mut projection = 0.0;
            for row in 0..m {
                projection += previous_q[row] * q[row];
            }
            for row in 0..m {
                q[row] -= projection * previous_q[row];
            }
        }
        let q_norm_sq = q.iter().map(|value| value * value).sum::<f64>();
        if q_norm_sq <= f64::EPSILON {
            return None;
        }
        let q_norm = q_norm_sq.sqrt();
        for value in &mut q {
            *value /= q_norm;
        }

        for col in rank..n {
            let mut projection = 0.0;
            for row in 0..m {
                projection += q[row] * residual_cols[col][row];
            }
            for row in 0..m {
                residual_cols[col][row] -= projection * q[row];
            }
            residual_norms[col] = residual_cols[col]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .max(0.0);
        }

        let mut r_row = vec![0.0; n];
        for col in 0..n {
            let original_col = permutation[col];
            let mut projection = 0.0;
            for row in 0..m {
                projection += q[row] * a[row * n + original_col];
            }
            r_row[col] = projection;
        }
        q_cols.push(q);
        r_rows.push(r_row);

        let active_rank = rank + 1;
        let mut rrt = low_rank_rrt(&r_rows, active_rank, n);
        let (lambdas, eigenvectors) = jacobi_eigendecomposition_cyclic(&mut rrt, active_rank);
        let sigma_max_sq = lambdas.iter().copied().fold(0.0_f64, f64::max);
        let cutoff = rcond_sq * sigma_max_sq;
        let remaining_trace: f64 = residual_norms.iter().skip(active_rank).sum();

        if sigma_max_sq > 0.0 && remaining_trace <= cutoff * PINV_LOW_RANK_RESIDUAL_MARGIN {
            return Some(low_rank_qr_pinv_result(
                &q_cols,
                &r_rows,
                &permutation,
                &lambdas,
                &eigenvectors,
                cutoff,
                m,
                n,
                active_rank,
            ));
        }
    }

    None
}

fn low_rank_rrt(r_rows: &[Vec<f64>], rank: usize, n: usize) -> Vec<f64> {
    let mut rrt = vec![0.0; rank * rank];
    for i in 0..rank {
        for j in i..rank {
            let mut dot = 0.0;
            for (&left, &right) in r_rows[i].iter().zip(&r_rows[j]).take(n) {
                dot += left * right;
            }
            rrt[i * rank + j] = dot;
            rrt[j * rank + i] = dot;
        }
    }
    rrt
}

#[expect(
    clippy::too_many_arguments,
    reason = "small linear-algebra kernel keeps dimensions explicit"
)]
fn low_rank_qr_pinv_result(
    q_cols: &[Vec<f64>],
    r_rows: &[Vec<f64>],
    permutation: &[usize],
    lambdas: &[f64],
    eigenvectors: &[f64],
    cutoff: f64,
    m: usize,
    n: usize,
    rank: usize,
) -> Vec<f64> {
    let mut r_plus = vec![0.0; n * rank];
    for eig in 0..rank {
        let lambda = lambdas[eig];
        if lambda <= cutoff || lambda <= 0.0 {
            continue;
        }
        let inv_lambda = 1.0 / lambda;
        for col in 0..n {
            let mut rt_u = 0.0;
            for row in 0..rank {
                rt_u += r_rows[row][col] * eigenvectors[row * rank + eig];
            }
            for q_idx in 0..rank {
                r_plus[col * rank + q_idx] += rt_u * eigenvectors[q_idx * rank + eig] * inv_lambda;
            }
        }
    }

    let mut result = vec![0.0; n * m];
    for permuted_col in 0..n {
        let original_col = permutation[permuted_col];
        for row in 0..m {
            let mut value = 0.0;
            for q_idx in 0..rank {
                value += r_plus[permuted_col * rank + q_idx] * q_cols[q_idx][row];
            }
            result[original_col * m + row] = value;
        }
    }
    result
}

/// SVD-based Moore–Penrose pseudoinverse, matching JAX's definition
/// `A⁺ = V diag(σ_i⁻¹ for σ_i > rcond·σ_max, else 0) Uᵀ`. Uses the high-relative-
/// accuracy one-sided Jacobi SVD ([`one_sided_jacobi_svd_real`]), so the rcond cutoff
/// acts on the TRUE singular values and the 1/σ weights stay accurate for
/// ill-conditioned inputs. The former AᵀA-Gram form computed `1/λ = 1/σ²` from squared
/// eigenvalues, giving small singular values only √ε relative accuracy — which propagated
/// into pinv (and lstsq) for ill-conditioned A. Algebraically equivalent for
/// well-conditioned inputs, more accurate near the cutoff, and JAX-exact.
fn pinv_svd(a: &[f64], m: usize, n: usize, rcond: f64) -> Vec<f64> {
    let k = m.min(n);
    let (sigma, u, v) = one_sided_jacobi_svd_real(m, n, a);
    let mut u_cols = vec![0.0_f64; k * m];
    for l in 0..k {
        for j in 0..m {
            u_cols[l * m + j] = u[j * k + l];
        }
    }
    // σ descending ⇒ σ_max = sigma[0]. JAX/NumPy cutoff: drop σ ≤ rcond·σ_max. A
    // non-finite rcond yields a NaN cutoff, so every σ is dropped (σ > NaN is false) and
    // the pseudoinverse is the zero matrix — matching the prior Gram behaviour.
    let sigma_max = sigma.first().copied().unwrap_or(0.0);
    let cutoff = rcond.abs() * sigma_max;
    // A⁺ = (V·Σ⁺)·Uᵀ over the KEPT singular values: `V_scaled[i,c] = V[i,l]/σ_l` (n×kk)
    // times `u_kept[c,j] = U[j,l]` (kk×m). Computing it as one cache-blocked, threaded
    // `matmul_2d` instead of the prior scalar `Σ_l` outer-product triple loop. The GEMM
    // accumulates each output's k-sum in ascending column order — the same ascending-`l`
    // order over kept σ as the old loop — so it is bit-for-bit identical.
    let kept: Vec<usize> = (0..k)
        .filter(|&l| sigma[l] > cutoff && sigma[l] > 0.0)
        .collect();
    let kk = kept.len();
    if kk == 0 {
        return vec![0.0_f64; n * m];
    }
    let mut v_scaled = vec![0.0_f64; n * kk];
    let mut u_kept = vec![0.0_f64; kk * m];
    for (c, &l) in kept.iter().enumerate() {
        let inv = 1.0 / sigma[l];
        for i in 0..n {
            v_scaled[i * kk + c] = v[i * n + l] * inv;
        }
        u_kept[c * m..c * m + m].copy_from_slice(&u_cols[l * m..l * m + m]);
    }
    crate::tensor_contraction::matmul_2d(&v_scaled, n, kk, &u_kept, m)
}

// ── Norms ───────────────────────────────────────────────────────────

/// Compute vector or matrix norms.
///
/// For vectors: ord=None or 2 gives L2 norm, ord=1 gives L1, ord=inf gives max.
/// For matrices: ord=None or "fro" gives Frobenius norm.
///
/// Matches `jnp.linalg.norm(x, ord)`.
pub fn vector_norm(x: &[f64], ord: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }

    if ord == 0.0 {
        // L0 "norm": count of non-zero elements
        x.iter().filter(|&&v| v != 0.0).count() as f64
    } else if ord == 1.0 {
        // L1 norm: sum of absolute values
        x.iter().map(|&v| v.abs()).sum()
    } else if ord == 2.0 {
        // L2 norm: Euclidean
        x.iter().map(|&v| v * v).sum::<f64>().sqrt()
    } else if ord == f64::INFINITY {
        // L-infinity norm: max absolute value
        x.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max)
    } else if ord == f64::NEG_INFINITY {
        // L-neg-infinity norm: min absolute value
        x.iter().map(|&v| v.abs()).fold(f64::INFINITY, f64::min)
    } else {
        // General Lp norm
        x.iter()
            .map(|&v| v.abs().powf(ord))
            .sum::<f64>()
            .powf(1.0 / ord)
    }
}

/// Frobenius norm of a matrix: sqrt(sum of squared elements).
pub fn frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Matrix 1-norm (max column sum of absolute values).
pub fn matrix_norm_1(a: &[f64], m: usize, n: usize) -> f64 {
    let mut max_col_sum = 0.0_f64;
    for j in 0..n {
        let col_sum: f64 = (0..m).map(|i| a[i * n + j].abs()).sum();
        max_col_sum = max_col_sum.max(col_sum);
    }
    max_col_sum
}

/// Matrix infinity-norm (max row sum of absolute values).
pub fn matrix_norm_inf(a: &[f64], m: usize, n: usize) -> f64 {
    let mut max_row_sum = 0.0_f64;
    for i in 0..m {
        let row_sum: f64 = (0..n).map(|j| a[i * n + j].abs()).sum();
        max_row_sum = max_row_sum.max(row_sum);
    }
    max_row_sum
}

/// Solve the linear system Ax = b using LU decomposition with partial pivoting.
///
/// Matches `jnp.linalg.solve(a, b)`.
///
/// Returns None if the matrix is singular.
/// LU-factorize an `n`x`n` matrix in place with partial (row) pivoting.
///
/// Returns `(lu, p)` where `lu` packs the unit-lower `L` (strictly below the
/// diagonal) and `U` (on/above the diagonal), and `p` is the row permutation.
/// Returns `None` only for a shape mismatch — a singular or near-singular `A`
/// is NOT rejected. Elimination divides through the (tiny or zero) pivot,
/// giving a finite large factor for near-singular `A` and inf/NaN for an
/// exactly-singular one, matching JAX's lu/solve (which never raise; NumPy
/// raises LinAlgError). The elimination order and pivot rule depend only on
/// `A`, so a single factor can drive any number of right-hand sides bit-identically.
fn lu_factor(a: &[f64], n: usize) -> Option<(Vec<f64>, Vec<usize>)> {
    if a.len() != n * n {
        return None;
    }

    let mut lu = a.to_vec();
    let mut p: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut max_idx = k;
        let mut max_val = lu[k * n + k].abs();
        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // No singularity bail-out: JAX's solve divides through a zero/near-zero
        // pivot (inf/NaN or finite-large) rather than raising. `max_val` is used
        // only to pick the partial-pivot row.
        let _ = max_val;

        if max_idx != k {
            p.swap(k, max_idx);
            for j in 0..n {
                lu.swap(k * n + j, max_idx * n + j);
            }
        }

        for i in (k + 1)..n {
            let factor = lu[i * n + k] / lu[k * n + k];
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    Some((lu, p))
}

/// Solve `Ax = b` from a precomputed LU factorization via forward then back
/// substitution. Arithmetic and ordering match the inline substitution `solve`
/// previously performed, so the result is bit-for-bit identical.
fn lu_solve(lu: &[f64], p: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[p[i]];
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= lu[i * n + j] * y[j];
        }
        y[i] = sum;
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu[i * n + j] * x[j];
        }
        x[i] = sum / lu[i * n + i];
    }

    x
}

/// LU factorization for the dense real `solve`/`solve_multi_rhs` path, returning
/// `(combined L\U row-major, swap-built row permutation)` consumed identically by
/// `lu_solve`. For small `n` (< `LU_BLOCK_THRESHOLD`) this is the bit-identical
/// scalar `lu_factor` the conformance goldens pin; at scale it switches to the
/// cache-blocked, GEMM-accelerated `lu_factor_real_blocked` (LAPACK `getrf` shape),
/// whose O(n³) trailing Schur update `A22 -= L21·U12` runs at `matmul_2d`
/// microkernel speed instead of strided scalar rank-1 sweeps. The blocked result is
/// numerically equivalent — `P·A = L·U` to machine precision, exactly JAX's own
/// blocked `getrf` guarantee — differing only in the last ulp; `solve` parity is
/// tolerance-based and its goldens cover only small `n` (same gate + rationale as
/// `eval_lu`). `lu_factor_real_blocked` builds `perm` by the identical
/// `swap(col, max_row)`-from-identity sequence, so converting it to `usize` feeds
/// `lu_solve` (`pb[i] = b[perm[i]]`) exactly as the scalar `p` does.
fn lu_factor_for_solve(a: &[f64], n: usize) -> Option<(Vec<f64>, Vec<usize>)> {
    if a.len() != n * n {
        return None;
    }
    if n >= LU_BLOCK_THRESHOLD {
        let mut lu = a.to_vec();
        let (_pivots, perm) = lu_factor_real_blocked(&mut lu, n, n);
        let p: Vec<usize> = perm.iter().map(|&row| row as usize).collect();
        Some((lu, p))
    } else {
        lu_factor(a, n)
    }
}

/// Below this `n` the f64 LU already dominates (the O(n³) trailing GEMM is too
/// small for the f32 GEMM's ~2× to pay against the f32-factor + refinement
/// overhead), so the mixed-precision solve only engages at scale.
const MIXED_PRECISION_SOLVE_MIN_N: usize = 512;
/// Refinement iteration cap; well-conditioned systems converge to f64 accuracy in
/// 2–3, so a higher cap only ever helps borderline cases before the f64 fallback.
const MIXED_REFINE_MAX_ITERS: usize = 6;

/// f32 blocked LU (LAPACK `getrf` shape), a single-precision mirror of
/// [`lu_factor_real_blocked`] with the trailing Schur update `A22 -= L21·U12` run
/// through the native-f32 GEMM (~2× the f64 kernel — see the f32-vs-f64 GEMM
/// diagnostic). For the mixed-precision solve ONLY (NOT `eval_lu`, whose goldens
/// need f64). `lu` holds the combined L\U on return; the returned `perm` is the
/// swap-from-identity row permutation consumed by [`lu_solve_f32`].
fn lu_factor_real_blocked_f32(lu: &mut [f32], n: usize) -> Vec<usize> {
    let m = n;
    let mut perm: Vec<usize> = (0..m).collect();
    let nb = LU_BLOCK_SIZE;
    let mut j = 0;
    while j < n {
        let jb = nb.min(n - j);
        let panel_end = j + jb;
        for col in j..panel_end {
            let mut max_val = lu[col * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..m {
                let v = lu[row * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_row != col {
                perm.swap(col, max_row);
                for jj in 0..n {
                    lu.swap(col * n + jj, max_row * n + jj);
                }
            }
            let diag = lu[col * n + col];
            if diag.abs() < f32::EPSILON * 1e-6 {
                continue;
            }
            for row in (col + 1)..m {
                let factor = lu[row * n + col] / diag;
                lu[row * n + col] = factor;
                for jj in (col + 1)..panel_end {
                    lu[row * n + jj] -= factor * lu[col * n + jj];
                }
            }
        }
        if panel_end < n {
            for ri in 1..jb {
                let r = j + ri;
                for t in 0..ri {
                    let l_rt = lu[r * n + (j + t)];
                    if l_rt != 0.0 {
                        let trow = j + t;
                        for jj in panel_end..n {
                            lu[r * n + jj] -= l_rt * lu[trow * n + jj];
                        }
                    }
                }
            }
        }
        let rows_below = m - panel_end;
        let cols_right = n - panel_end;
        if rows_below > 0 && cols_right > 0 {
            let mut l21 = Vec::with_capacity(rows_below * jb);
            for p in 0..rows_below {
                let src = (panel_end + p) * n + j;
                l21.extend_from_slice(&lu[src..src + jb]);
            }
            let mut u12 = Vec::with_capacity(jb * cols_right);
            for t in 0..jb {
                let src = (j + t) * n + panel_end;
                u12.extend_from_slice(&lu[src..src + cols_right]);
            }
            let prod = crate::tensor_contraction::batched_matmul_2d_f32_in(
                &l21, 1, rows_below, jb, &u12, cols_right,
            );
            for p in 0..rows_below {
                let row = (panel_end + p) * n + panel_end;
                let pr = p * cols_right;
                for q in 0..cols_right {
                    lu[row + q] -= prod[pr + q];
                }
            }
        }
        j = panel_end;
    }
    perm
}

/// Solve `L·U·x = P·b` with f32 combined-L\U factors (unit-lower `L`), all in f32.
fn lu_solve_f32(lu: &[f32], p: &[usize], b: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b[p[i]];
        for j in 0..i {
            sum -= lu[i * n + j] * y[j];
        }
        y[i] = sum;
    }
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu[i * n + j] * x[j];
        }
        x[i] = sum / lu[i * n + i];
    }
    x
}

/// Blocked (communication-avoiding) f32 multi-RHS triangular solve `L·U·X = P·B` —
/// the single-precision sibling of [`lu_solve_multi_blocked`]. The bulk cross-block
/// elimination `X_block −= L_block·X_solved` runs through the native-f32 GEMM
/// [`batched_matmul_2d_f32_in`] (~2× the f64 kernel and X streams from cache instead
/// of being re-read O(n) times by the scalar sweep); only the small diagonal block
/// is solved by scalar substitution. For the mixed-precision multi-RHS path, where
/// the result is refined to f64 accuracy against f64 residuals — so the f32 block
/// rounding is absorbed by refinement.
fn lu_solve_multi_blocked_f32(lu: &[f32], p: &[usize], b: &[f32], n: usize, m: usize) -> Vec<f32> {
    let nb = TRSM_BLOCK;
    let mut x = vec![0.0f32; n * m];
    for i in 0..n {
        let src = p[i] * m;
        x[i * m..i * m + m].copy_from_slice(&b[src..src + m]);
    }
    // Forward (L, unit lower): left-looking blocks ascending.
    let mut r0 = 0;
    while r0 < n {
        let r1 = (r0 + nb).min(n);
        let bsz = r1 - r0;
        if r0 > 0 {
            let mut l_sub = vec![0.0f32; bsz * r0];
            for ii in 0..bsz {
                let s = (r0 + ii) * n;
                l_sub[ii * r0..ii * r0 + r0].copy_from_slice(&lu[s..s + r0]);
            }
            let prod = crate::tensor_contraction::batched_matmul_2d_f32_in(
                &l_sub,
                1,
                bsz,
                r0,
                &x[0..r0 * m],
                m,
            );
            for ii in 0..bsz {
                let xb = (r0 + ii) * m;
                for col in 0..m {
                    x[xb + col] -= prod[ii * m + col];
                }
            }
        }
        for i in r0..r1 {
            for k in r0..i {
                let l = lu[i * n + k];
                let (head, tail) = x.split_at_mut(i * m);
                let xk = &head[k * m..k * m + m];
                let xi = &mut tail[..m];
                for col in 0..m {
                    xi[col] -= l * xk[col];
                }
            }
        }
        r0 = r1;
    }
    // Back (U, upper): left-looking blocks descending.
    let mut r1 = n;
    while r1 > 0 {
        let r0 = r1.saturating_sub(nb);
        let bsz = r1 - r0;
        if r1 < n {
            let cw = n - r1;
            let mut u_sub = vec![0.0f32; bsz * cw];
            for ii in 0..bsz {
                let s = (r0 + ii) * n + r1;
                u_sub[ii * cw..ii * cw + cw].copy_from_slice(&lu[s..s + cw]);
            }
            let prod = crate::tensor_contraction::batched_matmul_2d_f32_in(
                &u_sub,
                1,
                bsz,
                cw,
                &x[r1 * m..n * m],
                m,
            );
            for ii in 0..bsz {
                let xb = (r0 + ii) * m;
                for col in 0..m {
                    x[xb + col] -= prod[ii * m + col];
                }
            }
        }
        for i in (r0..r1).rev() {
            for k in (i + 1)..r1 {
                let u = lu[i * n + k];
                let (head, tail) = x.split_at_mut(k * m);
                let xi = &mut head[i * m..i * m + m];
                let xk = &tail[..m];
                for col in 0..m {
                    xi[col] -= u * xk[col];
                }
            }
            let diag = lu[i * n + i];
            let xi = &mut x[i * m..i * m + m];
            for v in xi.iter_mut() {
                *v /= diag;
            }
        }
        r1 = r0;
    }
    x
}

/// f32 batched multi-RHS triangular solve `L·U·X = P·B` (`B`, `X` row-major `n×m`)
/// from a combined f32 `lu` factor — the single-precision sibling of
/// [`lu_solve_multi`]. Each `lu[i][k]` is read once and applied across all `m`
/// columns (the inner column loop auto-vectorizes), so the shared factor streams
/// from cache once instead of once per column. Used by the mixed-precision
/// multi-RHS solve (initial f32 solve + each f32 refinement correction). At scale
/// (large `m`) the scalar column sweep is the bottleneck, so it switches to the
/// communication-avoiding [`lu_solve_multi_blocked_f32`] (f32 GEMM cross-block
/// elimination) — the f32 block rounding is absorbed by f64 residual refinement.
fn lu_solve_multi_f32(lu: &[f32], p: &[usize], b: &[f32], n: usize, m: usize) -> Vec<f32> {
    if n >= MIXED_PRECISION_SOLVE_MIN_N && m >= 64 {
        return lu_solve_multi_blocked_f32(lu, p, b, n, m);
    }
    let mut x = vec![0.0f32; n * m];
    // Forward (unit-lower L): y[i][:] = (P·B)[i][:] − Σ_{k<i} lu[i][k]·y[k][:].
    for i in 0..n {
        let xi = i * m;
        let src = p[i] * m;
        x[xi..xi + m].copy_from_slice(&b[src..src + m]);
        for k in 0..i {
            let l = lu[i * n + k];
            let (head, tail) = x.split_at_mut(xi);
            let yk = &head[k * m..k * m + m];
            let yi = &mut tail[..m];
            for col in 0..m {
                yi[col] -= l * yk[col];
            }
        }
    }
    // Back (upper U): x[i][:] = (y[i][:] − Σ_{k>i} lu[i][k]·x[k][:]) / lu[i][i].
    for i in (0..n).rev() {
        let xi = i * m;
        for k in (i + 1)..n {
            let u = lu[i * n + k];
            let (head, tail) = x.split_at_mut(k * m);
            let yi = &mut head[xi..xi + m];
            let xk = &tail[..m];
            for col in 0..m {
                yi[col] -= u * xk[col];
            }
        }
        let diag = lu[i * n + i];
        for col in 0..m {
            x[xi + col] /= diag;
        }
    }
    x
}

/// Mixed-precision iterative-refinement solve: factor `A` ONCE in f32 (≈2× faster
/// than f64 at scale), then recover f64 accuracy by refining against f64-precision
/// residuals (`r = b − A·x` in f64; solve `A·dx = r` with the cached f32 factors;
/// `x += dx`). Classical Wilkinson refinement: a backward-stable f32 factorization
/// plus f64 residuals converges to a result whose residual is at working (f64)
/// precision for well-conditioned `A`, so it matches the plain f64 solve to within
/// `solve`'s tolerance. Returns `None` (caller falls back to the f64 solve) when
/// the f32 factor is singular or refinement fails to reach f64-quality within
/// [`MIXED_REFINE_MAX_ITERS`] (ill-conditioned `A`) — so parity is never at risk.
fn solve_mixed_precision(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut lu32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let perm = lu_factor_real_blocked_f32(&mut lu32, n);
    for i in 0..n {
        let d = lu32[i * n + i];
        if !d.is_finite() || d == 0.0 {
            return None;
        }
    }
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if bnorm == 0.0 {
        return Some(vec![0.0; n]);
    }
    // `r = b − A·x` in f64; returns its 2-norm. The `A·x` matvec is the refinement's
    // dominant per-iteration cost (O(n²), and A streams from RAM), so it runs through
    // the cache-blocked, auto-threaded f64 `matmul_2d` (A as n×n, x as n×1) instead of
    // a serial dot-product loop — without it the refinement caps the solve well below
    // the f32 LU's ~2× speedup.
    let residual = |x: &[f64], r: &mut [f64]| -> f64 {
        let ax = crate::tensor_contraction::matmul_2d(a, n, n, x, 1);
        let mut s2 = 0.0f64;
        for i in 0..n {
            let s = b[i] - ax[i];
            r[i] = s;
            s2 += s * s;
        }
        s2.sqrt()
    };

    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let mut x: Vec<f64> = lu_solve_f32(&lu32, &perm, &b32, n)
        .iter()
        .map(|&v| v as f64)
        .collect();

    let accept = 1e-12 * (bnorm + 1.0); // f64-quality residual → matches the f64 solve
    let mut r = vec![0.0f64; n];
    let mut prev = f64::INFINITY;
    for _ in 0..MIXED_REFINE_MAX_ITERS {
        let rnorm = residual(&x, &mut r);
        if rnorm <= accept {
            return Some(x);
        }
        if rnorm >= prev {
            // refinement stalled (f32-factor / conditioning limit) — fail to f64.
            return None;
        }
        prev = rnorm;
        let r32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let dx = lu_solve_f32(&lu32, &perm, &r32, n);
        for i in 0..n {
            x[i] += dx[i] as f64;
        }
    }
    if residual(&x, &mut r) <= accept {
        Some(x)
    } else {
        None
    }
}

/// Mixed-precision iterative-refinement multi-RHS solve `A·X = B` (`B`, `X` are
/// `n×m` row-major) — the dense-RHS sibling of [`solve_mixed_precision`]. Factor
/// `A` ONCE in f32 (≈2× faster than f64 at scale), solve all `m` columns in f32,
/// then recover f64 accuracy by refining against f64-precision residuals
/// (`R = B − A·X` in f64; solve `A·dX = R` with the cached f32 factors; `X += dX`).
/// The dominant per-iteration cost — the `A·X` residual GEMM, O(n²·m) — runs
/// through the cache-blocked, auto-threaded f64 [`matmul_2d`] (`A` as `n×n`, `X` as
/// `n×m`); the f32 triangular re-solves reuse [`lu_solve_multi_f32`]. Returns `None`
/// (caller falls back to the f64 [`lu_solve_multi`]) when the f32 factor is singular
/// or refinement fails to reach f64-quality within [`MIXED_REFINE_MAX_ITERS`]
/// (ill-conditioned `A`), so parity is never at risk. Convergence is judged on the
/// Frobenius norm of the full residual, matching the single-RHS 2-norm target.
fn solve_multi_rhs_mixed_precision(a: &[f64], b: &[f64], n: usize, m: usize) -> Option<Vec<f64>> {
    if a.len() != n * n || b.len() != n * m {
        return None;
    }
    let mut lu32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let perm = lu_factor_real_blocked_f32(&mut lu32, n);
    for i in 0..n {
        let d = lu32[i * n + i];
        if !d.is_finite() || d == 0.0 {
            return None;
        }
    }
    let bnorm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if bnorm == 0.0 {
        return Some(vec![0.0; n * m]);
    }
    // `R = B − A·X` in f64; returns its Frobenius norm. `A·X` is the refinement's
    // dominant per-iteration cost (O(n²·m), and A streams from RAM), so it runs
    // through the cache-blocked, auto-threaded f64 `matmul_2d` (A as n×n, X as n×m).
    let residual = |x: &[f64], r: &mut [f64]| -> f64 {
        let ax = crate::tensor_contraction::matmul_2d(a, n, n, x, m);
        let mut s2 = 0.0f64;
        for i in 0..n * m {
            let s = b[i] - ax[i];
            r[i] = s;
            s2 += s * s;
        }
        s2.sqrt()
    };

    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let mut x: Vec<f64> = lu_solve_multi_f32(&lu32, &perm, &b32, n, m)
        .iter()
        .map(|&v| v as f64)
        .collect();

    let accept = 1e-12 * (bnorm + 1.0); // f64-quality residual → matches the f64 solve
    let mut r = vec![0.0f64; n * m];
    let mut prev = f64::INFINITY;
    for _ in 0..MIXED_REFINE_MAX_ITERS {
        let rnorm = residual(&x, &mut r);
        if rnorm <= accept {
            return Some(x);
        }
        if rnorm >= prev {
            // refinement stalled (f32-factor / conditioning limit) — fail to f64.
            return None;
        }
        prev = rnorm;
        let r32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
        let dx = lu_solve_multi_f32(&lu32, &perm, &r32, n, m);
        for i in 0..n * m {
            x[i] += dx[i] as f64;
        }
    }
    if residual(&x, &mut r) <= accept {
        Some(x)
    } else {
        None
    }
}

pub fn solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n || b.len() != n {
        return None;
    }
    // Mixed-precision iterative refinement at scale (falls through to f64 on
    // singular / ill-conditioned non-convergence so the result is always f64-quality).
    if n >= MIXED_PRECISION_SOLVE_MIN_N
        && let Some(x) = solve_mixed_precision(a, b, n)
    {
        return Some(x);
    }
    let (lu, p) = lu_factor_for_solve(a, n)?;
    Some(lu_solve(&lu, &p, b, n))
}

/// Block size for the communication-avoiding multi-RHS triangular solve.
const TRSM_BLOCK: usize = 64;

/// Blocked (communication-avoiding) multi-RHS triangular solve `L·U·X = P·B`.
/// Splits the substitution into `TRSM_BLOCK`-row blocks: the bulk cross-block
/// elimination `X_block −= L_block·X_solved` runs through the cache-blocked,
/// threaded `matmul_2d` (X streams from cache instead of being re-read O(n) times
/// by the scalar SAXPY sweep), and only the small diagonal block is solved by
/// scalar substitution. NOT bit-identical to the per-column / batched solve (the
/// GEMM sums a block's contributions before subtracting, vs the running scalar
/// subtraction) — but within solve's tolerance, so it is gated to large `n`/`m`
/// where the GEMM pays and the bit-exact small-`n` goldens keep the scalar path.
fn lu_solve_multi_blocked(lu: &[f64], p: &[usize], b: &[f64], n: usize, m: usize) -> Vec<f64> {
    let nb = TRSM_BLOCK;
    let mut x = vec![0.0f64; n * m];
    for i in 0..n {
        let src = p[i] * m;
        x[i * m..i * m + m].copy_from_slice(&b[src..src + m]);
    }
    // Forward (L, unit lower): left-looking blocks ascending.
    let mut r0 = 0;
    while r0 < n {
        let r1 = (r0 + nb).min(n);
        let bsz = r1 - r0;
        if r0 > 0 {
            let mut l_sub = vec![0.0f64; bsz * r0];
            for ii in 0..bsz {
                let s = (r0 + ii) * n;
                l_sub[ii * r0..ii * r0 + r0].copy_from_slice(&lu[s..s + r0]);
            }
            let prod = crate::tensor_contraction::matmul_2d(&l_sub, bsz, r0, &x[0..r0 * m], m);
            for ii in 0..bsz {
                let xb = (r0 + ii) * m;
                for col in 0..m {
                    x[xb + col] -= prod[ii * m + col];
                }
            }
        }
        for i in r0..r1 {
            for k in r0..i {
                let l = lu[i * n + k];
                let (head, tail) = x.split_at_mut(i * m);
                let xk = &head[k * m..k * m + m];
                let xi = &mut tail[..m];
                for col in 0..m {
                    xi[col] -= l * xk[col];
                }
            }
        }
        r0 = r1;
    }
    // Back (U, upper): left-looking blocks descending.
    let mut r1 = n;
    while r1 > 0 {
        let r0 = r1.saturating_sub(nb);
        let bsz = r1 - r0;
        if r1 < n {
            let cw = n - r1;
            let mut u_sub = vec![0.0f64; bsz * cw];
            for ii in 0..bsz {
                let s = (r0 + ii) * n + r1;
                u_sub[ii * cw..ii * cw + cw].copy_from_slice(&lu[s..s + cw]);
            }
            let prod = crate::tensor_contraction::matmul_2d(&u_sub, bsz, cw, &x[r1 * m..n * m], m);
            for ii in 0..bsz {
                let xb = (r0 + ii) * m;
                for col in 0..m {
                    x[xb + col] -= prod[ii * m + col];
                }
            }
        }
        for i in (r0..r1).rev() {
            for k in (i + 1)..r1 {
                let u = lu[i * n + k];
                let (head, tail) = x.split_at_mut(k * m);
                let xi = &mut head[i * m..i * m + m];
                let xk = &tail[..m];
                for col in 0..m {
                    xi[col] -= u * xk[col];
                }
            }
            let diag = lu[i * n + i];
            let xi = &mut x[i * m..i * m + m];
            for v in xi.iter_mut() {
                *v /= diag;
            }
        }
        r1 = r0;
    }
    x
}

/// Batched multi-RHS triangular solve `L·U·X = P·B` (B and X are n×m row-major)
/// from a combined `lu` factor. Reads each `lu[i][k]` ONCE and applies it across
/// ALL m columns (the inner column loop auto-vectorizes), so the shared factor
/// streams from cache once instead of once per column. BIT-IDENTICAL to running
/// `m` independent [`lu_solve`] calls: each column's forward/back substitution is
/// the same ascending-`k` fold, unchanged — only the loop nesting differs.
fn lu_solve_multi(lu: &[f64], p: &[usize], b: &[f64], n: usize, m: usize) -> Vec<f64> {
    // At scale, the communication-avoiding blocked TRSM (cross-block elimination via
    // matmul_2d) beats the scalar SAXPY sweep ~2-2.7x. It is not bit-identical (block
    // sum vs running subtract) — within solve tolerance — so the small-n bit-exact
    // path below is preserved (and the bit-identity goldens use n < this gate).
    if n >= MIXED_PRECISION_SOLVE_MIN_N && m >= 64 {
        return lu_solve_multi_blocked(lu, p, b, n, m);
    }
    let mut x = vec![0.0f64; n * m];
    // Forward: y[i][:] = (P·B)[i][:] − Σ_{k<i} lu[i][k]·y[k][:]  (stored in x).
    for i in 0..n {
        let xi = i * m;
        let src = p[i] * m;
        x[xi..xi + m].copy_from_slice(&b[src..src + m]);
        for k in 0..i {
            let l = lu[i * n + k];
            let (head, tail) = x.split_at_mut(xi);
            let yk = &head[k * m..k * m + m];
            let yi = &mut tail[..m];
            for col in 0..m {
                yi[col] -= l * yk[col];
            }
        }
    }
    // Back: x[i][:] = (y[i][:] − Σ_{k>i} lu[i][k]·x[k][:]) / lu[i][i].
    for i in (0..n).rev() {
        let xi = i * m;
        for k in (i + 1)..n {
            let u = lu[i * n + k];
            let (head, tail) = x.split_at_mut(k * m);
            let yi = &mut head[xi..xi + m];
            let xk = &tail[..m];
            for col in 0..m {
                yi[col] -= u * xk[col];
            }
        }
        let diag = lu[i * n + i];
        for col in 0..m {
            x[xi + col] /= diag;
        }
    }
    x
}

/// Solve the linear system Ax = b for multiple right-hand sides.
///
/// A is n x n, B is n x m, returns X which is n x m. The LU factorization is
/// computed once and reused across all `m` columns (previously `A` was
/// re-factorized per column: O(n^3 * m) -> O(n^3 + n^2 * m)). Bit-for-bit
/// identical: every column previously received the same factorization of the
/// same `A`.
pub fn solve_multi_rhs(a: &[f64], b: &[f64], n: usize, m: usize) -> Option<Vec<f64>> {
    // Mixed-precision iterative refinement in the factorization-dominated regime:
    // the O(n³) factor runs ~2× faster in f32 while the refinement overhead is
    // O(n²·m·iters) (an f64 residual GEMM + f32 re-solve per iteration). The f32-LU
    // win only survives the refinement tax when the cubic factor dominates the
    // quadratic refinement, i.e. `n ≥ 8·m`; beyond that the already-blocked f64
    // multi-RHS solve is hard to beat, so fall through. Always falls back to the
    // f64 path on singular / ill-conditioned non-convergence, so the result is
    // f64-quality regardless (parity preserved).
    if n >= MIXED_PRECISION_SOLVE_MIN_N
        && n >= 8 * m
        && let Some(x) = solve_multi_rhs_mixed_precision(a, b, n, m)
    {
        return Some(x);
    }
    let (lu, p) = lu_factor_for_solve(a, n)?;
    // Batched forward/back substitution: stream the shared L\U factor once and
    // apply across all m columns, instead of `m` per-column passes each re-reading
    // it (memory-bound at scale). Bit-identical to the per-column solve.
    Some(lu_solve_multi(&lu, &p, b, n, m))
}

/// Minimum-norm least-squares solution to Ax = b, matching `jnp.linalg.lstsq`.
///
/// Returns the `x` minimizing `||Ax - b||₂` and, among minimizers, the one with
/// smallest `||x||₂` — i.e. `x = A⁺ b` via the Moore-Penrose pseudoinverse.
///
/// The previous form solved the normal equations `solve(AᵀA, Aᵀb)`, which
/// returned `None` for rank-deficient `A` (the Gram matrix is singular) where
/// JAX returns the SVD min-norm solution, and squared the condition number. For
/// full-rank systems this yields the same unique least-squares solution.
pub fn lstsq(a: &[f64], m: usize, n: usize, b: &[f64]) -> Option<Vec<f64>> {
    if m == 0 || n == 0 {
        return Some(vec![0.0; n]);
    }

    // NumPy/JAX default cutoff: singular values <= rcond·σ_max are dropped.
    let rcond = (m.max(n) as f64) * f64::EPSILON;
    let p = pinv(a, m, n, rcond); // n×m

    // x = A⁺ b  (n-vector).
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..m {
            sum += p[i * m + j] * b[j];
        }
        x[i] = sum;
    }
    Some(x)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_contraction::matmul_2d;
    use std::collections::BTreeMap;

    /// Deterministic Hermitian n×n test matrix (H = (M + Mᴴ)/2, real diagonal).
    fn hermitian_test_matrix(n: usize) -> Vec<(f64, f64)> {
        let rnd = |i: usize, j: usize, salt: u64| -> f64 {
            let mut x = (i as u64)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add((j as u64).wrapping_mul(0x85eb_ca6b))
                .wrapping_add(salt)
                .wrapping_add(n as u64 * 0x1000_0001);
            x ^= x >> 16;
            x = x.wrapping_mul(0x7feb_352d);
            x ^= x >> 15;
            ((x % 2000) as f64) / 1000.0 - 1.0
        };
        let m: Vec<(f64, f64)> = (0..n * n)
            .map(|idx| (rnd(idx / n, idx % n, 1), rnd(idx / n, idx % n, 2)))
            .collect();
        let mut h = vec![(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                let a = m[i * n + j];
                let b = complex_conj(m[j * n + i]);
                h[i * n + j] = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
            }
        }
        for i in 0..n {
            h[i * n + i] = (h[i * n + i].0, 0.0);
        }
        h
    }

    #[test]
    fn complex_jacobi_diagonalizes_hermitian() {
        // Regression for frankenjax-complex-jacobi-wrong-eigenvalues-15mx4: the
        // complex Hermitian Jacobi must be a genuine unitary diagonalization, i.e.
        // Vᴴ H V = diag(w). Pre-fix this failed — e.g. n=2 returned [0.667,-0.0088]
        // for a matrix whose true eigenvalues are [0.767,-0.109] (trace preserved,
        // det/spectrum not). Phase-convention-independent on the eigenvectors.
        for &n in &[2usize, 3, 4, 8, 20, 33] {
            let h = hermitian_test_matrix(n);
            let mut a = h.clone();
            let (w, v) = complex_jacobi_eigendecomposition(&mut a, n);

            let mut hv = vec![(0.0_f64, 0.0_f64); n * n];
            for r in 0..n {
                for c in 0..n {
                    let mut acc = (0.0, 0.0);
                    for j in 0..n {
                        acc = complex_add(acc, complex_mul(h[r * n + j], v[j * n + c]));
                    }
                    hv[r * n + c] = acc;
                }
            }
            let mut off = 0.0_f64;
            let mut diag = 0.0_f64;
            for a2 in 0..n {
                for b in 0..n {
                    let mut acc = (0.0, 0.0);
                    for i in 0..n {
                        acc = complex_add(
                            acc,
                            complex_mul(complex_conj(v[i * n + a2]), hv[i * n + b]),
                        );
                    }
                    if a2 == b {
                        diag = diag.max((acc.0 - w[a2]).abs()).max(acc.1.abs());
                    } else {
                        off = off.max(complex_abs(acc));
                    }
                }
            }
            assert!(
                off < 1e-9 && diag < 1e-9,
                "n={n} not diagonalized: off={off:e} diag={diag:e}"
            );
        }
    }

    #[test]
    fn complex_jacobi_cyclic_matches_maxpivot() {
        // The row-cyclic kernel (production) must match the max-pivot reference
        // spectrum (both now correct → same true eigenvalues, sorted) and itself
        // diagonalize H. Isomorphism proof for routing complex eigh/SVD to cyclic.
        for &n in &[2usize, 3, 4, 8, 20, 33] {
            let h = hermitian_test_matrix(n);
            let (mut w_ref, _) = complex_jacobi_eigendecomposition(&mut h.clone(), n);
            let (w_cyc, v_cyc) = complex_jacobi_eigendecomposition_cyclic(&mut h.clone(), n);

            let mut w_cyc_s = w_cyc.clone();
            w_ref.sort_by(f64::total_cmp);
            w_cyc_s.sort_by(f64::total_cmp);
            for (r, c) in w_ref.iter().zip(w_cyc_s.iter()) {
                assert!((r - c).abs() < 1e-9, "n={n} spectrum {r} vs {c}");
            }

            // Cyclic must diagonalize: Vᴴ H V = diag(w_cyc).
            let mut hv = vec![(0.0_f64, 0.0_f64); n * n];
            for r in 0..n {
                for col in 0..n {
                    let mut acc = (0.0, 0.0);
                    for j in 0..n {
                        acc = complex_add(acc, complex_mul(h[r * n + j], v_cyc[j * n + col]));
                    }
                    hv[r * n + col] = acc;
                }
            }
            let mut off = 0.0_f64;
            for a2 in 0..n {
                for b in 0..n {
                    if a2 == b {
                        continue;
                    }
                    let mut acc = (0.0, 0.0);
                    for i in 0..n {
                        acc = complex_add(
                            acc,
                            complex_mul(complex_conj(v_cyc[i * n + a2]), hv[i * n + b]),
                        );
                    }
                    off = off.max(complex_abs(acc));
                }
            }
            assert!(off < 1e-9, "n={n} cyclic not diagonal: off={off:e}");
        }
    }

    #[test]
    #[ignore]
    fn complex_jacobi_cyclic_ab_timing() {
        use std::time::Instant;
        fn median_ms(mut v: Vec<f64>) -> f64 {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        }
        for &n in &[48usize, 96, 160] {
            let h = hermitian_test_matrix(n);
            let _ = complex_jacobi_eigendecomposition(&mut h.clone(), n);
            let _ = complex_jacobi_eigendecomposition_cyclic(&mut h.clone(), n);
            let (mut t_old, mut t_new) = (Vec::new(), Vec::new());
            for _ in 0..5 {
                let mut a = h.clone();
                let t = Instant::now();
                let _ = complex_jacobi_eigendecomposition(&mut a, n);
                t_old.push(t.elapsed().as_secs_f64() * 1e3);
                let mut a = h.clone();
                let t = Instant::now();
                let _ = complex_jacobi_eigendecomposition_cyclic(&mut a, n);
                t_new.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let (o, c) = (median_ms(t_old), median_ms(t_new));
            println!(
                "complex_jacobi n={n}: maxpivot {o:.3}ms | cyclic {c:.3}ms | speedup {:.2}x",
                o / c
            );
        }
    }

    /// Ground truth for frankenjax-eig-nonsymmetric-broken-n3-66pmy: build A = H·T·H
    /// with H a symmetric-orthogonal Householder and T block-diagonal — a 1×1 {2}
    /// plus a 2×2 rotation block {±3i}. A is a dense non-symmetric matrix similar to
    #[test]
    fn real_eig_cube_roots_of_unity_diag() {
        // Companion of z^3-1: eigenvalues are the cube roots of unity {1, e^{±2πi/3}},
        // all modulus 1. Real unshifted QR + 2x2-block extraction may fail to separate.
        let n = 3usize;
        let a = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (w, _) = eig_qr_iteration(&a, n);
        let want = [
            (1.0, 0.0),
            (-0.5, 0.8660254037844387),
            (-0.5, -0.8660254037844387),
        ];
        for &(er, ei) in &want {
            let f = w
                .iter()
                .any(|&(wr, wi)| (wr - er).abs() < 1e-5 && (wi - ei).abs() < 1e-5);
            assert!(f, "REAL EIG MISSING ({er},{ei}) in {w:?}");
        }
    }

    #[test]
    fn complex_eig_qr_handles_unit_circle_spectrum() {
        // Eigenvalues {1, i, -1, -i} all have modulus 1: the previous UNSHIFTED QR
        // never converged (subdiagonals shrink at rate ~1) and returned garbage
        // (≈{0.22−0.11i,…}). The Wilkinson shift breaks the degeneracy. A = H·T·H
        // with a complex Householder H, T = diag of the four eigenvalues.
        let n = 4usize;
        let eigs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        let mut t = vec![(0.0_f64, 0.0_f64); n * n];
        for i in 0..n {
            t[i * n + i] = eigs[i];
        }
        let v = [(1.0, 0.0), (0.0, 1.0), (1.0, -1.0), (1.0, 1.0)];
        let vhv: f64 = v.iter().map(|z| complex_abs(*z).powi(2)).sum();
        let mut h = vec![(0.0_f64, 0.0_f64); n * n];
        for r in 0..n {
            for c in 0..n {
                let id = if r == c { (1.0, 0.0) } else { (0.0, 0.0) };
                let vvh = complex_mul(v[r], complex_conj(v[c]));
                h[r * n + c] = complex_sub(id, (2.0 / vhv * vvh.0, 2.0 / vhv * vvh.1));
            }
        }
        let a = complex_matmul_n(&complex_matmul_n(&h, &t, n), &h, n);
        let (w, _) = complex_eig_qr(&a, n);
        for &(er, ei) in &eigs {
            let found = w
                .iter()
                .any(|&(wr, wi)| (wr - er).abs() < 1e-6 && (wi - ei).abs() < 1e-6);
            assert!(
                found,
                "missing eigenvalue ({er},{ei}) — QR did not converge"
            );
        }
    }

    #[test]
    fn complex_eig_qr_recovers_known_spectrum_n3() {
        // Frankenjax-eig-drops-complex-input-gx9mm: complex eig must use the full
        // complex matrix, not its real part. Build A = H·T·H with H a complex
        // Householder (unitary & Hermitian) and T = diag(2+i, −1+3i, −2i); A is a
        // dense complex non-Hermitian matrix similar to T, so eig must recover that
        // spectrum and satisfy A·v = λ·v.
        let n = 3usize;
        let eigs = [(2.0, 1.0), (-1.0, 3.0), (0.0, -2.0)];
        let mut t = vec![(0.0_f64, 0.0_f64); n * n];
        for i in 0..n {
            t[i * n + i] = eigs[i];
        }
        // v = [1, i, 1−i], vᴴv = 4, H = I − 0.5·v·vᴴ.
        let v = [(1.0, 0.0), (0.0, 1.0), (1.0, -1.0)];
        let mut h = vec![(0.0_f64, 0.0_f64); n * n];
        for r in 0..n {
            for c in 0..n {
                let id = if r == c { (1.0, 0.0) } else { (0.0, 0.0) };
                let vvh = complex_mul(v[r], complex_conj(v[c]));
                h[r * n + c] = complex_sub(id, (0.5 * vvh.0, 0.5 * vvh.1));
            }
        }
        let a = complex_matmul_n(&complex_matmul_n(&h, &t, n), &h, n);

        let (w, vecs) = complex_eig_qr(&a, n);
        for &(er, ei) in &eigs {
            assert!(
                w.iter()
                    .any(|&(wr, wi)| (wr - er).abs() < 1e-7 && (wi - ei).abs() < 1e-7),
                "missing eigenvalue ({er},{ei}) in {w:?}"
            );
        }
        for col in 0..n {
            for row in 0..n {
                let mut acc = (0.0_f64, 0.0_f64);
                for k in 0..n {
                    acc = complex_add(acc, complex_mul(a[row * n + k], vecs[k * n + col]));
                }
                let want = complex_mul(w[col], vecs[row * n + col]);
                assert!(
                    (acc.0 - want.0).abs() < 1e-7 && (acc.1 - want.1).abs() < 1e-7,
                    "A·v≠λ·v at ({row},{col}): {acc:?} vs {want:?}"
                );
            }
        }
    }

    /// Deterministic dense complex non-Hermitian matrix `A = H·T·Hᴴ` (H a complex
    /// Householder reflector, unitary; T diagonal with distinct complex eigenvalues),
    /// for exercising the complex Hessenberg eigenvector path at moderate n.
    fn complex_eig_test_matrix(n: usize) -> Vec<(f64, f64)> {
        let mut t = vec![(0.0_f64, 0.0_f64); n * n];
        for i in 0..n {
            let re = (i as f64) * 1.3 - 2.0;
            let im = (((i * 37 + 5) % 11) as f64) - 5.0;
            t[i * n + i] = (re, im);
        }
        let mut v = vec![(0.0_f64, 0.0_f64); n];
        for (r, slot) in v.iter_mut().enumerate() {
            *slot = (
                (((r * 13 + 3) % 17) as f64) - 8.0,
                (((r * 29 + 7) % 13) as f64) - 6.0,
            );
        }
        let vhv: f64 = v.iter().map(|z| complex_abs(*z).powi(2)).sum();
        let mut h = vec![(0.0_f64, 0.0_f64); n * n];
        for r in 0..n {
            for c in 0..n {
                let id = if r == c { (1.0, 0.0) } else { (0.0, 0.0) };
                let vvh = complex_mul(v[r], complex_conj(v[c]));
                h[r * n + c] = complex_sub(id, (2.0 / vhv * vvh.0, 2.0 / vhv * vvh.1));
            }
        }
        complex_matmul_n(&complex_matmul_n(&h, &t, n), &h, n)
    }

    #[test]
    fn complex_hessenberg_reduction_is_unitary_similarity() {
        // H must be upper-Hessenberg and A = Q0·H·Q0ᴴ with Q0 unitary.
        for &n in &[4usize, 6, 8] {
            let a = complex_eig_test_matrix(n);
            let (h, q0) = complex_hessenberg_reduction(&a, n);
            // (1) H upper-Hessenberg: entries below the first subdiagonal are zero.
            for row in 2..n {
                for col in 0..row - 1 {
                    assert!(
                        complex_abs(h[row * n + col]) < 1e-9,
                        "n={n}: H[{row}][{col}] not zero"
                    );
                }
            }
            // (2) Q0 unitary: Q0ᴴ·Q0 = I.
            for i in 0..n {
                for j in 0..n {
                    let mut acc = (0.0_f64, 0.0_f64);
                    for k in 0..n {
                        acc = complex_add(
                            acc,
                            complex_mul(complex_conj(q0[k * n + i]), q0[k * n + j]),
                        );
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (acc.0 - want).abs() < 1e-9 && acc.1.abs() < 1e-9,
                        "n={n}: Q0ᴴQ0[{i}][{j}]={acc:?}"
                    );
                }
            }
            // (3) Reconstruction Q0·H·Q0ᴴ = A.
            let q0h: Vec<(f64, f64)> = {
                let mut out = vec![(0.0_f64, 0.0_f64); n * n];
                for i in 0..n {
                    for j in 0..n {
                        out[i * n + j] = complex_conj(q0[j * n + i]);
                    }
                }
                out
            };
            let recon = complex_matmul_n(&complex_matmul_n(&q0, &h, n), &q0h, n);
            for idx in 0..n * n {
                assert!(
                    (recon[idx].0 - a[idx].0).abs() < 1e-7
                        && (recon[idx].1 - a[idx].1).abs() < 1e-7,
                    "n={n}: reconstruction mismatch at {idx}: {:?} vs {:?}",
                    recon[idx],
                    a[idx]
                );
            }
        }
    }

    #[test]
    fn complex_eig_eigenvector_hessenberg_matches_full_reference() {
        // The O(n²) complex Hessenberg eigenvector path must (a) satisfy A·v = λ·v and
        // (b) span the same direction as the O(n³) full-matrix oracle
        // `complex_eig_eigenvector` (|⟨v_hess, v_full⟩| ≈ 1, up to scale/phase).
        for &n in &[4usize, 6, 8] {
            let a = complex_eig_test_matrix(n);
            let (w, v) = complex_eig_qr(&a, n);
            // (a) public reconstruction contract on the production output.
            for col in 0..n {
                for row in 0..n {
                    let mut acc = (0.0_f64, 0.0_f64);
                    for k in 0..n {
                        acc = complex_add(acc, complex_mul(a[row * n + k], v[k * n + col]));
                    }
                    let want = complex_mul(w[col], v[row * n + col]);
                    assert!(
                        complex_abs(complex_sub(acc, want)) < 1e-7,
                        "n={n}: A·v≠λ·v at ({row},{col})"
                    );
                }
            }
            // (b) direction match vs the full-matrix oracle.
            for (col, &lambda) in w.iter().enumerate() {
                let v_full = complex_eig_eigenvector(&a, n, lambda);
                let mut dot = (0.0_f64, 0.0_f64);
                for k in 0..n {
                    dot = complex_add(dot, complex_mul(v[k * n + col], complex_conj(v_full[k])));
                }
                assert!(
                    (complex_abs(dot) - 1.0).abs() < 1e-5,
                    "n={n} col={col}: |⟨v_hess,v_full⟩|={}",
                    complex_abs(dot)
                );
            }
        }
    }

    #[test]
    fn complex_qr_mgs_is_valid_factorization() {
        // Oracle check on the retained reference QR: Q unitary and Q·R = A.
        let n = 5usize;
        let a = complex_eig_test_matrix(n);
        let (q, r) = complex_qr_mgs(&a, n);
        // Q unitary: QᴴQ = I.
        for i in 0..n {
            for j in 0..n {
                let mut acc = (0.0_f64, 0.0_f64);
                for k in 0..n {
                    acc = complex_add(acc, complex_mul(complex_conj(q[k * n + i]), q[k * n + j]));
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc.0 - want).abs() < 1e-9 && acc.1.abs() < 1e-9,
                    "QᴴQ[{i}][{j}]={acc:?}"
                );
            }
        }
        // R upper-triangular and Q·R = A.
        for row in 1..n {
            for col in 0..row {
                assert!(
                    complex_abs(r[row * n + col]) < 1e-12,
                    "R[{row}][{col}] not zero"
                );
            }
        }
        let recon = complex_matmul_n(&q, &r, n);
        for idx in 0..n * n {
            assert!(
                complex_abs(complex_sub(recon[idx], a[idx])) < 1e-9,
                "Q·R ≠ A at {idx}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_complex_eig_eigenvectors_hessenberg_vs_full() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let run = |n: usize| {
            let a = complex_eig_test_matrix(n);
            let (w, _v) = complex_eig_qr(&a, n);
            let full = best_time(|| {
                for &lambda in &w {
                    std::hint::black_box(complex_eig_eigenvector(&a, n, lambda));
                }
            });
            let (h, q0) = complex_hessenberg_reduction(&a, n);
            let hess = best_time(|| {
                for &lambda in &w {
                    std::hint::black_box(complex_eig_eigenvector_hessenberg(&h, &q0, n, lambda));
                }
            });
            println!(
                "BENCH complex eig eigenvectors n={n}: full-LU O(n⁴) {:.3}ms -> Hessenberg O(n³) {:.3}ms = {:.2}x",
                full * 1e3,
                hess * 1e3,
                full / hess
            );
        };
        run(48);
        run(96);
        run(160);
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_complex_eig_qr_iteration_hessenberg_vs_mgs() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // Wilkinson/exceptional shift for the trailing 2×2 of the active leading p×p
        // (identical to complex_eig_qr's; the only thing the two QR steps share).
        let shift = |t: &[(f64, f64)], n: usize, p: usize, since_deflate: usize| -> (f64, f64) {
            let aa = t[(p - 2) * n + (p - 2)];
            let bb = t[(p - 2) * n + (p - 1)];
            let cc = t[(p - 1) * n + (p - 2)];
            let dd = t[(p - 1) * n + (p - 1)];
            if since_deflate.is_multiple_of(10) {
                let s = complex_abs(t[(p - 1) * n + (p - 2)])
                    + if p >= 3 {
                        complex_abs(t[(p - 2) * n + (p - 3)])
                    } else {
                        0.0
                    };
                let s = if s > 0.0 { s } else { 1.0 };
                complex_add(dd, (0.75 * s, 0.31 * s))
            } else {
                let mid = (0.5 * (aa.0 + dd.0), 0.5 * (aa.1 + dd.1));
                let hd = (0.5 * (aa.0 - dd.0), 0.5 * (aa.1 - dd.1));
                let disc = complex_add(complex_mul(hd, hd), complex_mul(bb, cc));
                let sq = complex_sqrt(disc);
                let m1 = complex_add(mid, sq);
                let m2 = complex_sub(mid, sq);
                if complex_abs(complex_sub(m1, dd)) <= complex_abs(complex_sub(m2, dd)) {
                    m1
                } else {
                    m2
                }
            }
        };
        let run = |n: usize| {
            let a = complex_eig_test_matrix(n);
            let (h0, _q0) = complex_hessenberg_reduction(&a, n);
            // (a) OLD per-sweep: copy active block, full MGS QR + dense RQ matmul (O(p³)).
            let mgs = best_time(|| {
                let mut t = h0.clone();
                let (mut p, mut iters, mut since) = (n, 0usize, 0usize);
                while p > 1 {
                    let sub = complex_abs(t[(p - 1) * n + (p - 2)]);
                    let ds = complex_abs(t[(p - 2) * n + (p - 2)])
                        + complex_abs(t[(p - 1) * n + (p - 1)]);
                    if sub <= f64::EPSILON * ds.max(f64::MIN_POSITIVE) {
                        t[(p - 1) * n + (p - 2)] = (0.0, 0.0);
                        p -= 1;
                        since = 0;
                        continue;
                    }
                    if iters >= 100 * n + 100 {
                        break;
                    }
                    iters += 1;
                    since += 1;
                    let mu = shift(&t, n, p, since);
                    let mut b = vec![(0.0_f64, 0.0_f64); p * p];
                    for i in 0..p {
                        for j in 0..p {
                            b[i * p + j] = t[i * n + j];
                        }
                    }
                    for i in 0..p {
                        b[i * p + i] = complex_sub(b[i * p + i], mu);
                    }
                    let (q, r) = complex_qr_mgs(&b, p);
                    let mut rq = complex_matmul_n(&r, &q, p);
                    for i in 0..p {
                        rq[i * p + i] = complex_add(rq[i * p + i], mu);
                    }
                    for i in 0..p {
                        for j in 0..p {
                            t[i * n + j] = rq[i * p + j];
                        }
                    }
                }
                std::hint::black_box(&t);
            });
            // (b) NEW per-sweep: O(p²) Hessenberg complex-Givens step.
            let hess = best_time(|| {
                let mut t = h0.clone();
                let (mut p, mut iters, mut since) = (n, 0usize, 0usize);
                while p > 1 {
                    let sub = complex_abs(t[(p - 1) * n + (p - 2)]);
                    let ds = complex_abs(t[(p - 2) * n + (p - 2)])
                        + complex_abs(t[(p - 1) * n + (p - 1)]);
                    if sub <= f64::EPSILON * ds.max(f64::MIN_POSITIVE) {
                        t[(p - 1) * n + (p - 2)] = (0.0, 0.0);
                        p -= 1;
                        since = 0;
                        continue;
                    }
                    if iters >= 100 * n + 100 {
                        break;
                    }
                    iters += 1;
                    since += 1;
                    let mu = shift(&t, n, p, since);
                    complex_hessenberg_shifted_qr_step(&mut t, n, p, mu);
                }
                std::hint::black_box(&t);
            });
            println!(
                "BENCH complex eig QR-iter n={n}: MGS O(p³)/sweep {:.3}ms -> Hessenberg-Givens O(p²) {:.3}ms = {:.2}x",
                mgs * 1e3,
                hess * 1e3,
                mgs / hess
            );
        };
        run(48);
        run(96);
        run(160);
    }

    /// T, so eig must recover the spectrum {2, 3i, -3i}. The pre-fix kernel read only
    /// the diagonal and returned ≈{2,0,0}, dropping the complex pair.
    #[test]
    fn eig_qr_recovers_complex_eigenvalues_n3() {
        let n = 3;
        let t = [2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 3.0, 0.0];
        let v = [1.0, 2.0, -1.0];
        let vtv = 6.0_f64;
        let mut h = [0.0_f64; 9];
        for r in 0..n {
            for c in 0..n {
                let id = if r == c { 1.0 } else { 0.0 };
                h[r * n + c] = id - 2.0 / vtv * v[r] * v[c];
            }
        }
        let matmul3 = |x: &[f64; 9], y: &[f64; 9]| -> [f64; 9] {
            let mut out = [0.0_f64; 9];
            for r in 0..n {
                for c in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += x[r * n + k] * y[k * n + c];
                    }
                    out[r * n + c] = s;
                }
            }
            out
        };
        let a = matmul3(&matmul3(&h, &t), &h);
        let (w, v) = eig_qr_iteration(&a, n);
        for &(er, ei) in &[(2.0, 0.0), (0.0, 3.0), (0.0, -3.0)] {
            let found = w
                .iter()
                .any(|&(wr, wi)| (wr - er).abs() < 1e-7 && (wi - ei).abs() < 1e-7);
            assert!(found, "missing eigenvalue ({er},{ei}) in {w:?}");
        }
        // Eigenvector residual A·v[:,k] = w[k]·v[:,k] (v column-major in row-major).
        assert_eig_residual_complex(&a, n, &w, &v, 1e-7);
    }

    /// A·v[:,k] − w[k]·v[:,k] ≈ 0 for each column k, with `v[row*n+col]` the
    /// eigenvector for `w[col]` (the layout eval_eig emits). Mirrors the conformance
    /// `assert_eig_residual` but local to fj-lax's `(re,im)` tuples.
    fn assert_eig_residual_complex(
        a: &[f64],
        n: usize,
        w: &[(f64, f64)],
        v: &[(f64, f64)],
        tol: f64,
    ) {
        for col in 0..n {
            let norm: f64 = (0..n)
                .map(|row| complex_abs(v[row * n + col]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(norm > 1e-12, "eigenvector column {col} is zero");
            for row in 0..n {
                let mut av = (0.0_f64, 0.0_f64);
                for k in 0..n {
                    av = complex_add(av, complex_mul((a[row * n + k], 0.0), v[k * n + col]));
                }
                let lv = complex_mul(w[col], v[row * n + col]);
                let res = complex_abs(complex_sub(av, lv));
                assert!(
                    res <= tol,
                    "residual {res} at ({row},{col}) > {tol}; w={:?}",
                    w[col]
                );
            }
        }
    }

    #[test]
    fn eig_qr_eigenvector_residual_n4_mixed() {
        // 4×4 with a real eigenvalue, a repeated real, and a complex pair: T =
        // diag-blocks {5} {-1} and rotation {1±2i}; A = H·T·H with a 4-vector
        // Householder. Checks both eigenvalues and A·v = λ·v residual.
        let n = 4;
        let t = [
            5.0, 0.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, -2.0, //
            0.0, 0.0, 2.0, 1.0,
        ];
        let vv = [1.0_f64, -2.0, 1.0, 3.0];
        let vtv: f64 = vv.iter().map(|x| x * x).sum();
        let mut h = [0.0_f64; 16];
        for r in 0..n {
            for c in 0..n {
                let id = if r == c { 1.0 } else { 0.0 };
                h[r * n + c] = id - 2.0 / vtv * vv[r] * vv[c];
            }
        }
        let matmul4 = |x: &[f64; 16], y: &[f64; 16]| -> [f64; 16] {
            let mut out = [0.0_f64; 16];
            for r in 0..n {
                for c in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += x[r * n + k] * y[k * n + c];
                    }
                    out[r * n + c] = s;
                }
            }
            out
        };
        let a = matmul4(&matmul4(&h, &t), &h);
        let (w, v) = eig_qr_iteration(&a, n);
        for &(er, ei) in &[(5.0, 0.0), (-1.0, 0.0), (1.0, 2.0), (1.0, -2.0)] {
            assert!(
                w.iter()
                    .any(|&(wr, wi)| (wr - er).abs() < 1e-6 && (wi - ei).abs() < 1e-6),
                "missing eigenvalue ({er},{ei}) in {w:?}"
            );
        }
        assert_eig_residual_complex(&a, n, &w, &v, 1e-6);
    }

    /// Deterministic non-symmetric n×n matrix with a well-spread real+complex
    /// spectrum (block-diagonal T similarity-transformed by a dense orthogonal-ish
    /// matrix), used to exercise the Hessenberg eigenvector path at larger n.
    fn eig_test_matrix(n: usize) -> Vec<f64> {
        // T: alternating isolated real eigenvalues and 2×2 complex-pair blocks.
        let mut t = vec![0.0f64; n * n];
        let mut i = 0;
        let mut tag = 1.0f64;
        while i < n {
            if i + 1 < n && i % 3 == 1 {
                // 2×2 rotation block -> complex pair tag ± (tag/2)i.
                t[i * n + i] = tag;
                t[i * n + i + 1] = -(tag / 2.0);
                t[(i + 1) * n + i] = tag / 2.0;
                t[(i + 1) * n + i + 1] = tag;
                i += 2;
            } else {
                t[i * n + i] = tag + 0.7;
                i += 1;
            }
            tag += 1.3;
        }
        // Dense similarity S T S⁻¹ with S = I + small structured perturbation, kept
        // well-conditioned. Use a Householder H (orthogonal, H⁻¹=H) so eigenvalues
        // are preserved exactly: A = H T H.
        let mut vv = vec![0.0f64; n];
        for (r, slot) in vv.iter_mut().enumerate() {
            *slot = ((r * 37 + 11) % 23) as f64 - 11.0;
        }
        let vtv: f64 = vv.iter().map(|x| x * x).sum();
        let mut h = vec![0.0f64; n * n];
        for r in 0..n {
            for c in 0..n {
                let id = if r == c { 1.0 } else { 0.0 };
                h[r * n + c] = id - 2.0 / vtv * vv[r] * vv[c];
            }
        }
        let matmul = |x: &[f64], y: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0f64; n * n];
            for r in 0..n {
                for c in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += x[r * n + k] * y[k * n + c];
                    }
                    out[r * n + c] = s;
                }
            }
            out
        };
        matmul(&matmul(&h, &t), &h)
    }

    #[test]
    fn eig_hessenberg_eigenvector_matches_full_reference() {
        // The O(n²) Hessenberg inverse-iteration eigenvector path must produce
        // eigenvectors that (a) satisfy the public reconstruction contract A·v=λ·v
        // and (b) span the SAME direction as the O(n³) full-matrix reference
        // `eig_eigenvector` (|⟨v_hess, v_full⟩| ≈ ‖·‖² — equal up to scale/phase).
        for &n in &[5usize, 7, 8] {
            let a = eig_test_matrix(n);
            let (w, v) = eig_qr_iteration(&a, n);
            // (a) public contract on the production output.
            assert_eig_residual_complex(&a, n, &w, &v, 1e-8);
            // (b) direction match vs the full-matrix oracle, per eigenvalue.
            for (col, &lambda) in w.iter().enumerate() {
                let v_full = eig_eigenvector(&a, n, lambda);
                let v_hess: Vec<(f64, f64)> = (0..n).map(|r| v[r * n + col]).collect();
                // |⟨v_hess, conj(v_full)⟩| should equal 1 (both unit, same direction).
                let mut dot = (0.0f64, 0.0f64);
                for k in 0..n {
                    dot = complex_add(dot, complex_mul(v_hess[k], (v_full[k].0, -v_full[k].1)));
                }
                assert!(
                    (complex_abs(dot) - 1.0).abs() < 1e-6,
                    "n={n} col={col}: |⟨v_hess,v_full⟩|={} (λ={lambda:?})",
                    complex_abs(dot)
                );
            }
        }
    }

    #[test]
    fn francis_qr_converges_without_fallback_and_reconstructs() {
        // The Francis implicit double-shift drives even complex-conjugate-pair
        // spectra (which STALLED the old unshifted QR → O(n⁴) complex_eig_qr
        // fallback) to quasi-triangular form in O(n) sweeps. Verify at larger n:
        // (a) francis_qr_schur converges (returns true → no fallback) and leaves a
        //     genuinely quasi-triangular matrix; (b) the production eig output
        //     reconstructs A·v = λ·v to tolerance and recovers the known spectrum.
        let neg = |t: &[f64], n: usize, j: usize| -> bool {
            let scale = t[(j - 1) * n + j - 1].abs() + t[j * n + j].abs();
            t[j * n + j - 1].abs() <= f64::EPSILON * scale.max(f64::MIN_POSITIVE)
        };
        for &n in &[16usize, 32, 48, 64] {
            let a = eig_test_matrix(n);
            // (a) Francis converges on the Hessenberg form, no fallback.
            let (mut h, _q) = hessenberg_reduction(&a, n);
            assert!(
                francis_qr_schur(&mut h, n),
                "n={n}: Francis QR did not converge (would force the O(n⁴) fallback)"
            );
            assert!(
                (2..n).all(|j| neg(&h, n, j) || neg(&h, n, j - 1)),
                "n={n}: Francis result is not quasi-triangular"
            );
            // (b) Full production path: reconstruction + spectrum.
            let (w, v) = eig_qr_iteration(&a, n);
            assert_eq!(w.len(), n, "n={n}: expected {n} eigenvalues");
            assert_eig_residual_complex(&a, n, &w, &v, 1e-6);
            // eig_test_matrix's spectrum: isolated reals `tag+0.7` and complex pairs
            // `tag ± (tag/2)i`. Just check every eigenvalue is finite and the trace
            // (Σλ) matches tr(A) — a similarity invariant independent of ordering.
            let trace_a: f64 = (0..n).map(|i| a[i * n + i]).sum();
            let trace_w: f64 = w.iter().map(|&(re, _)| re).sum();
            assert!(
                (trace_a - trace_w).abs() <= 1e-6 * trace_a.abs().max(1.0),
                "n={n}: Σλ {trace_w} != tr(A) {trace_a}"
            );
            assert!(
                w.iter().all(|&(re, im)| re.is_finite() && im.is_finite()),
                "n={n}: non-finite eigenvalue"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_eig_eigenvectors_hessenberg_vs_full() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let run = |n: usize| {
            let a = eig_test_matrix(n);
            let (w, _v) = eig_qr_iteration(&a, n);
            // Reproduce the production Hessenberg factor + Q0.
            let (mut t, q0) = hessenberg_reduction(&a, n);
            // (t is the Hessenberg form here since we don't QR-iterate it in the bench)
            let mut h_hess = t.clone();
            for row in 2..n {
                for col in 0..row - 1 {
                    h_hess[row * n + col] = 0.0;
                }
            }
            let _ = &mut t;
            // Eigenvector phase only (the O(n⁴) vs O(n³) difference).
            let full = best_time(|| {
                for &lambda in &w {
                    std::hint::black_box(eig_eigenvector(&a, n, lambda));
                }
            });
            let hess = best_time(|| {
                for &lambda in &w {
                    std::hint::black_box(eig_eigenvector_hessenberg(&h_hess, &q0, n, lambda));
                }
            });
            println!(
                "BENCH eig eigenvectors n={n}: full-LU O(n^4) {:.3}ms -> Hessenberg O(n^3) {:.3}ms = {:.2}x",
                full * 1e3,
                hess * 1e3,
                full / hess
            );
        };
        run(64);
        run(128);
        run(192);
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_eig_qr_iteration_window_vs_full_sweep() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let neg = |t: &[f64], n: usize, j: usize| -> bool {
            let scale = t[(j - 1) * n + j - 1].abs() + t[j * n + j].abs();
            t[j * n + j - 1].abs() <= f64::EPSILON * scale.max(f64::MIN_POSITIVE)
        };
        let run = |n: usize| {
            let a = eig_test_matrix(n);
            let (h0, _q0) = hessenberg_reduction(&a, n);
            // (a) OLD full-sweep loop: sweep the whole n×n each iteration.
            let full = best_time(|| {
                let mut t = h0.clone();
                for _ in 0..(200 * n) {
                    hessenberg_qr_step(&mut t, None, n);
                    if (2..n).all(|j| neg(&t, n, j) || neg(&t, n, j - 1)) {
                        break;
                    }
                }
                std::hint::black_box(&t);
            });
            // (b) NEW deflation-window loop: sweep only the active leading p×p.
            let window = best_time(|| {
                let mut t = h0.clone();
                let mut p = n;
                let mut iters = 0usize;
                while p > 2 {
                    let prev_p = p;
                    loop {
                        if p > 1 && neg(&t, n, p - 1) {
                            t[(p - 1) * n + (p - 2)] = 0.0;
                            p -= 1;
                        } else if p > 2 && neg(&t, n, p - 2) {
                            t[(p - 2) * n + (p - 3)] = 0.0;
                            p -= 2;
                        } else {
                            break;
                        }
                    }
                    if p <= 2 {
                        break;
                    }
                    if prev_p == p && iters >= 200 * n {
                        break;
                    }
                    hessenberg_qr_step_leading(&mut t, n, p);
                    iters += 1;
                }
                std::hint::black_box(&t);
            });
            // (c) Francis implicit double-shift QR: production candidate.
            let francis = best_time(|| {
                let mut t = h0.clone();
                std::hint::black_box(francis_qr_schur(&mut t, n));
                std::hint::black_box(&t);
            });
            println!(
                "BENCH eig QR-iter n={n}: full-sweep O(n²)/it {:.3}ms -> deflation-window {:.3}ms = {:.2}x -> Francis {:.3}ms = {:.2}x vs window",
                full * 1e3,
                window * 1e3,
                full / window,
                francis * 1e3,
                window / francis
            );
        };
        // Sizes where unshifted QR converges within the cap (the regime the real
        // path serves — larger/clustered matrices that stall fall back to
        // complex_eig_qr, where windowing is neutral and never regresses).
        run(32);
        run(48);
        run(64);
    }

    fn make_matrix(m: usize, n: usize, data: &[f64]) -> Value {
        let elements: Vec<Literal> = data.iter().map(|&v| Literal::from_f64(v)).collect();
        let shape = Shape {
            dims: vec![m as u32, n as u32],
        };
        Value::Tensor(TensorValue::new(DType::F64, shape, elements).unwrap())
    }

    /// Informational head-to-head timing of fj-lax dense f64 linalg vs JAX/LAPACK
    /// (compare against /data/projects/.scratch/jax_linalg_headtohead.py). Run with
    /// `--ignored --nocapture`. Pure measurement — no asserts on timing.
    #[test]
    #[ignore = "informational head-to-head timing; run with --ignored --nocapture"]
    fn bench_linalg_vs_jax_sizes() {
        use std::collections::BTreeMap;
        use std::time::Instant;
        let p = BTreeMap::new();
        let best = |f: &dyn Fn()| -> f64 {
            let mut b = f64::MAX;
            for _ in 0..3 {
                let t0 = Instant::now();
                f();
                b = b.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            b
        };
        for &n in &[128usize, 256, 512] {
            // Deterministic pseudo-random A, symmetric S = A+Aᵀ, SPD = S + 4n·I.
            let a: Vec<f64> = (0..n * n)
                .map(|i| (i as f64 * 0.123).sin() + (i as f64 * 0.0457).cos())
                .collect();
            let mut sym = vec![0.0f64; n * n];
            for r in 0..n {
                for c in 0..n {
                    sym[r * n + c] = a[r * n + c] + a[c * n + r];
                }
            }
            let mut spd = sym.clone();
            for d in 0..n {
                spd[d * n + d] += 4.0 * n as f64;
            }
            let am = make_matrix(n, n, &a);
            let symm = make_matrix(n, n, &sym);
            let spdm = make_matrix(n, n, &spd);
            let run = |prim: Primitive, m: &Value| -> f64 {
                best(&|| {
                    let _ = crate::eval_primitive(prim, std::slice::from_ref(m), &p).unwrap();
                })
            };
            eprintln!(
                "[linalg n={n}] svd={:.2}ms eigh={:.2}ms qr={:.2}ms cholesky={:.2}ms",
                run(Primitive::Svd, &am),
                run(Primitive::Eigh, &symm),
                run(Primitive::Qr, &am),
                run(Primitive::Cholesky, &spdm),
            );
        }
    }

    /// Same-binary A/B for the apply_householder_left cache-layout fix (ur4h3): the OLD
    /// column-strided left-apply vs the NEW row-contiguous one, inside a full 512×512
    /// Hessenberg reduction. Asserts H and Q are bit-identical. Run `--ignored --nocapture`.
    #[test]
    #[ignore = "informational same-binary A/B; run with --ignored --nocapture"]
    fn bench_householder_left_cache_layout_ab() {
        // The pre-fix column-strided left-apply, kept here as the A/B baseline only.
        fn apply_left_strided(
            matrix: &mut [f64],
            n: usize,
            row_start: usize,
            col_start: usize,
            v: &[f64],
            beta: f64,
        ) {
            for col in col_start..n {
                let mut dot = 0.0;
                for (offset, &v_i) in v.iter().enumerate() {
                    dot += v_i * matrix[(row_start + offset) * n + col];
                }
                let scale = beta * dot;
                for (offset, &v_i) in v.iter().enumerate() {
                    matrix[(row_start + offset) * n + col] -= scale * v_i;
                }
            }
        }
        // Full Hessenberg reduction with a selectable left-apply (right-applies unchanged).
        fn reduce(a: &[f64], n: usize, strided: bool) -> (Vec<f64>, Vec<f64>) {
            let mut h = a.to_vec();
            let mut q = vec![0.0_f64; n * n];
            for i in 0..n {
                q[i * n + i] = 1.0;
            }
            if n < 3 {
                return (h, q);
            }
            for k in 0..n - 2 {
                let start = k + 1;
                let len = n - start;
                let mut norm = 0.0;
                for row in start..n {
                    let val = h[row * n + k];
                    norm += val * val;
                }
                norm = norm.sqrt();
                if norm <= 1e-15 {
                    continue;
                }
                let mut v = vec![0.0; len];
                for row in start..n {
                    v[row - start] = h[row * n + k];
                }
                if v[0] >= 0.0 {
                    v[0] += norm;
                } else {
                    v[0] -= norm;
                }
                let mut vnsq = 0.0;
                for x in &v {
                    vnsq += x * x;
                }
                if vnsq <= 1e-30 {
                    continue;
                }
                let beta = 2.0 / vnsq;
                if strided {
                    apply_left_strided(&mut h, n, start, k, &v, beta);
                } else {
                    apply_householder_left(&mut h, n, start, k, &v, beta);
                }
                apply_householder_right(&mut h, n, 0, start, &v, beta);
                apply_householder_right(&mut q, n, 0, start, &v, beta);
                for row in start + 1..n {
                    h[row * n + k] = 0.0;
                }
            }
            (h, q)
        }
        let n = 512usize;
        let a: Vec<f64> = (0..n * n)
            .map(|i| (i as f64 * 0.123).sin() + (i as f64 * 0.0457).cos())
            .collect();
        let mut sym = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                sym[r * n + c] = a[r * n + c] + a[c * n + r];
            }
        }
        let best = |strided: bool| -> f64 {
            let mut b = f64::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                let r = reduce(&sym, n, strided);
                std::hint::black_box(&r);
                b = b.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            b
        };
        let old = best(true);
        let new = best(false);
        let (ho, qo) = reduce(&sym, n, true);
        let (hn, qn) = reduce(&sym, n, false);
        assert!(
            ho.iter().zip(&hn).all(|(a, b)| a.to_bits() == b.to_bits()),
            "H must be bit-identical between strided and contiguous left-apply"
        );
        assert!(
            qo.iter().zip(&qn).all(|(a, b)| a.to_bits() == b.to_bits()),
            "Q must be bit-identical between strided and contiguous left-apply"
        );
        eprintln!(
            "[hessenberg {n}x{n}] strided-left={old:.1}ms contiguous-left={new:.1}ms ratio={:.2}x",
            old / new
        );
    }

    /// Same-binary A/B for the QL eigenvector-accumulation cache fix (ur4h3 step 2): the
    /// OLD row-major (column-strided) rotation vs the NEW column-major (contiguous) one,
    /// on the tridiagonal+vectors of a real 512×512 reduction. Asserts eigenvalues AND
    /// eigenvectors are bit-identical. Run `--ignored --nocapture`.
    #[test]
    #[ignore = "informational same-binary A/B; run with --ignored --nocapture"]
    fn bench_ql_eigvec_accumulation_layout_ab() {
        // Pre-fix row-major QL (column-strided eigenvector rotation), A/B baseline only.
        fn ql_rowmajor(d: &mut [f64], e: &mut [f64], z: &mut [f64], n: usize) {
            if n <= 1 {
                return;
            }
            let eps = f64::EPSILON;
            for l in 0..n {
                let mut iter = 0usize;
                loop {
                    let mut m = l;
                    while m + 1 < n {
                        let dd = d[m].abs() + d[m + 1].abs();
                        if e[m].abs() <= eps * dd {
                            break;
                        }
                        m += 1;
                    }
                    if m == l {
                        break;
                    }
                    iter += 1;
                    if iter > 50 {
                        break;
                    }
                    let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                    let mut r = g.hypot(1.0);
                    let sgn = if g >= 0.0 { r.abs() } else { -r.abs() };
                    g = d[m] - d[l] + e[l] / (g + sgn);
                    let mut s = 1.0_f64;
                    let mut c = 1.0_f64;
                    let mut p = 0.0_f64;
                    let mut zeroed = false;
                    let mut i = m;
                    while i > l {
                        i -= 1;
                        let mut f = s * e[i];
                        let b = c * e[i];
                        r = f.hypot(g);
                        e[i + 1] = r;
                        if r == 0.0 {
                            d[i + 1] -= p;
                            e[m] = 0.0;
                            zeroed = true;
                            break;
                        }
                        s = f / r;
                        c = g / r;
                        g = d[i + 1] - p;
                        r = (d[i] - g) * s + 2.0 * c * b;
                        p = s * r;
                        d[i + 1] = g + p;
                        g = c * r - b;
                        for k in 0..n {
                            f = z[k * n + (i + 1)];
                            z[k * n + (i + 1)] = s * z[k * n + i] + c * f;
                            z[k * n + i] = c * z[k * n + i] - s * f;
                        }
                    }
                    if zeroed {
                        continue;
                    }
                    d[l] -= p;
                    e[l] = g;
                    e[m] = 0.0;
                }
            }
        }
        let n = 512usize;
        let a: Vec<f64> = (0..n * n)
            .map(|i| (i as f64 * 0.137).sin() + (i as f64 * 0.0411).cos())
            .collect();
        let mut sym = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                sym[r * n + c] = a[r * n + c] + a[c * n + r];
            }
        }
        let (h, q) = hessenberg_reduction(&sym, n);
        let mut d0 = vec![0.0; n];
        let mut e0 = vec![0.0; n];
        for i in 0..n {
            d0[i] = h[i * n + i];
            e0[i] = if i + 1 < n { h[(i + 1) * n + i] } else { 0.0 };
        }
        // OLD: row-major QL directly on q.
        let best = |run: &dyn Fn() -> (Vec<f64>, Vec<f64>)| -> (f64, (Vec<f64>, Vec<f64>)) {
            let mut b = f64::MAX;
            let mut last = (vec![], vec![]);
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                last = run();
                b = b.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            (b, last)
        };
        let (old_ms, (do_, zo)) = best(&|| {
            let (mut d, mut e, mut z) = (d0.clone(), e0.clone(), q.clone());
            ql_rowmajor(&mut d, &mut e, &mut z, n);
            (d, z)
        });
        // NEW: column-major QL (transpose in/out), matching production.
        let (new_ms, (dn, zn)) = best(&|| {
            let (mut d, mut e) = (d0.clone(), e0.clone());
            let mut zc = vec![0.0; n * n];
            for row in 0..n {
                for col in 0..n {
                    zc[col * n + row] = q[row * n + col];
                }
            }
            symmetric_tridiagonal_ql(&mut d, &mut e, &mut zc, n);
            let mut z = vec![0.0; n * n];
            for row in 0..n {
                for col in 0..n {
                    z[row * n + col] = zc[col * n + row];
                }
            }
            (d, z)
        });
        assert!(
            do_.iter().zip(&dn).all(|(a, b)| a.to_bits() == b.to_bits()),
            "QL eigenvalues must be bit-identical (row-major vs column-major)"
        );
        assert!(
            zo.iter().zip(&zn).all(|(a, b)| a.to_bits() == b.to_bits()),
            "QL eigenvectors must be bit-identical (row-major vs column-major)"
        );
        eprintln!(
            "[QL eigvec-accum {n}x{n}] row-major-strided={old_ms:.1}ms column-major-contig={new_ms:.1}ms ratio={:.2}x",
            old_ms / new_ms
        );
    }

    fn extract_f64_elements(v: &Value) -> Vec<f64> {
        match v {
            Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
            Value::Scalar(l) => vec![l.as_f64().unwrap()],
        }
    }

    fn assert_real_qr_matches_complex_zero_imag(n: usize, a: &[f64]) {
        let real_input = make_matrix(n, n, a);
        let complex_input = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                a.iter()
                    .map(|&v| Literal::from_complex128(v, 0.0))
                    .collect(),
            )
            .unwrap(),
        );

        let real_result = eval_qr(&[real_input], &BTreeMap::new()).unwrap();
        let complex_result = eval_qr(&[complex_input], &BTreeMap::new()).unwrap();
        assert_eq!(real_result.len(), complex_result.len());

        for (output_idx, (real_output, complex_output)) in
            real_result.iter().zip(complex_result.iter()).enumerate()
        {
            assert!(
                matches!(real_output, Value::Tensor(_)),
                "real QR output {output_idx} is not a tensor"
            );
            assert!(
                matches!(complex_output, Value::Tensor(_)),
                "complex QR output {output_idx} is not a tensor"
            );

            let Value::Tensor(real_tensor) = real_output else {
                return;
            };
            let Value::Tensor(complex_tensor) = complex_output else {
                return;
            };

            assert_eq!(real_tensor.shape, complex_tensor.shape);
            assert_eq!(real_tensor.elements.len(), complex_tensor.elements.len());

            for idx in 0..real_tensor.elements.len() {
                assert!(
                    matches!(real_tensor.elements[idx], Literal::F64Bits(_)),
                    "real QR output element is not F64Bits"
                );
                assert!(
                    matches!(complex_tensor.elements[idx], Literal::Complex128Bits(..)),
                    "complex QR output element is not Complex128Bits"
                );

                let real_bits = match real_tensor.elements[idx] {
                    Literal::F64Bits(bits) => bits,
                    _ => 0,
                };
                let (complex_re, complex_im) = match complex_tensor.elements[idx] {
                    Literal::Complex128Bits(re, im) => (re, im),
                    _ => (0, 0),
                };
                assert_eq!(real_bits, complex_re, "output {output_idx} real bits {idx}");
                assert_eq!(
                    f64::from_bits(complex_im),
                    0.0,
                    "output {output_idx} imag bits {idx}"
                );
            }
        }
    }

    #[test]
    fn cholesky_2x2_identity() {
        let identity = make_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = eval_cholesky(&[identity], &BTreeMap::new()).unwrap();
        let elems = extract_f64_elements(&result);
        assert!((elems[0] - 1.0).abs() < 1e-10);
        assert!((elems[1]).abs() < 1e-10);
        assert!((elems[2]).abs() < 1e-10);
        assert!((elems[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_2x2_spd() {
        // A = [[4, 2], [2, 3]]  =>  L = [[2, 0], [1, sqrt(2)]]
        let a = make_matrix(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let result = eval_cholesky(&[a], &BTreeMap::new()).unwrap();
        let elems = extract_f64_elements(&result);
        assert!((elems[0] - 2.0).abs() < 1e-10, "L[0,0]");
        assert!((elems[1]).abs() < 1e-10, "L[0,1]");
        assert!((elems[2] - 1.0).abs() < 1e-10, "L[1,0]");
        assert!((elems[3] - 2.0_f64.sqrt()).abs() < 1e-10, "L[1,1]");
    }

    #[test]
    fn cholesky_3x3_spd() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        let a = make_matrix(
            3,
            3,
            &[4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        );
        let result = eval_cholesky(&[a], &BTreeMap::new()).unwrap();
        let elems = extract_f64_elements(&result);
        let expected = [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0];
        for (i, (&got, &exp)) in elems.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "L[{},{}]: got {got}, expected {exp}",
                i / 3,
                i % 3
            );
        }
    }

    #[test]
    fn svd_real_fast_path_matches_complex_spectrum_and_reconstructs() {
        // The cyclic-Jacobi real fast path must (1) produce the same singular
        // spectrum as the complex max-pivot kernel to machine precision and
        // (2) yield a valid SVD: U·diag(S)·Vᵀ = A with U and Vᵀ orthonormal.
        // Covers tall (m>n), wide (m<n), square × thin and full_matrices.
        for &(m, n) in &[(6usize, 4usize), (4usize, 6usize), (5usize, 5usize)] {
            for full in [false, true] {
                let mut data = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        data[i * n + j] = ((i * 13 + j * 7) % 11) as f64 - 5.0 + 0.25 * (i as f64);
                    }
                }
                let k = m.min(n);
                let u_cols = if full { m } else { k };
                let vt_rows = if full { n } else { k };
                let mut params = BTreeMap::new();
                if full {
                    params.insert("full_matrices".to_owned(), "true".to_owned());
                }
                let a_real = make_matrix(m, n, &data);
                let a_complex = Value::Tensor(
                    TensorValue::new(
                        DType::Complex128,
                        Shape {
                            dims: vec![m as u32, n as u32],
                        },
                        data.iter()
                            .map(|&v| Literal::from_complex128(v, 0.0))
                            .collect(),
                    )
                    .unwrap(),
                );

                let real_out = eval_svd(&[a_real], &params).unwrap();
                let cplx_out = eval_svd(&[a_complex], &params).unwrap();
                assert_eq!(real_out.len(), 3, "{m}x{n} full={full}: expected U,S,Vh");

                let u = extract_f64_elements(&real_out[0]); // m×u_cols
                let s = extract_f64_elements(&real_out[1]); // k
                let vh = extract_f64_elements(&real_out[2]); // vt_rows×n
                let s_cplx = extract_f64_elements(&cplx_out[1]); // k

                // Output shapes must match the complex reference path.
                if let (Value::Tensor(ru), Value::Tensor(cu)) = (&real_out[0], &cplx_out[0]) {
                    assert_eq!(ru.shape.dims, cu.shape.dims, "{m}x{n} full={full}: U shape");
                }
                if let (Value::Tensor(rv), Value::Tensor(cv)) = (&real_out[2], &cplx_out[2]) {
                    assert_eq!(
                        rv.shape.dims, cv.shape.dims,
                        "{m}x{n} full={full}: Vh shape"
                    );
                }

                assert_eq!(s.len(), k, "{m}x{n} full={full}: S length");
                // (1) spectrum parity vs the reference complex kernel.
                for t in 0..k {
                    assert!(
                        (s[t] - s_cplx[t]).abs() < 1e-9,
                        "{m}x{n} full={full}: singular value {t} {} vs complex {} differ",
                        s[t],
                        s_cplx[t]
                    );
                    if t > 0 {
                        assert!(
                            s[t - 1] >= s[t],
                            "{m}x{n} full={full}: singular values not descending"
                        );
                    }
                }

                // (2a) reconstruction U·diag(S)·Vᵀ = A (only first k columns of U
                // and rows of Vᵀ carry nonzero singular values).
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for t in 0..k {
                            acc += u[i * u_cols + t] * s[t] * vh[t * n + j];
                        }
                        assert!(
                            (acc - data[i * n + j]).abs() < 1e-9,
                            "{m}x{n} full={full}: reconstruction[{i}][{j}] {acc} vs {}",
                            data[i * n + j]
                        );
                    }
                }

                // (2b) U columns orthonormal (Uᵀ·U = I_{u_cols}).
                for c1 in 0..u_cols {
                    for c2 in 0..u_cols {
                        let mut dot = 0.0;
                        for i in 0..m {
                            dot += u[i * u_cols + c1] * u[i * u_cols + c2];
                        }
                        let expected = if c1 == c2 { 1.0 } else { 0.0 };
                        assert!(
                            (dot - expected).abs() < 1e-9,
                            "{m}x{n} full={full}: U col dot[{c1}][{c2}] = {dot}, expected {expected}"
                        );
                    }
                }

                // (2c) Vᵀ rows orthonormal (Vh·Vhᵀ = I_{vt_rows}).
                for s1 in 0..vt_rows {
                    for s2 in 0..vt_rows {
                        let mut dot = 0.0;
                        for j in 0..n {
                            dot += vh[s1 * n + j] * vh[s2 * n + j];
                        }
                        let expected = if s1 == s2 { 1.0 } else { 0.0 };
                        assert!(
                            (dot - expected).abs() < 1e-9,
                            "{m}x{n} full={full}: Vh row dot[{s1}][{s2}] = {dot}, expected {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cholesky_blocked_matches_scalar_and_reconstructs() {
        // n >= CHOLESKY_BLOCK_THRESHOLD exercises the cache-blocked right-looking
        // kernel. The Cholesky factor is unique (positive diagonal), so the
        // blocked result must match the scalar kernel to near machine precision
        // and reconstruct A = L Lᵀ.
        let n = 300usize;
        let base: Vec<f64> = (0..n * n)
            .map(|idx| (((idx % 13) as f64) - 6.0) * 0.25)
            .collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += base[k * n + i] * base[k * n + j];
                }
                a[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
            }
        }
        assert!(n >= CHOLESKY_BLOCK_THRESHOLD, "n must hit the blocked path");

        let l_blocked = cholesky_real_blocked(n, &a);
        let l_scalar = cholesky_real_scalar(n, &a);

        // Strict upper triangle is zero.
        for i in 0..n {
            for jj in (i + 1)..n {
                assert_eq!(l_blocked[i * n + jj], 0.0, "upper[{i}][{jj}] nonzero");
            }
        }

        // Blocked vs scalar agree to near machine precision (unique factor).
        for idx in 0..n * n {
            let denom = 1.0 + l_scalar[idx].abs();
            assert!(
                (l_blocked[idx] - l_scalar[idx]).abs() <= 1e-9 * denom,
                "blocked vs scalar differ at {idx}: {} vs {}",
                l_blocked[idx],
                l_scalar[idx]
            );
        }

        // Reconstruction L Lᵀ ≈ A on the lower triangle.
        let scale = a.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        for i in 0..n {
            for j in 0..=i {
                let mut acc = 0.0;
                for k in 0..=j {
                    acc += l_blocked[i * n + k] * l_blocked[j * n + k];
                }
                assert!(
                    (acc - a[i * n + j]).abs() <= 1e-9 * scale,
                    "reconstruction[{i}][{j}] {acc} vs {}",
                    a[i * n + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_blocked_path_golden_output_digest() -> Result<(), Box<dyn std::error::Error>> {
        let n = CHOLESKY_BLOCK_THRESHOLD;
        let base: Vec<f64> = (0..n * n)
            .map(|idx| (((idx * 17 + 3) % 31) as f64 - 15.0) * 0.03125)
            .collect();
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += base[k * n + i] * base[k * n + j];
                }
                a[i * n + j] = s + if i == j { n as f64 + 17.0 } else { 0.0 };
            }
        }

        let result = eval_cholesky(&[make_matrix(n, n, &a)], &BTreeMap::new())?;
        let output_bits: Vec<u64> = extract_f64_elements(&result)
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let digest = fj_test_utils::fixture_id_from_json(&output_bits)?;
        assert_eq!(
            digest, "4ed70745461cf775a4ea667bf87bae3c6fbc5326c5095cad6435d0069d1c7540",
            "blocked Cholesky golden output digest changed"
        );
        Ok(())
    }

    #[test]
    fn cholesky_lower_schur_update_matches_full_gemm_on_consumed_triangle() {
        let rem = 9usize;
        let jb = 7usize;
        let n = 13usize;
        let base = 4usize;
        let mut a_full: Vec<f64> = (0..n * n)
            .map(|idx| ((idx % 19) as f64 - 9.0) * 0.125)
            .collect();
        let mut a_lower = a_full.clone();
        let l21: Vec<f64> = (0..rem * jb)
            .map(|idx| ((idx % 11) as f64 - 5.0) * 0.25)
            .collect();
        let mut l21t = vec![0.0_f64; jb * rem];
        for p in 0..rem {
            for c in 0..jb {
                l21t[c * rem + p] = l21[p * jb + c];
            }
        }

        let prod = matmul_2d(&l21, rem, jb, &l21t, rem);
        for p in 0..rem {
            let row = (base + p) * n + base;
            let pr = p * rem;
            for q in 0..rem {
                a_full[row + q] -= prod[pr + q];
            }
        }
        cholesky_schur_update_lower(&mut a_lower, n, base, &l21, rem, jb);

        for p in 0..rem {
            for q in 0..=p {
                let idx = (base + p) * n + base + q;
                assert_eq!(
                    a_lower[idx].to_bits(),
                    a_full[idx].to_bits(),
                    "lower Schur entry [{p},{q}] diverged"
                );
            }
        }
        for p in 0..rem {
            for q in (p + 1)..rem {
                let idx = (base + p) * n + base + q;
                assert_eq!(
                    a_lower[idx].to_bits(),
                    (((idx % 19) as f64 - 9.0) * 0.125).to_bits(),
                    "upper Schur entry [{p},{q}] should be untouched"
                );
            }
        }
    }

    #[test]
    fn one_sided_jacobi_svd_real_reconstructs_tall_profile_shape() {
        let (m, n) = (18usize, 9usize);
        let data: Vec<f64> = (0..m * n)
            .map(|idx| {
                let row = idx / n;
                let col = idx % n;
                let off_diag = (((row * 17 + col * 31) % 13) as f64 - 6.0) * 0.01;
                if row == col {
                    4.0 + col as f64 + off_diag
                } else {
                    off_diag + (row as f64) * 0.001
                }
            })
            .collect();
        let (sigma, u, v) = one_sided_jacobi_svd_real(m, n, &data);

        for row in 0..m {
            for col in 0..n {
                let mut reconstructed = 0.0;
                for axis in 0..n {
                    reconstructed += u[row * n + axis] * sigma[axis] * v[col * n + axis];
                }
                assert!(
                    (reconstructed - data[row * n + col]).abs() < 1e-9,
                    "reconstruction[{row},{col}] = {reconstructed}, expected {}",
                    data[row * n + col]
                );
            }
        }

        for left in 0..n {
            for right in 0..n {
                let mut u_dot = 0.0;
                let mut v_dot = 0.0;
                for row in 0..m {
                    u_dot += u[row * n + left] * u[row * n + right];
                }
                for row in 0..n {
                    v_dot += v[row * n + left] * v[row * n + right];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (u_dot - expected).abs() < 1e-9,
                    "U dot[{left},{right}] = {u_dot}, expected {expected}"
                );
                assert!(
                    (v_dot - expected).abs() < 1e-9,
                    "V dot[{left},{right}] = {v_dot}, expected {expected}"
                );
            }
        }
    }

    #[derive(Debug)]
    struct JacobiSvdCounterReport {
        sweeps: usize,
        visited_pairs: usize,
        accepted_rotations: usize,
        skipped_rotations: usize,
        per_sweep_pair_visits: Vec<usize>,
        per_sweep_active_pairs: Vec<usize>,
        zero_singular_columns: usize,
        near_zero_singular_columns: usize,
    }

    fn svd_48_profile_matrix_data() -> Vec<f64> {
        (0..48 * 48)
            .map(|i| {
                let x = i as f64;
                (x * 0.125).sin() + (x * 0.03125).cos()
            })
            .collect()
    }

    #[allow(clippy::too_many_lines)]
    fn one_sided_jacobi_svd_real_counter_report(
        m: usize,
        n: usize,
        a: &[f64],
    ) -> JacobiSvdCounterReport {
        let mut w = vec![0.0_f64; m * n];
        let mut initial_norm_sq = vec![0.0_f64; n];
        for row in 0..m {
            for col in 0..n {
                let value = a[row * n + col];
                w[col * m + row] = value;
                initial_norm_sq[col] += value * value;
            }
        }
        let mut v = vec![0.0_f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        let mut initial_order: Vec<usize> = (0..n).collect();
        initial_order.sort_by(|&left, &right| {
            initial_norm_sq[right]
                .total_cmp(&initial_norm_sq[left])
                .then_with(|| left.cmp(&right))
        });
        if initial_order
            .iter()
            .enumerate()
            .any(|(idx, &col)| idx != col)
        {
            let mut ordered_w = vec![0.0_f64; m * n];
            let mut ordered_v = vec![0.0_f64; n * n];
            for (new_col, &old_col) in initial_order.iter().enumerate() {
                let src = old_col * m;
                let dst = new_col * m;
                ordered_w[dst..dst + m].copy_from_slice(&w[src..src + m]);
                ordered_v[new_col * n + old_col] = 1.0;
            }
            w = ordered_w;
            v = ordered_v;
        }

        let eps = f64::EPSILON;
        let max_sweeps = 60;
        let mut visited_pairs = 0usize;
        let mut accepted_rotations = 0usize;
        let mut skipped_rotations = 0usize;
        let mut per_sweep_pair_visits = Vec::new();
        let mut per_sweep_active_pairs = Vec::new();
        for _ in 0..max_sweeps {
            let mut converged = true;
            let mut sweep_pair_visits = 0usize;
            let mut sweep_active_pairs = 0usize;
            for p in 0..n.saturating_sub(1) {
                for q in (p + 1)..n {
                    visited_pairs += 1;
                    sweep_pair_visits += 1;
                    let mut alpha = 0.0_f64;
                    let mut beta = 0.0_f64;
                    let mut gamma = 0.0_f64;
                    for i in 0..m {
                        let wip = w[p * m + i];
                        let wiq = w[q * m + i];
                        alpha += wip * wip;
                        beta += wiq * wiq;
                        gamma += wip * wiq;
                    }
                    if gamma.abs() <= eps * (alpha * beta).sqrt() {
                        skipped_rotations += 1;
                        continue;
                    }
                    converged = false;
                    accepted_rotations += 1;
                    sweep_active_pairs += 1;
                    let tau = (beta - alpha) / (2.0 * gamma);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    for i in 0..m {
                        let wip = w[p * m + i];
                        let wiq = w[q * m + i];
                        w[p * m + i] = c * wip - s * wiq;
                        w[q * m + i] = s * wip + c * wiq;
                    }
                    for i in 0..n {
                        let vip = v[p * n + i];
                        let viq = v[q * n + i];
                        v[p * n + i] = c * vip - s * viq;
                        v[q * n + i] = s * vip + c * viq;
                    }
                }
            }
            per_sweep_pair_visits.push(sweep_pair_visits);
            per_sweep_active_pairs.push(sweep_active_pairs);
            if converged {
                break;
            }
        }

        let mut zero_singular_columns = 0usize;
        let mut near_zero_singular_columns = 0usize;
        for j in 0..n {
            let mut s2 = 0.0_f64;
            for i in 0..m {
                let wij = w[j * m + i];
                s2 += wij * wij;
            }
            let sigma = s2.sqrt();
            if sigma == 0.0 {
                zero_singular_columns += 1;
            } else if sigma <= f64::EPSILON * 1e4 {
                near_zero_singular_columns += 1;
            }
        }

        JacobiSvdCounterReport {
            sweeps: per_sweep_pair_visits.len(),
            visited_pairs,
            accepted_rotations,
            skipped_rotations,
            per_sweep_pair_visits,
            per_sweep_active_pairs,
            zero_singular_columns,
            near_zero_singular_columns,
        }
    }

    #[test]
    fn one_sided_jacobi_svd_real_48x48_profile_counters() {
        let data = svd_48_profile_matrix_data();
        let report = one_sided_jacobi_svd_real_counter_report(48, 48, &data);

        eprintln!("SVD 48x48 profile counters: {report:#?}");
        assert_eq!(
            report.visited_pairs,
            report.accepted_rotations + report.skipped_rotations
        );
        assert_eq!(report.per_sweep_pair_visits.len(), report.sweeps);
        assert_eq!(report.per_sweep_active_pairs.len(), report.sweeps);
        assert!(
            report.accepted_rotations > 0,
            "profile target should exercise Jacobi rotations"
        );
        assert_eq!(report.sweeps, 13);
        assert_eq!(report.visited_pairs, 14_664);
        assert_eq!(report.accepted_rotations, 10_264);
        assert_eq!(report.skipped_rotations, 4_400);
        assert_eq!(
            report.per_sweep_active_pairs,
            vec![
                1128, 1128, 1128, 1128, 1059, 957, 955, 961, 956, 711, 149, 4, 0
            ]
        );
        assert_eq!(
            report.zero_singular_columns, 0,
            "benchmark fixture should have no exactly zero singular columns"
        );
        assert_eq!(
            report.near_zero_singular_columns, 44,
            "profile target's near-zero columns are the active-pair compaction signal"
        );
    }

    #[test]
    fn svd_48x48_f64_golden_output_digest() -> Result<(), Box<dyn std::error::Error>> {
        let data = svd_48_profile_matrix_data();
        let result = eval_svd(&[make_matrix(48, 48, &data)], &BTreeMap::new())?;
        let u = extract_f64_elements(&result[0]);
        let s = extract_f64_elements(&result[1]);
        let vh = extract_f64_elements(&result[2]);
        for (idx, pair) in s.windows(2).enumerate() {
            assert!(
                pair[0] >= pair[1],
                "singular values not descending at {idx}: {} < {}",
                pair[0],
                pair[1]
            );
        }
        for row in 0..48 {
            for col in 0..48 {
                let mut reconstructed = 0.0;
                for axis in 0..48 {
                    reconstructed += u[row * 48 + axis] * s[axis] * vh[axis * 48 + col];
                }
                assert!(
                    (reconstructed - data[row * 48 + col]).abs() < 1e-9,
                    "SVD 48x48 reconstruction[{row},{col}] = {reconstructed}, expected {}",
                    data[row * 48 + col]
                );
            }
        }
        let output_bits: Vec<Vec<u64>> = result
            .iter()
            .map(|value| {
                extract_f64_elements(value)
                    .iter()
                    .map(|element| element.to_bits())
                    .collect()
            })
            .collect();
        let digest = fj_test_utils::fixture_id_from_json(&output_bits)?;
        eprintln!("SVD 48x48 f64 golden digest: {digest}");
        assert_eq!(
            digest, "5d9fc335e204bea56ea3e086abf97a7e271c9d70aaad4352503d8de8457f197f",
            "SVD 48x48 f64 golden output digest changed"
        );
        Ok(())
    }

    fn one_sided_jacobi_svd_real_rowmajor_v_reference(
        m: usize,
        n: usize,
        a: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let k = m.min(n);
        let mut w = vec![0.0_f64; m * n];
        for row in 0..m {
            for col in 0..n {
                w[col * m + row] = a[row * n + col];
            }
        }
        let mut v = vec![0.0_f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        let eps = f64::EPSILON;
        let max_sweeps = 60;
        for _ in 0..max_sweeps {
            let mut converged = true;
            for p in 0..n.saturating_sub(1) {
                for q in (p + 1)..n {
                    let mut alpha = 0.0_f64;
                    let mut beta = 0.0_f64;
                    let mut gamma = 0.0_f64;
                    for i in 0..m {
                        let wip = w[p * m + i];
                        let wiq = w[q * m + i];
                        alpha += wip * wip;
                        beta += wiq * wiq;
                        gamma += wip * wiq;
                    }
                    if gamma.abs() <= eps * (alpha * beta).sqrt() {
                        continue;
                    }
                    converged = false;
                    let tau = (beta - alpha) / (2.0 * gamma);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    for i in 0..m {
                        let wip = w[p * m + i];
                        let wiq = w[q * m + i];
                        w[p * m + i] = c * wip - s * wiq;
                        w[q * m + i] = s * wip + c * wiq;
                    }
                    for i in 0..n {
                        let vip = v[i * n + p];
                        let viq = v[i * n + q];
                        v[i * n + p] = c * vip - s * viq;
                        v[i * n + q] = s * vip + c * viq;
                    }
                }
            }
            if converged {
                break;
            }
        }

        let mut col_norm = vec![0.0_f64; n];
        for j in 0..n {
            let mut s2 = 0.0_f64;
            for i in 0..m {
                let wij = w[j * m + i];
                s2 += wij * wij;
            }
            col_norm[j] = s2.sqrt();
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| col_norm[b].total_cmp(&col_norm[a]));

        let mut sigma = vec![0.0_f64; k];
        let mut v_sorted = vec![0.0_f64; n * n];
        let mut u = vec![0.0_f64; m * k];
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..n {
                v_sorted[row * n + new_col] = v[row * n + old_col];
            }
            if new_col < k {
                let sg = col_norm[old_col];
                sigma[new_col] = sg;
                if sg > f64::EPSILON * 1e4 {
                    for row in 0..m {
                        u[row * k + new_col] = w[old_col * m + row] / sg;
                    }
                }
            }
        }

        (sigma, u, v_sorted)
    }

    #[test]
    fn one_sided_jacobi_svd_real_initial_norm_order_preserves_contract() {
        let (m, n) = (11usize, 7usize);
        let data: Vec<f64> = (0..m * n)
            .map(|idx| {
                let row = idx / n;
                let col = idx % n;
                let base = ((row * 23 + col * 29) % 17) as f64 - 8.0;
                base * 0.025 + (row as f64 * 0.003) - (col as f64 * 0.007)
            })
            .collect();

        let (old_sigma, _, _) = one_sided_jacobi_svd_real_rowmajor_v_reference(m, n, &data);
        let (new_sigma, new_u, new_v) = one_sided_jacobi_svd_real(m, n, &data);
        let (repeat_sigma, repeat_u, repeat_v) = one_sided_jacobi_svd_real(m, n, &data);

        for (idx, ((&old, &new), &repeat)) in old_sigma
            .iter()
            .zip(&new_sigma)
            .zip(&repeat_sigma)
            .enumerate()
        {
            let tolerance = 1e-12 * (1.0 + old.abs());
            assert!(
                (old - new).abs() <= tolerance,
                "sigma contract drift at {idx}: old={old:?} new={new:?}"
            );
            assert_eq!(
                new.to_bits(),
                repeat.to_bits(),
                "sigma determinism drift at {idx}: first={new:?} repeat={repeat:?}"
            );
        }
        for (idx, (&first, &repeat)) in new_u.iter().zip(&repeat_u).enumerate() {
            assert_eq!(
                first.to_bits(),
                repeat.to_bits(),
                "U determinism drift at {idx}: first={first:?} repeat={repeat:?}"
            );
        }
        for (idx, (&first, &repeat)) in new_v.iter().zip(&repeat_v).enumerate() {
            assert_eq!(
                first.to_bits(),
                repeat.to_bits(),
                "V determinism drift at {idx}: first={first:?} repeat={repeat:?}"
            );
        }

        for row in 0..m {
            for col in 0..n {
                let mut reconstructed = 0.0;
                for axis in 0..n {
                    reconstructed +=
                        new_u[row * n + axis] * new_sigma[axis] * new_v[col * n + axis];
                }
                assert!(
                    (reconstructed - data[row * n + col]).abs() <= 1e-11,
                    "reconstruction[{row},{col}] = {reconstructed}, expected {}",
                    data[row * n + col]
                );
            }
        }

        for left in 0..n {
            for right in 0..n {
                let expected = if left == right { 1.0 } else { 0.0 };
                let mut u_dot = 0.0;
                let mut v_dot = 0.0;
                for row in 0..m {
                    u_dot += new_u[row * n + left] * new_u[row * n + right];
                }
                for row in 0..n {
                    v_dot += new_v[row * n + left] * new_v[row * n + right];
                }
                assert!(
                    (u_dot - expected).abs() <= 1e-11,
                    "U dot[{left},{right}] = {u_dot}, expected {expected}"
                );
                assert!(
                    (v_dot - expected).abs() <= 1e-11,
                    "V dot[{left},{right}] = {v_dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn cholesky_real_fast_path_bit_identical_to_complex_path() {
        // Factor the same real SPD matrix via the real fast path (F64 input)
        // and the complex path (same values as Complex128, imag 0). The factor
        // real parts must be bit-for-bit identical and the complex path's imag
        // parts must be exactly zero.
        let n = 6usize;
        let mut base = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                base[i * n + j] = ((i * 7 + j * 3) % 5) as f64 - 2.0;
            }
        }
        // A = base^T base + n*I  (symmetric positive definite, deterministic).
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += base[k * n + i] * base[k * n + j];
                }
                a[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
            }
        }

        let a_real = make_matrix(n, n, &a);
        let a_complex = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                a.iter()
                    .map(|&v| Literal::from_complex128(v, 0.0))
                    .collect(),
            )
            .unwrap(),
        );

        let Value::Tensor(lr) = eval_cholesky(&[a_real], &BTreeMap::new()).unwrap() else {
            panic!("real result not a tensor");
        };
        let Value::Tensor(lc) = eval_cholesky(&[a_complex], &BTreeMap::new()).unwrap() else {
            panic!("complex result not a tensor");
        };
        for idx in 0..n * n {
            let re_real = match lr.elements[idx] {
                Literal::F64Bits(b) => b,
                other => panic!("real factor element not F64Bits: {other:?}"),
            };
            let (re_cplx, im_cplx) = match lc.elements[idx] {
                Literal::Complex128Bits(re, im) => (re, im),
                other => panic!("complex factor element not Complex128Bits: {other:?}"),
            };
            assert_eq!(re_real, re_cplx, "real-part bits differ at {idx}");
            assert_eq!(
                im_cplx,
                0.0_f64.to_bits(),
                "complex factor imag nonzero at {idx}"
            );
        }
    }

    #[test]
    fn cholesky_f64_dense_output_matches_literal_materialization() {
        let n = 6usize;
        let mut base = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                base[i * n + j] = ((i * 11 + j * 5) % 7) as f64 - 3.0;
            }
        }
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += base[k * n + i] * base[k * n + j];
                }
                a[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
            }
        }

        let Value::Tensor(actual) = eval_cholesky(&[make_matrix(n, n, &a)], &BTreeMap::new())
            .expect("cholesky should succeed")
        else {
            panic!("expected tensor output");
        };
        let expected_values = cholesky_real_scalar(n, &a);
        let Value::Tensor(expected) = matrix_to_value(n, n, &expected_values, DType::F64)
            .expect("literal materialization should succeed")
        else {
            panic!("expected tensor output");
        };

        assert_eq!(actual.dtype, DType::F64);
        assert_eq!(actual.shape.dims, vec![n as u32, n as u32]);
        assert!(
            actual.elements.as_f64_slice().is_some(),
            "F64 Cholesky should return dense F64 storage"
        );
        assert_eq!(actual.elements.as_slice(), expected.elements.as_slice());
        for (idx, (&dense, &expected)) in actual
            .elements
            .as_f64_slice()
            .expect("dense F64 output")
            .iter()
            .zip(expected_values.iter())
            .enumerate()
        {
            assert_eq!(
                dense.to_bits(),
                expected.to_bits(),
                "dense output bits differ at {idx}"
            );
        }
    }

    #[test]
    fn cholesky_real_fast_path_preserves_f32_dtype_and_shape() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_f32(4.0),
                    Literal::from_f32(2.0),
                    Literal::from_f32(2.0),
                    Literal::from_f32(3.0),
                ],
            )
            .unwrap(),
        );

        let Value::Tensor(t) = eval_cholesky(&[input], &BTreeMap::new()).unwrap() else {
            panic!("expected tensor");
        };

        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape.dims, vec![2, 2]);
        t.validate_dtype_consistency()
            .expect("F32 Cholesky output must contain only F32Bits elements");
        assert!(
            t.elements
                .iter()
                .all(|literal| matches!(literal, Literal::F32Bits(_)))
        );
    }

    #[test]
    fn cholesky_not_positive_definite_returns_nan() {
        // jnp.linalg.cholesky returns NaN (not an error) for non-PD input,
        // unlike NumPy which raises LinAlgError. A = [[1,2],[2,1]] has
        // eigenvalues 3 and -1, so L[1][1] = sqrt(1 - 2^2) = sqrt(-3) = NaN.
        let a = make_matrix(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let result =
            eval_cholesky(&[a], &BTreeMap::new()).expect("cholesky must not raise for non-PD");
        let Value::Tensor(t) = result else {
            panic!("expected tensor");
        };
        let vals: Vec<f64> = t.elements.iter().map(|e| e.as_f64().unwrap()).collect();
        assert!(
            vals.iter().any(|v| v.is_nan()),
            "non-PD cholesky must contain NaN, got {vals:?}"
        );
    }

    #[test]
    fn cholesky_non_square_rejected() {
        let a = make_matrix(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(eval_cholesky(&[a], &BTreeMap::new()).is_err());
    }

    #[test]
    fn cholesky_roundtrip_llt() {
        // Verify L * L^T = A for a known SPD matrix
        let a_data = [4.0, 2.0, 2.0, 3.0];
        let a = make_matrix(2, 2, &a_data);
        let result = eval_cholesky(&[a], &BTreeMap::new()).unwrap();
        let l = extract_f64_elements(&result);
        // Compute L * L^T
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l[i * n + k] * l[j * n + k];
                }
                assert!(
                    (sum - a_data[i * n + j]).abs() < 1e-10,
                    "LLT[{i},{j}] = {sum}, expected {}",
                    a_data[i * n + j]
                );
            }
        }
    }

    #[test]
    fn triangular_solve_lower_2x2() {
        // L = [[2, 0], [1, 3]], b = [[4], [7]]
        // Forward substitution: x0 = 4/2 = 2, x1 = (7 - 1*2)/3 = 5/3
        let l = make_matrix(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let b = make_matrix(2, 1, &[4.0, 7.0]);
        let result = eval_triangular_solve(&[l, b], &BTreeMap::new()).unwrap();
        let elems = extract_f64_elements(&result);
        assert!((elems[0] - 2.0).abs() < 1e-10, "x[0]");
        assert!((elems[1] - 5.0 / 3.0).abs() < 1e-10, "x[1]");
    }

    #[test]
    fn triangular_solve_singular_diagonal_returns_non_finite_not_error() {
        // jnp.linalg-style triangular solve does NOT raise for a zero/near-zero
        // diagonal (NumPy raises). Zero diagonal => non-finite (division by
        // zero); near-zero diagonal => finite large value.
        let l = make_matrix(2, 2, &[0.0, 0.0, 1.0, 3.0]); // L[0][0] = 0 (singular)
        let b = make_matrix(2, 1, &[4.0, 7.0]);
        let result = eval_triangular_solve(&[l, b], &BTreeMap::new())
            .expect("triangular_solve must not raise for a singular diagonal");
        let elems = extract_f64_elements(&result);
        assert!(
            elems.iter().any(|v| !v.is_finite()),
            "singular triangular solve must be non-finite, got {elems:?}"
        );

        // Near-singular: L[0][0] = 1e-13 => x0 = 1/1e-13 = 1e13 (finite), matching
        // JAX's divide-through behavior rather than the old < ~2e-12 error.
        let l2 = make_matrix(2, 2, &[1e-13, 0.0, 1.0, 3.0]);
        let b2 = make_matrix(2, 1, &[1.0, 7.0]);
        let r2 = eval_triangular_solve(&[l2, b2], &BTreeMap::new())
            .expect("triangular_solve must not raise for a near-singular diagonal");
        let e2 = extract_f64_elements(&r2);
        assert!(
            e2.iter().all(|v| v.is_finite()),
            "near-singular triangular solve must stay finite, got {e2:?}"
        );
        assert!((e2[0] - 1e13).abs() < 1e3, "x[0] ~ 1e13, got {}", e2[0]);
    }

    #[test]
    fn triangular_solve_upper() {
        // U = [[2, 1], [0, 3]], b = [[5], [6]]
        // Back substitution: x1 = 6/3 = 2, x0 = (5 - 1*2)/2 = 1.5
        let u = make_matrix(2, 2, &[2.0, 1.0, 0.0, 3.0]);
        let b = make_matrix(2, 1, &[5.0, 6.0]);
        let mut params = BTreeMap::new();
        params.insert("lower".to_owned(), "false".to_owned());
        let result = eval_triangular_solve(&[u, b], &params).unwrap();
        let elems = extract_f64_elements(&result);
        assert!((elems[0] - 1.5).abs() < 1e-10, "x[0]");
        assert!((elems[1] - 2.0).abs() < 1e-10, "x[1]");
    }

    #[test]
    fn triangular_solve_multi_rhs() {
        // L = [[1, 0], [2, 1]], B = [[1, 2], [4, 6]]
        // Col 0: x0=1, x1=4-2*1=2 | Col 1: x0=2, x1=6-2*2=2
        let l = make_matrix(2, 2, &[1.0, 0.0, 2.0, 1.0]);
        let b = make_matrix(2, 2, &[1.0, 2.0, 4.0, 6.0]);
        let result = eval_triangular_solve(&[l, b], &BTreeMap::new()).unwrap();
        let elems = extract_f64_elements(&result);
        assert!((elems[0] - 1.0).abs() < 1e-10, "X[0,0]");
        assert!((elems[1] - 2.0).abs() < 1e-10, "X[0,1]");
        assert!((elems[2] - 2.0).abs() < 1e-10, "X[1,0]");
        assert!((elems[3] - 2.0).abs() < 1e-10, "X[1,1]");
    }

    #[test]
    fn triangular_solve_transpose_lower() {
        // L = [[2, 0], [1, 3]], so L^T = [[2, 1], [0, 3]]
        // Solve L^T x = b where b = [[5], [6]]
        // Back sub: x1 = 6/3 = 2, x0 = (5 - 1*2)/2 = 1.5
        let l = make_matrix(2, 2, &[2.0, 0.0, 1.0, 3.0]);
        let b = make_matrix(2, 1, &[5.0, 6.0]);
        let mut params = BTreeMap::new();
        params.insert("lower".to_owned(), "true".to_owned());
        params.insert("transpose_a".to_owned(), "true".to_owned());
        let result = eval_triangular_solve(&[l, b], &params).unwrap();
        let elems = extract_f64_elements(&result);
        assert!((elems[0] - 1.5).abs() < 1e-10, "x[0]");
        assert!((elems[1] - 2.0).abs() < 1e-10, "x[1]");
    }

    #[test]
    fn triangular_solve_unit_diagonal() {
        // L = [[99, 0], [2, 99]], b = [[3], [8]] with unit_diagonal=true
        // Diagonal treated as 1: x0 = 3/1 = 3, x1 = (8 - 2*3)/1 = 2
        let l = make_matrix(2, 2, &[99.0, 0.0, 2.0, 99.0]);
        let b = make_matrix(2, 1, &[3.0, 8.0]);
        let mut params = BTreeMap::new();
        params.insert("unit_diagonal".to_owned(), "true".to_owned());
        let result = eval_triangular_solve(&[l, b], &params).unwrap();
        let elems = extract_f64_elements(&result);
        assert!((elems[0] - 3.0).abs() < 1e-10, "x[0]");
        assert!((elems[1] - 2.0).abs() < 1e-10, "x[1]");
    }

    #[test]
    fn triangular_solve_cholesky_roundtrip() {
        // A = [[4, 2], [2, 3]], b = [[1], [2]]
        // Cholesky: L, then solve L y = b, then solve L^T x = y => A x = b
        let a_data = [4.0, 2.0, 2.0, 3.0];
        let a = make_matrix(2, 2, &a_data);
        let b = make_matrix(2, 1, &[1.0, 2.0]);

        // Step 1: Cholesky
        let l = eval_cholesky(&[a], &BTreeMap::new()).unwrap();

        // Step 2: Forward solve L y = b
        let y = eval_triangular_solve(&[l.clone(), b.clone()], &BTreeMap::new()).unwrap();

        // Step 3: Back solve L^T x = y
        let mut params = BTreeMap::new();
        params.insert("transpose_a".to_owned(), "true".to_owned());
        let x = eval_triangular_solve(&[l, y], &params).unwrap();
        let x_elems = extract_f64_elements(&x);

        // Verify A * x = b
        let b_elems = [1.0, 2.0];
        for i in 0..2 {
            let mut row_sum = 0.0;
            for j in 0..2 {
                row_sum += a_data[i * 2 + j] * x_elems[j];
            }
            assert!(
                (row_sum - b_elems[i]).abs() < 1e-10,
                "A*x[{i}] = {row_sum}, expected {}",
                b_elems[i]
            );
        }
    }

    // Inline per-column lower/no-transpose REAL triangular solve reference (matching
    // the real fast path's `a·x` / `x/d` f64 arithmetic), so the batched-vs-per-column
    // comparison is bit-exact for real inputs.
    fn tri_solve_per_column_lower(adata: &[f64], bdata: &[f64], n: usize, n_b: usize) -> Vec<f64> {
        let mut x = vec![0.0f64; n * n_b];
        let mut b_col = vec![0.0f64; n];
        for col in 0..n_b {
            for row in 0..n {
                b_col[row] = bdata[row * n_b + col];
            }
            for i in 0..n {
                for k in 0..i {
                    b_col[i] -= adata[i * n + k] * x[k * n_b + col];
                }
                x[i * n_b + col] = b_col[i] / adata[i * n + i];
            }
        }
        x
    }

    #[test]
    fn triangular_solve_batched_bit_identical_to_per_column() {
        // The real-fast-path eval_triangular_solve must equal the per-column real
        // substitution bit-for-bit (only loop nesting differs) at a size/column count
        // that exercises the vectorized inner loop.
        let n = 160usize;
        let n_b = 40usize;
        // Lower-triangular, diagonally dominant A.
        let mut adata = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                adata[i * n + j] = (((i * 7 + j * 5 + 1) % 9) as f64 - 4.0) * 0.1;
            }
            adata[i * n + i] = (i as f64) * 0.05 + 3.0;
        }
        let bdata: Vec<f64> = (0..n * n_b)
            .map(|k| (((k * 53 + 11) % 37) as f64) * 0.1 - 1.8)
            .collect();
        let a = make_matrix(n, n, &adata);
        let b = make_matrix(n, n_b, &bdata);
        let got = extract_f64_elements(&eval_triangular_solve(&[a, b], &BTreeMap::new()).unwrap());
        let want = tri_solve_per_column_lower(&adata, &bdata, n, n_b);
        for k in 0..n * n_b {
            assert_eq!(
                got[k].to_bits(),
                want[k].to_bits(),
                "batched tri-solve diverged at {k}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_triangular_solve_real_vs_complex() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // Real fast path (production for real inputs) vs the complex batched kernel
        // (the prior path), on identical real data.
        for &(n, n_b) in &[(1024usize, 64usize), (1024, 256), (2048, 128)] {
            let mut adata = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..=i {
                    adata[i * n + j] = (((i * 7 + j * 5 + 1) % 9) as f64 - 4.0) * 0.1;
                }
                adata[i * n + i] = (i as f64) * 0.001 + 3.0;
            }
            let bdata: Vec<f64> = (0..n * n_b)
                .map(|k| (((k * 53 + 11) % 37) as f64) * 0.1 - 1.8)
                .collect();
            let ac: Vec<(f64, f64)> = adata.iter().map(|&v| (v, 0.0)).collect();
            let t_complex = best_time(|| {
                let mut x: Vec<(f64, f64)> = bdata.iter().map(|&v| (v, 0.0)).collect();
                super::batched_tri_solve(&ac, &mut x, n, n_b, true, false, false);
                std::hint::black_box(x);
            });
            let t_real = best_time(|| {
                let mut x = bdata.clone();
                super::batched_tri_solve_real(&adata, &mut x, n, n_b, true, false, false);
                std::hint::black_box(x);
            });
            println!(
                "BENCH triangular_solve real vs complex n={n} n_b={n_b}: complex {:.2}ms -> real {:.2}ms = {:.2}x",
                t_complex * 1e3,
                t_real * 1e3,
                t_complex / t_real
            );
        }
    }

    // ── QR tests ──────────────────────────────────────────────

    #[test]
    fn qr_2x2_identity() {
        let a = make_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = eval_qr(&[a], &BTreeMap::new()).unwrap();
        assert_eq!(result.len(), 2, "QR should return [Q, R]");
        let q = extract_f64_elements(&result[0]);
        let r = extract_f64_elements(&result[1]);
        // Q and R should both be identity (possibly with sign flips)
        // Check Q^T Q = I
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += q[k * 2 + i] * q[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "QTQ[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
        // Check QR = A
        let a_data = [1.0, 0.0, 0.0, 1.0];
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += q[i * 2 + k] * r[k * 2 + j];
                }
                assert!(
                    (val - a_data[i * 2 + j]).abs() < 1e-10,
                    "QR[{i},{j}] = {val}, expected {}",
                    a_data[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn qr_2x2_roundtrip() {
        let a_data = [1.0, -1.0, 1.0, 1.0];
        let a = make_matrix(2, 2, &a_data);
        let result = eval_qr(&[a], &BTreeMap::new()).unwrap();
        let q = extract_f64_elements(&result[0]);
        let r = extract_f64_elements(&result[1]);

        // Verify Q^T Q = I
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += q[k * 2 + i] * q[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "QTQ[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }

        // Verify R is upper triangular
        assert!(r[2].abs() < 1e-10, "R[1,0] should be zero, got {}", r[2]);

        // Verify QR = A
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += q[i * 2 + k] * r[k * 2 + j];
                }
                assert!(
                    (val - a_data[i * 2 + j]).abs() < 1e-10,
                    "QR[{i},{j}] = {val}, expected {}",
                    a_data[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn qr_3x2_thin() {
        // Thin QR: 3x2 → Q 3x2, R 2x2
        let a_data = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let a = make_matrix(3, 2, &a_data);
        let result = eval_qr(&[a], &BTreeMap::new()).unwrap();
        assert_eq!(result.len(), 2);

        let q = extract_f64_elements(&result[0]);
        let r = extract_f64_elements(&result[1]);

        // Q should be 3x2
        assert_eq!(q.len(), 6, "Q should have 3*2=6 elements");
        // R should be 2x2
        assert_eq!(r.len(), 4, "R should have 2*2=4 elements");

        // Verify Q^T Q = I (2x2)
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += q[k * 2 + i] * q[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "QTQ[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }

        // Verify R is upper triangular
        assert!(r[2].abs() < 1e-10, "R[1,0] should be zero, got {}", r[2]);

        // Verify QR = A
        for i in 0..3 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += q[i * 2 + k] * r[k * 2 + j];
                }
                assert!(
                    (val - a_data[i * 2 + j]).abs() < 1e-10,
                    "QR[{i},{j}] = {val}, expected {}",
                    a_data[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn qr_real_fast_path_bit_identical_to_complex_zero_imag_path() {
        let n = 7usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    (n as f64) + (i as f64) * 0.375 + 5.0
                } else {
                    (((i * 17 + j * 29 + 3) % 23) as f64 + 0.5) * 0.125 - 1.25
                };
            }
        }

        assert_real_qr_matches_complex_zero_imag(n, &a);
    }

    #[test]
    fn qr_real_zero_input_preserves_complex_zero_imag_path_bits() {
        let n = 5usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    (n as f64) + 2.0
                } else if (i + j) % 3 == 0 {
                    0.0
                } else {
                    ((i * 11 + j * 7 + 1) % 17) as f64 * 0.125 - 0.75
                };
            }
        }

        assert_real_qr_matches_complex_zero_imag(n, &a);
    }

    #[test]
    fn qr_blocked_reconstructs_and_orthonormal() {
        // Force the WY-blocked path (qr_real_bench blocked=true) at a small n so the
        // test is cheap: A = Q·R must hold and Q must be orthonormal (Qᵀ·Q = I), both
        // to tolerance. The blocked path reassociates vs the scalar reflector loop, so
        // this is a tolerance check (QR JAX-parity is 1e-12), not bit-identity.
        let n = 300usize;
        // Diagonally dominant => full rank with no tiny pivots, so QR with positive-
        // diagonal R is UNIQUE and the blocked path must match the scalar path (avoids
        // the genuine column-direction ambiguity of a near-dependent column).
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                let off = ((idx as f64) * 0.012_34).sin() * 0.7 + ((idx % 13) as f64) * 0.05 - 0.3;
                if i == j { off + n as f64 } else { off }
            })
            .collect();
        let blocked = super::qr_real_bench(a.clone(), n, n, true);
        let scalar = super::qr_real_bench(a.clone(), n, n, false);
        let qd: Vec<f64> = blocked[0]
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();
        let rd: Vec<f64> = blocked[1]
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();
        // Blocked must also match the trusted scalar factorization to tolerance.
        let qs: Vec<f64> = scalar[0]
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();
        let mut max_res = 0.0_f64;
        let mut max_orth = 0.0_f64;
        let mut max_vs_scalar = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let mut qr = 0.0;
                let mut qtq = 0.0;
                for kk in 0..n {
                    qr += qd[i * n + kk] * rd[kk * n + j];
                    qtq += qd[kk * n + i] * qd[kk * n + j];
                }
                max_res = max_res.max((qr - a[i * n + j]).abs());
                let want = if i == j { 1.0 } else { 0.0 };
                max_orth = max_orth.max((qtq - want).abs());
                max_vs_scalar = max_vs_scalar.max((qd[i * n + j] - qs[i * n + j]).abs());
            }
        }
        assert!(max_res < 1e-9, "QR reconstruction residual {max_res}");
        assert!(max_orth < 1e-9, "Q orthonormality residual {max_orth}");
        assert!(
            max_vs_scalar < 1e-9,
            "blocked Q vs scalar Q {max_vs_scalar}"
        );
    }

    #[test]
    fn qr_real_path_handles_zeros_reconstructs_and_orthonormal() {
        // The real QR path must work for real matrices CONTAINING ZEROS (the
        // previous `a.iter().all(|v| *v != 0.0)` gate was overly broad — Householder
        // QR handles zero pivots/columns fine). Verify A=Q·R and Qᵀ·Q=I to tolerance
        // for several zero patterns, on the real path directly.
        for n in [4usize, 17, 64] {
            // Diagonally dominant (full rank) but with many deliberate zeros.
            let a: Vec<f64> = (0..n * n)
                .map(|idx| {
                    let (i, j) = (idx / n, idx % n);
                    if i == j {
                        (n as f64) + (i as f64) * 0.1 + 2.0
                    } else if idx % 3 == 0 {
                        0.0 // scatter zeros through the off-diagonal
                    } else {
                        (((i * 31 + j * 17) % 11) as f64 - 5.0) * 0.2
                    }
                })
                .collect();
            let qr = super::qr_real_bench(a.clone(), n, n, false);
            let qd = extract_f64_elements(&qr[0]);
            let rd = extract_f64_elements(&qr[1]);
            let mut max_res = 0.0f64;
            let mut max_orth = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let mut qrij = 0.0;
                    let mut qtq = 0.0;
                    for kk in 0..n {
                        qrij += qd[i * n + kk] * rd[kk * n + j];
                        qtq += qd[kk * n + i] * qd[kk * n + j];
                    }
                    max_res = max_res.max((qrij - a[i * n + j]).abs());
                    max_orth = max_orth.max((qtq - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            assert!(
                max_res < 1e-9,
                "n={n} real-QR-with-zeros reconstruction {max_res:e}"
            );
            assert!(
                max_orth < 1e-9,
                "n={n} real-QR-with-zeros orthonormality {max_orth:e}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_qr_real_vs_complex_path() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // Real matrix WITH zeros (previously forced onto the complex QR path).
        for &n in &[256usize, 512] {
            let a: Vec<f64> = (0..n * n)
                .map(|idx| {
                    let (i, j) = (idx / n, idx % n);
                    if i == j {
                        (n as f64) + (i as f64) * 0.01 + 2.0
                    } else if idx % 3 == 0 {
                        0.0
                    } else {
                        (((i * 31 + j * 17) % 11) as f64 - 5.0) * 0.1
                    }
                })
                .collect();
            let t_real = best_time(|| {
                std::hint::black_box(super::qr_real_bench(a.clone(), n, n, true));
            });
            let celems: Vec<Literal> = a
                .iter()
                .map(|&v| Literal::from_complex128(v, 0.0))
                .collect();
            let cval = Value::Tensor(
                TensorValue::new(
                    DType::Complex128,
                    Shape {
                        dims: vec![n as u32, n as u32],
                    },
                    celems,
                )
                .unwrap(),
            );
            let t_complex = best_time(|| {
                std::hint::black_box(
                    eval_qr(std::slice::from_ref(&cval), &BTreeMap::new()).unwrap(),
                );
            });
            println!(
                "BENCH QR real vs complex-path n={n}: complex {:.1}ms -> real {:.1}ms = {:.2}x",
                t_complex * 1e3,
                t_real * 1e3,
                t_complex / t_real
            );
        }
    }

    #[test]
    fn qr_tiled_reflector_bit_identical_to_scalar_column_loop() {
        let rows = 11usize;
        let cols = 13usize;
        let row_start = 3usize;
        let col_start = 2usize;
        let col_end = cols;
        let tau = 0.3125f64;
        let v: Vec<f64> = (row_start..rows)
            .map(|i| ((i * 17 + 5) as f64).sin() * 0.5 + 1.0)
            .collect();
        let mut scalar: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx * 19 + 7) as f64).cos() * 0.25 + idx as f64 * 0.001)
            .collect();
        let mut tiled = scalar.clone();

        for col in col_start..col_end {
            let mut dot = 0.0;
            for (row_offset, &vi) in v.iter().enumerate() {
                dot += vi * scalar[(row_start + row_offset) * cols + col];
            }
            let tau_dot = tau * dot;
            for (row_offset, &vi) in v.iter().enumerate() {
                scalar[(row_start + row_offset) * cols + col] -= vi * tau_dot;
            }
        }

        apply_real_householder_columns(&mut tiled, cols, row_start, col_start, col_end, &v, tau);

        for (idx, (expected, actual)) in scalar.iter().zip(tiled.iter()).enumerate() {
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "tiled reflector changed bits at element {idx}"
            );
        }
    }

    #[test]
    fn qr_real_path_golden_output_digest() {
        let n = 7usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    (n as f64) + (i as f64) * 0.375 + 5.0
                } else {
                    (((i * 17 + j * 29 + 3) % 23) as f64 + 0.5) * 0.125 - 1.25
                };
            }
        }

        let input = make_matrix(n, n, &a);
        let result = eval_qr(&[input], &BTreeMap::new()).unwrap();
        let mut output_bits = Vec::new();
        for value in &result {
            let Value::Tensor(tensor) = value else {
                panic!("QR output must be tensor");
            };
            output_bits.extend(tensor.elements.iter().map(|literal| {
                literal
                    .as_f64()
                    .expect("real QR output must contain f64")
                    .to_bits()
            }));
        }

        assert_eq!(
            fj_test_utils::fixture_id_from_json(&output_bits).unwrap(),
            "6119fc5cf4759d8cdcd9c34d89a79de89d205203730814fc06aa52bf57ff262b"
        );
    }

    #[test]
    fn qr_rejects_scalar() {
        let scalar = Value::Scalar(Literal::from_f64(1.0));
        assert!(eval_qr(&[scalar], &BTreeMap::new()).is_err());
    }

    // ── LU tests ──────────────────────────────────────────────

    #[test]
    fn lu_2x2_identity() {
        let a = make_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = eval_lu(&[a], &BTreeMap::new()).unwrap();
        assert_eq!(result.len(), 3, "LU should return [lu, pivots, perm]");

        let lu = extract_f64_elements(&result[0]);
        assert!((lu[0] - 1.0).abs() < 1e-10, "lu[0,0] = 1");
        assert!((lu[1]).abs() < 1e-10, "lu[0,1] = 0");
        assert!((lu[2]).abs() < 1e-10, "lu[1,0] = 0");
        assert!((lu[3] - 1.0).abs() < 1e-10, "lu[1,1] = 1");
    }

    #[test]
    fn lu_2x2_roundtrip() {
        let a = make_matrix(2, 2, &[4.0, 3.0, 6.0, 3.0]);
        let result = eval_lu(std::slice::from_ref(&a), &BTreeMap::new()).unwrap();
        assert_eq!(result.len(), 3);

        let lu = extract_f64_elements(&result[0]);
        let pivots = match &result[1] {
            Value::Tensor(t) => t
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect::<Vec<_>>(),
            other => {
                assert!(matches!(other, Value::Tensor(_)), "pivots should be tensor");
                Vec::new()
            }
        };
        let perm = match &result[2] {
            Value::Tensor(t) => t
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect::<Vec<_>>(),
            other => {
                assert!(matches!(other, Value::Tensor(_)), "perm should be tensor");
                Vec::new()
            }
        };

        assert_eq!(pivots.len(), 2);
        assert_eq!(perm.len(), 2);

        let l11 = 1.0;
        let l21 = lu[2];
        let u11 = lu[0];
        let u12 = lu[1];
        let u22 = lu[3];

        let p0 = perm[0] as usize;
        let p1 = perm[1] as usize;

        let a_orig = extract_f64_elements(&a);
        let a_perm = [
            a_orig[p0 * 2],
            a_orig[p0 * 2 + 1],
            a_orig[p1 * 2],
            a_orig[p1 * 2 + 1],
        ];

        let reconstructed = [l11 * u11, l11 * u12, l21 * u11, l21 * u12 + u22];

        for i in 0..4 {
            assert!(
                (a_perm[i] - reconstructed[i]).abs() < 1e-10,
                "PA = LU failed at index {}: {} vs {}",
                i,
                a_perm[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn lu_blocked_path_reconstructs_and_matches_scalar() {
        // k = 300 >= LU_BLOCK_THRESHOLD exercises the cache-blocked LU. The
        // blocked factorization must (a) numerically match the scalar kernel and
        // (b) satisfy P·A = L·U to machine precision. Blocked is NOT required to
        // be bit-identical (the GEMM reorders the trailing sum), only equivalent.
        let n = 300usize;
        assert!(n >= LU_BLOCK_THRESHOLD, "n must hit the blocked path");
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for jc in 0..n {
                a[i * n + jc] = if i == jc {
                    (n as f64) + (i as f64) * 0.125 + 3.0
                } else {
                    (((i * 31 + jc * 17 + 5) % 29) as f64 - 14.0) * 0.05
                };
            }
        }

        // (a) Scalar vs blocked factorization agree to tolerance. The matrix is
        //     diagonally dominant, so partial pivoting takes the diagonal in both
        //     and the pivot/permutation vectors are identical.
        let mut a_scalar = a.clone();
        let (piv_s, perm_s) = lu_factor_real_scalar(&mut a_scalar, n, n);
        let mut a_blocked = a.clone();
        let (piv_b, perm_b) = lu_factor_real_blocked(&mut a_blocked, n, n);
        assert_eq!(piv_s, piv_b, "blocked pivots diverge from scalar");
        assert_eq!(perm_s, perm_b, "blocked perm diverges from scalar");
        let mut max_diff = 0.0f64;
        for idx in 0..n * n {
            max_diff = max_diff.max((a_scalar[idx] - a_blocked[idx]).abs());
        }
        assert!(max_diff < 1e-9, "blocked vs scalar LU differ by {max_diff}");

        // (b) Reconstruction P·A = L·U from the blocked output.
        let lu = &a_blocked;
        let perm = &perm_b;
        let mut residual = 0.0f64;
        for i in 0..n {
            for jc in 0..n {
                let mut s = 0.0f64;
                let lim = i.min(jc);
                for t in 0..=lim {
                    let l_it = if t == i { 1.0 } else { lu[i * n + t] };
                    s += l_it * lu[t * n + jc];
                }
                let pa = a[perm[i] as usize * n + jc];
                residual = residual.max((pa - s).abs());
            }
        }
        assert!(
            residual < 1e-7,
            "blocked P·A = L·U residual too large: {residual}"
        );
    }

    #[test]
    fn lu_blocked_path_golden_output_digest() -> Result<(), Box<dyn std::error::Error>> {
        let n = LU_BLOCK_THRESHOLD;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for jc in 0..n {
                a[i * n + jc] = if i == jc {
                    (n as f64) + (i as f64) * 0.25 + 7.0
                } else {
                    (((i * 19 + jc * 23 + 11) % 37) as f64 - 18.0) * 0.03125
                };
            }
        }

        let result = eval_lu(&[make_matrix(n, n, &a)], &BTreeMap::new())?;
        let mut output_words = Vec::new();
        for value in &result {
            let Value::Tensor(tensor) = value else {
                return Err("LU output must be tensors".into());
            };
            match tensor.dtype {
                DType::F64 => {
                    output_words.push(0xf64f_64f6_4f64_f64f);
                    for &literal in &tensor.elements {
                        let Some(value) = literal.as_f64() else {
                            return Err("LU matrix output must contain f64 literals".into());
                        };
                        output_words.push(value.to_bits());
                    }
                }
                DType::I32 => {
                    output_words.push(0x1321_1321_1321_1321);
                    for &literal in &tensor.elements {
                        let Some(value) = literal.as_i64() else {
                            return Err("LU index output must contain integer literals".into());
                        };
                        output_words.push(value as u64);
                    }
                }
                _ => return Err("unexpected LU output dtype".into()),
            }
        }

        let digest = fj_test_utils::fixture_id_from_json(&output_words)?;
        assert_eq!(
            digest, "4015f89e43b02bad7dc3f84df97617fd1d93332a81682e3bada8da779af55a91",
            "blocked LU golden output digest changed"
        );
        Ok(())
    }

    #[test]
    fn lu_real_fast_path_bit_identical_to_complex_zero_imag_path() {
        fn assert_real_lu_matches_complex_zero_imag(m: usize, n: usize, a: Vec<f64>) {
            let real_result = eval_lu_real_matrix(m, n, a.clone(), DType::F64).unwrap();
            let complex_zero_imag = a.iter().map(|&value| (value, 0.0)).collect();
            let reference_result =
                eval_lu_complex_matrix(m, n, complex_zero_imag, DType::F64).unwrap();

            assert_eq!(real_result.len(), reference_result.len());
            for (output_idx, (real_output, reference_output)) in
                real_result.iter().zip(reference_result.iter()).enumerate()
            {
                if let (Value::Tensor(real_tensor), Value::Tensor(reference_tensor)) =
                    (real_output, reference_output)
                {
                    assert_eq!(real_tensor.dtype, reference_tensor.dtype);
                    assert_eq!(real_tensor.shape, reference_tensor.shape);
                    assert_eq!(real_tensor.elements.len(), reference_tensor.elements.len());
                    for idx in 0..real_tensor.elements.len() {
                        assert_eq!(
                            real_tensor.elements[idx], reference_tensor.elements[idx],
                            "output {output_idx} element {idx}"
                        );
                    }
                } else {
                    assert!(
                        matches!(
                            (real_output, reference_output),
                            (Value::Tensor(_), Value::Tensor(_))
                        ),
                        "LU output {output_idx} should be tensors"
                    );
                }
            }
        }

        let (m, n) = (7usize, 5usize);
        let mut pivoting = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                pivoting[i * n + j] = if i == j {
                    (m as f64) + (i as f64) * 0.375 + 1.5
                } else {
                    (((i * 17 + j * 29 + 3) % 23) as f64 + 0.25) * 0.125 - 1.375
                };
            }
        }
        pivoting.swap(0, 2 * n);
        pivoting.swap(1, 2 * n + 1);
        assert_real_lu_matches_complex_zero_imag(m, n, pivoting);

        let adversarial = vec![
            3.0,
            -0.0,
            2.0,
            f64::INFINITY,
            0.0,
            -5.0,
            -0.0,
            -1.25,
            4.0,
            f64::NEG_INFINITY,
            0.0,
            -0.5,
        ];
        assert_real_lu_matches_complex_zero_imag(4, 3, adversarial);
    }

    #[test]
    fn lu_rejects_scalar() {
        let scalar = Value::Scalar(Literal::from_f64(1.0));
        assert!(eval_lu(&[scalar], &BTreeMap::new()).is_err());
    }

    // ── SVD tests ─────────────────────────────────────────────

    #[test]
    fn svd_2x2_identity() {
        let a = make_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = eval_svd(&[a], &BTreeMap::new()).unwrap();
        assert_eq!(result.len(), 3, "SVD should return [U, S, Vt]");

        let u = extract_f64_elements(&result[0]);
        let s = extract_f64_elements(&result[1]);
        let vt = extract_f64_elements(&result[2]);

        // Singular values should be [1, 1]
        assert!((s[0] - 1.0).abs() < 1e-10, "s[0]");
        assert!((s[1] - 1.0).abs() < 1e-10, "s[1]");

        // U and Vt should form orthogonal matrices
        // Verify U diag(S) Vt = A
        let a_data = [1.0, 0.0, 0.0, 1.0];
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                }
                assert!(
                    (val - a_data[i * 2 + j]).abs() < 1e-10,
                    "U*S*Vt[{i},{j}] = {val}, expected {}",
                    a_data[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn svd_2x2_diagonal() {
        let a = make_matrix(2, 2, &[3.0, 0.0, 0.0, -2.0]);
        let result = eval_svd(&[a], &BTreeMap::new()).unwrap();
        let s = extract_f64_elements(&result[1]);

        // Singular values should be [3, 2] (descending, positive)
        assert!((s[0] - 3.0).abs() < 1e-10, "s[0] = {}", s[0]);
        assert!((s[1] - 2.0).abs() < 1e-10, "s[1] = {}", s[1]);

        // Verify roundtrip
        let u = extract_f64_elements(&result[0]);
        let vt = extract_f64_elements(&result[2]);
        let a_data = [3.0, 0.0, 0.0, -2.0];
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                }
                assert!(
                    (val - a_data[i * 2 + j]).abs() < 1e-10,
                    "U*S*Vt[{i},{j}] = {val}, expected {}",
                    a_data[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn svd_rejects_scalar() {
        let scalar = Value::Scalar(Literal::from_f64(1.0));
        assert!(eval_svd(&[scalar], &BTreeMap::new()).is_err());
    }

    #[test]
    fn svd_nan_input_does_not_panic_during_sort() {
        let a = make_matrix(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]);
        let outcome = std::panic::catch_unwind(|| eval_svd(&[a], &BTreeMap::new()));
        assert!(outcome.is_ok(), "SVD eigenvalue sorting should not panic");
    }

    // ── Eigh tests ────────────────────────────────────────────

    #[test]
    fn eigh_2x2_identity() {
        let a = make_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let result = eval_eigh(&[a], &BTreeMap::new()).unwrap();
        assert_eq!(result.len(), 2, "Eigh should return [W, V]");

        let w = extract_f64_elements(&result[0]);
        assert!((w[0] - 1.0).abs() < 1e-10, "w[0]");
        assert!((w[1] - 1.0).abs() < 1e-10, "w[1]");
    }

    #[test]
    fn eigh_2x2_diagonal() {
        // A = [[5, 0], [0, 3]] → eigenvalues [3, 5] ascending
        let a = make_matrix(2, 2, &[5.0, 0.0, 0.0, 3.0]);
        let result = eval_eigh(&[a], &BTreeMap::new()).unwrap();
        let w = extract_f64_elements(&result[0]);
        let v = extract_f64_elements(&result[1]);

        // Eigenvalues ascending
        assert!((w[0] - 3.0).abs() < 1e-10, "w[0] = {}", w[0]);
        assert!((w[1] - 5.0).abs() < 1e-10, "w[1] = {}", w[1]);

        // Verify V diag(W) V^T = A
        let a_data = [5.0, 0.0, 0.0, 3.0];
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += v[i * 2 + k] * w[k] * v[j * 2 + k];
                }
                assert!(
                    (val - a_data[i * 2 + j]).abs() < 1e-10,
                    "V*W*Vt[{i},{j}] = {val}, expected {}",
                    a_data[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn analytic_eigh_3x3_matches_jacobi_general() {
        // Non-diagonal, distinct-eigenvalue 3×3 (the hot batched-eigh shape).
        // The analytic path must (a) succeed, (b) return ascending eigenvalues,
        // (c) be orthonormal, and (d) reconstruct A = V diag(w) Vᵀ — and match
        // the iterative Jacobi eigenvalues, all within 1e-10.
        let a_rows = [2.0, 0.1, 0.0, 0.1, 3.0, 0.2, 0.0, 0.2, 4.0];
        let (w, v) = analytic_eigh_3x3(&a_rows).expect("analytic path should accept");

        // Ascending.
        assert!(w[0] <= w[1] && w[1] <= w[2], "ascending {w:?}");

        // Orthonormality VᵀV = I.
        for i in 0..3 {
            for j in 0..3 {
                let mut d = 0.0;
                for r in 0..3 {
                    d += v[r * 3 + i] * v[r * 3 + j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((d - target).abs() < 1e-10, "VtV[{i},{j}]={d}");
            }
        }

        // Reconstruction V diag(w) Vᵀ = A.
        for i in 0..3 {
            for j in 0..3 {
                let mut val = 0.0;
                for k in 0..3 {
                    val += v[i * 3 + k] * w[k] * v[j * 3 + k];
                }
                assert!(
                    (val - a_rows[i * 3 + j]).abs() < 1e-10,
                    "recon[{i},{j}]={val} exp {}",
                    a_rows[i * 3 + j]
                );
            }
        }

        // Eigenvalues agree with the iterative Jacobi reference.
        let mut jac = a_rows;
        let (mut jw, _) = jacobi_eigendecomposition(&mut jac, 3);
        jw.sort_by(f64::total_cmp);
        for k in 0..3 {
            assert!(
                (w[k] - jw[k]).abs() < 1e-10,
                "eig[{k}] {} vs jacobi {}",
                w[k],
                jw[k]
            );
        }
    }

    /// Deterministic dense real SYMMETRIC n×n matrix with a spread spectrum
    /// (A = QᵀDQ would be ideal but we just build a symmetric matrix directly).
    fn symmetric_test_matrix(n: usize) -> Vec<f64> {
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = if i == j {
                    (i as f64) * 1.7 + 3.0
                } else {
                    (((i * 31 + j * 17 + 5) % 19) as f64 - 9.0) * 0.05
                };
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    #[test]
    fn tridiag_ql_eigh_matches_jacobi_and_reconstructs() {
        // The tridiagonalization + implicit-QL eigh must (a) reconstruct
        // A = V diag(w) Vᵀ, (b) have orthonormal V, and (c) recover the same
        // spectrum (sorted) as the cyclic-Jacobi reference — all to tolerance.
        for &n in &[5usize, 8, 16, 24] {
            let a = symmetric_test_matrix(n);
            let (w, v) = tridiag_ql_eigendecomposition(&a, n);

            // (a) reconstruction V diag(w) Vᵀ = A.
            for i in 0..n {
                for j in 0..n {
                    let mut val = 0.0;
                    for k in 0..n {
                        val += v[i * n + k] * w[k] * v[j * n + k];
                    }
                    assert!(
                        (val - a[i * n + j]).abs() < 1e-9,
                        "n={n} recon[{i},{j}]={val} exp {}",
                        a[i * n + j]
                    );
                }
            }
            // (b) orthonormality VᵀV = I.
            for i in 0..n {
                for j in 0..n {
                    let mut d = 0.0;
                    for r in 0..n {
                        d += v[r * n + i] * v[r * n + j];
                    }
                    let target = if i == j { 1.0 } else { 0.0 };
                    assert!((d - target).abs() < 1e-9, "n={n} VtV[{i},{j}]={d}");
                }
            }
            // (c) spectrum matches the cyclic-Jacobi reference (sorted).
            let mut jac = a.clone();
            let (mut jw, _) = jacobi_eigendecomposition_cyclic(&mut jac, n);
            jw.sort_by(f64::total_cmp);
            let mut ql = w.clone();
            ql.sort_by(f64::total_cmp);
            for k in 0..n {
                assert!(
                    (ql[k] - jw[k]).abs() < 1e-9,
                    "n={n} eig[{k}] QL {} vs Jacobi {}",
                    ql[k],
                    jw[k]
                );
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_eigh_tridiag_ql_vs_jacobi() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let run = |n: usize| {
            let a = symmetric_test_matrix(n);
            let jac = best_time(|| {
                let mut work = a.clone();
                std::hint::black_box(jacobi_eigendecomposition_cyclic(&mut work, n));
            });
            let ql = best_time(|| {
                std::hint::black_box(tridiag_ql_eigendecomposition(&a, n));
            });
            println!(
                "BENCH eigh n={n}: cyclic-Jacobi {:.3}ms -> tridiag+QL {:.3}ms = {:.2}x",
                jac * 1e3,
                ql * 1e3,
                jac / ql
            );
        };
        run(48);
        run(96);
        run(160);
    }

    #[test]
    fn analytic_eigh_3x3_handles_repeated_eigenvalue() {
        // Rotated [2,2,5]: a degenerate 2-D eigenspace. Must still be orthonormal
        // and reconstruct (sign/rotation within the eigenspace is free).
        let a_rows = [2.0, 0.0, 0.0, 0.0, 3.5, 1.5, 0.0, 1.5, 3.5]; // eigs 2,2,5
        let (w, v) = analytic_eigh_3x3(&a_rows).expect("analytic path should accept");
        assert!(
            (w[0] - 2.0).abs() < 1e-10 && (w[1] - 2.0).abs() < 1e-10 && (w[2] - 5.0).abs() < 1e-10,
            "{w:?}"
        );
        for i in 0..3 {
            for j in 0..3 {
                let mut val = 0.0;
                for k in 0..3 {
                    val += v[i * 3 + k] * w[k] * v[j * 3 + k];
                }
                assert!(
                    (val - a_rows[i * 3 + j]).abs() < 1e-10,
                    "recon[{i},{j}]={val}"
                );
            }
        }
    }

    #[test]
    fn eigh_rejects_scalar() {
        let scalar = Value::Scalar(Literal::from_f64(1.0));
        assert!(eval_eigh(&[scalar], &BTreeMap::new()).is_err());
    }

    #[test]
    fn eigh_rejects_non_square() {
        let a = make_matrix(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(eval_eigh(&[a], &BTreeMap::new()).is_err());
    }

    #[test]
    fn eigh_nan_input_does_not_panic_during_sort() {
        let a = make_matrix(2, 2, &[f64::NAN, 0.0, 0.0, 1.0]);
        let outcome = std::panic::catch_unwind(|| eval_eigh(&[a], &BTreeMap::new()));
        assert!(outcome.is_ok(), "Eigh eigenvalue sorting should not panic");
    }

    // ── Determinant tests ───────────────────────────────────────────

    #[test]
    fn det_2x2_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let d = det(&a, 2);
        assert!((d - (-2.0)).abs() < 1e-10, "det = {d}, expected -2");
    }

    #[test]
    fn det_3x3_identity() {
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let d = det(&a, 3);
        assert!((d - 1.0).abs() < 1e-10, "det(I) = {d}, expected 1");
    }

    #[test]
    fn det_singular() {
        let a = [1.0, 2.0, 2.0, 4.0]; // Rows are linearly dependent
        let d = det(&a, 2);
        assert!(d.abs() < 1e-10, "det of singular matrix should be ~0");
    }

    #[test]
    fn det_small_pivot_not_collapsed_to_zero() {
        // diag(1e-16, 2, 3): a small but nonzero pivot -> small nonzero det.
        // jnp.linalg.det returns 6e-16; the old `< 1e-15` pivot threshold wrongly
        // collapsed it to 0.
        let a = [1e-16, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let d = det(&a, 3);
        assert!(d != 0.0, "small-pivot determinant must not collapse to 0");
        assert!(
            (d / 6e-16 - 1.0).abs() < 1e-9,
            "det should be ~6e-16, got {d}"
        );

        // An exactly-zero pivot column is structurally singular -> det 0 (no NaN).
        let z = [0.0, 1.0, 0.0, 1.0]; // column 0 is all-zero
        let dz = det(&z, 2);
        assert_eq!(
            dz, 0.0,
            "structurally singular det must be exactly 0, got {dz}"
        );
    }

    #[test]
    fn det_1x1() {
        let a = [5.0];
        let d = det(&a, 1);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn det_empty() {
        let d = det(&[], 0);
        assert!((d - 1.0).abs() < 1e-10, "det of empty matrix is 1");
    }

    #[test]
    fn slogdet_positive() {
        let a = [2.0, 0.0, 0.0, 3.0];
        let (sign, logabsdet) = slogdet(&a, 2);
        assert!((sign - 1.0).abs() < 1e-10);
        assert!((logabsdet - (6.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn slogdet_negative() {
        let a = [1.0, 2.0, 3.0, 4.0]; // det = -2
        let (sign, logabsdet) = slogdet(&a, 2);
        assert!((sign - (-1.0)).abs() < 1e-10);
        assert!((logabsdet - (2.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn slogdet_sign_from_negative_diagonal() {
        // Odd count of negative diagonal entries => sign -1 (det = -6).
        let a = [-2.0, 0.0, 0.0, 3.0];
        let (sign, logabsdet) = slogdet(&a, 2);
        assert!((sign - (-1.0)).abs() < 1e-10);
        assert!((logabsdet - (6.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn slogdet_no_overflow_large_determinant() {
        // diag(100) over 200 dims => det = 100^200 = 1e400, which OVERFLOWS f64.
        // Forming det() then log() yields +inf; slogdet must accumulate logs and
        // return the finite 200*ln(100), matching jnp.linalg.slogdet.
        let n = 200;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 100.0;
        }
        let (sign, logabsdet) = slogdet(&a, n);
        assert_eq!(sign, 1.0);
        assert!(
            logabsdet.is_finite(),
            "logabsdet must be finite, got {logabsdet}"
        );
        let expected = n as f64 * 100.0_f64.ln();
        assert!((logabsdet - expected).abs() < 1e-6);
        // The naive log(det) path would have failed: det() overflows to +inf.
        assert!(det(&a, n).is_infinite());
    }

    #[test]
    fn slogdet_no_underflow_tiny_determinant() {
        // diag(0.01) over 200 dims => det = 1e-400 UNDERFLOWS to 0; slogdet must
        // return the finite 200*ln(0.01).
        let n = 200;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 0.01;
        }
        let (sign, logabsdet) = slogdet(&a, n);
        assert_eq!(sign, 1.0);
        let expected = n as f64 * 0.01_f64.ln();
        assert!((logabsdet - expected).abs() < 1e-6);
        // The naive log(det) path would have failed: det() underflows to 0.
        assert_eq!(det(&a, n), 0.0);
    }

    /// Naive scalar reference for the blocked-LU det path: BLAS-2 right-looking
    /// elimination, the exact algorithm `det` runs below `LU_BLOCK_THRESHOLD`.
    fn naive_det_ref(a: &[f64], n: usize) -> f64 {
        let mut lu = a.to_vec();
        let mut sign = 1.0_f64;
        for k in 0..n {
            let mut max_val = lu[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            if max_val == 0.0 {
                return 0.0;
            }
            if max_row != k {
                for j in 0..n {
                    lu.swap(k * n + j, max_row * n + j);
                }
                sign = -sign;
            }
            for i in (k + 1)..n {
                let factor = lu[i * n + k] / lu[k * n + k];
                for j in (k + 1)..n {
                    lu[i * n + j] -= factor * lu[k * n + j];
                }
            }
        }
        let mut result = sign;
        for i in 0..n {
            result *= lu[i * n + i];
        }
        result
    }

    #[test]
    fn blocked_det_matches_naive_within_tolerance() {
        // n ≥ LU_BLOCK_THRESHOLD routes `det`/`slogdet` through the cache-blocked
        // GEMM LU. The block-reordered Schur update is numerically equivalent to
        // the scalar elimination (P·A = L·U to machine precision) but not bit-
        // identical, so verify the determinant and its sign+log agree to tolerance.
        let n = 300usize;
        assert!(n >= LU_BLOCK_THRESHOLD, "must exercise the blocked path");
        // Strongly diagonally-dominant, well-conditioned matrix whose determinant
        // (≈ product of diagonals in [1.0, 1.3]) stays finite — the dominant-
        // diagonal solve_test_system would give det ≈ 300^300 = +inf.
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (((i * 131 + j * 17 + 7) % 13) as f64 - 6.0) * 1e-5;
            }
            a[i * n + i] = 1.0 + ((i % 7) as f64) * 0.05;
        }

        let d_blocked = det(&a, n);
        let d_ref = naive_det_ref(&a, n);
        assert!(
            d_blocked.is_finite() && d_ref != 0.0,
            "det must be finite/nonzero"
        );
        // Block-reordered GEMM Schur update is numerically equivalent, agreeing to ~ulp.
        assert!(
            (d_blocked / d_ref - 1.0).abs() < 1e-9,
            "blocked det {d_blocked} vs naive {d_ref}"
        );

        let (sign_b, log_b) = slogdet(&a, n);
        let sign_ref = d_ref.signum();
        assert_eq!(sign_b, sign_ref, "slogdet sign mismatch");
        let log_ref = d_ref.abs().ln();
        assert!(
            (log_b - log_ref).abs() < 1e-9,
            "blocked logabsdet {log_b} vs naive {log_ref}"
        );
    }

    #[test]
    fn small_det_stays_bit_identical_to_naive() {
        // Below the threshold, `det` must be the SAME unblocked elimination the
        // conformance goldens pin — bit-for-bit identical to the scalar reference.
        let n = 64usize;
        assert!(n < LU_BLOCK_THRESHOLD);
        let (a, _b) = solve_test_system(n);
        assert_eq!(
            det(&a, n).to_bits(),
            naive_det_ref(&a, n).to_bits(),
            "small-n det must stay bit-identical to the scalar kernel"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_det_blocked_vs_naive() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let run = |n: usize| {
            let (a, _b) = solve_test_system(n);
            let naive = best_time(|| {
                std::hint::black_box(naive_det_ref(std::hint::black_box(&a), n));
            });
            let blocked = best_time(|| {
                std::hint::black_box(det(std::hint::black_box(&a), n));
            });
            println!(
                "BENCH det n={n}: naive-LU {:.3}ms -> blocked-GEMM-LU {:.3}ms = {:.2}x",
                naive * 1e3,
                blocked * 1e3,
                naive / blocked
            );
        };
        run(512);
        run(1024);
    }

    // ── Blocked complex LU (solve/det) tests ────────────────────────

    /// Well-conditioned complex system with a strongly dominant real diagonal.
    #[allow(clippy::type_complexity)]
    fn complex_solve_test_system(n: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let mut a = vec![(0.0f64, 0.0f64); n * n];
        for i in 0..n {
            for j in 0..n {
                let re = (((i * 131 + j * 17 + 7) % 1000) as f64 / 500.0 - 1.0) * 0.01;
                let im = (((i * 43 + j * 91 + 3) % 1000) as f64 / 500.0 - 1.0) * 0.01;
                a[i * n + j] = (re, im);
            }
            a[i * n + i] = (n as f64, 0.0);
        }
        let b: Vec<(f64, f64)> = (0..n)
            .map(|i| (((i * 13 % 97) as f64) - 48.0, ((i * 7 % 53) as f64) - 26.0))
            .collect();
        (a, b)
    }

    /// Naive augmented-elimination complex solve — the exact algorithm
    /// `complex_solve_system` runs below `LU_BLOCK_THRESHOLD`.
    fn naive_complex_solve_ref(
        a: &[(f64, f64)],
        b: &[(f64, f64)],
        n: usize,
        ncols: usize,
    ) -> Vec<(f64, f64)> {
        let w = n + ncols;
        let mut m = vec![(0.0_f64, 0.0_f64); n * w];
        for i in 0..n {
            m[i * w..i * w + n].copy_from_slice(&a[i * n..i * n + n]);
            m[i * w + n..i * w + w].copy_from_slice(&b[i * ncols..i * ncols + ncols]);
        }
        for col in 0..n {
            let mut piv = col;
            let mut best = complex_abs(m[col * w + col]);
            for r in (col + 1)..n {
                let mag = complex_abs(m[r * w + col]);
                if mag > best {
                    best = mag;
                    piv = r;
                }
            }
            if piv != col {
                for c in 0..w {
                    m.swap(col * w + c, piv * w + c);
                }
            }
            let pivot = m[col * w + col];
            for r in (col + 1)..n {
                let factor = complex_div(m[r * w + col], pivot);
                for c in col..w {
                    m[r * w + c] = complex_sub(m[r * w + c], complex_mul(factor, m[col * w + c]));
                }
            }
        }
        let mut x = vec![(0.0_f64, 0.0_f64); n * ncols];
        for jcol in 0..ncols {
            for row in (0..n).rev() {
                let mut s = m[row * w + n + jcol];
                for c in (row + 1)..n {
                    s = complex_sub(s, complex_mul(m[row * w + c], x[c * ncols + jcol]));
                }
                x[row * ncols + jcol] = complex_div(s, m[row * w + row]);
            }
        }
        x
    }

    #[test]
    fn blocked_complex_solve_matches_naive_within_tolerance_and_residual() {
        // n ≥ LU_BLOCK_THRESHOLD routes `complex_solve_system` through the cache-
        // blocked complex GEMM LU. Verify it (a) reconstructs b and (b) matches the
        // naive augmented-elimination reference to tolerance (P·A=L·U, ulp-level).
        let n = 300usize;
        assert!(n >= LU_BLOCK_THRESHOLD, "must exercise the blocked path");
        let ncols = 3usize;
        let (a, bvec) = complex_solve_test_system(n);
        // matrix RHS: ncols columns built from the vector b plus a shift.
        let mut b = vec![(0.0f64, 0.0f64); n * ncols];
        for i in 0..n {
            for j in 0..ncols {
                b[i * ncols + j] = (bvec[i].0 + j as f64, bvec[i].1 - j as f64);
            }
        }

        let x = complex_solve_system(&a, &b, n, ncols);

        // (a) residual ‖A·x − b‖_∞.
        let mut max_res = 0.0f64;
        for i in 0..n {
            for j in 0..ncols {
                let mut s = (0.0f64, 0.0f64);
                for k in 0..n {
                    s = complex_add(s, complex_mul(a[i * n + k], x[k * ncols + j]));
                }
                let d = complex_sub(s, b[i * ncols + j]);
                max_res = max_res.max(complex_abs(d));
            }
        }
        assert!(max_res < 1e-9, "complex residual too large: {max_res}");

        // (b) vs naive augmented elimination.
        let xref = naive_complex_solve_ref(&a, &b, n, ncols);
        let mut max_diff = 0.0f64;
        for i in 0..(n * ncols) {
            max_diff = max_diff.max(complex_abs(complex_sub(x[i], xref[i])));
        }
        assert!(
            max_diff < 1e-9,
            "blocked vs naive complex solve diff: {max_diff}"
        );
    }

    #[test]
    fn blocked_complex_det_matches_naive_within_tolerance() {
        // Strongly diagonally-dominant complex matrix whose determinant stays
        // finite (diagonals in [1.0, 1.3], tiny off-diagonals).
        let n = 300usize;
        assert!(n >= LU_BLOCK_THRESHOLD);
        let mut a = vec![(0.0f64, 0.0f64); n * n];
        for i in 0..n {
            for j in 0..n {
                let re = (((i * 131 + j * 17 + 7) % 13) as f64 - 6.0) * 1e-5;
                let im = (((i * 29 + j * 53 + 1) % 11) as f64 - 5.0) * 1e-5;
                a[i * n + j] = (re, im);
            }
            a[i * n + i] = (1.0 + ((i % 7) as f64) * 0.05, ((i % 5) as f64) * 0.01);
        }

        let d_blocked = complex_det(&a, n);
        // Naive scalar complex-LU det reference (the < threshold path).
        let mut m = a.clone();
        let mut d_ref = (1.0_f64, 0.0_f64);
        for col in 0..n {
            let mut piv = col;
            let mut best = complex_abs(m[col * n + col]);
            for r in (col + 1)..n {
                let mag = complex_abs(m[r * n + col]);
                if mag > best {
                    best = mag;
                    piv = r;
                }
            }
            if piv != col {
                for c in 0..n {
                    m.swap(col * n + c, piv * n + c);
                }
                d_ref = (-d_ref.0, -d_ref.1);
            }
            let pivot = m[col * n + col];
            d_ref = complex_mul(d_ref, pivot);
            for r in (col + 1)..n {
                let factor = complex_div(m[r * n + col], pivot);
                for c in (col + 1)..n {
                    m[r * n + c] = complex_sub(m[r * n + c], complex_mul(factor, m[col * n + c]));
                }
            }
        }
        let rel = complex_abs(complex_sub(d_blocked, d_ref)) / complex_abs(d_ref).max(1e-300);
        assert!(
            rel < 1e-9,
            "blocked complex det {d_blocked:?} vs naive {d_ref:?} rel {rel}"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_complex_solve_blocked_vs_naive() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let run = |n: usize| {
            let (a, bvec) = complex_solve_test_system(n);
            let b: Vec<(f64, f64)> = bvec.clone();
            let naive = best_time(|| {
                std::hint::black_box(naive_complex_solve_ref(std::hint::black_box(&a), &b, n, 1));
            });
            let blocked = best_time(|| {
                let mut lu = a.clone();
                let perm = complex_lu_factor_blocked(&mut lu, n);
                std::hint::black_box(complex_lu_solve(&lu, &perm, &b, n, 1));
            });
            println!(
                "BENCH complex solve n={n}: naive-LU {:.3}ms -> blocked-GEMM-LU {:.3}ms = {:.2}x",
                naive * 1e3,
                blocked * 1e3,
                naive / blocked
            );
        };
        run(512);
        run(1024);
    }

    // ── Inverse tests ───────────────────────────────────────────────

    #[test]
    fn inv_2x2_basic() {
        let a = [4.0, 7.0, 2.0, 6.0];
        let inv_a = inv(&a, 2).expect("should be invertible");
        // Verify A * A^{-1} = I
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += a[i * 2 + k] * inv_a[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < 1e-10, "A*A^-1[{i},{j}] = {val}");
            }
        }
    }

    #[test]
    fn inv_singular_returns_none() {
        let a = [1.0, 2.0, 2.0, 4.0];
        assert!(inv(&a, 2).is_none());
    }

    #[test]
    fn inv_identity() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let inv_a = inv(&a, 2).expect("identity invertible");
        assert!((inv_a[0] - 1.0).abs() < 1e-10);
        assert!(inv_a[1].abs() < 1e-10);
        assert!(inv_a[2].abs() < 1e-10);
        assert!((inv_a[3] - 1.0).abs() < 1e-10);
    }

    // ── Pseudoinverse tests ─────────────────────────────────────────

    #[test]
    fn pinv_svd_column_major_reconstruction_matches_row_major_bits() {
        let (m, n) = (6usize, 4usize);
        let a: Vec<f64> = (0..m * n)
            .map(|idx| {
                let x = idx as f64;
                (x * 0.125).sin() + (x * 0.03125).cos()
            })
            .collect();
        let rcond = 1e-15;
        let k = m.min(n);
        let (sigma, u, v) = one_sided_jacobi_svd_real(m, n, &a);
        let sigma_max = sigma.first().copied().unwrap_or(0.0);
        let cutoff = rcond * sigma_max;

        let mut row_major_result = vec![0.0_f64; n * m];
        for l in 0..k {
            let sg = sigma[l];
            if sg > cutoff && sg > 0.0 {
                let inv = 1.0 / sg;
                for i in 0..n {
                    let vil = v[i * n + l];
                    if vil == 0.0 {
                        continue;
                    }
                    let vil_inv = vil * inv;
                    for j in 0..m {
                        row_major_result[i * m + j] += vil_inv * u[j * k + l];
                    }
                }
            }
        }

        let mut u_cols = vec![0.0_f64; k * m];
        for l in 0..k {
            for j in 0..m {
                u_cols[l * m + j] = u[j * k + l];
            }
        }
        let mut col_major_result = vec![0.0_f64; n * m];
        for l in 0..k {
            let sg = sigma[l];
            if sg > cutoff && sg > 0.0 {
                let inv = 1.0 / sg;
                for i in 0..n {
                    let vil = v[i * n + l];
                    if vil == 0.0 {
                        continue;
                    }
                    let vil_inv = vil * inv;
                    for j in 0..m {
                        col_major_result[i * m + j] += vil_inv * u_cols[l * m + j];
                    }
                }
            }
        }

        for (idx, (&old, &new)) in row_major_result.iter().zip(&col_major_result).enumerate() {
            assert_eq!(
                old.to_bits(),
                new.to_bits(),
                "pinv reconstruction bit drift at {idx}: old={old:?} new={new:?}"
            );
        }
    }

    #[test]
    fn pinv_low_rank_profile_shape_golden_digest() -> Result<(), Box<dyn std::error::Error>> {
        let (m, n) = (256usize, 128usize);
        let a: Vec<f64> = (0..m * n)
            .map(|idx| {
                let x = idx as f64;
                (x * 0.125).sin() + (x * 0.03125).cos()
            })
            .collect();

        let got = pinv(&a, m, n, 1e-15);
        assert_eq!(got.len(), n * m);
        let bits: Vec<u64> = got.iter().map(|value| value.to_bits()).collect();
        let digest = fj_test_utils::fixture_id_from_json(&bits)?;
        assert_eq!(
            digest, "aa6633cd1c02444eca8fa95d72870710cafc7dbc9503e9c4056640f3118d492f",
            "profile-shaped low-rank pinv output digest changed"
        );
        Ok(())
    }

    #[test]
    fn pinv_ill_conditioned_2x2_svd_accurate() {
        // A = U Σ Vᵀ with Σ = diag(1, 1e-6) and non-axis-aligned rotations, so the
        // small singular value lives in a rotated direction. The SVD-based pinv recovers
        // A⁺ = V Σ⁻¹ Uᵀ to full relative accuracy. The former AᵀA-Gram path lost ~√ε in
        // the small σ: forming AᵀA perturbs σ_min²=1e-12 by ~ε ⇒ ~2e-4 relative error in
        // σ_min ⇒ ~200 absolute error in the ~1e6 pinv entries — which a 1e-7 relative
        // bound rejects. Golden proof of the one-sided-Jacobi accuracy propagating to
        // pinv/lstsq (frankenjax-96i7w follow-on).
        let (th, ph) = (0.6_f64, 1.1_f64);
        let (ct, st) = (th.cos(), th.sin());
        let (cp, sp) = (ph.cos(), ph.sin());
        let vmat = [ct, -st, st, ct]; // V (columns are right singular vectors)
        let umat = [cp, -sp, sp, cp]; // U
        let sig = [1.0_f64, 1e-6_f64];
        let mut a = [0.0_f64; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for l in 0..2 {
                    acc += umat[i * 2 + l] * sig[l] * vmat[j * 2 + l];
                }
                a[i * 2 + j] = acc;
            }
        }
        // Analytic A⁺ = V Σ⁻¹ Uᵀ.
        let mut expected = [0.0_f64; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for l in 0..2 {
                    acc += vmat[i * 2 + l] * (1.0 / sig[l]) * umat[j * 2 + l];
                }
                expected[i * 2 + j] = acc;
            }
        }
        let got = pinv(&a, 2, 2, 2.0 * f64::EPSILON);
        for idx in 0..4 {
            let rel = (got[idx] - expected[idx]).abs() / expected[idx].abs().max(1.0);
            assert!(
                rel < 1e-7,
                "pinv[{idx}] rel err {rel}: got {}, expected {}",
                got[idx],
                expected[idx]
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_pinv_reconstruction_scalar_vs_matmul() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // The pinv reconstruction A⁺ = V_scaled (n×kk) · u_kept (kk×m): isolate the
        // scalar outer-product triple loop vs matmul_2d on the same data.
        for &(n, m, kk) in &[(1024usize, 1024, 1024), (1500, 800, 800)] {
            let v_scaled: Vec<f64> = (0..n * kk)
                .map(|i| (i as f64 * 0.0007).sin() - 0.3)
                .collect();
            let u_kept: Vec<f64> = (0..kk * m)
                .map(|i| (i as f64 * 0.0009).cos() + 0.2)
                .collect();
            let t_matmul = best_time(|| {
                std::hint::black_box(crate::tensor_contraction::matmul_2d(
                    &v_scaled, n, kk, &u_kept, m,
                ));
            });
            let t_scalar = best_time(|| {
                let mut result = vec![0.0f64; n * m];
                for c in 0..kk {
                    for i in 0..n {
                        let vic = v_scaled[i * kk + c];
                        if vic == 0.0 {
                            continue;
                        }
                        for j in 0..m {
                            result[i * m + j] += vic * u_kept[c * m + j];
                        }
                    }
                }
                std::hint::black_box(result);
            });
            println!(
                "BENCH pinv reconstruction n={n} m={m} kk={kk}: scalar {:.1}ms -> matmul {:.1}ms = {:.2}x",
                t_scalar * 1e3,
                t_matmul * 1e3,
                t_scalar / t_matmul
            );
        }
    }

    #[test]
    fn pinv_square_invertible() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let p = pinv(&a, 2, 2, 1e-15);
        // A * A^+ * A ≈ A for pseudoinverse
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    for l in 0..2 {
                        val += a[i * 2 + k] * p[k * 2 + l] * a[l * 2 + j];
                    }
                }
                assert!(
                    (val - a[i * 2 + j]).abs() < 1e-8,
                    "A*A^+*A[{i},{j}] = {val}, expected {}",
                    a[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn pinv_tall_matrix() {
        // 3x2 matrix
        let a = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let p = pinv(&a, 3, 2, 1e-15);
        assert_eq!(p.len(), 2 * 3); // n x m
    }

    fn low_rank_square_matrix(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let u0 = 1.0 + (i as f64) * 0.125;
                let u1 = ((i * 7 + 3) % 13) as f64 - 6.0;
                let v0 = 0.5 + (j as f64) * 0.0625;
                let v1 = ((j * 5 + 1) % 17) as f64 - 8.0;
                a[i * n + j] = u0 * v0 + u1 * v1 * 0.01;
            }
        }
        a
    }

    fn test_matmul(a: &[f64], rows: usize, inner: usize, b: &[f64], cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..inner {
                    sum += a[i * inner + k] * b[k * cols + j];
                }
                out[i * cols + j] = sum;
            }
        }
        out
    }

    fn assert_matrix_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "matrix[{idx}] = {actual}, expected {expected}"
            );
        }
    }

    fn assert_symmetric(matrix: &[f64], n: usize, tolerance: f64) {
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (matrix[i * n + j] - matrix[j * n + i]).abs() <= tolerance,
                    "matrix must be symmetric at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn pinv_low_rank_qr_satisfies_moore_penrose_identities() {
        let n = PINV_LOW_RANK_MIN_DIM;
        let a = low_rank_square_matrix(n);
        let fast = pinv_m_ge_n_low_rank_qr(&a, n, n, 1e-12).expect("low-rank certificate passes");
        let public = pinv(&a, n, n, 1e-12);

        assert_eq!(public, fast, "public pinv must return the certified path");

        let a_p = test_matmul(&a, n, n, &fast, n);
        let p_a = test_matmul(&fast, n, n, &a, n);
        let a_p_a = test_matmul(&a_p, n, n, &a, n);
        let p_a_p = test_matmul(&p_a, n, n, &fast, n);

        assert_matrix_close(&a_p_a, &a, 1e-8);
        assert_matrix_close(&p_a_p, &fast, 1e-8);
        assert_symmetric(&a_p, n, 1e-8);
        assert_symmetric(&p_a, n, 1e-8);
    }

    #[test]
    fn pinv_low_rank_qr_rejects_full_rank_matrix() {
        let n = PINV_LOW_RANK_MIN_DIM;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    (n as f64) + 3.0 + (i as f64) * 0.125
                } else {
                    (((i * 17 + j * 31) % 11) as f64 - 5.0) * 0.01
                };
            }
        }

        assert!(
            pinv_m_ge_n_low_rank_qr(&a, n, n, 1e-12).is_none(),
            "full-rank input must fall back to the Gram path"
        );
    }

    #[test]
    fn pinv_low_rank_qr_rejects_nan_rcond() {
        let a = [1.0, 0.0, 0.0, 1.0];
        assert!(pinv_m_ge_n_low_rank_qr(&a, 2, 2, f64::NAN).is_none());

        let p = pinv(&a, 2, 2, f64::NAN);
        assert!(
            p.iter().all(|value| *value == 0.0),
            "current Gram semantics with NaN rcond retain no singular values"
        );
    }

    #[test]
    fn pinv_rank_deficient_satisfies_moore_penrose() {
        // Rank-1 3x2 matrix (column 2 = 2 * column 1). AᵀA is singular, so the
        // old (AᵀA)⁻¹Aᵀ form returned all-zeros; the SVD/eigendecomposition pinv
        // returns the true Moore-Penrose inverse, which must satisfy A A⁺ A = A.
        let m = 3;
        let n = 2;
        let a = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]; // rows: (1,2),(2,4),(3,6)
        let p = pinv(&a, m, n, 1e-12);
        assert_eq!(p.len(), n * m);

        // A A⁺ A == A  (Moore-Penrose identity 1).
        for i in 0..m {
            for j in 0..n {
                // (A A⁺)[i][l] = Σ_k A[i][k] A⁺[k][l]; then · A[l][j].
                let mut val = 0.0;
                for l in 0..m {
                    let mut aap = 0.0;
                    for k in 0..n {
                        aap += a[i * n + k] * p[k * m + l];
                    }
                    val += aap * a[l * n + j];
                }
                assert!(
                    (val - a[i * n + j]).abs() < 1e-9,
                    "A A+ A[{i},{j}] = {val}, expected {}",
                    a[i * n + j]
                );
            }
        }
        // And the pseudoinverse is non-trivial (the old path returned all-zeros).
        assert!(
            p.iter().any(|&x| x.abs() > 1e-6),
            "pinv must not be all-zeros"
        );
    }

    // ── Norm tests ──────────────────────────────────────────────────

    #[test]
    fn vector_norm_l2() {
        let x = [3.0, 4.0];
        let n = vector_norm(&x, 2.0);
        assert!((n - 5.0).abs() < 1e-10);
    }

    #[test]
    fn vector_norm_l1() {
        let x = [-3.0, 4.0];
        let n = vector_norm(&x, 1.0);
        assert!((n - 7.0).abs() < 1e-10);
    }

    #[test]
    fn vector_norm_linf() {
        let x = [-3.0, 4.0, -5.0];
        let n = vector_norm(&x, f64::INFINITY);
        assert!((n - 5.0).abs() < 1e-10);
    }

    #[test]
    fn frobenius_norm_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let n = frobenius_norm(&a);
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!((n - expected).abs() < 1e-10);
    }

    #[test]
    fn matrix_norm_1_basic() {
        let a = [1.0, -2.0, 3.0, 4.0];
        let n = matrix_norm_1(&a, 2, 2);
        // Column sums: |1|+|3|=4, |-2|+|4|=6 → max = 6
        assert!((n - 6.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_norm_inf_basic() {
        let a = [1.0, -2.0, 3.0, 4.0];
        let n = matrix_norm_inf(&a, 2, 2);
        // Row sums: |1|+|-2|=3, |3|+|4|=7 → max = 7
        assert!((n - 7.0).abs() < 1e-10);
    }

    // ── Solve tests ─────────────────────────────────────────────────

    #[test]
    fn solve_2x2_identity() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 4.0];
        let x = solve(&a, &b, 2).expect("identity solvable");
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn solve_2x2_general() {
        // A = [[2, 1], [1, 3]], b = [5, 7]
        // Solution: x = [8/5, 9/5] = [1.6, 1.8]
        let a = [2.0, 1.0, 1.0, 3.0];
        let b = [5.0, 7.0];
        let x = solve(&a, &b, 2).expect("solvable");
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn solve_3x3() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        // b = [14, 32, 53]
        // Ax = b has solution x = [1, 2, 3]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let b = [14.0, 32.0, 53.0];
        let x = solve(&a, &b, 3).expect("solvable");
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 2.0).abs() < 1e-8);
        assert!((x[2] - 3.0).abs() < 1e-8);
    }

    // Deterministic, diagonally-dominant (well-conditioned) n×n system + RHS.
    fn solve_test_system(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = (((i * 131 + j * 17 + 7) % 1000) as f64 / 500.0) - 1.0;
            }
            a[i * n + i] += n as f64; // dominant diagonal → well-conditioned
        }
        let b: Vec<f64> = (0..n).map(|i| ((i * 13 % 97) as f64) - 48.0).collect();
        (a, b)
    }

    #[test]
    fn blocked_solve_matches_naive_within_tolerance_and_residual() {
        // n ≥ LU_BLOCK_THRESHOLD routes `solve` through the cache-blocked GEMM LU.
        // Verify the blocked solution (a) reconstructs b with a tiny residual and
        // (b) matches the naive scalar-LU reference to tolerance — numerically
        // equivalent (P·A = L·U to machine precision), differing only at ulp level.
        let n = 300usize;
        assert!(n >= LU_BLOCK_THRESHOLD, "must exercise the blocked path");
        let (a, b) = solve_test_system(n);

        let x = solve(&a, &b, n).expect("blocked solve");

        // (a) residual ‖Ax − b‖_∞
        let mut max_res = 0.0f64;
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += a[i * n + j] * x[j];
            }
            max_res = max_res.max((s - b[i]).abs());
        }
        assert!(max_res < 1e-9, "residual too large: {max_res}");

        // (b) vs naive scalar-LU reference (call the unblocked kernel directly).
        let (lu, p) = lu_factor(&a, n).expect("naive lu");
        let xref = lu_solve(&lu, &p, &b, n);
        let mut max_diff = 0.0f64;
        for i in 0..n {
            max_diff = max_diff.max((x[i] - xref[i]).abs());
        }
        assert!(max_diff < 1e-9, "blocked vs naive solve diff: {max_diff}");
    }

    #[test]
    fn small_solve_stays_bit_identical_to_naive_lu() {
        // Below the threshold, `solve` must be the SAME unblocked factorization the
        // conformance goldens pin — bit-for-bit identical to the naive lu_factor path.
        let n = 64usize;
        assert!(n < LU_BLOCK_THRESHOLD);
        let (a, b) = solve_test_system(n);
        let x = solve(&a, &b, n).expect("solve");
        let (lu, p) = lu_factor(&a, n).expect("naive lu");
        let xref = lu_solve(&lu, &p, &b, n);
        for i in 0..n {
            assert_eq!(
                x[i].to_bits(),
                xref[i].to_bits(),
                "small-n solve must stay bit-identical at {i}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_solve_blocked_lu_vs_naive() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let run = |n: usize| {
            let (a, b) = solve_test_system(n);
            let naive = best_time(|| {
                let (lu, p) = lu_factor(&a, n).unwrap();
                std::hint::black_box(lu_solve(&lu, &p, &b, n));
            });
            let blocked = best_time(|| {
                let mut lu = a.clone();
                let (_piv, perm) = lu_factor_real_blocked(&mut lu, n, n);
                let p: Vec<usize> = perm.iter().map(|&r| r as usize).collect();
                std::hint::black_box(lu_solve(&lu, &p, &b, n));
            });
            println!(
                "BENCH solve n={n}: naive-LU {:.3}ms -> blocked-GEMM-LU {:.3}ms = {:.2}x",
                naive * 1e3,
                blocked * 1e3,
                naive / blocked
            );
        };
        run(512);
        run(1024);
    }

    #[test]
    fn solve_singular_returns_non_finite_not_none() {
        // JAX's solve does not raise/fail for a singular A — its LU divides
        // through the zero pivot and yields inf/NaN (NumPy raises LinAlgError).
        let a = [1.0, 2.0, 2.0, 4.0]; // singular (row2 = 2*row1)
        let b = [3.0, 6.0];
        let x = solve(&a, &b, 2).expect("solve must not fail for a singular matrix");
        assert!(
            x.iter().any(|v| !v.is_finite()),
            "singular solve must be non-finite, got {x:?}"
        );
    }

    #[test]
    fn solve_verifies_ax_eq_b() {
        let a = [3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let x = solve(&a, &b, 3).expect("solvable");
        // Verify A*x = b
        for i in 0..3 {
            let mut row_sum = 0.0;
            for j in 0..3 {
                row_sum += a[i * 3 + j] * x[j];
            }
            assert!(
                (row_sum - b[i]).abs() < 1e-8,
                "A*x[{i}] = {row_sum}, expected {}",
                b[i]
            );
        }
    }

    #[test]
    fn eval_solve_preserves_f32_dtype() {
        // eval_solve previously emitted F64Bits literals under the declared
        // output_dtype, so an F32 system produced a tensor whose declared
        // dtype (F32) disagreed with its literals. Pin dtype-consistent F32.
        let a = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_f32(2.0),
                    Literal::from_f32(1.0),
                    Literal::from_f32(1.0),
                    Literal::from_f32(3.0),
                ],
            )
            .unwrap(),
        );
        let b = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: vec![2] },
                vec![Literal::from_f32(5.0), Literal::from_f32(7.0)],
            )
            .unwrap(),
        );
        let out = eval_solve(&[a, b], &BTreeMap::new()).expect("solve");
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32, "must preserve F32, not widen to F64");
                t.validate_dtype_consistency()
                    .expect("declared dtype must match element literal kinds");
                let x: Vec<f32> = t
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::F32Bits(bits) => f32::from_bits(*bits),
                        other => panic!("element not F32: {other:?}"),
                    })
                    .collect();
                // A x = b -> x = [1.6, 1.8]
                assert!((x[0] - 1.6).abs() < 1e-5, "x0 = {}", x[0]);
                assert!((x[1] - 1.8).abs() < 1e-5, "x1 = {}", x[1]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn eval_cholesky_preserves_f32_dtype() {
        // Cholesky factors in f64 internally but must emit the input's dtype.
        // SPD f32 A = [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]]. Pin dtype-consistent F32
        // (regression guard for the f32 linalg-forward family — matrix_to_value +
        // linalg_literal_from_f64).
        let a = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_f32(4.0),
                    Literal::from_f32(2.0),
                    Literal::from_f32(2.0),
                    Literal::from_f32(3.0),
                ],
            )
            .unwrap(),
        );
        let out = eval_cholesky(&[a], &BTreeMap::new()).expect("cholesky");
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32, "cholesky must preserve F32, not widen");
                t.validate_dtype_consistency()
                    .expect("declared dtype must match element literal kinds");
                let l: Vec<f32> = t
                    .elements
                    .iter()
                    .map(|x| match x {
                        Literal::F32Bits(b) => f32::from_bits(*b),
                        other => panic!("element not F32: {other:?}"),
                    })
                    .collect();
                assert!((l[0] - 2.0).abs() < 1e-5, "L00={}", l[0]);
                assert!((l[2] - 1.0).abs() < 1e-5, "L10={}", l[2]);
                assert!(
                    (l[3] - std::f32::consts::SQRT_2).abs() < 1e-5,
                    "L11={}",
                    l[3]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn eval_qr_svd_preserve_f32_dtype() {
        // QR (Q,R) and SVD (U,S,Vt) of an f32 matrix must return f32 outputs (JAX:
        // f32 in -> f32 out). Completes the f32 linalg-forward family dtype coverage
        // (solve/cholesky/det/slogdet already pinned).
        let mk = || {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims: vec![3, 2] },
                    [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                        .iter()
                        .map(|&v| Literal::from_f32(v))
                        .collect(),
                )
                .unwrap(),
            )
        };
        for (label, out) in [
            ("qr", eval_qr(&[mk()], &BTreeMap::new()).expect("qr")),
            ("svd", eval_svd(&[mk()], &BTreeMap::new()).expect("svd")),
        ] {
            for (i, v) in out.iter().enumerate() {
                if let Value::Tensor(t) = v {
                    assert_eq!(
                        t.dtype,
                        DType::F32,
                        "{label} output[{i}] must preserve f32, got {:?}",
                        t.dtype
                    );
                    t.validate_dtype_consistency().unwrap_or_else(|e| {
                        panic!("{label} output[{i}] dtype/literal mismatch: {e:?}")
                    });
                }
            }
        }
    }

    #[test]
    fn eval_lu_triangular_solve_preserve_f32_dtype() {
        // LU factor (f32 matrix, i32 pivots/perm) and triangular_solve (f32 in -> f32 out)
        // must preserve dtype. Closes the last gap in the f32 linalg-forward family coverage
        // (qr/svd/cholesky/det/slogdet/solve already pinned); det/slogdet had real
        // f32->f64 widening bugs, so the whole family is guarded explicitly.
        let f32_mat = |dims: Vec<u32>, vals: &[f32]| {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims },
                    vals.iter().map(|&v| Literal::from_f32(v)).collect(),
                )
                .unwrap(),
            )
        };

        // LU of [[2,1],[1,3]] (f32): lu -> F32, pivots/perm -> I32.
        let lu_out = eval_lu(
            &[f32_mat(vec![2, 2], &[2.0, 1.0, 1.0, 3.0])],
            &BTreeMap::new(),
        )
        .expect("lu");
        assert_eq!(lu_out.len(), 3, "lu returns [lu, pivots, perm]");
        let expect_dtype = [DType::F32, DType::I32, DType::I32];
        for (i, v) in lu_out.iter().enumerate() {
            if let Value::Tensor(t) = v {
                assert_eq!(
                    t.dtype, expect_dtype[i],
                    "lu output[{i}] dtype, got {:?}",
                    t.dtype
                );
                t.validate_dtype_consistency()
                    .unwrap_or_else(|e| panic!("lu output[{i}] dtype/literal mismatch: {e:?}"));
            }
        }

        // triangular_solve: lower A=[[2,0],[1,3]], b=[[2],[5]] (f32) -> x f32 [2,1].
        let a = f32_mat(vec![2, 2], &[2.0, 0.0, 1.0, 3.0]);
        let b = f32_mat(vec![2, 1], &[2.0, 5.0]);
        let x = eval_triangular_solve(&[a, b], &BTreeMap::new()).expect("triangular_solve");
        match x {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, DType::F32, "triangular_solve must preserve f32");
                t.validate_dtype_consistency()
                    .unwrap_or_else(|e| panic!("triangular_solve dtype/literal mismatch: {e:?}"));
            }
            other => panic!("triangular_solve must return an f32 tensor, got {other:?}"),
        }
    }

    #[test]
    fn eval_det_slogdet_preserve_f32_dtype() {
        // det/slogdet of an f32 matrix must return f32 scalars (JAX: f32 in -> f32 out).
        // The real paths previously hardcoded Value::scalar_f64, widening f32 -> f64.
        // A = [[2,1],[1,3]] (f32): det = 5, slogdet = (+1, ln 5).
        let a = || {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape { dims: vec![2, 2] },
                    vec![
                        Literal::from_f32(2.0),
                        Literal::from_f32(1.0),
                        Literal::from_f32(1.0),
                        Literal::from_f32(3.0),
                    ],
                )
                .unwrap(),
            )
        };
        let det = eval_det(&[a()], &BTreeMap::new()).expect("det");
        match det {
            Value::Scalar(Literal::F32Bits(b)) => {
                assert!(
                    (f32::from_bits(b) - 5.0).abs() < 1e-4,
                    "det = {}",
                    f32::from_bits(b)
                );
            }
            other => panic!("det must be an F32 scalar, got {other:?}"),
        }
        let sld = eval_slogdet(&[a()], &BTreeMap::new()).expect("slogdet");
        assert_eq!(sld.len(), 2);
        match (&sld[0], &sld[1]) {
            (Value::Scalar(Literal::F32Bits(s)), Value::Scalar(Literal::F32Bits(l))) => {
                assert!(
                    (f32::from_bits(*s) - 1.0).abs() < 1e-4,
                    "sign = {}",
                    f32::from_bits(*s)
                );
                assert!(
                    (f32::from_bits(*l) - 5.0_f32.ln()).abs() < 1e-4,
                    "logabsdet = {}",
                    f32::from_bits(*l)
                );
            }
            other => panic!("slogdet must be (F32, F32) scalars, got {other:?}"),
        }
    }

    #[test]
    fn eval_solve_complex_vector_matches_closed_form() {
        // A = diag(1+i, 1+i), b = (2, 4). Since (1+i)(1−i)=2, the solution is
        // x = (1−i, 2−2i). Complex solve must now compute it (jnp.linalg.solve
        // supports complex) instead of failing closed.
        let a = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_complex64(1.0, 1.0),
                    Literal::from_complex64(0.0, 0.0),
                    Literal::from_complex64(0.0, 0.0),
                    Literal::from_complex64(1.0, 1.0),
                ],
            )
            .unwrap(),
        );
        let b = Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                vec![
                    Literal::from_complex64(2.0, 0.0),
                    Literal::from_complex64(4.0, 0.0),
                ],
            )
            .unwrap(),
        );
        let Value::Tensor(t) = eval_solve(&[a, b], &BTreeMap::new()).expect("complex solve") else {
            panic!("expected tensor");
        };
        // C64 + C64 → C64; shape [2].
        assert_eq!(t.dtype, DType::Complex64);
        assert_eq!(t.shape.dims, vec![2]);
        let got: Vec<(f64, f64)> = t
            .elements
            .iter()
            .map(|l| {
                let (re, im) = l.as_complex64().unwrap();
                (f64::from(re), f64::from(im))
            })
            .collect();
        for (g, w) in got.iter().zip(&[(1.0, -1.0), (2.0, -2.0)]) {
            assert!(
                (g.0 - w.0).abs() < 1e-5 && (g.1 - w.1).abs() < 1e-5,
                "got {got:?}"
            );
        }
    }

    #[test]
    fn eval_solve_complex_matrix_rhs_reconstructs() {
        // General 3×3 complex A and a 3×2 complex B: solve, then verify A·X = B to
        // tolerance and the promoted dtype. A is well-conditioned (diagonally
        // dominant) so the system is non-singular.
        let n = 3usize;
        let c = |re: f64, im: f64| Literal::from_complex128(re, im);
        let a_raw = [
            (4.0, 1.0),
            (0.5, -0.3),
            (0.2, 0.1), //
            (-0.4, 0.2),
            (5.0, -1.0),
            (0.6, 0.4), //
            (0.1, -0.2),
            (0.3, 0.5),
            (6.0, 0.7),
        ];
        let b_raw = [
            (1.0, 2.0),
            (3.0, -1.0), //
            (-2.0, 0.5),
            (0.0, 1.0), //
            (1.5, -1.5),
            (2.0, 2.0),
        ];
        let a = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: vec![n as u32, n as u32],
                },
                a_raw.iter().map(|&(r, i)| c(r, i)).collect(),
            )
            .unwrap(),
        );
        let b = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape {
                    dims: vec![n as u32, 2],
                },
                b_raw.iter().map(|&(r, i)| c(r, i)).collect(),
            )
            .unwrap(),
        );
        let Value::Tensor(t) = eval_solve(&[a, b], &BTreeMap::new()).expect("complex solve") else {
            panic!("expected tensor");
        };
        assert_eq!(t.dtype, DType::Complex128);
        assert_eq!(t.shape.dims, vec![n as u32, 2]);
        let x: Vec<(f64, f64)> = t
            .elements
            .iter()
            .map(|l| l.as_complex128().unwrap())
            .collect();
        // A·X must equal B (residual < 1e-10).
        for col in 0..2 {
            for row in 0..n {
                let mut acc = (0.0_f64, 0.0_f64);
                for k in 0..n {
                    acc = complex_add(acc, complex_mul(a_raw[row * n + k], x[k * 2 + col]));
                }
                let want = b_raw[row * 2 + col];
                assert!(
                    (acc.0 - want.0).abs() < 1e-10 && (acc.1 - want.1).abs() < 1e-10,
                    "A·X≠B at ({row},{col}): {acc:?} vs {want:?}"
                );
            }
        }
    }

    // Inline per-column complex LU solve reference (the prior structure) for the
    // batched-vs-per-column bit-identity test + bench.
    fn complex_lu_solve_per_column(
        lu: &[(f64, f64)],
        perm: &[usize],
        b: &[(f64, f64)],
        n: usize,
        ncols: usize,
    ) -> Vec<(f64, f64)> {
        let mut x = vec![(0.0_f64, 0.0_f64); n * ncols];
        for jcol in 0..ncols {
            let mut y = vec![(0.0_f64, 0.0_f64); n];
            for i in 0..n {
                let mut s = b[perm[i] * ncols + jcol];
                for k in 0..i {
                    s = complex_sub(s, complex_mul(lu[i * n + k], y[k]));
                }
                y[i] = s;
            }
            for i in (0..n).rev() {
                let mut s = y[i];
                for k in (i + 1)..n {
                    s = complex_sub(s, complex_mul(lu[i * n + k], x[k * ncols + jcol]));
                }
                x[i * ncols + jcol] = complex_div(s, lu[i * n + i]);
            }
        }
        x
    }

    #[allow(clippy::type_complexity)]
    fn complex_lu_system(n: usize, ncols: usize) -> (Vec<(f64, f64)>, Vec<usize>, Vec<(f64, f64)>) {
        let mut a = vec![(0.0_f64, 0.0_f64); n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    ((n as f64) + 4.0, 1.0)
                } else {
                    (
                        (((i * 31 + j * 17) % 11) as f64 - 5.0) * 0.1,
                        (((i + j) % 7) as f64 - 3.0) * 0.05,
                    )
                };
            }
        }
        let perm = super::complex_lu_factor_blocked(&mut a, n);
        let b: Vec<(f64, f64)> = (0..n * ncols)
            .map(|k| {
                (
                    (((k * 53 + 11) % 37) as f64) * 0.1 - 1.8,
                    (((k * 13) % 19) as f64 - 9.0) * 0.05,
                )
            })
            .collect();
        (a, perm, b)
    }

    #[test]
    fn complex_lu_solve_batched_bit_identical_to_per_column() {
        let n = 160usize;
        let ncols = 40usize;
        let (lu, perm, b) = complex_lu_system(n, ncols);
        let got = super::complex_lu_solve(&lu, &perm, &b, n, ncols);
        let want = complex_lu_solve_per_column(&lu, &perm, &b, n, ncols);
        for k in 0..n * ncols {
            assert_eq!(
                got[k].0.to_bits(),
                want[k].0.to_bits(),
                "re diverged at {k}"
            );
            assert_eq!(
                got[k].1.to_bits(),
                want[k].1.to_bits(),
                "im diverged at {k}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_complex_lu_solve_batched_vs_per_column() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        for &(n, ncols) in &[(1024usize, 64usize), (1024, 256), (2048, 128)] {
            let (lu, perm, b) = complex_lu_system(n, ncols);
            let t_batched = best_time(|| {
                std::hint::black_box(super::complex_lu_solve(&lu, &perm, &b, n, ncols));
            });
            let t_percol = best_time(|| {
                std::hint::black_box(complex_lu_solve_per_column(&lu, &perm, &b, n, ncols));
            });
            println!(
                "BENCH complex_lu_solve n={n} ncols={ncols}: per-column {:.2}ms -> batched {:.2}ms = {:.2}x",
                t_percol * 1e3,
                t_batched * 1e3,
                t_percol / t_batched
            );
        }
    }

    #[test]
    fn eval_det_and_slogdet_complex_closed_form() {
        // det(diag(1+i, 1+i)) = (1+i)^2 = 2i; and an upper-triangular case
        // det([[2+i,3-i],[0,1+2i]]) = (2+i)(1+2i) = 5i (off-diagonal irrelevant).
        let mk = |data: &[(f64, f64)]| {
            Value::Tensor(
                TensorValue::new(
                    DType::Complex128,
                    Shape { dims: vec![2, 2] },
                    data.iter()
                        .map(|&(r, i)| Literal::from_complex128(r, i))
                        .collect(),
                )
                .unwrap(),
            )
        };
        let diag = mk(&[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]);
        let tri = mk(&[(2.0, 1.0), (3.0, -1.0), (0.0, 0.0), (1.0, 2.0)]);

        for (a, want) in [(&diag, (0.0, 2.0)), (&tri, (0.0, 5.0))] {
            let Value::Scalar(lit) = eval_det(std::slice::from_ref(a), &BTreeMap::new()).unwrap()
            else {
                panic!("det scalar");
            };
            let (re, im) = lit.as_complex128().unwrap();
            assert!(
                (re - want.0).abs() < 1e-10 && (im - want.1).abs() < 1e-10,
                "det = ({re},{im}), want {want:?}"
            );
        }

        // slogdet(diag) : sign = 2i/|2i| = i, logabsdet = ln 2.
        let out = eval_slogdet(&[diag], &BTreeMap::new()).unwrap();
        let Value::Scalar(sl) = &out[0] else {
            panic!("sign scalar");
        };
        let (sre, sim) = sl.as_complex128().unwrap();
        let Value::Scalar(ll) = &out[1] else {
            panic!("logabsdet scalar");
        };
        let lad = ll.as_f64().unwrap();
        assert!(
            (sre - 0.0).abs() < 1e-12 && (sim - 1.0).abs() < 1e-12,
            "sign ({sre},{sim})"
        );
        assert!((lad - 2.0_f64.ln()).abs() < 1e-12, "logabsdet {lad}");
    }

    #[test]
    fn eval_solve_singular_returns_non_finite_tensor_not_error() {
        // The dispatched Solve primitive must not raise for a singular A — JAX
        // returns inf/NaN (NumPy raises). Previously eval_solve mapped the
        // lu_factor None to EvalError::Unsupported "singular matrix".
        let a = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(4.0),
                ],
            )
            .unwrap(),
        );
        let b = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(3.0), Literal::from_f64(6.0)],
            )
            .unwrap(),
        );
        let result = eval_solve(&[a, b], &BTreeMap::new()).expect("singular solve must not raise");
        let Value::Tensor(t) = result else {
            panic!("expected tensor");
        };
        let vals: Vec<f64> = t.elements.iter().map(|e| e.as_f64().unwrap()).collect();
        assert!(
            vals.iter().any(|v| !v.is_finite()),
            "singular solve output must be non-finite, got {vals:?}"
        );
    }

    // Build a deterministic, well-conditioned (diagonally dominant) n×n system and
    // its dense RHS for the mixed-precision tests/bench.
    fn mixed_solve_system(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    (n as f64) + (i as f64) * 0.01 + 4.0
                } else {
                    (((i * 131 + j * 71 + 7) % 23) as f64 - 11.0) * 0.05
                };
            }
        }
        let b: Vec<f64> = (0..n)
            .map(|i| ((i * 37 + 5) % 29) as f64 * 0.1 - 1.4)
            .collect();
        (a, b)
    }

    #[test]
    fn mixed_precision_solve_matches_f64_within_tolerance() {
        // At n ≥ MIXED_PRECISION_SOLVE_MIN_N the production `solve` runs the
        // mixed-precision path; its result must match the f64 reference solve to
        // within the solve tolerance (iterative refinement recovers f64 accuracy).
        let n = super::MIXED_PRECISION_SOLVE_MIN_N + 13;
        let (a, b) = mixed_solve_system(n);
        // Direct mixed-precision result.
        let mixed = super::solve_mixed_precision(&a, &b, n).expect("mixed should converge");
        // f64 reference (the pre-mixed path).
        let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
        let f64ref = super::lu_solve(&lu, &p, &b, n);
        let mut max_rel = 0.0f64;
        for i in 0..n {
            let d = (mixed[i] - f64ref[i]).abs();
            max_rel = max_rel.max(d / (f64ref[i].abs() + 1.0));
        }
        assert!(
            max_rel < 1e-9,
            "mixed-precision solve diverged from f64: max_rel={max_rel:.3e}"
        );
        // And the production `solve` (which routes through mixed) also matches.
        let prod = super::solve(&a, &b, n).unwrap();
        for i in 0..n {
            assert!((prod[i] - f64ref[i]).abs() / (f64ref[i].abs() + 1.0) < 1e-9);
        }
    }

    // Multi-column RHS for the mixed-precision multi-RHS tests/bench: column `c`
    // is the single-RHS vector cyclically shifted, so each column is a distinct,
    // well-posed system against the same diagonally-dominant `A`.
    fn mixed_solve_system_multi(n: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
        let (a, b1) = mixed_solve_system(n);
        let mut b = vec![0.0f64; n * m];
        for i in 0..n {
            for c in 0..m {
                b[i * m + c] = b1[(i + c) % n] + (c as f64) * 0.013;
            }
        }
        (a, b)
    }

    #[test]
    fn mixed_precision_solve_multi_matches_f64_within_tolerance() {
        // At n ≥ MIXED_PRECISION_SOLVE_MIN_N `solve_multi_rhs` runs the
        // mixed-precision multi-RHS path; its result must match the f64 reference
        // (batched f64 solve) to within solve tolerance for every column.
        let n = super::MIXED_PRECISION_SOLVE_MIN_N + 9;
        let m = 32; // n ≥ 8·m so production `solve_multi_rhs` routes through mixed.
        let (a, b) = mixed_solve_system_multi(n, m);
        let mixed = super::solve_multi_rhs_mixed_precision(&a, &b, n, m)
            .expect("mixed-multi should converge");
        // f64 reference (the pre-mixed batched path).
        let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
        let f64ref = super::lu_solve_multi(&lu, &p, &b, n, m);
        let mut max_rel = 0.0f64;
        for i in 0..n * m {
            let d = (mixed[i] - f64ref[i]).abs();
            max_rel = max_rel.max(d / (f64ref[i].abs() + 1.0));
        }
        assert!(
            max_rel < 1e-9,
            "mixed-precision multi-RHS solve diverged from f64: max_rel={max_rel:.3e}"
        );
        // And the production `solve_multi_rhs` (which routes through mixed) matches.
        let prod = super::solve_multi_rhs(&a, &b, n, m).unwrap();
        for i in 0..n * m {
            assert!((prod[i] - f64ref[i]).abs() / (f64ref[i].abs() + 1.0) < 1e-9);
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_mixed_precision_solve_multi_vs_f64() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // Factorization-dominated regime (n ≥ 8·m), where the production
        // `solve_multi_rhs` gate engages the mixed-precision path.
        for &(n, m) in &[
            (1024usize, 16usize),
            (1024, 64),
            (1024, 128),
            (2048, 64),
            (2048, 128),
            (2048, 256),
        ] {
            let (a, b) = mixed_solve_system_multi(n, m);
            let t_mixed = best_time(|| {
                std::hint::black_box(super::solve_multi_rhs_mixed_precision(&a, &b, n, m).unwrap());
            });
            let t_f64 = best_time(|| {
                let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
                std::hint::black_box(super::lu_solve_multi(&lu, &p, &b, n, m));
            });
            println!(
                "BENCH mixed-precision multi-RHS solve n={n} m={m}: f64 {:.2}ms -> mixed {:.2}ms = {:.2}x",
                t_f64 * 1e3,
                t_mixed * 1e3,
                t_f64 / t_mixed,
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_mixed_precision_solve_vs_f64() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        for &n in &[512usize, 1024, 2048] {
            let (a, b) = mixed_solve_system(n);
            let t_mixed = best_time(|| {
                std::hint::black_box(super::solve_mixed_precision(&a, &b, n).unwrap());
            });
            let t_f64 = best_time(|| {
                let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
                std::hint::black_box(super::lu_solve(&lu, &p, &b, n));
            });
            // LU-only ratios (same-invocation → reliable even on a noisy worker):
            // f64 blocked vs f32 blocked vs f32 recursive.
            let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let t_lu_f64 = best_time(|| {
                std::hint::black_box(super::lu_factor_for_solve(&a, n).unwrap());
            });
            let t_lu_blk = best_time(|| {
                let mut lu = a32.clone();
                std::hint::black_box(super::lu_factor_real_blocked_f32(&mut lu, n));
            });
            println!(
                "BENCH mixed-precision solve n={n}: f64 {:.2}ms -> mixed {:.2}ms = {:.2}x | LU-only f64 {:.1}ms / f32 {:.1}ms = {:.2}x",
                t_f64 * 1e3,
                t_mixed * 1e3,
                t_f64 / t_mixed,
                t_lu_f64 * 1e3,
                t_lu_blk * 1e3,
                t_lu_f64 / t_lu_blk,
            );
        }
    }

    #[test]
    fn lu_solve_multi_bit_identical_to_per_column_large() {
        // The batched multi-RHS triangular solve must be BIT-IDENTICAL to running
        // the per-column `lu_solve` (only the loop nesting differs) — at a size and
        // column count that exercise the vectorized inner loop.
        let n = 200usize;
        let m = 48usize;
        let (a, _b) = mixed_solve_system(n);
        let bmat: Vec<f64> = (0..n * m)
            .map(|k| (((k * 53 + 11) % 37) as f64) * 0.1 - 1.8)
            .collect();
        let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
        let batched = super::lu_solve_multi(&lu, &p, &bmat, n, m);
        for j in 0..m {
            let rhs: Vec<f64> = (0..n).map(|i| bmat[i * m + j]).collect();
            let col = super::lu_solve(&lu, &p, &rhs, n);
            for i in 0..n {
                assert_eq!(
                    batched[i * m + j].to_bits(),
                    col[i].to_bits(),
                    "batched multi-RHS solve diverged at i={i} j={j}"
                );
            }
        }
    }

    #[test]
    fn lu_solve_multi_blocked_matches_batched_within_tolerance() {
        // The blocked (GEMM cross-block) TRSM is not bit-identical to the scalar
        // batched solve (block-sum vs running subtract) but must agree to tolerance.
        let n = 600usize;
        let m = 32usize;
        let (a, _b) = mixed_solve_system(n);
        let bmat: Vec<f64> = (0..n * m)
            .map(|k| (((k * 53 + 11) % 37) as f64) * 0.1 - 1.8)
            .collect();
        let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
        let batched = super::lu_solve_multi(&lu, &p, &bmat, n, m);
        let blocked = super::lu_solve_multi_blocked(&lu, &p, &bmat, n, m);
        let mut max_rel = 0.0f64;
        for k in 0..n * m {
            max_rel = max_rel.max((blocked[k] - batched[k]).abs() / (batched[k].abs() + 1.0));
        }
        assert!(
            max_rel < 1e-9,
            "blocked TRSM diverged from batched: {max_rel:.3e}"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_lu_solve_multi_blocked_vs_batched() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        for &(n, m) in &[(1024usize, 256usize), (1024, 512), (2048, 256)] {
            let (a, _b) = mixed_solve_system(n);
            let bmat: Vec<f64> = (0..n * m)
                .map(|k| (((k * 53 + 11) % 37) as f64) * 0.1 - 1.8)
                .collect();
            let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
            let t_batched = best_time(|| {
                std::hint::black_box(super::lu_solve_multi(&lu, &p, &bmat, n, m));
            });
            let t_blocked = best_time(|| {
                std::hint::black_box(super::lu_solve_multi_blocked(&lu, &p, &bmat, n, m));
            });
            println!(
                "BENCH lu_solve_multi n={n} m={m}: batched {:.2}ms -> blocked {:.2}ms = {:.2}x",
                t_batched * 1e3,
                t_blocked * 1e3,
                t_batched / t_blocked
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_lu_solve_multi_vs_per_column() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        for &(n, m) in &[(1024usize, 64usize), (1024, 256), (2048, 128)] {
            let (a, _b) = mixed_solve_system(n);
            let bmat: Vec<f64> = (0..n * m)
                .map(|k| (((k * 53 + 11) % 37) as f64) * 0.1 - 1.8)
                .collect();
            let (lu, p) = super::lu_factor_for_solve(&a, n).unwrap();
            let t_batched = best_time(|| {
                std::hint::black_box(super::lu_solve_multi(&lu, &p, &bmat, n, m));
            });
            let t_percol = best_time(|| {
                let mut res = vec![0.0; n * m];
                for j in 0..m {
                    let rhs: Vec<f64> = (0..n).map(|i| bmat[i * m + j]).collect();
                    let x = super::lu_solve(&lu, &p, &rhs, n);
                    for i in 0..n {
                        res[i * m + j] = x[i];
                    }
                }
                std::hint::black_box(res);
            });
            println!(
                "BENCH multi-RHS triangular solve n={n} m={m}: per-column {:.2}ms -> batched {:.2}ms = {:.2}x",
                t_percol * 1e3,
                t_batched * 1e3,
                t_percol / t_batched
            );
        }
    }

    #[test]
    fn solve_multi_rhs_basic() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0, 4.0]; // 2x2 (two RHS columns)
        let x = solve_multi_rhs(&a, &b, 2, 2).expect("solvable");
        assert_eq!(x.len(), 4);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
        assert!((x[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn solve_multi_rhs_bit_identical_to_per_column_solve() {
        // The shared-factorization multi-RHS path must produce bit-for-bit the
        // same result as solving each RHS column independently (which is what
        // the previous per-column-refactor implementation did).
        let n = 7usize;
        let m = 5usize;
        // Deterministic, well-conditioned (diagonally dominant) A.
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    (n as f64) + (i as f64) * 0.5 + 3.0
                } else {
                    ((i * 31 + j * 17) % 11) as f64 * 0.25 - 1.0
                };
            }
        }
        let mut b = vec![0.0f64; n * m];
        for i in 0..n {
            for j in 0..m {
                b[i * m + j] = ((i * 13 + j * 7 + 1) % 19) as f64 * 0.5 - 4.0;
            }
        }

        let multi = solve_multi_rhs(&a, &b, n, m).expect("non-singular");
        for j in 0..m {
            let rhs: Vec<f64> = (0..n).map(|i| b[i * m + j]).collect();
            let single = solve(&a, &rhs, n).expect("non-singular");
            for i in 0..n {
                assert_eq!(
                    multi[i * m + j].to_bits(),
                    single[i].to_bits(),
                    "mismatch at row {i} col {j}"
                );
            }
        }
    }

    #[test]
    fn qr_decomposition_contiguous_bit_identical_to_rowmajor() {
        let n = 9usize;
        let a: Vec<f64> = (0..n * n)
            .map(|i| (i as f64 * 0.211).sin() * 4.0 + (i as f64 * 0.037).cos())
            .collect();

        let (q_new, r_new) = qr_decomposition(&a, n);

        // Inline reference: the prior row-major Gram-Schmidt.
        let mut q = vec![(0.0f64, 0.0f64); n * n];
        let mut r = vec![0.0f64; n * n];
        for j in 0..n {
            let mut v: Vec<f64> = (0..n).map(|i| a[i * n + j]).collect();
            for i in 0..j {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += q[k * n + i].0 * a[k * n + j];
                }
                r[i * n + j] = dot;
                for k in 0..n {
                    v[k] -= dot * q[k * n + i].0;
                }
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r[j * n + j] = norm;
            if norm > 1e-15 {
                for k in 0..n {
                    q[k * n + j] = (v[k] / norm, 0.0);
                }
            }
        }

        for idx in 0..n * n {
            assert_eq!(q_new[idx].to_bits(), q[idx].0.to_bits(), "q.re {idx}");
            assert_eq!(r_new[idx].to_bits(), r[idx].to_bits(), "r {idx}");
        }
    }

    #[test]
    fn matrix_mul_ikj_bit_identical_to_ijk() {
        let n = 11usize;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.137).sin() * 2.0).collect();
        let b_re: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.071).cos() * 1.5).collect();
        let b_im: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.091).sin() * 0.5).collect();
        let b: Vec<(f64, f64)> = b_re
            .iter()
            .zip(b_im.iter())
            .map(|(&r, &i)| (r, i))
            .collect();
        let a_cplx: Vec<(f64, f64)> = a.iter().map(|&v| (v, v * 0.25)).collect();

        // Reference: i-j-k accumulation.
        let mut ref_real = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a[i * n + k] * b[k * n + j].0;
                }
                ref_real[i * n + j] = s;
            }
        }
        let mut ref_cplx = vec![(0.0f64, 0.0f64); n * n];
        for i in 0..n {
            for j in 0..n {
                let (mut re, mut im) = (0.0, 0.0);
                for k in 0..n {
                    let (ar, ai) = a_cplx[i * n + k];
                    let (br, bi) = b[k * n + j];
                    re += ar * br - ai * bi;
                    im += ar * bi + ai * br;
                }
                ref_cplx[i * n + j] = (re, im);
            }
        }

        let got_real = matrix_mul(&a, &b_re, n);
        for idx in 0..n * n {
            assert_eq!(
                got_real[idx].to_bits(),
                ref_real[idx].to_bits(),
                "real {idx}"
            );
        }
        let got_cplx = matrix_mul_complex(&a_cplx, &b, n);
        for idx in 0..n * n {
            assert_eq!(
                got_cplx[idx].0.to_bits(),
                ref_cplx[idx].0.to_bits(),
                "re {idx}"
            );
            assert_eq!(
                got_cplx[idx].1.to_bits(),
                ref_cplx[idx].1.to_bits(),
                "im {idx}"
            );
        }
    }

    #[test]
    fn upper_triangular_matrix_mul_bit_identical_to_dense_zero_lower() {
        let n = 13usize;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for k in i..n {
                a[i * n + k] = ((i * 17 + k * 11 + 3) as f64 * 0.019).cos();
            }
        }
        let b: Vec<f64> = (0..n * n)
            .map(|idx| ((idx * 23 + 5) as f64 * 0.013).sin())
            .collect();

        let dense = matrix_mul(&a, &b, n);
        let triangular = upper_triangular_matrix_mul(&a, &b, n);
        for idx in 0..n * n {
            assert_eq!(
                triangular[idx].to_bits(),
                dense[idx].to_bits(),
                "triangular multiply mismatch at {idx}"
            );
        }
    }

    #[test]
    fn hessenberg_reduction_forms_orthogonal_similarity() {
        let n = 6usize;
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let row = idx / n;
                let col = idx % n;
                if row == col {
                    n as f64 + row as f64 * 0.25 + 3.0
                } else {
                    ((row * 5 + col * 7 + 2) % 9) as f64 * 0.125 - 0.5
                }
            })
            .collect();

        let (h, q) = hessenberg_reduction(&a, n);
        for row in 2..n {
            for col in 0..row - 1 {
                assert!(h[row * n + col].abs() <= 1e-12, "H[{row},{col}]");
            }
        }

        let qt = transpose_square(&q, n);
        let qt_a = matrix_mul(&qt, &a, n);
        let qt_a_q = matrix_mul(&qt_a, &q, n);
        for idx in 0..n * n {
            assert!(
                (qt_a_q[idx] - h[idx]).abs() <= 1e-10,
                "Q^T A Q mismatch at {idx}: {} vs {}",
                qt_a_q[idx],
                h[idx]
            );
        }
    }

    #[test]
    fn eig_qr_iteration_preserves_diagonal_deflation_contract() {
        let n = 3usize;
        let a = [
            7.0, 0.0, 0.0, //
            0.0, 3.0, 0.0, //
            0.0, 0.0, -2.0,
        ];

        let (values, vectors) = eig_qr_iteration(&a, n);
        let expected_values: [(f64, f64); 3] = [(7.0, 0.0), (3.0, 0.0), (-2.0, 0.0)];

        // Eigenvalues of a diagonal matrix deflate exactly.
        for idx in 0..n {
            assert_eq!(values[idx].0.to_bits(), expected_values[idx].0.to_bits());
            assert_eq!(values[idx].1.to_bits(), expected_values[idx].1.to_bits());
        }
        // Eigenvectors now come from inverse iteration — valid up to scale/phase
        // (not bit-exact identity as the old Schur-vector path happened to emit for
        // diagonal input). Assert the real contract: A·v = λ·v, and each column is
        // axis-aligned (|v[row][col]| ≈ δ(row,col)) to tolerance.
        assert_eig_residual_complex(&a, n, &values, &vectors, 1e-12);
        for col in 0..n {
            for row in 0..n {
                let want = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (complex_abs(vectors[row * n + col]) - want).abs() < 1e-9,
                    "diagonal eig vector[{row}][{col}] |·|={} not ≈{want}",
                    complex_abs(vectors[row * n + col])
                );
            }
        }
    }

    fn transpose_square(matrix: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                out[col * n + row] = matrix[row * n + col];
            }
        }
        out
    }

    // ── Least squares tests ─────────────────────────────────────────

    #[test]
    fn lstsq_exact_square() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 4.0];
        let x = lstsq(&a, 2, 2, &b).expect("solvable");
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn lstsq_overdetermined() {
        // 3 equations, 2 unknowns: y = x line
        // Points: (0,0), (1,1), (2,2)
        // A = [[1, 0], [1, 1], [1, 2]], b = [0, 1, 2]
        // Least squares: a=0, b=1 (y = 0 + 1*x)
        let a = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let b = [0.0, 1.0, 2.0];
        let x = lstsq(&a, 3, 2, &b).expect("solvable");
        // x should be [0, 1] for intercept and slope
        assert!((x[0]).abs() < 1e-8, "intercept should be ~0, got {}", x[0]);
        assert!(
            (x[1] - 1.0).abs() < 1e-8,
            "slope should be ~1, got {}",
            x[1]
        );
    }

    #[test]
    fn lstsq_verifies_normal_equations() {
        // Verify A^T * A * x = A^T * b
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let b = [1.0, 2.0, 3.0];
        let x = lstsq(&a, 3, 2, &b).expect("solvable");

        // Compute A^T * A * x and A^T * b
        let mut ata_x = [0.0; 2];
        let mut atb = [0.0; 2];
        for i in 0..2 {
            for j in 0..2 {
                let mut ata_ij = 0.0;
                for k in 0..3 {
                    ata_ij += a[k * 2 + i] * a[k * 2 + j];
                }
                ata_x[i] += ata_ij * x[j];
            }
            for k in 0..3 {
                atb[i] += a[k * 2 + i] * b[k];
            }
        }
        assert!((ata_x[0] - atb[0]).abs() < 1e-8);
        assert!((ata_x[1] - atb[1]).abs() < 1e-8);
    }

    #[test]
    fn lstsq_rank_deficient_returns_min_norm_solution() {
        // Rank-1 3x2 system (column 2 = 2*column 1): AᵀA is singular, so the old
        // solve(AᵀA, Aᵀb) returned None. JAX returns the SVD min-norm solution;
        // the pinv-based path must return a finite x satisfying the least-squares
        // optimality condition AᵀA x = Aᵀ b.
        let a = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]; // 3x2, rank 1
        let b = [1.0, 2.0, 3.0];
        let x = lstsq(&a, 3, 2, &b).expect("rank-deficient lstsq must still return a solution");
        assert_eq!(x.len(), 2);
        assert!(x.iter().all(|v| v.is_finite()));

        // AᵀA x ≈ Aᵀ b (normal equations hold for any least-squares minimizer).
        for i in 0..2 {
            let mut atax = 0.0;
            let mut atb = 0.0;
            for j in 0..2 {
                let mut ata = 0.0;
                for k in 0..3 {
                    ata += a[k * 2 + i] * a[k * 2 + j];
                }
                atax += ata * x[j];
            }
            for k in 0..3 {
                atb += a[k * 2 + i] * b[k];
            }
            assert!(
                (atax - atb).abs() < 1e-8,
                "normal equation {i} violated: {atax} vs {atb}"
            );
        }
    }
}
