#![forbid(unsafe_code)]

//! Linear algebra primitives: Cholesky, triangular solve, and QR decomposition.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;
use crate::type_promotion::promote_dtype;

type ComplexScalar = (f64, f64);
type ComplexVector = Vec<ComplexScalar>;
type EigQrResult = (ComplexVector, ComplexVector);

// ── Complex Arithmetic Helpers ──────────────────────────────────────

fn complex_abs(z: (f64, f64)) -> f64 {
    (z.0 * z.0 + z.1 * z.1).sqrt()
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

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
    let elements: Vec<Literal> = data
        .iter()
        .map(|&v| linalg_literal_from_f64(dtype, v))
        .collect();
    let shape = Shape {
        dims: vec![m as u32, n as u32],
    };
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

    matrix_to_value(m, m, &l, dtype)
}

// ── Triangular solve ────────────────────────────────────────────────

/// Solve a triangular linear system: find X such that A * X = B (or A^T * X = B).
///
/// By default, A is assumed lower-triangular and `left_side=true`.
/// Params:
///   - `lower` (default "true"): if "true", A is lower-triangular; else upper.
///   - `transpose_a` (default "false"): if "true", solve A^T X = B instead.
///   - `unit_diagonal` (default "false"): if "true", diagonal of A is assumed 1.
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
    let one = (1.0, 0.0);
    let mut x = vec![(0.0_f64, 0.0_f64); n * n_b];
    let mut b_col = vec![(0.0_f64, 0.0_f64); n];

    for col in 0..n_b {
        for row in 0..n {
            b_col[row] = b[row * n_b + col];
        }

        if lower && !transpose_a {
            for i in 0..n {
                for k in 0..i {
                    b_col[i] = complex_sub(b_col[i], complex_mul(a[i * n + k], x[k * n_b + col]));
                }
                let diag = if unit_diagonal { one } else { a[i * n + i] };
                // JAX's triangular_solve does not raise for a zero/near-zero
                // diagonal — complex_div yields inf/nan (singular) or a finite
                // large value (near-singular), matching jnp (NumPy raises).
                x[i * n_b + col] = complex_div(b_col[i], diag);
            }
        } else if !lower && !transpose_a {
            for i in (0..n).rev() {
                for k in (i + 1)..n {
                    b_col[i] = complex_sub(b_col[i], complex_mul(a[i * n + k], x[k * n_b + col]));
                }
                let diag = if unit_diagonal { one } else { a[i * n + i] };
                // JAX's triangular_solve does not raise for a zero/near-zero
                // diagonal — complex_div yields inf/nan (singular) or a finite
                // large value (near-singular), matching jnp (NumPy raises).
                x[i * n_b + col] = complex_div(b_col[i], diag);
            }
        } else if lower && transpose_a {
            for i in (0..n).rev() {
                for k in (i + 1)..n {
                    b_col[i] = complex_sub(b_col[i], complex_mul(a[k * n + i], x[k * n_b + col]));
                }
                let diag = if unit_diagonal { one } else { a[i * n + i] };
                // JAX's triangular_solve does not raise for a zero/near-zero
                // diagonal — complex_div yields inf/nan (singular) or a finite
                // large value (near-singular), matching jnp (NumPy raises).
                x[i * n_b + col] = complex_div(b_col[i], diag);
            }
        } else {
            for i in 0..n {
                for k in 0..i {
                    b_col[i] = complex_sub(b_col[i], complex_mul(a[k * n + i], x[k * n_b + col]));
                }
                let diag = if unit_diagonal { one } else { a[i * n + i] };
                // JAX's triangular_solve does not raise for a zero/near-zero
                // diagonal — complex_div yields inf/nan (singular) or a finite
                // large value (near-singular), matching jnp (NumPy raises).
                x[i * n_b + col] = complex_div(b_col[i], diag);
            }
        }
    }

    let out_dtype = promote_dtype(dtype_a, dtype_b);
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

    if !matches!(inputs[0].dtype(), DType::Complex64 | DType::Complex128) {
        let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
        if a.iter().all(|v| *v != 0.0) {
            return eval_qr_real_matrix(m, n, a, dtype, full_matrices);
        }
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
    let k = m.min(n);

    let mut r = a;
    let mut v_store: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut tau_store: Vec<f64> = Vec::with_capacity(k);

    for j in 0..k {
        let mut v: Vec<f64> = (j..m).map(|i| r[i * n + j]).collect();
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

            for col in j..n {
                let mut dot = 0.0;
                for (vi, row) in v.iter().zip(j..m) {
                    dot += *vi * r[row * n + col];
                }
                let tau_dot = tau * dot;
                for (vi, row) in v.iter().zip(j..m) {
                    r[row * n + col] -= *vi * tau_dot;
                }
            }

            v_store.push(v);
            tau_store.push(tau);
        } else {
            v_store.push(vec![0.0; m - j]);
            tau_store.push(0.0);
        }
    }

    let q_cols = if full_matrices { m } else { k };
    let mut q = vec![0.0; m * q_cols];

    for i in 0..q_cols.min(m) {
        q[i * q_cols + i] = 1.0;
    }

    for j in (0..k).rev() {
        let v = &v_store[j];
        let tau = tau_store[j];
        if (tau * tau).sqrt() < f64::EPSILON {
            continue;
        }

        for col in j..q_cols {
            let mut dot = 0.0;
            for (vi, row) in v.iter().zip(j..m) {
                dot += *vi * q[row * q_cols + col];
            }
            let tau_dot = tau * dot;
            for (vi, row) in v.iter().zip(j..m) {
                q[row * q_cols + col] -= *vi * tau_dot;
            }
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

fn eval_lu_real_matrix(
    m: usize,
    n: usize,
    mut lu: Vec<f64>,
    dtype: DType,
) -> Result<Vec<Value>, EvalError> {
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

    // Step 1: Compute A^H A (n×n Hermitian matrix)
    let mut aha = vec![zero; n * n];
    for i in 0..n {
        for j in i..n {
            let mut dot = zero;
            for row in 0..m {
                dot = complex_add(
                    dot,
                    complex_mul(complex_conj(a[row * n + i]), a[row * n + j]),
                );
            }
            aha[i * n + j] = dot;
            aha[j * n + i] = complex_conj(dot);
        }
    }

    // Step 2: Eigendecompose A^H A via Jacobi rotations → V, eigenvalues σ²
    let (eigenvalues, v) = complex_jacobi_eigendecomposition(&mut aha, n);

    // Step 3: Sort eigenvalues (and corresponding V columns) in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

    let mut sigma = vec![0.0_f64; k];
    let mut v_sorted = vec![zero; n * n];
    for (new_col, &old_col) in indices.iter().enumerate() {
        if new_col < k {
            sigma[new_col] = eigenvalues[old_col].max(0.0).sqrt();
        }
        for row in 0..n {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
    }

    // Step 4: Compute U = A V Σ^{-1} (thin: m×k)
    let mut u = vec![zero; m * k];
    for i in 0..m {
        for j in 0..k {
            if sigma[j] > f64::EPSILON * 1e4 {
                let mut val = zero;
                for col in 0..n {
                    val = complex_add(val, complex_mul(a[i * n + col], v_sorted[col * n + j]));
                }
                u[i * k + j] = complex_div(val, (sigma[j], 0.0));
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

    // Step 1: A^T A (n×n symmetric).
    let mut ata = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0_f64;
            for row in 0..m {
                dot += a[row * n + i] * a[row * n + j];
            }
            ata[i * n + j] = dot;
            ata[j * n + i] = dot;
        }
    }

    // Step 2: symmetric eigendecomposition via row-cyclic Jacobi → V, σ².
    let (eigenvalues, v) = jacobi_eigendecomposition_cyclic(&mut ata, n);

    // Step 3: sort eigenvalues (and V columns) descending.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

    let mut sigma = vec![0.0_f64; k];
    let mut v_sorted = vec![0.0_f64; n * n];
    for (new_col, &old_col) in indices.iter().enumerate() {
        if new_col < k {
            sigma[new_col] = eigenvalues[old_col].max(0.0).sqrt();
        }
        for row in 0..n {
            v_sorted[row * n + new_col] = v[row * n + old_col];
        }
    }

    // Step 4: U = A V Σ⁻¹ (thin: m×k).
    let mut u = vec![0.0_f64; m * k];
    for i in 0..m {
        for j in 0..k {
            if sigma[j] > f64::EPSILON * 1e4 {
                let mut val = 0.0_f64;
                for col in 0..n {
                    val += a[i * n + col] * v_sorted[col * n + j];
                }
                u[i * k + j] = val / sigma[j];
            }
        }
    }

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
    let s_tensor = TensorValue::new(dtype, Shape { dims: vec![k as u32] }, s_elements)
        .map_err(EvalError::InvalidTensor)?;
    let s_val = Value::Tensor(s_tensor);

    let vh_val = matrix_to_value(vt_rows, n, &vh, dtype)?;

    Ok(vec![u_val, s_val, vh_val])
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

        // Unitary rotation U:
        // U = [[c, -s*e^{-iφ}], [s*e^{iφ}, c]]
        // U^H = [[c, s*e^{-iφ}], [-s*e^{iφ}, c]]
        let phase_conj = complex_conj(phase);
        let neg_s_phase_conj = complex_mul((-s, 0.0), phase_conj); // -s*e^{-iφ}
        let s_phase = complex_mul((s, 0.0), phase); // s*e^{iφ}
        let s_phase_conj = complex_mul((s, 0.0), phase_conj); // s*e^{-iφ}
        let neg_s_phase = complex_mul((-s, 0.0), phase); // -s*e^{iφ}

        // Apply U^H from left
        let row_p: Vec<_> = (0..n).map(|j| a[p * n + j]).collect();
        let row_q: Vec<_> = (0..n).map(|j| a[q * n + j]).collect();

        for j in 0..n {
            // (U^H * A)[p][j] = c * A[p][j] + s*e^{-iφ} * A[q][j]
            a[p * n + j] = complex_add(
                complex_mul((c, 0.0), row_p[j]),
                complex_mul(s_phase_conj, row_q[j]),
            );
            // (U^H * A)[q][j] = (-s*e^{iφ}) * A[p][j] + c * A[q][j]
            a[q * n + j] = complex_add(
                complex_mul(neg_s_phase, row_p[j]),
                complex_mul((c, 0.0), row_q[j]),
            );
        }

        // Apply U from right
        let col_p: Vec<_> = (0..n).map(|i| a[i * n + p]).collect();
        let col_q: Vec<_> = (0..n).map(|i| a[i * n + q]).collect();

        for i in 0..n {
            // (A * U)[i][p] = A[i][p] * c + A[i][q] * s*e^{iφ}
            a[i * n + p] = complex_add(
                complex_mul((c, 0.0), col_p[i]),
                complex_mul(s_phase, col_q[i]),
            );
            // (A * U)[i][q] = A[i][p] * (-s*e^{-iφ}) + A[i][q] * c
            a[i * n + q] = complex_add(
                complex_mul(neg_s_phase_conj, col_p[i]),
                complex_mul((c, 0.0), col_q[i]),
            );
        }

        // Diagonal elements should be real; off-diagonal (p,q) should be zero
        a[p * n + p] = (a[p * n + p].0, 0.0);
        a[q * n + q] = (a[q * n + q].0, 0.0);
        a[p * n + q] = zero;
        a[q * n + p] = zero;

        // Update eigenvector matrix: V' = V * U
        let vp: Vec<_> = (0..n).map(|i| v[i * n + p]).collect();
        let vq: Vec<_> = (0..n).map(|i| v[i * n + q]).collect();
        for i in 0..n {
            // V'[i][p] = V[i][p]*c + V[i][q]*s*e^{iφ}
            v[i * n + p] = complex_add(complex_mul((c, 0.0), vp[i]), complex_mul(s_phase, vq[i]));
            // V'[i][q] = V[i][p]*(-s*e^{-iφ}) + V[i][q]*c
            v[i * n + q] = complex_add(
                complex_mul(neg_s_phase_conj, vp[i]),
                complex_mul((c, 0.0), vq[i]),
            );
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i].0).collect();
    (eigenvalues, v)
}

/// Jacobi eigendecomposition of a symmetric n×n matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column-major in an n×n array.
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
            complex_jacobi_eigendecomposition(&mut a_work, m)
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
        let mut a_work = a;
        // Row-cyclic Jacobi (O(n³·sweeps)) instead of the classic max-pivot
        // sweep (O(n⁴): an O(n²) off-diagonal search before every rotation).
        // Same spectrum to machine precision; eigenvectors differ only within
        // the eigendecomposition's intrinsic sign/rotation freedom, and eigh
        // conformance is reconstruction + spectrum based (V diag(w) Vᵀ = A).
        let (eigenvalues, eigenvectors) = jacobi_eigendecomposition_cyclic(&mut a_work, m);

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

    // Complex linear solve is unsupported in V1 — fail closed instead of
    // silently solving the real part only (the previous code did
    // `a.iter().map(|(re, _im)| re)`, corrupting any complex input). The
    // result of a solve is inexact, so integer/bool inputs promote to a
    // floating dtype.
    let dtype_b = inputs[1].dtype();
    if matches!(dtype_a, DType::Complex64 | DType::Complex128)
        || matches!(dtype_b, DType::Complex64 | DType::Complex128)
    {
        return Err(EvalError::Unsupported {
            primitive,
            detail: "complex linear solve is unsupported in V1".to_owned(),
        });
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

    let (m, n, a, _dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "det requires square matrix",
        });
    }

    let a_real: Vec<f64> = a.iter().map(|(re, _im)| *re).collect();
    let d = det(&a_real, n);
    Ok(Value::scalar_f64(d))
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

    let (m, n, a, _dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "slogdet requires square matrix",
        });
    }

    let a_real: Vec<f64> = a.iter().map(|(re, _im)| *re).collect();
    let (sign, logabsdet) = slogdet(&a_real, n);
    Ok(vec![Value::scalar_f64(sign), Value::scalar_f64(logabsdet)])
}

// ── General Eigendecomposition (eig) ────────────────────────────────

/// Evaluate general eigendecomposition for non-symmetric matrices.
///
/// Input: A square matrix [n, n]
/// Output: (eigenvalues [n] complex, eigenvectors [n, n] complex)
///
/// Uses QR iteration for small matrices. Returns complex eigenvalues/eigenvectors
/// since non-symmetric matrices can have complex eigenvalues.
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

    let (m, n, a, _dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    if m != n {
        return Err(EvalError::TypeMismatch {
            primitive,
            detail: "eig requires square matrix",
        });
    }

    let a_real: Vec<f64> = a.iter().map(|(re, _im)| *re).collect();

    // Compute eigenvalues and eigenvectors via QR iteration
    let (eigenvalues, eigenvectors) = eig_qr_iteration(&a_real, n);

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

    // For larger matrices, use simple QR iteration (V1: limited to real eigenvalues)
    let mut t = a.to_vec();
    // q_total stays purely real: it starts as the identity and is only ever
    // multiplied by the real orthogonal factor Q from qr_decomposition (whose
    // imaginary parts are exactly zero). Accumulate it as f64 via the real
    // `matrix_mul` (2 flops/element) instead of `matrix_mul_complex` (6
    // flops/element computing guaranteed-zero imaginary parts). The real-part
    // accumulation is identical (`matrix_mul_complex`'s re = sum(ar*br - ai*bi)
    // collapses to sum(ar*br) when ai = bi = 0), so the eigenvectors are
    // bit-for-bit identical.
    let mut q_total = vec![0.0_f64; n * n];
    for i in 0..n {
        q_total[i * n + i] = 1.0;
    }

    for _iter in 0..100 {
        let (q, r) = qr_decomposition(&t, n);

        // T = R * Q. R is upper triangular by construction, so the lower
        // exact-zero half of this multiply can be skipped without changing
        // the ascending-k accumulation order for any nonzero term.
        t = upper_triangular_matrix_mul(&r, &q, n);

        // Accumulate Q (real * real -> real).
        q_total = matrix_mul(&q_total, &q, n);

        // Check convergence (off-diagonal elements)
        let mut max_off_diag = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    max_off_diag = max_off_diag.max(t[i * n + j].abs());
                }
            }
        }
        if max_off_diag < 1e-10 {
            break;
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<(f64, f64)> = (0..n).map(|i| (t[i * n + i], 0.0)).collect();
    let eigenvectors: Vec<(f64, f64)> = q_total.into_iter().map(|v| (v, 0.0)).collect();

    (eigenvalues, eigenvectors)
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
/// Uses the symmetric eigendecomposition of the smaller Gram matrix (`AᵀA` when
/// `m >= n`, else `AAᵀ`): the eigenvalues are the squared singular values, so
/// `A⁺ = (AᵀA)⁺ Aᵀ` (resp. `Aᵀ (AAᵀ)⁺`) where the Gram pseudoinverse drops every
/// eigenvalue `λ` with singular value `√λ <= rcond·σ_max` (i.e. `λ <= rcond²·λ_max`),
/// exactly as JAX's SVD-based `pinv` truncates small singular values.
///
/// The previous `(AᵀA)⁻¹ Aᵀ` normal-equations form (a) ignored `rcond` entirely,
/// (b) returned all-zeros for rank-deficient inputs (because `inv(AᵀA)` failed on
/// the singular Gram matrix) where JAX returns the true pseudoinverse, and
/// (c) squared the condition number. For full-rank inputs the result is the same
/// unique Moore-Penrose inverse.
pub fn pinv(a: &[f64], m: usize, n: usize, rcond: f64) -> Vec<f64> {
    if m == 0 || n == 0 {
        return vec![0.0; n * m];
    }

    if m >= n {
        // Gram matrix G = AᵀA (n×n, symmetric PSD).
        let mut g = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += a[k * n + i] * a[k * n + j];
                }
                g[i * n + j] = sum;
            }
        }
        let g_pinv = gram_pseudoinverse(&mut g, n, rcond);
        // A⁺ = G⁺ Aᵀ  (n×m): result[r][col] = Σ_c G⁺[r][c]·Aᵀ[c][col] = Σ_c G⁺[r][c]·a[col][c]
        let mut result = vec![0.0; n * m];
        for r in 0..n {
            for col in 0..m {
                let mut sum = 0.0;
                for c in 0..n {
                    sum += g_pinv[r * n + c] * a[col * n + c];
                }
                result[r * m + col] = sum;
            }
        }
        result
    } else {
        // Gram matrix G = AAᵀ (m×m, symmetric PSD).
        let mut g = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i * n + k] * a[j * n + k];
                }
                g[i * m + j] = sum;
            }
        }
        let g_pinv = gram_pseudoinverse(&mut g, m, rcond);
        // A⁺ = Aᵀ G⁺  (n×m): result[r][col] = Σ_c Aᵀ[r][c]·G⁺[c][col] = Σ_c a[c][r]·G⁺[c][col]
        let mut result = vec![0.0; n * m];
        for r in 0..n {
            for col in 0..m {
                let mut sum = 0.0;
                for c in 0..m {
                    sum += a[c * n + r] * g_pinv[c * m + col];
                }
                result[r * m + col] = sum;
            }
        }
        result
    }
}

/// Pseudoinverse of a symmetric PSD `d×d` Gram matrix via Jacobi
/// eigendecomposition: `G⁺ = Σ_i (1/λ_i) vᵢ vᵢᵀ` over eigenvalues with
/// `√λ_i > rcond·√λ_max` (singular-value cutoff, matching JAX). `g` is consumed
/// (overwritten by the eigensolver).
fn gram_pseudoinverse(g: &mut [f64], d: usize, rcond: f64) -> Vec<f64> {
    let (lambdas, v) = jacobi_eigendecomposition(g, d);
    let lambda_max = lambdas.iter().copied().fold(0.0_f64, f64::max);
    // √λ_i > rcond·√λ_max  ⇔  λ_i > rcond²·λ_max
    let cutoff = rcond * rcond * lambda_max;
    let mut g_pinv = vec![0.0; d * d];
    for i in 0..d {
        if lambdas[i] > cutoff && lambdas[i] > 0.0 {
            let inv_l = 1.0 / lambdas[i];
            for r in 0..d {
                let vri = v[r * d + i];
                if vri == 0.0 {
                    continue;
                }
                for c in 0..d {
                    g_pinv[r * d + c] += inv_l * vri * v[c * d + i];
                }
            }
        }
    }
    g_pinv
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

pub fn solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n || b.len() != n {
        return None;
    }
    let (lu, p) = lu_factor(a, n)?;
    Some(lu_solve(&lu, &p, b, n))
}

/// Solve the linear system Ax = b for multiple right-hand sides.
///
/// A is n x n, B is n x m, returns X which is n x m. The LU factorization is
/// computed once and reused across all `m` columns (previously `A` was
/// re-factorized per column: O(n^3 * m) -> O(n^3 + n^2 * m)). Bit-for-bit
/// identical: every column previously received the same factorization of the
/// same `A`.
pub fn solve_multi_rhs(a: &[f64], b: &[f64], n: usize, m: usize) -> Option<Vec<f64>> {
    let (lu, p) = lu_factor(a, n)?;
    let mut result = vec![0.0; n * m];
    for j in 0..m {
        let rhs: Vec<f64> = (0..n).map(|i| b[i * m + j]).collect();
        let x = lu_solve(&lu, &p, &rhs, n);
        for i in 0..n {
            result[i * m + j] = x[i];
        }
    }
    Some(result)
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
    use std::collections::BTreeMap;

    fn make_matrix(m: usize, n: usize, data: &[f64]) -> Value {
        let elements: Vec<Literal> = data.iter().map(|&v| Literal::from_f64(v)).collect();
        let shape = Shape {
            dims: vec![m as u32, n as u32],
        };
        Value::Tensor(TensorValue::new(DType::F64, shape, elements).unwrap())
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
                    assert_eq!(rv.shape.dims, cv.shape.dims, "{m}x{n} full={full}: Vh shape");
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
    fn eval_solve_rejects_complex_input() {
        // Complex solve is not implemented; it must fail closed rather than
        // silently solving the real part only.
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
        let err = eval_solve(&[a, b], &BTreeMap::new())
            .expect_err("complex solve must be rejected, not silently real-dropped");
        assert!(
            matches!(
                err,
                EvalError::Unsupported {
                    primitive: Primitive::Solve,
                    ..
                }
            ),
            "expected Unsupported for complex solve, got {err:?}"
        );
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
    fn eig_qr_iteration_upper_triangular_rq_bit_identical_to_dense_reference() {
        let n = 6usize;
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                if i == j {
                    n as f64 + i as f64 * 0.25 + 3.0
                } else {
                    ((i * 5 + j * 7 + 2) % 9) as f64 * 0.125 - 0.5
                }
            })
            .collect();

        let (got_values, got_vectors) = eig_qr_iteration(&a, n);
        let (want_values, want_vectors) = {
            let mut t = a.clone();
            let mut q_total = vec![0.0_f64; n * n];
            for i in 0..n {
                q_total[i * n + i] = 1.0;
            }

            for _iter in 0..100 {
                let (q, r) = qr_decomposition(&t, n);
                t = matrix_mul(&r, &q, n);
                q_total = matrix_mul(&q_total, &q, n);

                let mut max_off_diag = 0.0f64;
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            max_off_diag = max_off_diag.max(t[i * n + j].abs());
                        }
                    }
                }
                if max_off_diag < 1e-10 {
                    break;
                }
            }

            let eigenvalues: Vec<(f64, f64)> = (0..n).map(|i| (t[i * n + i], 0.0)).collect();
            let eigenvectors: Vec<(f64, f64)> = q_total.into_iter().map(|v| (v, 0.0)).collect();
            (eigenvalues, eigenvectors)
        };

        for idx in 0..n {
            assert_eq!(got_values[idx].0.to_bits(), want_values[idx].0.to_bits());
            assert_eq!(got_values[idx].1.to_bits(), want_values[idx].1.to_bits());
        }
        for idx in 0..n * n {
            assert_eq!(got_vectors[idx].0.to_bits(), want_vectors[idx].0.to_bits());
            assert_eq!(got_vectors[idx].1.to_bits(), want_vectors[idx].1.to_bits());
        }
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
