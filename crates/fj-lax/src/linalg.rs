#![forbid(unsafe_code)]

//! Linear algebra primitives: Cholesky, triangular solve, and QR decomposition.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;
use crate::type_promotion::promote_dtype;

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
            DType::BF16 => Literal::from_bf16_f32(re as f32),
            DType::F16 => Literal::from_f16_f32(re as f32),
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
        DType::BF16 => Literal::from_bf16_f32(value as f32),
        DType::F16 => Literal::from_f16_f32(value as f32),
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
                if diag_val.0 <= 0.0 || diag_val.1.abs() > 1e-10 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "matrix is not positive definite (diagonal element {i} = {:?})",
                            diag_val
                        ),
                    });
                }
                l[i * m + j] = (diag_val.0.sqrt(), 0.0);
            } else {
                let numer = complex_sub(a[i * m + j], sum);
                l[i * m + j] = complex_div(numer, l[j * m + j]);
            }
        }
    }

    complex_matrix_to_value(m, m, &l, dtype)
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
                if complex_abs(diag) < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
                x[i * n_b + col] = complex_div(b_col[i], diag);
            }
        } else if !lower && !transpose_a {
            for i in (0..n).rev() {
                for k in (i + 1)..n {
                    b_col[i] = complex_sub(b_col[i], complex_mul(a[i * n + k], x[k * n_b + col]));
                }
                let diag = if unit_diagonal { one } else { a[i * n + i] };
                if complex_abs(diag) < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
                x[i * n_b + col] = complex_div(b_col[i], diag);
            }
        } else if lower && transpose_a {
            for i in (0..n).rev() {
                for k in (i + 1)..n {
                    b_col[i] = complex_sub(b_col[i], complex_mul(a[k * n + i], x[k * n_b + col]));
                }
                let diag = if unit_diagonal { one } else { a[i * n + i] };
                if complex_abs(diag) < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
                x[i * n_b + col] = complex_div(b_col[i], diag);
            }
        } else {
            for i in 0..n {
                for k in 0..i {
                    b_col[i] = complex_sub(b_col[i], complex_mul(a[k * n + i], x[k * n_b + col]));
                }
                let diag = if unit_diagonal { one } else { a[i * n + i] };
                if complex_abs(diag) < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
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

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    let k = m.min(n);
    let zero = (0.0, 0.0);
    let one = (1.0, 0.0);

    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");

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

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
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

    let (m, n, a, dtype) = extract_complex_matrix(primitive, &inputs[0])?;
    let k = m.min(n);
    let zero = (0.0, 0.0);

    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");

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
#[allow(dead_code)]
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

    let mut a_work = a;
    let (eigenvalues, eigenvectors) = jacobi_eigendecomposition(&mut a_work, m);

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
    let output_dtype = promote_dtype(dtype_a, dtype_a);

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

    // Extract A as f64 (ignoring complex for now - use real part)
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

    // Build output tensor
    let elements: Vec<Literal> = result.iter().map(|&v| Literal::F64Bits(v.to_bits())).collect();

    let shape = if b_cols == 1 {
        Shape { dims: vec![n as u32] }
    } else {
        Shape { dims: vec![n as u32, b_cols as u32] }
    };

    let tensor = TensorValue::new(output_dtype, shape, elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
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

        if max_val < 1e-15 {
            return 0.0; // Singular matrix
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
pub fn slogdet(a: &[f64], n: usize) -> (f64, f64) {
    let d = det(a, n);
    if d == 0.0 {
        (0.0, f64::NEG_INFINITY)
    } else if d > 0.0 {
        (1.0, d.ln())
    } else {
        (-1.0, (-d).ln())
    }
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

/// Compute the Moore-Penrose pseudoinverse via SVD.
///
/// Matches `jnp.linalg.pinv(a, rcond)`.
pub fn pinv(a: &[f64], m: usize, n: usize, rcond: f64) -> Vec<f64> {
    if m == 0 || n == 0 {
        return vec![0.0; n * m];
    }

    // For very small matrices, use direct formula
    if m == 1 && n == 1 {
        let val = a[0];
        if val.abs() < rcond {
            return vec![0.0];
        }
        return vec![1.0 / val];
    }

    // Compute A^T * A or A * A^T depending on which is smaller
    if m >= n {
        // A^+ = (A^T A)^{-1} A^T for tall/square matrices
        let mut ata = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    sum += a[k * n + i] * a[k * n + j];
                }
                ata[i * n + j] = sum;
            }
        }

        if let Some(ata_inv) = inv(&ata, n) {
            // A^+ = (A^T A)^{-1} A^T
            let mut result = vec![0.0; n * m];
            for i in 0..n {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += ata_inv[i * n + k] * a[j * n + k];
                    }
                    result[i * m + j] = sum;
                }
            }
            result
        } else {
            vec![0.0; n * m]
        }
    } else {
        // A^+ = A^T (A A^T)^{-1} for wide matrices
        let mut aat = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i * n + k] * a[j * n + k];
                }
                aat[i * m + j] = sum;
            }
        }

        if let Some(aat_inv) = inv(&aat, m) {
            // A^+ = A^T (A A^T)^{-1}
            let mut result = vec![0.0; n * m];
            for i in 0..n {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..m {
                        sum += a[k * n + i] * aat_inv[k * m + j];
                    }
                    result[i * m + j] = sum;
                }
            }
            result
        } else {
            vec![0.0; n * m]
        }
    }
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
pub fn solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n || b.len() != n {
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

        if max_val < 1e-14 {
            return None;
        }

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

    Some(x)
}

/// Solve the linear system Ax = b for multiple right-hand sides.
///
/// A is n x n, B is n x m, returns X which is n x m.
pub fn solve_multi_rhs(a: &[f64], b: &[f64], n: usize, m: usize) -> Option<Vec<f64>> {
    let mut result = vec![0.0; n * m];
    for j in 0..m {
        let rhs: Vec<f64> = (0..n).map(|i| b[i * m + j]).collect();
        let x = solve(a, &rhs, n)?;
        for i in 0..n {
            result[i * m + j] = x[i];
        }
    }
    Some(result)
}

/// Least-squares solution to Ax = b using the normal equations.
///
/// Matches `jnp.linalg.lstsq(a, b)` for overdetermined systems.
///
/// Returns the least-squares solution x that minimizes ||Ax - b||^2.
pub fn lstsq(a: &[f64], m: usize, n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    let mut atb = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += a[k * n + i] * b[k];
        }
        atb[i] = sum;
    }

    solve(&ata, &atb, n)
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
    fn cholesky_not_positive_definite() {
        let a = make_matrix(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(eval_cholesky(&[a], &BTreeMap::new()).is_err());
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
    fn solve_singular_returns_none() {
        let a = [1.0, 2.0, 2.0, 4.0]; // singular
        let b = [3.0, 6.0];
        assert!(solve(&a, &b, 2).is_none());
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
}
