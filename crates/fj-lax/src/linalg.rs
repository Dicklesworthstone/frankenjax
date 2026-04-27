#![forbid(unsafe_code)]

//! Linear algebra primitives: Cholesky, triangular solve, and QR decomposition.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;
use crate::type_promotion::promote_dtype;

// ── Helpers ──────────────────────────────────────────────────────────

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

/// Build a Value::Tensor from row-major f64 data and shape.
fn matrix_to_value(m: usize, n: usize, data: &[f64], dtype: DType) -> Result<Value, EvalError> {
    let elements: Vec<Literal> = data.iter().map(|&v| Literal::from_f64(v)).collect();
    let shape = Shape {
        dims: vec![m as u32, n as u32],
    };
    let tensor = TensorValue::new(dtype, shape, elements).map_err(EvalError::InvalidTensor)?;
    Ok(Value::Tensor(tensor))
}

// ── Cholesky decomposition ──────────────────────────────────────────

/// Compute the lower-triangular Cholesky factor L such that A = L * L^T.
///
/// Uses the standard row-by-row algorithm (Cholesky–Banachiewicz).
/// Input must be a symmetric positive-definite matrix.
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

    let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;

    if m != n {
        return Err(EvalError::Unsupported {
            primitive,
            detail: format!("Cholesky requires a square matrix, got {m}x{n}"),
        });
    }

    let mut l = vec![0.0_f64; m * m];

    for i in 0..m {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i * m + k] * l[j * m + k];
            }

            if i == j {
                let diag = a[i * m + i] - sum;
                if diag <= 0.0 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: format!(
                            "matrix is not positive definite (diagonal element {i} = {diag})"
                        ),
                    });
                }
                l[i * m + j] = diag.sqrt();
            } else {
                l[i * m + j] = (a[i * m + j] - sum) / l[j * m + j];
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

    let (m_a, n_a, a, dtype_a) = extract_matrix(primitive, &inputs[0])?;
    let (m_b, n_b, b, dtype_b) = extract_matrix(primitive, &inputs[1])?;

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

    let n = m_a; // system size

    // Solve column by column: for each column j of B, solve A x_j = b_j
    let mut x = vec![0.0_f64; n * n_b];

    for col in 0..n_b {
        // Extract column `col` from B
        let mut b_col: Vec<f64> = (0..n).map(|row| b[row * n_b + col]).collect();

        if lower && !transpose_a {
            // Forward substitution (lower triangular)
            for i in 0..n {
                for k in 0..i {
                    b_col[i] -= a[i * n + k] * x[k * n_b + col];
                }
                let diag = if unit_diagonal { 1.0 } else { a[i * n + i] };
                if diag.abs() < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
                x[i * n_b + col] = b_col[i] / diag;
            }
        } else if !lower && !transpose_a {
            // Back substitution (upper triangular)
            for i in (0..n).rev() {
                for k in (i + 1)..n {
                    b_col[i] -= a[i * n + k] * x[k * n_b + col];
                }
                let diag = if unit_diagonal { 1.0 } else { a[i * n + i] };
                if diag.abs() < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
                x[i * n_b + col] = b_col[i] / diag;
            }
        } else if lower && transpose_a {
            // Back substitution with L^T (effectively upper triangular)
            for i in (0..n).rev() {
                for k in (i + 1)..n {
                    b_col[i] -= a[k * n + i] * x[k * n_b + col]; // a[k][i] = L^T[i][k]
                }
                let diag = if unit_diagonal { 1.0 } else { a[i * n + i] };
                if diag.abs() < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
                x[i * n_b + col] = b_col[i] / diag;
            }
        } else {
            // !lower && transpose_a: forward substitution with U^T
            for i in 0..n {
                for k in 0..i {
                    b_col[i] -= a[k * n + i] * x[k * n_b + col]; // a[k][i] = U^T[i][k]
                }
                let diag = if unit_diagonal { 1.0 } else { a[i * n + i] };
                if diag.abs() < f64::EPSILON * 1e4 {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "singular or near-singular triangular matrix".to_owned(),
                    });
                }
                x[i * n_b + col] = b_col[i] / diag;
            }
        }
    }

    let out_dtype = promote_dtype(dtype_a, dtype_b);
    matrix_to_value(n, n_b, &x, out_dtype)
}

// ── QR decomposition ───────────────────────────────────────────────

/// Compute the thin QR decomposition A = Q R using Householder reflections.
///
/// Returns `[Q, R]` where Q is m×k orthogonal and R is k×n upper-triangular,
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

    let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
    let k = m.min(n);

    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");

    // Work on a copy of A; we'll transform it into R in-place using Householder reflections.
    let mut r = a;

    // Store Householder vectors for building Q later.
    let mut v_store: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut tau_store: Vec<f64> = Vec::with_capacity(k);

    for j in 0..k {
        // Extract the column vector a[j:m, j]
        let mut v: Vec<f64> = (j..m).map(|i| r[i * n + j]).collect();
        let norm_v = v.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Householder reflection: v = x - alpha*e1, tau = 2 / (v^T v)
        let alpha = if v[0] >= 0.0 { -norm_v } else { norm_v };
        v[0] -= alpha;
        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();

        if v_norm_sq > f64::EPSILON * 1e4 {
            let tau = 2.0 / v_norm_sq;

            // Apply H = I - tau * v * v^T to R[j:m, j:n]
            for col in j..n {
                let mut dot = 0.0;
                for (vi, row) in v.iter().zip(j..m) {
                    dot += vi * r[row * n + col];
                }
                for (vi, row) in v.iter().zip(j..m) {
                    r[row * n + col] -= tau * vi * dot;
                }
            }

            v_store.push(v);
            tau_store.push(tau);
        } else {
            // Zero column — no reflection needed
            v_store.push(vec![0.0; m - j]);
            tau_store.push(0.0);
        }
    }

    // Build Q by accumulating Householder reflections.
    let q_cols = if full_matrices { m } else { k };
    let mut q = vec![0.0_f64; m * q_cols];

    // Initialize Q to identity (m × q_cols)
    for i in 0..q_cols.min(m) {
        q[i * q_cols + i] = 1.0;
    }

    // Apply reflections in reverse order: Q = H_0 * H_1 * ... * H_{k-1}
    for j in (0..k).rev() {
        let v = &v_store[j];
        let tau = tau_store[j];
        if tau.abs() < f64::EPSILON {
            continue;
        }

        // Apply H_j to Q[j:m, j:q_cols]
        for col in j..q_cols {
            let mut dot = 0.0;
            for (vi, row) in v.iter().zip(j..m) {
                dot += vi * q[row * q_cols + col];
            }
            for (vi, row) in v.iter().zip(j..m) {
                q[row * q_cols + col] -= tau * vi * dot;
            }
        }
    }

    // Build R output (upper triangular portion)
    let r_rows = if full_matrices { m } else { k };
    let mut r_out = vec![0.0_f64; r_rows * n];
    for i in 0..r_rows {
        for j in i..n {
            r_out[i * n + j] = r[i * n + j];
        }
    }

    // Sign correction: ensure R has non-negative diagonal entries.
    // If R[i,i] < 0, flip signs of row i in R and column i in Q.
    for i in 0..k {
        if r_out[i * n + i] < 0.0 {
            // Flip row i of R
            for j in 0..n {
                r_out[i * n + j] = -r_out[i * n + j];
            }
            // Flip column i of Q
            for row in 0..m {
                q[row * q_cols + i] = -q[row * q_cols + i];
            }
        }
    }

    let q_val = matrix_to_value(m, q_cols, &q, dtype)?;
    let r_val = matrix_to_value(r_rows, n, &r_out, dtype)?;

    Ok(vec![q_val, r_val])
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

    let (m, n, a, dtype) = extract_matrix(primitive, &inputs[0])?;
    let k = m.min(n);

    let full_matrices = params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");

    // Step 1: Compute A^T A (n×n symmetric matrix)
    let mut ata = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0;
            for row in 0..m {
                dot += a[row * n + i] * a[row * n + j];
            }
            ata[i * n + j] = dot;
            ata[j * n + i] = dot;
        }
    }

    // Step 2: Eigendecompose A^T A via Jacobi rotations → V, eigenvalues σ²
    let (eigenvalues, v) = jacobi_eigendecomposition(&mut ata, n);

    // Step 3: Sort eigenvalues (and corresponding V columns) in descending order
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

    // Step 4: Compute U = A V Σ^{-1} (thin: m×k)
    let mut u = vec![0.0_f64; m * k];
    for i in 0..m {
        for j in 0..k {
            if sigma[j] > f64::EPSILON * 1e4 {
                let mut val = 0.0;
                for col in 0..n {
                    val += a[i * n + col] * v_sorted[col * n + j];
                }
                u[i * k + j] = val / sigma[j];
            }
        }
    }

    // Build outputs
    let u_cols = if full_matrices { m } else { k };
    let vt_rows = if full_matrices { n } else { k };

    // For full_matrices, extend U to m×m orthogonal matrix
    let u_out = if full_matrices && u_cols > k {
        extend_orthogonal_columns(&u, m, k, u_cols)
    } else {
        u
    };

    // Build V^T (transposed)
    let vt_out = if full_matrices {
        // Full V^T is n×n
        let mut vt = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                vt[i * n + j] = v_sorted[j * n + i]; // transpose
            }
        }
        vt
    } else {
        // Thin V^T is k×n
        let mut vt = vec![0.0_f64; k * n];
        for i in 0..k {
            for j in 0..n {
                vt[i * n + j] = v_sorted[j * n + i]; // transpose
            }
        }
        vt
    };

    let u_val = matrix_to_value(m, u_cols, &u_out, dtype)?;

    // S is a 1D vector
    let s_elements: Vec<Literal> = sigma.iter().map(|&v| Literal::from_f64(v)).collect();
    let s_shape = Shape {
        dims: vec![k as u32],
    };
    let s_tensor =
        TensorValue::new(dtype, s_shape, s_elements).map_err(EvalError::InvalidTensor)?;
    let s_val = Value::Tensor(s_tensor);

    let vt_val = matrix_to_value(vt_rows, n, &vt_out, dtype)?;

    Ok(vec![u_val, s_val, vt_val])
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

// ── Eigh (Symmetric Eigendecomposition) ────────────────────────────

/// Compute the eigendecomposition of a symmetric matrix: A = V diag(W) V^T.
///
/// Returns `[W, V]` where W is a vector of eigenvalues (ascending) and V
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

    // W is a 1D vector of eigenvalues
    let w_elements: Vec<Literal> = w_sorted.iter().map(|&v| Literal::from_f64(v)).collect();
    let w_shape = Shape {
        dims: vec![m as u32],
    };
    let w_tensor =
        TensorValue::new(dtype, w_shape, w_elements).map_err(EvalError::InvalidTensor)?;
    let w_val = Value::Tensor(w_tensor);

    let v_val = matrix_to_value(m, m, &v_sorted, dtype)?;

    Ok(vec![w_val, v_val])
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
}
