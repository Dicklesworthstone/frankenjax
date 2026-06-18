//! Linear algebra primitive oracle tests.
//!
//! Tests Cholesky, QR, SVD, Eigh, and TriangularSolve against
//! hand-verified analytical expected values.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive_multi;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write;

type ComplexF64 = (f64, f64);
type EigOutput = (Vec<ComplexF64>, Vec<ComplexF64>);

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn make_f64_matrix(rows: u32, cols: u32, data: &[f64]) -> Value {
    assert_eq!(data.len(), (rows * cols) as usize);
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![rows, cols],
            },
            data.iter().map(|&v| Literal::from_f64(v)).collect(),
        )
        .unwrap(),
    )
}

fn transpose(rows: usize, cols: usize, data: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = data[row * cols + col];
        }
    }
    out
}

fn extract_f64_matrix(val: &Value) -> Vec<f64> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_f64_vec_from_value(val: &Value) -> Vec<f64> {
    match val {
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
    }
}

fn extract_complex128_elements(val: &Value) -> Vec<ComplexF64> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_complex128().unwrap())
        .collect()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{context}[{i}]: got {a}, expected {e} (tol={tol})"
        );
    }
}

fn matmul(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

fn complex_sub((ar, ai): ComplexF64, (br, bi): ComplexF64) -> ComplexF64 {
    (ar - br, ai - bi)
}

fn complex_mul((ar, ai): ComplexF64, (br, bi): ComplexF64) -> ComplexF64 {
    (ar * br - ai * bi, ar * bi + ai * br)
}

fn complex_abs((re, im): ComplexF64) -> f64 {
    re.hypot(im)
}

fn assert_complex_unordered_close(
    actual: &[ComplexF64],
    expected: &[ComplexF64],
    tol: f64,
    context: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    let mut used = vec![false; actual.len()];
    for &(expected_re, expected_im) in expected {
        let mut best_idx = None;
        let mut best_dist = f64::INFINITY;
        for (idx, &(actual_re, actual_im)) in actual.iter().enumerate() {
            if used[idx] {
                continue;
            }
            let dist = (actual_re - expected_re).hypot(actual_im - expected_im);
            if dist < best_dist {
                best_idx = Some(idx);
                best_dist = dist;
            }
        }
        let idx = best_idx.unwrap_or_else(|| panic!("{context}: no unmatched eigenvalue"));
        assert!(
            best_dist <= tol,
            "{context}: eigenvalue {:?} is not within {tol} of expected ({expected_re}, {expected_im}); best actual {:?}",
            (expected_re, expected_im),
            actual[idx]
        );
        used[idx] = true;
    }
}

fn assert_eig_output_shapes(result: &[Value], n: u32, context: &str) {
    assert_eq!(result.len(), 2, "{context}: Eig should return W and V");
    for (label, value, dims) in [("W", &result[0], vec![n]), ("V", &result[1], vec![n, n])] {
        let tensor = value
            .as_tensor()
            .unwrap_or_else(|| panic!("{context}: {label} should be a tensor"));
        assert_eq!(tensor.dtype, DType::Complex128, "{context}: {label} dtype");
        assert_eq!(tensor.shape.dims, dims, "{context}: {label} shape");
        tensor
            .validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("{context}: {label} dtype/element invariant: {e}"));
    }
}

fn assert_eig_residual(a: &[f64], n: usize, w: &[ComplexF64], v: &[ComplexF64], tol: f64) -> f64 {
    let mut max_residual = 0.0_f64;
    for col in 0..n {
        let mut column_norm_sq = 0.0;
        for row in 0..n {
            column_norm_sq += complex_abs(v[row * n + col]).powi(2);
        }
        assert!(
            column_norm_sq.sqrt() > 1e-12,
            "eig eigenvector column {col} should be nonzero"
        );

        for row in 0..n {
            let mut av = (0.0, 0.0);
            for k in 0..n {
                let (v_re, v_im) = v[k * n + col];
                av.0 += a[row * n + k] * v_re;
                av.1 += a[row * n + k] * v_im;
            }
            let lambda_v = complex_mul(w[col], v[row * n + col]);
            max_residual = max_residual.max(complex_abs(complex_sub(av, lambda_v)));
        }
    }
    assert!(
        max_residual <= tol,
        "eig residual max {max_residual} exceeds tolerance {tol}"
    );
    max_residual
}

fn eig2x2_expected(a: &[f64; 4]) -> Vec<ComplexF64> {
    let trace = a[0] + a[3];
    let det = a[0] * a[3] - a[1] * a[2];
    let disc = trace * trace - 4.0 * det;
    if disc >= 0.0 {
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
    }
}

fn canonicalize_complex_values(values: &[ComplexF64], tol: f64) -> Vec<ComplexF64> {
    let mut indexed: Vec<(usize, ComplexF64)> = values
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, (re, im))| {
            let re = if re.abs() <= tol { 0.0 } else { re };
            let im = if im.abs() <= tol { 0.0 } else { im };
            (idx, (re, im))
        })
        .collect();
    indexed.sort_by(
        |(left_idx, (left_re, left_im)), (right_idx, (right_re, right_im))| {
            right_re
                .total_cmp(left_re)
                .then_with(|| right_im.total_cmp(left_im))
                .then_with(|| left_idx.cmp(right_idx))
        },
    );
    indexed.into_iter().map(|(_, value)| value).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

// ======================== Cholesky ========================

#[test]
fn oracle_det_value_and_multiplicativity() {
    // det forward VALUE is only incidentally exercised at the conformance layer (via
    // the AD VJP/JVP tests); pin a direct 2x2 value plus the multiplicativity identity
    // det(A@B) == det(A)*det(B) (oracle-free for the latter).
    let scalar = |v: &Value| -> f64 {
        match v {
            Value::Scalar(l) => l.as_f64().unwrap(),
            Value::Tensor(t) => t.elements[0].as_f64().unwrap(),
        }
    };
    let det = |m: &Value| -> f64 {
        scalar(&eval_primitive_multi(Primitive::Det, std::slice::from_ref(m), &no_params()).unwrap()[0])
    };
    let a = make_f64_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = make_f64_matrix(2, 2, &[2.0, 1.0, 0.0, 3.0]);
    let det_a = det(&a);
    let det_b = det(&b);
    assert!((det_a + 2.0).abs() < 1e-10, "det([[1,2],[3,4]]) = -2, got {det_a}");
    assert!((det_b - 6.0).abs() < 1e-10, "det([[2,1],[0,3]]) = 6, got {det_b}");
    let ab = eval_primitive_multi(Primitive::Dot, &[a, b], &no_params()).unwrap();
    let det_ab = det(&ab[0]);
    assert!(
        (det_ab - det_a * det_b).abs() < 1e-9,
        "det(A@B)={det_ab} must equal det(A)*det(B)={}",
        det_a * det_b
    );
}

#[test]
fn oracle_slogdet_reconstructs_det() {
    // slogdet(A) = (sign, log|det A|); sign * exp(log|det A|) must reconstruct det(A).
    // Cross-validates Slogdet against Det — slogdet forward value is otherwise untested
    // at the conformance layer. JAX/impl output order is (sign, logabsdet).
    let scalar = |v: &Value| -> f64 {
        match v {
            Value::Scalar(l) => l.as_f64().unwrap(),
            Value::Tensor(t) => t.elements[0].as_f64().unwrap(),
        }
    };
    let a = make_f64_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]); // det = -2
    let sl = eval_primitive_multi(Primitive::Slogdet, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(sl.len(), 2, "slogdet returns (sign, logabsdet)");
    let sign = scalar(&sl[0]);
    let logabsdet = scalar(&sl[1]);
    let det = scalar(
        &eval_primitive_multi(Primitive::Det, std::slice::from_ref(&a), &no_params()).unwrap()[0],
    );
    assert!((det + 2.0).abs() < 1e-10, "det([[1,2],[3,4]]) = -2");
    assert!(
        (sign * logabsdet.exp() - det).abs() < 1e-9,
        "sign*exp(logabsdet)={} must reconstruct det={det}",
        sign * logabsdet.exp()
    );
}

#[test]
fn oracle_det_singular_is_zero() {
    // det of a singular matrix is 0 (must not error or return NaN). Rows of A are
    // proportional ([1,2] and [2,4]), so det = 1*4 - 2*2 = 0 exactly. JAX returns 0.
    let scalar = |v: &Value| -> f64 {
        match v {
            Value::Scalar(l) => l.as_f64().unwrap(),
            Value::Tensor(t) => t.elements[0].as_f64().unwrap(),
        }
    };
    let a = make_f64_matrix(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    let det = scalar(
        &eval_primitive_multi(Primitive::Det, std::slice::from_ref(&a), &no_params()).unwrap()[0],
    );
    assert!(
        det.abs() < 1e-12,
        "det of a singular matrix must be 0, got {det}"
    );
}

#[test]
fn oracle_cholesky_2x2_identity() {
    // Cholesky of I₂ = I₂
    let a = make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let result =
        eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(result.len(), 1);
    let l = extract_f64_matrix(&result[0]);
    assert_close(&l, &[1.0, 0.0, 0.0, 1.0], 1e-12, "cholesky(I₂)");
}

#[test]
fn oracle_cholesky_2x2_spd() {
    // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, √2]]
    let a = make_f64_matrix(2, 2, &[4.0, 2.0, 2.0, 3.0]);
    let result =
        eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params()).unwrap();
    let l = extract_f64_matrix(&result[0]);
    let expected = [2.0, 0.0, 1.0, 2.0_f64.sqrt()];
    assert_close(&l, &expected, 1e-12, "cholesky([[4,2],[2,3]])");

    // Verify L @ L^T = A
    let lt = [l[0], l[2], l[1], l[3]]; // transpose
    let reconstructed = matmul(2, 2, 2, &l, &lt);
    assert_close(
        &reconstructed,
        &[4.0, 2.0, 2.0, 3.0],
        1e-12,
        "L@L^T should equal A",
    );
}

#[test]
fn oracle_cholesky_3x3() {
    // A = [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]
    // Known Cholesky: L = [[5, 0, 0], [3, 3, 0], [-1, 1, 3]]
    let a = make_f64_matrix(3, 3, &[25.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0]);
    let result =
        eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params()).unwrap();
    let l = extract_f64_matrix(&result[0]);
    let expected = [5.0, 0.0, 0.0, 3.0, 3.0, 0.0, -1.0, 1.0, 3.0];
    assert_close(&l, &expected, 1e-12, "cholesky(3×3)");
}

#[test]
fn oracle_cholesky_4x4_known_factor() {
    // Build A = L @ L^T from a known lower-triangular factor and recover L.
    let expected_l = [
        2.0, 0.0, 0.0, 0.0, //
        -1.0, 3.0, 0.0, 0.0, //
        4.0, 2.0, 1.0, 0.0, //
        3.0, -2.0, 5.0, 2.0,
    ];
    let lt = transpose(4, 4, &expected_l);
    let a = matmul(4, 4, 4, &expected_l, &lt);
    let result = eval_primitive_multi(
        Primitive::Cholesky,
        &[make_f64_matrix(4, 4, &a)],
        &no_params(),
    )
    .unwrap();
    let l = extract_f64_matrix(&result[0]);
    assert_close(&l, &expected_l, 1e-10, "cholesky(4x4 known factor)");
}

// ======================== QR Decomposition ========================

#[test]
fn oracle_qr_identity() {
    // QR of I₂: Q=I (or -I), R=I (or -I), Q@R=I
    let a = make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let result =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(result.len(), 2);
    let q = extract_f64_matrix(&result[0]);
    let r = extract_f64_matrix(&result[1]);

    // Q@R should reconstruct A
    let reconstructed = matmul(2, 2, 2, &q, &r);
    assert_close(&reconstructed, &[1.0, 0.0, 0.0, 1.0], 1e-12, "Q@R = I₂");

    // Q should be orthogonal: Q^T @ Q = I
    let qt = [q[0], q[2], q[1], q[3]];
    let qtq = matmul(2, 2, 2, &qt, &q);
    assert_close(&qtq, &[1.0, 0.0, 0.0, 1.0], 1e-12, "Q^T@Q = I");
}

#[test]
fn oracle_qr_2x2() {
    // QR of [[1, -1], [1, 1]]
    let a = make_f64_matrix(2, 2, &[1.0, -1.0, 1.0, 1.0]);
    let result =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &no_params()).unwrap();
    let q = extract_f64_matrix(&result[0]);
    let r = extract_f64_matrix(&result[1]);

    // Q@R = A
    let reconstructed = matmul(2, 2, 2, &q, &r);
    assert_close(&reconstructed, &[1.0, -1.0, 1.0, 1.0], 1e-12, "Q@R = A");

    // Q orthogonal
    let qt = [q[0], q[2], q[1], q[3]];
    let qtq = matmul(2, 2, 2, &qt, &q);
    assert_close(&qtq, &[1.0, 0.0, 0.0, 1.0], 1e-12, "Q^T@Q = I");

    // R upper triangular: r[1][0] = 0
    assert!(
        r[2].abs() < 1e-12,
        "R should be upper triangular, r[1][0] = {}",
        r[2]
    );
}

#[test]
fn oracle_qr_4x3_tall_matrix() {
    let a_data = [
        1.0, 2.0, 0.0, //
        0.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, //
        2.0, 1.0, 3.0,
    ];
    let a = make_f64_matrix(4, 3, &a_data);
    let result =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &no_params()).unwrap();
    let q = extract_f64_matrix(&result[0]);
    let r = extract_f64_matrix(&result[1]);

    let reconstructed = matmul(4, 3, 3, &q, &r);
    assert_close(&reconstructed, &a_data, 1e-10, "Q@R = A for tall matrix");

    let qt = transpose(4, 3, &q);
    let qtq = matmul(3, 4, 3, &qt, &q);
    assert_close(
        &qtq,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        1e-10,
        "Q^T@Q = I for tall matrix",
    );

    for row in 1..3 {
        for col in 0..row {
            assert!(
                r[row * 3 + col].abs() < 1e-10,
                "R[{row},{col}] should be zero, got {}",
                r[row * 3 + col]
            );
        }
    }
}

// ======================== SVD ========================

#[test]
fn oracle_svd_identity() {
    // SVD of I₂: U=I, S=[1,1], V^T=I (up to sign)
    let a = make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let result =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(result.len(), 3);
    let u = extract_f64_matrix(&result[0]);
    let s = extract_f64_vec_from_value(&result[1]);
    let vt = extract_f64_matrix(&result[2]);

    // Singular values should be [1, 1]
    assert_close(&s, &[1.0, 1.0], 1e-12, "svd(I₂) singular values");

    // U @ diag(S) @ V^T = A
    let us = [u[0] * s[0], u[1] * s[1], u[2] * s[0], u[3] * s[1]];
    let reconstructed = matmul(2, 2, 2, &us, &vt);
    assert_close(&reconstructed, &[1.0, 0.0, 0.0, 1.0], 1e-12, "U@S@V^T = I₂");
}

#[test]
fn oracle_svd_2x2() {
    // SVD of [[3, 0], [0, -2]]: singular values should be [3, 2]
    let a = make_f64_matrix(2, 2, &[3.0, 0.0, 0.0, -2.0]);
    let result =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    let u = extract_f64_matrix(&result[0]);
    let s = extract_f64_vec_from_value(&result[1]);
    let vt = extract_f64_matrix(&result[2]);

    // Singular values in descending order
    assert!(s[0] >= s[1], "singular values should be sorted descending");
    assert_close(&s, &[3.0, 2.0], 1e-12, "svd singular values");

    // Reconstruct
    let us = [u[0] * s[0], u[1] * s[1], u[2] * s[0], u[3] * s[1]];
    let reconstructed = matmul(2, 2, 2, &us, &vt);
    assert_close(&reconstructed, &[3.0, 0.0, 0.0, -2.0], 1e-12, "U@S@V^T = A");
}

#[test]
fn oracle_svd_3x2_rectangular() {
    let a_data = [
        3.0, 1.0, //
        0.0, 2.0, //
        0.0, 0.0,
    ];
    let a = make_f64_matrix(3, 2, &a_data);
    let result =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    let u = extract_f64_matrix(&result[0]);
    let s = extract_f64_vec_from_value(&result[1]);
    let vt = extract_f64_matrix(&result[2]);

    assert_eq!(s.len(), 2);
    assert!(s[0] >= s[1], "singular values should be sorted descending");

    let us = [
        u[0] * s[0],
        u[1] * s[1],
        u[2] * s[0],
        u[3] * s[1],
        u[4] * s[0],
        u[5] * s[1],
    ];
    let reconstructed = matmul(3, 2, 2, &us, &vt);
    assert_close(
        &reconstructed,
        &a_data,
        1e-10,
        "U@diag(S)@V^T = A for rectangular matrix",
    );
}

#[test]
fn oracle_svd_wide_2x4_reconstructs() {
    // Wide matrix (m<n): k=min(m,n)=2 ⇒ U[2×2], S[2], Vt[2×4]; U·diag(S)·Vt = A.
    // Previously only tall/square SVD shapes were covered.
    let a_data = [1.0, 2.0, 0.5, -1.0, 0.3, 1.5, 2.0, 0.7];
    let a = make_f64_matrix(2, 4, &a_data);
    let result =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    let u = extract_f64_matrix(&result[0]);
    let s = extract_f64_vec_from_value(&result[1]);
    let vt = extract_f64_matrix(&result[2]);
    assert_eq!(s.len(), 2, "wide SVD: k=min(m,n)=2 singular values");
    assert!(s[0] >= s[1], "singular values sorted descending");
    let us = [u[0] * s[0], u[1] * s[1], u[2] * s[0], u[3] * s[1]];
    let reconstructed = matmul(2, 2, 4, &us, &vt);
    assert_close(&reconstructed, &a_data, 1e-10, "U·diag(S)·Vt = A (wide)");
}

#[test]
fn oracle_svd_rank_deficient_3x3_reconstructs() {
    // Rank-2 3×3 (row 2 = row 0 + row 1): smallest singular value ≈ 0, and the
    // decomposition must still reconstruct A.
    let a_data = [
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
        5.0, 7.0, 9.0,
    ];
    let a = make_f64_matrix(3, 3, &a_data);
    let result =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    let u = extract_f64_matrix(&result[0]);
    let s = extract_f64_vec_from_value(&result[1]);
    let vt = extract_f64_matrix(&result[2]);
    assert_eq!(s.len(), 3);
    assert!(
        s[0] >= s[1] && s[1] >= s[2],
        "singular values sorted descending"
    );
    // The exact-zero singular value is recovered to ~ε·‖A‖ (≈1e-15) because the SVD
    // uses one-sided Jacobi (orthogonalizes A's columns directly). The old AᵀA /
    // normal-equations path squared the condition number and gave ≈2.7e-8 here, which
    // this 1e-10 bound provably rejects — a regression guard for the high-relative-
    // accuracy property (frankenjax-96i7w).
    assert!(
        s[2] < 1e-10,
        "rank-2 matrix zero singular value should be ~ε·‖A‖, got {} (s0={})",
        s[2],
        s[0]
    );
    let mut us = [0.0_f64; 9];
    for r in 0..3 {
        for c in 0..3 {
            us[r * 3 + c] = u[r * 3 + c] * s[c];
        }
    }
    let reconstructed = matmul(3, 3, 3, &us, &vt);
    assert_close(
        &reconstructed,
        &a_data,
        1e-7,
        "U·diag(S)·Vt = A (rank-deficient)",
    );
}

// ======================== Eigh (Symmetric Eigendecomposition) ========================

#[test]
fn oracle_eigh_identity() {
    // Eigh of I₂: eigenvalues [1, 1], eigenvectors = I (up to sign/ordering)
    let a = make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let result =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(result.len(), 2);
    let w = extract_f64_vec_from_value(&result[0]); // eigenvalues
    let v = extract_f64_matrix(&result[1]); // eigenvectors (columns)

    assert_close(&w, &[1.0, 1.0], 1e-12, "eigh(I₂) eigenvalues");

    // V @ diag(w) @ V^T = A
    let vw = [v[0] * w[0], v[1] * w[1], v[2] * w[0], v[3] * w[1]];
    let vt = [v[0], v[2], v[1], v[3]];
    let reconstructed = matmul(2, 2, 2, &vw, &vt);
    assert_close(
        &reconstructed,
        &[1.0, 0.0, 0.0, 1.0],
        1e-12,
        "V@diag(w)@V^T = I₂",
    );
}

#[test]
fn oracle_eigh_symmetric() {
    // Eigh of [[2, 1], [1, 2]]: eigenvalues [1, 3]
    let a = make_f64_matrix(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let result =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params()).unwrap();
    let w = extract_f64_vec_from_value(&result[0]);
    let v = extract_f64_matrix(&result[1]);

    // Eigenvalues should be 1 and 3 (ascending order)
    assert_close(&w, &[1.0, 3.0], 1e-12, "eigh eigenvalues");

    // Reconstruct: V @ diag(w) @ V^T = A
    let vw = [v[0] * w[0], v[1] * w[1], v[2] * w[0], v[3] * w[1]];
    let vt = [v[0], v[2], v[1], v[3]];
    let reconstructed = matmul(2, 2, 2, &vw, &vt);
    assert_close(
        &reconstructed,
        &[2.0, 1.0, 1.0, 2.0],
        1e-12,
        "V@diag(w)@V^T = A",
    );

    // V should be orthogonal
    let vtv = matmul(2, 2, 2, &vt, &v[..]);
    assert_close(&vtv, &[1.0, 0.0, 0.0, 1.0], 1e-12, "V^T@V = I");
}

#[test]
fn oracle_eigh_3x3_diagonal_repeated_eigenvalue() {
    let a_data = [
        2.0, 0.0, 0.0, //
        0.0, 2.0, 0.0, //
        0.0, 0.0, 5.0,
    ];
    let a = make_f64_matrix(3, 3, &a_data);
    let result =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params()).unwrap();
    let w = extract_f64_vec_from_value(&result[0]);
    let v = extract_f64_matrix(&result[1]);

    assert_close(&w, &[2.0, 2.0, 5.0], 1e-12, "eigh repeated eigenvalues");

    let vt = transpose(3, 3, &v);
    let vtv = matmul(3, 3, 3, &vt, &v);
    assert_close(
        &vtv,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        1e-10,
        "V^T@V = I for repeated-eigenvalue case",
    );

    let vw = [
        v[0] * w[0],
        v[1] * w[1],
        v[2] * w[2],
        v[3] * w[0],
        v[4] * w[1],
        v[5] * w[2],
        v[6] * w[0],
        v[7] * w[1],
        v[8] * w[2],
    ];
    let reconstructed = matmul(3, 3, 3, &vw, &vt);
    assert_close(
        &reconstructed,
        &a_data,
        1e-10,
        "V@diag(w)@V^T = A for repeated-eigenvalue case",
    );
}

// ======================== Eig (General Eigendecomposition) ========================

fn eval_eig_case_n(n: u32, a_data: &[f64], context: &str) -> EigOutput {
    let a = make_f64_matrix(n, n, a_data);
    let result =
        eval_primitive_multi(Primitive::Eig, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eig_output_shapes(&result, n, context);
    (
        extract_complex128_elements(&result[0]),
        extract_complex128_elements(&result[1]),
    )
}

fn eval_eig_case(a_data: &[f64; 4], context: &str) -> EigOutput {
    eval_eig_case_n(2, a_data, context)
}

#[test]
fn oracle_eig_real_diagonal_dominant_contract() {
    let a_data = [
        6.0, 0.25, //
        0.5, 2.0,
    ];
    let (w, v) = eval_eig_case(&a_data, "real diagonal-dominant eig");
    let expected = eig2x2_expected(&a_data);

    assert_complex_unordered_close(&w, &expected, 1e-10, "diagonal-dominant eig eigenvalues");
    assert_eig_residual(&a_data, 2, &w, &v, 1e-10);
}

#[test]
fn oracle_eig_nonsymmetric_real_contract() {
    let a_data = [
        3.0, 4.0, //
        1.0, 2.0,
    ];
    let (w, v) = eval_eig_case(&a_data, "nonsymmetric real eig");
    let expected = eig2x2_expected(&a_data);

    assert_complex_unordered_close(&w, &expected, 1e-10, "nonsymmetric eig eigenvalues");
    assert_eig_residual(&a_data, 2, &w, &v, 1e-10);
}

#[test]
fn oracle_eig_repeated_and_near_repeated_contract() {
    let repeated = [
        2.0, 0.0, //
        0.0, 2.0,
    ];
    let (w_repeated, v_repeated) = eval_eig_case(&repeated, "repeated eig");
    assert_complex_unordered_close(
        &w_repeated,
        &[(2.0, 0.0), (2.0, 0.0)],
        1e-12,
        "repeated eig eigenvalues",
    );
    assert_eig_residual(&repeated, 2, &w_repeated, &v_repeated, 1e-12);

    let near_repeated = [
        1.0 + 1e-8,
        0.0, //
        0.0,
        1.0,
    ];
    let (w_near, v_near) = eval_eig_case(&near_repeated, "near-repeated eig");
    assert_complex_unordered_close(
        &w_near,
        &[(1.0 + 1e-8, 0.0), (1.0, 0.0)],
        1e-8,
        "near-repeated eig eigenvalues",
    );
    assert_eig_residual(&near_repeated, 2, &w_near, &v_near, 1e-8);
}

#[test]
fn oracle_eig_complex_conjugate_contract() {
    let a_data = [
        0.5, -2.0, //
        2.0, 0.5,
    ];
    let (w, v) = eval_eig_case(&a_data, "complex-conjugate eig");
    let expected = eig2x2_expected(&a_data);

    assert_complex_unordered_close(&w, &expected, 1e-12, "complex-pair eig eigenvalues");
    assert!(
        (w[0].0 - w[1].0).abs() <= 1e-12,
        "complex conjugate pair should share a real part: {w:?}"
    );
    assert!(
        (w[0].1 + w[1].1).abs() <= 1e-12,
        "complex conjugate pair imaginary parts should cancel: {w:?}"
    );
    for (idx, value) in v.iter().enumerate() {
        assert!(
            value.0.is_finite() && value.1.is_finite(),
            "complex-pair eigenvector element {idx} should be finite: {value:?}"
        );
    }
}

#[test]
fn oracle_eig_three_by_three_diagonal_deflation_contract() {
    let a_data = [
        7.0, 0.0, 0.0, //
        0.0, 3.0, 0.0, //
        0.0, 0.0, -2.0,
    ];
    let (w, v) = eval_eig_case_n(3, &a_data, "3x3 diagonal eig");

    assert_complex_unordered_close(
        &w,
        &[(7.0, 0.0), (3.0, 0.0), (-2.0, 0.0)],
        1e-12,
        "3x3 diagonal eig eigenvalues",
    );
    assert_eig_residual(&a_data, 3, &w, &v, 1e-12);
}

fn eig_contract_summary() -> String {
    let cases = [
        (
            "real_diagonal_dominant",
            2,
            [
                6.0, 0.25, //
                0.5, 2.0,
            ]
            .as_slice(),
            None,
            true,
            1e-10,
        ),
        (
            "nonsymmetric_real",
            2,
            [
                3.0, 4.0, //
                1.0, 2.0,
            ]
            .as_slice(),
            None,
            true,
            1e-10,
        ),
        (
            "repeated",
            2,
            [
                2.0, 0.0, //
                0.0, 2.0,
            ]
            .as_slice(),
            None,
            true,
            1e-12,
        ),
        (
            "near_repeated",
            2,
            [
                1.0 + 1e-8,
                0.0, //
                0.0,
                1.0,
            ]
            .as_slice(),
            Some(vec![(1.0 + 1e-8, 0.0), (1.0, 0.0)]),
            true,
            1e-8,
        ),
        (
            "complex_conjugate",
            2,
            [
                0.5, -2.0, //
                2.0, 0.5,
            ]
            .as_slice(),
            None,
            false,
            1e-12,
        ),
        (
            "three_by_three_diagonal_deflation",
            3,
            [
                7.0, 0.0, 0.0, //
                0.0, 3.0, 0.0, //
                0.0, 0.0, -2.0,
            ]
            .as_slice(),
            Some(vec![(7.0, 0.0), (3.0, 0.0), (-2.0, 0.0)]),
            true,
            1e-12,
        ),
    ];

    let mut summary = String::new();
    writeln!(
        &mut summary,
        "Primitive::Eig contract v1; public order unspecified; canonical order=real_desc,imag_desc,original_index_tie"
    )
    .unwrap();
    for (name, n, a_data, expected, check_residual, tol) in cases {
        let (w, v) = eval_eig_case_n(n, a_data, name);
        let expected = expected.unwrap_or_else(|| {
            assert_eq!(n, 2, "{name}: analytic helper only covers 2x2 eig cases");
            eig2x2_expected(&[a_data[0], a_data[1], a_data[2], a_data[3]])
        });
        assert_complex_unordered_close(&w, &expected, tol, name);
        let residual_status = if check_residual {
            assert_eig_residual(a_data, n as usize, &w, &v, tol);
            "checked_within_tol"
        } else {
            "not_asserted_v1_complex_pair"
        };
        let canonical = canonicalize_complex_values(&expected, tol);
        let conjugate_status = if name == "complex_conjugate" {
            assert!(
                (canonical[0].0 - canonical[1].0).abs() <= tol
                    && (canonical[0].1 + canonical[1].1).abs() <= tol,
                "complex_conjugate summary requires a conjugate pair: {canonical:?}"
            );
            "checked"
        } else {
            "not_applicable"
        };
        write!(
            &mut summary,
            "case={name};tol={tol:.0e};w_dtype=Complex128;w_shape=[{n}];v_dtype=Complex128;v_shape=[{n},{n}];eigenvalue_multiset=checked;canonical_slots={};expected=",
            canonical.len()
        )
        .unwrap();
        for (idx, (re, im)) in canonical.iter().enumerate() {
            if idx > 0 {
                summary.push(',');
            }
            write!(&mut summary, "({re:.12e},{im:.12e})").unwrap();
        }
        summary.push(';');
        writeln!(
            &mut summary,
            "residual={residual_status};conjugate_pair={conjugate_status}"
        )
        .unwrap();
    }
    summary
}

#[test]
fn golden_eig_general_contract_sha256() {
    let summary = eig_contract_summary();
    assert_eq!(
        sha256_hex(summary.as_bytes()),
        "9b84bf2da55576f8de5832828ae6cb271d82a59dc11212513acd95ec29603791",
        "Primitive::Eig contract summary:\n{summary}"
    );
}

// ======================== TriangularSolve ========================

#[test]
fn oracle_triangular_solve_lower_identity() {
    // Solve I₂ @ X = B → X = B
    let a = make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let b = make_f64_matrix(2, 2, &[3.0, 4.0, 5.0, 6.0]);
    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());
    let result = eval_primitive_multi(Primitive::TriangularSolve, &[a, b], &params).unwrap();
    assert_eq!(result.len(), 1);
    let x = extract_f64_matrix(&result[0]);
    assert_close(&x, &[3.0, 4.0, 5.0, 6.0], 1e-12, "I@X=B → X=B");
}

#[test]
fn oracle_triangular_solve_lower_2x2() {
    // L = [[2, 0], [1, 3]], B = [[4], [7]]
    // L @ X = B: X[0] = 4/2 = 2, X[1] = (7 - 1*2)/3 = 5/3
    let a = make_f64_matrix(2, 2, &[2.0, 0.0, 1.0, 3.0]);
    let b = make_f64_matrix(2, 1, &[4.0, 7.0]);
    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());
    let result = eval_primitive_multi(Primitive::TriangularSolve, &[a, b], &params).unwrap();
    let x = extract_f64_matrix(&result[0]);
    assert_close(&x, &[2.0, 5.0 / 3.0], 1e-12, "triangular solve L@X=B");
}

#[test]
fn oracle_triangular_solve_upper_2x2() {
    // U = [[3, 1], [0, 2]], B = [[5], [4]]
    // U @ X = B: X[1] = 4/2 = 2, X[0] = (5 - 1*2)/3 = 1
    let a = make_f64_matrix(2, 2, &[3.0, 1.0, 0.0, 2.0]);
    let b = make_f64_matrix(2, 1, &[5.0, 4.0]);
    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "false".to_owned());
    let result = eval_primitive_multi(Primitive::TriangularSolve, &[a, b], &params).unwrap();
    let x = extract_f64_matrix(&result[0]);
    assert_close(&x, &[1.0, 2.0], 1e-12, "triangular solve U@X=B");
}

#[test]
fn oracle_triangular_solve_transpose_unit_diagonal_multiple_rhs() {
    // L has implicit unit diagonal; solve L^T X = B with two RHS columns.
    let a = make_f64_matrix(
        3,
        3,
        &[
            1.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, //
            -1.0, 3.0, 1.0,
        ],
    );
    let expected_x = [
        2.0, -1.0, //
        4.0, 3.0, //
        5.0, -2.0,
    ];
    let lt = transpose(3, 3, &extract_f64_matrix(&a));
    let b = matmul(3, 3, 2, &lt, &expected_x);
    let rhs = make_f64_matrix(3, 2, &b);
    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());
    params.insert("transpose_a".to_owned(), "true".to_owned());
    params.insert("unit_diagonal".to_owned(), "true".to_owned());

    let result = eval_primitive_multi(Primitive::TriangularSolve, &[a, rhs], &params).unwrap();
    let x = extract_f64_matrix(&result[0]);
    assert_close(
        &x,
        &expected_x,
        1e-12,
        "triangular solve with transpose_a and unit_diagonal",
    );
}

// ======================== 1x1 Matrix Edge Cases ========================

#[test]
fn oracle_cholesky_1x1() {
    // Cholesky of [[4]] = [[2]]
    let a = make_f64_matrix(1, 1, &[4.0]);
    let result =
        eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params()).unwrap();
    let l = extract_f64_matrix(&result[0]);
    assert_close(&l, &[2.0], 1e-12, "cholesky([[4]]) = [[2]]");
}

#[test]
fn oracle_qr_1x1() {
    // QR of [[3]]: Q = [[1]] or [[-1]], R = [[3]] or [[-3]]
    let a = make_f64_matrix(1, 1, &[3.0]);
    let result =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(result.len(), 2);
    let q = extract_f64_matrix(&result[0]);
    let r = extract_f64_matrix(&result[1]);
    // Q*R should equal A
    let qr = q[0] * r[0];
    assert_close(&[qr], &[3.0], 1e-12, "Q*R = [[3]]");
    // |Q| = 1
    assert_close(&[q[0].abs()], &[1.0], 1e-12, "|Q| = 1");
}

#[test]
fn oracle_svd_1x1() {
    // SVD of [[5]]: U = [[±1]], S = [[5]], V = [[±1]]
    let a = make_f64_matrix(1, 1, &[5.0]);
    let result =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(result.len(), 3);
    let u = extract_f64_matrix(&result[0]);
    let s = extract_f64_vec_from_value(&result[1]);
    let vt = extract_f64_matrix(&result[2]);
    // U * S * V^T = A
    let usv = u[0] * s[0] * vt[0];
    assert_close(&[usv], &[5.0], 1e-12, "U*S*V^T = [[5]]");
    // S = 5
    assert_close(&[s[0]], &[5.0], 1e-12, "singular value = 5");
}

#[test]
fn oracle_eigh_1x1() {
    // Eigh of [[7]]: eigenvalue = 7, eigenvector = [[1]] (or [[-1]])
    // Output order is [W, V] where W=eigenvalues, V=eigenvectors
    let a = make_f64_matrix(1, 1, &[7.0]);
    let result =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params()).unwrap();
    assert_eq!(result.len(), 2);
    let eigenvalues = extract_f64_vec_from_value(&result[0]);
    let eigenvectors = extract_f64_matrix(&result[1]);
    assert_close(&eigenvalues, &[7.0], 1e-12, "eigenvalue = 7");
    assert_close(&[eigenvectors[0].abs()], &[1.0], 1e-12, "|eigenvector| = 1");
}

#[test]
fn oracle_triangular_solve_1x1() {
    // Solve [[2]] @ x = [[6]] → x = [[3]]
    let a = make_f64_matrix(1, 1, &[2.0]);
    let b = make_f64_matrix(1, 1, &[6.0]);
    let result = eval_primitive_multi(Primitive::TriangularSolve, &[a, b], &no_params()).unwrap();
    let x = extract_f64_matrix(&result[0]);
    assert_close(&x, &[3.0], 1e-12, "[[2]] @ x = [[6]] → x = [[3]]");
}

// ======================== Metamorphic Tests ========================

#[test]
fn metamorphic_qr_reconstruction() {
    // Q @ R = A
    let a = make_f64_matrix(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
    let result =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &no_params()).unwrap();
    let q = extract_f64_matrix(&result[0]);
    let r = extract_f64_matrix(&result[1]);
    let qr = matmul(3, 3, 3, &q, &r);
    assert_close(
        &qr,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        1e-10,
        "Q @ R should reconstruct A",
    );
}

#[test]
fn metamorphic_svd_reconstruction() {
    // U @ diag(S) @ V^T = A
    let a = make_f64_matrix(2, 2, &[3.0, 1.0, 1.0, 2.0]);
    let result =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    let u = extract_f64_matrix(&result[0]);
    let s = extract_f64_vec_from_value(&result[1]);
    let vt = extract_f64_matrix(&result[2]);

    // Build diag(S) * V^T
    let svt = [s[0] * vt[0], s[0] * vt[1], s[1] * vt[2], s[1] * vt[3]];
    let usv = matmul(2, 2, 2, &u, &svt);
    assert_close(
        &usv,
        &[3.0, 1.0, 1.0, 2.0],
        1e-10,
        "U @ diag(S) @ V^T should reconstruct A",
    );
}

#[test]
fn metamorphic_cholesky_reconstruction() {
    // L @ L^T = A for positive definite A
    let a = make_f64_matrix(2, 2, &[4.0, 2.0, 2.0, 3.0]);
    let result =
        eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params()).unwrap();
    let l = extract_f64_matrix(&result[0]);
    let lt = transpose(2, 2, &l);
    let llt = matmul(2, 2, 2, &l, &lt);
    assert_close(
        &llt,
        &[4.0, 2.0, 2.0, 3.0],
        1e-10,
        "L @ L^T should reconstruct A",
    );
}

#[test]
fn metamorphic_eigh_reconstruction() {
    // V @ diag(W) @ V^T = A for symmetric A
    let a = make_f64_matrix(2, 2, &[3.0, 1.0, 1.0, 2.0]);
    let result =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params()).unwrap();
    let w = extract_f64_vec_from_value(&result[0]);
    let v = extract_f64_matrix(&result[1]);
    let vt = transpose(2, 2, &v);

    // Build V @ diag(W)
    let vw = [v[0] * w[0], v[1] * w[1], v[2] * w[0], v[3] * w[1]];
    let reconstructed = matmul(2, 2, 2, &vw, &vt);
    assert_close(
        &reconstructed,
        &[3.0, 1.0, 1.0, 2.0],
        1e-10,
        "V @ diag(W) @ V^T should reconstruct A",
    );
}

// ======================== Scaling Metamorphic Tests ========================

#[test]
fn metamorphic_svd_scaling() {
    // If A has SVD U*S*V^T, then c*A has singular values c*S (for c > 0)
    let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let scale = 3.0;
    let scaled_data: Vec<f64> = a_data.iter().map(|&x| x * scale).collect();

    let a = make_f64_matrix(2, 3, &a_data);
    let scaled_a = make_f64_matrix(2, 3, &scaled_data);

    let result_a =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params()).unwrap();
    let result_scaled = eval_primitive_multi(
        Primitive::Svd,
        std::slice::from_ref(&scaled_a),
        &no_params(),
    )
    .unwrap();

    let s_a = extract_f64_vec_from_value(&result_a[1]);
    let s_scaled = extract_f64_vec_from_value(&result_scaled[1]);

    let expected_scaled_s: Vec<f64> = s_a.iter().map(|&x| x * scale).collect();
    assert_close(
        &s_scaled,
        &expected_scaled_s,
        1e-10,
        "SVD singular values should scale linearly",
    );
}

#[test]
fn metamorphic_cholesky_scaling() {
    // If A = L*L^T, then c²*A = (c*L)*(c*L)^T (for c > 0)
    let a_data = [4.0, 2.0, 2.0, 5.0];
    let scale = 2.0;
    let scaled_data: Vec<f64> = a_data.iter().map(|&x| x * scale * scale).collect();

    let a = make_f64_matrix(2, 2, &a_data);
    let scaled_a = make_f64_matrix(2, 2, &scaled_data);

    let result_a =
        eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params()).unwrap();
    let result_scaled = eval_primitive_multi(
        Primitive::Cholesky,
        std::slice::from_ref(&scaled_a),
        &no_params(),
    )
    .unwrap();

    let l_a = extract_f64_matrix(&result_a[0]);
    let l_scaled = extract_f64_matrix(&result_scaled[0]);

    let expected_scaled_l: Vec<f64> = l_a.iter().map(|&x| x * scale).collect();
    assert_close(
        &l_scaled,
        &expected_scaled_l,
        1e-10,
        "Cholesky L should scale by sqrt of matrix scale factor",
    );
}

#[test]
fn metamorphic_eigh_scaling() {
    // If A has eigenvalues W, then c*A has eigenvalues c*W
    let a_data = [3.0, 1.0, 1.0, 2.0];
    let scale = 4.0;
    let scaled_data: Vec<f64> = a_data.iter().map(|&x| x * scale).collect();

    let a = make_f64_matrix(2, 2, &a_data);
    let scaled_a = make_f64_matrix(2, 2, &scaled_data);

    let result_a =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params()).unwrap();
    let result_scaled = eval_primitive_multi(
        Primitive::Eigh,
        std::slice::from_ref(&scaled_a),
        &no_params(),
    )
    .unwrap();

    let w_a = extract_f64_vec_from_value(&result_a[0]);
    let w_scaled = extract_f64_vec_from_value(&result_scaled[0]);

    let expected_scaled_w: Vec<f64> = w_a.iter().map(|&x| x * scale).collect();
    assert_close(
        &w_scaled,
        &expected_scaled_w,
        1e-10,
        "Eigh eigenvalues should scale linearly",
    );
}

// ======================== F32 dtype preservation (f8871c7) ========================

fn make_f32_matrix(rows: u32, cols: u32, data: &[f32]) -> Value {
    assert_eq!(data.len(), (rows * cols) as usize);
    Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape {
                dims: vec![rows, cols],
            },
            data.iter().map(|&v| Literal::from_f32(v)).collect(),
        )
        .unwrap(),
    )
}

// `matrix_to_value` (shared by Cholesky, QR, SVD, Eigh, TriangularSolve)
// previously unconditionally emitted Literal::from_f64, producing tensors
// that declared F32 but stored F64Bits — a dtype/element invariant
// violation. Plus the 1-D S (SVD) and W (Eigh) output vectors had the
// same defect.
#[test]
fn oracle_cholesky_f32_preserves_dtype() {
    // SPD 2x2 in F32.
    let a = make_f32_matrix(2, 2, &[4.0, 0.0, 0.0, 9.0]);
    let result =
        eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params()).unwrap();
    let Value::Tensor(t) = &result[0] else {
        panic!("expected tensor");
    };
    assert_eq!(t.dtype, DType::F32);
    t.validate_dtype_consistency()
        .expect("F32 Cholesky output dtype/element invariant");
}

#[test]
fn oracle_qr_f32_preserves_dtype() {
    let a = make_f32_matrix(2, 2, &[3.0, 4.0, 0.0, 5.0]);
    let result =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &no_params()).unwrap();
    for (label, val) in ["Q", "R"].iter().zip(result.iter()) {
        let Value::Tensor(t) = val else {
            panic!("{label}: expected tensor");
        };
        assert_eq!(t.dtype, DType::F32, "{label} dtype");
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("{label} dtype/element invariant: {e}"));
    }
}

#[test]
fn oracle_eigh_f32_preserves_dtype() {
    // Symmetric 2x2 in F32.
    let a = make_f32_matrix(2, 2, &[2.0, 1.0, 1.0, 2.0]);
    let result =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params()).unwrap();
    for (label, val) in ["W", "V"].iter().zip(result.iter()) {
        let Value::Tensor(t) = val else {
            panic!("{label}: expected tensor");
        };
        assert_eq!(t.dtype, DType::F32, "{label} dtype");
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("{label} dtype/element invariant: {e}"));
    }
}

// Property sweep across BF16/F16/F32/F64. The fix in f8871c7 routes all
// linalg outputs through `linalg_literal_from_f64` which dispatches per
// dtype. Run Cholesky on a 2×2 identity (exactly representable in every
// float dtype) and assert dtype preservation across the full family.
#[test]
fn property_cholesky_preserves_all_float_dtypes() {
    fn make_matrix(dtype: DType, data: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape { dims: vec![2, 2] },
                data.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    // Identity matrix is exactly representable in BF16/F16/F32/F64.
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_matrix(dtype, &identity);
        let result =
            eval_primitive_multi(Primitive::Cholesky, std::slice::from_ref(&a), &no_params())
                .unwrap_or_else(|e| panic!("Cholesky {dtype:?} failed: {e}"));
        let Value::Tensor(t) = &result[0] else {
            panic!("Cholesky {dtype:?}: expected tensor output");
        };
        assert_eq!(t.dtype, dtype, "Cholesky {dtype:?}: declared dtype");
        t.validate_dtype_consistency().unwrap_or_else(|e| {
            panic!("Cholesky {dtype:?}: validate_dtype_consistency failed: {e}")
        });
    }
}

#[test]
fn property_qr_preserves_all_float_dtypes() {
    fn make_matrix(dtype: DType, data: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape { dims: vec![2, 2] },
                data.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_matrix(dtype, &identity);
        let result = eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &no_params())
            .unwrap_or_else(|e| panic!("QR {dtype:?} failed: {e}"));
        assert_eq!(result.len(), 2, "QR should return Q and R for {dtype:?}");
        for (idx, output) in result.iter().enumerate() {
            let Value::Tensor(t) = output else {
                panic!("QR {dtype:?} output {idx}: expected tensor");
            };
            assert_eq!(t.dtype, dtype, "QR {dtype:?} output {idx}: dtype mismatch");
            t.validate_dtype_consistency()
                .unwrap_or_else(|e| panic!("QR {dtype:?} output {idx}: validate failed: {e}"));
        }
    }
}

#[test]
fn property_svd_preserves_all_float_dtypes() {
    fn make_matrix(dtype: DType, data: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape { dims: vec![2, 2] },
                data.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_matrix(dtype, &identity);
        let result = eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params())
            .unwrap_or_else(|e| panic!("SVD {dtype:?} failed: {e}"));
        assert_eq!(result.len(), 3, "SVD should return U, S, Vt for {dtype:?}");
        for (idx, output) in result.iter().enumerate() {
            let Value::Tensor(t) = output else {
                panic!("SVD {dtype:?} output {idx}: expected tensor");
            };
            assert_eq!(t.dtype, dtype, "SVD {dtype:?} output {idx}: dtype mismatch");
            t.validate_dtype_consistency()
                .unwrap_or_else(|e| panic!("SVD {dtype:?} output {idx}: validate failed: {e}"));
        }
    }
}

#[test]
fn property_eigh_preserves_all_float_dtypes() {
    fn make_matrix(dtype: DType, data: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape { dims: vec![2, 2] },
                data.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_matrix(dtype, &identity);
        let result = eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params())
            .unwrap_or_else(|e| panic!("Eigh {dtype:?} failed: {e}"));
        assert_eq!(
            result.len(),
            2,
            "Eigh should return eigenvalues and eigenvectors for {dtype:?}"
        );
        for (idx, output) in result.iter().enumerate() {
            let Value::Tensor(t) = output else {
                panic!("Eigh {dtype:?} output {idx}: expected tensor");
            };
            assert_eq!(
                t.dtype, dtype,
                "Eigh {dtype:?} output {idx}: dtype mismatch"
            );
            t.validate_dtype_consistency()
                .unwrap_or_else(|e| panic!("Eigh {dtype:?} output {idx}: validate failed: {e}"));
        }
    }
}

#[test]
fn property_triangular_solve_preserves_all_float_dtypes() {
    fn make_matrix(dtype: DType, data: &[f64]) -> Value {
        let lit_for = |v: f64| match dtype {
            DType::BF16 => Literal::from_bf16_f32(v as f32),
            DType::F16 => Literal::from_f16_f32(v as f32),
            DType::F32 => Literal::from_f32(v as f32),
            DType::F64 => Literal::from_f64(v),
            _ => unreachable!(),
        };
        Value::Tensor(
            TensorValue::new(
                dtype,
                Shape { dims: vec![2, 2] },
                data.iter().copied().map(lit_for).collect(),
            )
            .unwrap(),
        )
    }

    let lower_triangular = [1.0_f64, 0.0, 0.5, 1.0];
    let rhs = [1.0_f64, 0.0, 0.0, 1.0];

    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let a = make_matrix(dtype, &lower_triangular);
        let b = make_matrix(dtype, &rhs);
        let result = eval_primitive_multi(Primitive::TriangularSolve, &[a, b], &no_params())
            .unwrap_or_else(|e| panic!("TriangularSolve {dtype:?} failed: {e}"));
        assert_eq!(
            result.len(),
            1,
            "TriangularSolve should return one output for {dtype:?}"
        );
        let Value::Tensor(t) = &result[0] else {
            panic!("TriangularSolve {dtype:?}: expected tensor output");
        };
        assert_eq!(t.dtype, dtype, "TriangularSolve {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .unwrap_or_else(|e| panic!("TriangularSolve {dtype:?}: validate failed: {e}"));
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_matrix(rows: u32, cols: u32, data: &[(f32, f32)]) -> Value {
    assert_eq!(data.len(), (rows * cols) as usize);
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: vec![rows, cols],
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_matrix(rows: u32, cols: u32, data: &[(f64, f64)]) -> Value {
    assert_eq!(data.len(), (rows * cols) as usize);
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: vec![rows, cols],
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_complex64_matrix(val: &Value) -> Vec<(f32, f32)> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_complex64().unwrap())
        .collect()
}

fn extract_complex128_matrix(val: &Value) -> Vec<(f64, f64)> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_complex128().unwrap())
        .collect()
}

#[test]
fn oracle_triangular_solve_complex64_lower_2x2() {
    // Solve L * X = B where L is lower triangular
    // L = [[1+0i, 0], [0.5+0i, 1+0i]], B = I (identity)
    // Solution X = L^-1 = [[1, 0], [-0.5, 1]]
    let lower = make_complex64_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]);
    let rhs = make_complex64_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::TriangularSolve, &[lower, rhs], &no_params())
        .expect("TriangularSolve complex64 should succeed");
    assert_eq!(result.len(), 1);
    let vals = extract_complex64_matrix(&result[0]);
    assert!(
        (vals[0].0 - 1.0).abs() < 1e-5,
        "expected 1, got {:?}",
        vals[0]
    );
    assert!(
        (vals[2].0 - (-0.5)).abs() < 1e-5,
        "expected -0.5, got {:?}",
        vals[2]
    );
    assert!(
        (vals[3].0 - 1.0).abs() < 1e-5,
        "expected 1, got {:?}",
        vals[3]
    );
}

#[test]
fn oracle_triangular_solve_complex64_with_imaginary() {
    // L = [[1+i, 0], [0, 1+i]], B = [[1+i, 0], [0, 1+i]]
    // Solution X = I (identity)
    let lower = make_complex64_matrix(2, 2, &[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]);
    let rhs = make_complex64_matrix(2, 2, &[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]);
    let result = eval_primitive_multi(Primitive::TriangularSolve, &[lower, rhs], &no_params())
        .expect("TriangularSolve complex64 with imaginary should succeed");
    let vals = extract_complex64_matrix(&result[0]);
    assert!(
        (vals[0].0 - 1.0).abs() < 1e-5,
        "expected 1+0i, got {:?}",
        vals[0]
    );
    assert!(vals[0].1.abs() < 1e-5);
    assert!(
        (vals[3].0 - 1.0).abs() < 1e-5,
        "expected 1+0i, got {:?}",
        vals[3]
    );
    assert!(vals[3].1.abs() < 1e-5);
}

#[test]
fn oracle_triangular_solve_complex128_lower_2x2() {
    // Same as complex64 but with higher precision
    let lower = make_complex128_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]);
    let rhs = make_complex128_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::TriangularSolve, &[lower, rhs], &no_params())
        .expect("TriangularSolve complex128 should succeed");
    let vals = extract_complex128_matrix(&result[0]);
    assert!((vals[0].0 - 1.0).abs() < 1e-10);
    assert!((vals[2].0 - (-0.5)).abs() < 1e-10);
    assert!((vals[3].0 - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_cholesky_complex64_hermitian_2x2() {
    // Hermitian positive-definite matrix: [[2, 1-i], [1+i, 2]]
    // This is A where A = L * L^H (conjugate transpose)
    // L should be approximately [[sqrt(2), 0], [(1+i)/sqrt(2), sqrt(1/2)]]
    let matrix = make_complex64_matrix(2, 2, &[(2.0, 0.0), (1.0, -1.0), (1.0, 1.0), (2.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::Cholesky, &[matrix], &no_params())
        .expect("Cholesky complex64 hermitian should succeed");
    assert_eq!(result.len(), 1);
    let vals = extract_complex64_matrix(&result[0]);
    let sqrt2 = 2.0_f32.sqrt();
    assert!(
        (vals[0].0 - sqrt2).abs() < 1e-4,
        "L[0,0] expected sqrt(2)={sqrt2}, got {:?}",
        vals[0]
    );
}

#[test]
fn oracle_qr_complex64_2x2() {
    // QR decomposition of a simple complex matrix
    let matrix = make_complex64_matrix(2, 2, &[(1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (1.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::Qr, &[matrix], &no_params())
        .expect("QR complex64 should succeed");
    assert!(result.len() >= 2, "QR should return Q and R");
    let q_dtype = result[0].dtype();
    let r_dtype = result[1].dtype();
    assert_eq!(
        q_dtype,
        DType::Complex64,
        "Q should preserve complex64 dtype"
    );
    assert_eq!(
        r_dtype,
        DType::Complex64,
        "R should preserve complex64 dtype"
    );
}

#[test]
fn property_triangular_solve_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let (lower, rhs) = match dtype {
            DType::Complex64 => (
                make_complex64_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]),
                make_complex64_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]),
            ),
            DType::Complex128 => (
                make_complex128_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]),
                make_complex128_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]),
            ),
            _ => unreachable!(),
        };
        let result = eval_primitive_multi(Primitive::TriangularSolve, &[lower, rhs], &no_params())
            .unwrap_or_else(|e| panic!("TriangularSolve {dtype:?} failed: {e}"));
        assert_eq!(
            result[0].dtype(),
            dtype,
            "TriangularSolve {dtype:?}: dtype mismatch"
        );
    }
}

// ======================== Complex SVD ========================

#[test]
fn oracle_svd_complex64_2x2_real_values() {
    // SVD of a complex matrix with real entries should match real SVD
    // A = [[3, 0], [0, 2]] has singular values 3, 2
    let a = make_complex64_matrix(2, 2, &[(3.0, 0.0), (0.0, 0.0), (0.0, 0.0), (2.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params())
        .expect("svd complex64 should succeed");
    assert_eq!(result.len(), 3);
    // S should have singular values 3, 2 (in descending order)
    let s = result[1].as_tensor().unwrap();
    assert_eq!(s.dtype, DType::F32);
    let s_vals: Vec<f64> = s.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    assert!(
        (s_vals[0] - 3.0).abs() < 1e-4,
        "s[0] = {}, expected 3",
        s_vals[0]
    );
    assert!(
        (s_vals[1] - 2.0).abs() < 1e-4,
        "s[1] = {}, expected 2",
        s_vals[1]
    );
}

#[test]
fn oracle_svd_complex64_with_imaginary() {
    // A = [[1+i, 0], [0, 1-i]]
    // |1+i| = |1-i| = sqrt(2), so singular values should be sqrt(2), sqrt(2)
    let a = make_complex64_matrix(2, 2, &[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (1.0, -1.0)]);
    let result = eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params())
        .expect("svd complex64 with imaginary should succeed");
    let s = result[1].as_tensor().unwrap();
    let s_vals: Vec<f64> = s.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    let sqrt2 = std::f64::consts::SQRT_2;
    assert!(
        (s_vals[0] - sqrt2).abs() < 1e-5,
        "s[0] = {}, expected sqrt(2)",
        s_vals[0]
    );
    assert!(
        (s_vals[1] - sqrt2).abs() < 1e-5,
        "s[1] = {}, expected sqrt(2)",
        s_vals[1]
    );
}

#[test]
fn oracle_svd_complex128_rank_deficient_reconstructs() {
    // Rank-2 complex 3×3 (row 2 = row 0 + row 1 exactly) ⇒ smallest singular value
    // is exactly 0. One-sided complex Jacobi recovers it to ~ε·‖A‖ (<1e-10); the
    // old AᴴA / normal-equations path squared the condition number and would land
    // near √ε·‖A‖ (~1e-7), which the 1e-10 bound rejects (frankenjax-4kx6m).
    let a_data: [(f64, f64); 9] = [
        (1.0, 1.0),
        (2.0, 0.0),
        (3.0, -1.0), //
        (0.0, 2.0),
        (1.0, 1.0),
        (2.0, 0.0), //
        (1.0, 3.0),
        (3.0, 1.0),
        (5.0, -1.0),
    ];
    let a = make_complex128_matrix(3, 3, &a_data);
    let result = eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params())
        .expect("svd complex128 rank-deficient should succeed");
    let u = extract_complex128_matrix(&result[0]); // 3×3
    let s: Vec<f64> = result[1]
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect();
    let vh = extract_complex128_matrix(&result[2]); // 3×3
    assert_eq!(s.len(), 3);
    assert!(s[0] >= s[1] && s[1] >= s[2], "singular values descending");
    assert!(
        s[2] < 1e-10,
        "rank-2 complex matrix zero singular value should be ~ε·‖A‖, got {} (s0={})",
        s[2],
        s[0]
    );
    // Reconstruct A = U·diag(S)·Vᴴ (complex) and check ‖recon − A‖∞.
    let cmul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = (0.0_f64, 0.0_f64);
            for l in 0..3 {
                let usl = (u[i * 3 + l].0 * s[l], u[i * 3 + l].1 * s[l]);
                let term = cmul(usl, vh[l * 3 + j]);
                acc = (acc.0 + term.0, acc.1 + term.1);
            }
            let expected = a_data[i * 3 + j];
            assert!(
                (acc.0 - expected.0).abs() < 1e-9 && (acc.1 - expected.1).abs() < 1e-9,
                "recon[{i},{j}] = {acc:?}, expected {expected:?}"
            );
        }
    }
}

#[test]
fn property_svd_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let a = match dtype {
            DType::Complex64 => {
                make_complex64_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)])
            }
            DType::Complex128 => {
                make_complex128_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)])
            }
            _ => unreachable!(),
        };
        let result = eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &no_params())
            .unwrap_or_else(|e| panic!("SVD {dtype:?} failed: {e}"));
        // U and Vt should be complex, S should be real
        assert_eq!(result[0].dtype(), dtype, "SVD {dtype:?}: U dtype mismatch");
        let s_dtype = if dtype == DType::Complex64 {
            DType::F32
        } else {
            DType::F64
        };
        assert_eq!(
            result[1].dtype(),
            s_dtype,
            "SVD {dtype:?}: S dtype mismatch"
        );
        assert_eq!(result[2].dtype(), dtype, "SVD {dtype:?}: Vt dtype mismatch");
    }
}

// ======================== Complex Eigh ========================

#[test]
fn oracle_eigh_complex64_hermitian_2x2() {
    // Hermitian matrix: A = [[2, 1-i], [1+i, 3]]
    // Eigenvalues can be computed: trace = 5, det = 6 - |1-i|^2 = 6 - 2 = 4
    // λ² - 5λ + 4 = 0 → λ = 1, 4
    let a = make_complex64_matrix(2, 2, &[(2.0, 0.0), (1.0, -1.0), (1.0, 1.0), (3.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params())
        .expect("eigh complex64 should succeed");
    assert_eq!(result.len(), 2);
    // Eigenvalues should be real (ascending order: 1, 4)
    let w = result[0].as_tensor().unwrap();
    assert_eq!(w.dtype, DType::F32);
    let w_vals: Vec<f64> = w.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    assert!(
        (w_vals[0] - 1.0).abs() < 1e-3,
        "w[0] = {}, expected 1",
        w_vals[0]
    );
    assert!(
        (w_vals[1] - 4.0).abs() < 1e-3,
        "w[1] = {}, expected 4",
        w_vals[1]
    );
}

#[test]
fn oracle_eigh_complex64_diagonal() {
    // Diagonal Hermitian: eigenvalues = [1, 4] (diagonal elements)
    let a = make_complex64_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (4.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params())
        .expect("eigh complex64 diagonal should succeed");
    let w = result[0].as_tensor().unwrap();
    let w_vals: Vec<f64> = w.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    assert!(
        (w_vals[0] - 1.0).abs() < 1e-4,
        "w[0] = {}, expected 1",
        w_vals[0]
    );
    assert!(
        (w_vals[1] - 4.0).abs() < 1e-4,
        "w[1] = {}, expected 4",
        w_vals[1]
    );
}

#[test]
fn oracle_eigh_complex128_hermitian_identity() {
    // Hermitian identity: eigenvalues = [1, 1]
    let a = make_complex128_matrix(2, 2, &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params())
        .expect("eigh complex128 should succeed");
    let w = result[0].as_tensor().unwrap();
    assert_eq!(w.dtype, DType::F64);
    let w_vals: Vec<f64> = w.elements.iter().map(|l| l.as_f64().unwrap()).collect();
    assert!(
        (w_vals[0] - 1.0).abs() < 1e-10,
        "w[0] = {}, expected 1",
        w_vals[0]
    );
    assert!(
        (w_vals[1] - 1.0).abs() < 1e-10,
        "w[1] = {}, expected 1",
        w_vals[1]
    );
}

#[test]
fn property_eigh_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let a = match dtype {
            DType::Complex64 => {
                make_complex64_matrix(2, 2, &[(2.0, 0.0), (1.0, -1.0), (1.0, 1.0), (3.0, 0.0)])
            }
            DType::Complex128 => {
                make_complex128_matrix(2, 2, &[(2.0, 0.0), (1.0, -1.0), (1.0, 1.0), (3.0, 0.0)])
            }
            _ => unreachable!(),
        };
        let result = eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &no_params())
            .unwrap_or_else(|e| panic!("Eigh {dtype:?} failed: {e}"));
        // Eigenvalues should be real, eigenvectors should be complex
        let w_dtype = if dtype == DType::Complex64 {
            DType::F32
        } else {
            DType::F64
        };
        assert_eq!(
            result[0].dtype(),
            w_dtype,
            "Eigh {dtype:?}: W dtype mismatch"
        );
        assert_eq!(result[1].dtype(), dtype, "Eigh {dtype:?}: V dtype mismatch");
    }
}
