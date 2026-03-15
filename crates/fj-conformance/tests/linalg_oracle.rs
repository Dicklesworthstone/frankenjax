//! Linear algebra primitive oracle tests.
//!
//! Tests Cholesky, QR, SVD, Eigh, and TriangularSolve against
//! hand-verified analytical expected values.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive_multi;
use std::collections::BTreeMap;

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

// ======================== Cholesky ========================

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
