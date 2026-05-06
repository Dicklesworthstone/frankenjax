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
    let result_scaled =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&scaled_a), &no_params())
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

    let result_a = eval_primitive_multi(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        &no_params(),
    )
    .unwrap();
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
    let result_scaled =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&scaled_a), &no_params())
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
