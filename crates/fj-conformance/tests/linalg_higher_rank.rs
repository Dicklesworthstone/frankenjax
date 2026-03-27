//! Higher-rank linalg correctness tests.
//!
//! Verifies mathematical identities (A = L*L^T, A = Q*R, etc.) for
//! larger matrices beyond the 2x2/3x3 oracle fixtures. These are not
//! oracle-backed but validate implementation correctness via invariants.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{eval_primitive, eval_primitive_multi};
use std::collections::BTreeMap;

fn make_f64_matrix(rows: usize, cols: usize, data: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            data.iter().map(|&v| Literal::from_f64(v)).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(val: &Value) -> Vec<f64> {
    match val {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(l) => vec![l.as_f64().unwrap()],
    }
}

/// Matrix multiply C = A * B (naive, for verification only)
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
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

/// Transpose a matrix
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

/// Check ||A - B||_max < tol
fn assert_matrices_close(a: &[f64], b: &[f64], tol: f64, context: &str) {
    assert_eq!(a.len(), b.len(), "{context}: length mismatch");
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < tol,
        "{context}: max diff {max_diff:.2e} exceeds tolerance {tol:.2e}"
    );
}

// ======================== Cholesky ========================

fn verify_cholesky(n: usize, a_data: &[f64], tol: f64, name: &str) {
    let a = make_f64_matrix(n, n, a_data);
    let result = eval_primitive_multi(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        &BTreeMap::new(),
    )
    .unwrap();
    let l = extract_f64_vec(&result[0]);

    // Zero out upper triangle (Cholesky should be lower triangular)
    let mut l_lower = l.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                l_lower[i * n + j].abs() < 1e-12,
                "{name}: L[{i},{j}] = {} should be zero",
                l_lower[i * n + j]
            );
            l_lower[i * n + j] = 0.0;
        }
    }

    // Verify A = L * L^T
    let lt = transpose(&l_lower, n, n);
    let reconstructed = matmul(&l_lower, &lt, n, n, n);
    assert_matrices_close(a_data, &reconstructed, tol, &format!("{name}: A = L*L^T"));
}

#[test]
fn cholesky_4x4_spd() {
    #[rustfmt::skip]
    let a = [
        10.0,  3.0,  1.0,  0.5,
         3.0,  8.0,  2.0,  1.0,
         1.0,  2.0,  6.0,  1.5,
         0.5,  1.0,  1.5,  5.0,
    ];
    verify_cholesky(4, &a, 1e-10, "cholesky_4x4");
}

#[test]
fn cholesky_5x5_spd() {
    #[rustfmt::skip]
    let a = [
        20.0,  4.0,  2.0,  1.0,  0.5,
         4.0, 15.0,  3.0,  1.5,  1.0,
         2.0,  3.0, 12.0,  2.5,  1.5,
         1.0,  1.5,  2.5, 10.0,  2.0,
         0.5,  1.0,  1.5,  2.0,  8.0,
    ];
    verify_cholesky(5, &a, 1e-10, "cholesky_5x5");
}

// ======================== QR ========================

fn verify_qr(m: usize, n: usize, a_data: &[f64], tol: f64, name: &str) {
    let a = make_f64_matrix(m, n, a_data);
    let results =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();
    let q = extract_f64_vec(&results[0]);
    let r = extract_f64_vec(&results[1]);

    let k = m.min(n);

    // Verify A = Q * R
    let reconstructed = matmul(&q, &r, m, k, n);
    assert_matrices_close(a_data, &reconstructed, tol, &format!("{name}: A = Q*R"));

    // Verify Q^T * Q = I (orthogonality)
    let qt = transpose(&q, m, k);
    let qtq = matmul(&qt, &q, k, m, k);
    let mut identity = vec![0.0; k * k];
    for i in 0..k {
        identity[i * k + i] = 1.0;
    }
    assert_matrices_close(&qtq, &identity, tol, &format!("{name}: Q^T*Q = I"));
}

#[test]
fn qr_4x3_tall() {
    #[rustfmt::skip]
    let a = [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
        1.0, 3.0, 5.0,
    ];
    verify_qr(4, 3, &a, 1e-10, "qr_4x3");
}

#[test]
fn qr_4x4_square() {
    #[rustfmt::skip]
    let a = [
        2.0,  1.0,  0.0, -1.0,
        1.0,  3.0,  1.0,  0.0,
        0.0,  1.0,  4.0,  2.0,
       -1.0,  0.0,  2.0,  5.0,
    ];
    verify_qr(4, 4, &a, 1e-10, "qr_4x4");
}

// ======================== SVD ========================

fn verify_svd(m: usize, n: usize, a_data: &[f64], tol: f64, name: &str) {
    let a = make_f64_matrix(m, n, a_data);
    let results =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();
    let u = extract_f64_vec(&results[0]);
    let s = extract_f64_vec(&results[1]);
    let vt = extract_f64_vec(&results[2]);

    let k = m.min(n);

    // Build diagonal S matrix (k x k)
    let mut s_diag = vec![0.0; k * k];
    for i in 0..k {
        s_diag[i * k + i] = s[i];
    }

    // Verify A = U * diag(S) * Vt
    let us = matmul(&u, &s_diag, m, k, k);
    let reconstructed = matmul(&us, &vt, m, k, n);
    assert_matrices_close(a_data, &reconstructed, tol, &format!("{name}: A = U*S*Vt"));

    // Verify singular values are non-negative and descending
    for i in 0..k {
        assert!(
            s[i] >= -1e-12,
            "{name}: s[{i}] = {} should be non-negative",
            s[i]
        );
        if i > 0 {
            assert!(
                s[i - 1] >= s[i] - 1e-12,
                "{name}: singular values not descending: s[{}]={} > s[{i}]={}",
                i - 1,
                s[i - 1],
                s[i]
            );
        }
    }
}

#[test]
fn svd_3x3_general() {
    #[rustfmt::skip]
    let a = [
        3.0, 2.0, 1.0,
        1.0, 4.0, 2.0,
        0.5, 1.0, 3.0,
    ];
    verify_svd(3, 3, &a, 1e-10, "svd_3x3");
}

#[test]
fn svd_4x3_tall() {
    #[rustfmt::skip]
    let a = [
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
        1.0, 3.0, 5.0,
    ];
    verify_svd(4, 3, &a, 1e-10, "svd_4x3");
}

// ======================== Eigh ========================

fn verify_eigh(n: usize, a_data: &[f64], tol: f64, name: &str) {
    let a = make_f64_matrix(n, n, a_data);
    let results =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();
    let eigenvalues = extract_f64_vec(&results[0]);
    let eigenvectors = extract_f64_vec(&results[1]);

    // Verify eigenvalues are sorted ascending
    for i in 1..n {
        assert!(
            eigenvalues[i] >= eigenvalues[i - 1] - 1e-10,
            "{name}: eigenvalues not ascending: w[{}]={} > w[{i}]={}",
            i - 1,
            eigenvalues[i - 1],
            eigenvalues[i]
        );
    }

    // Verify A*V = V*diag(w): for each eigenvector v_i, A*v_i = w_i*v_i
    for col in 0..n {
        let mut v = vec![0.0; n];
        for row in 0..n {
            v[row] = eigenvectors[row * n + col];
        }

        // A * v
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += a_data[i * n + j] * v[j];
            }
        }

        // w * v
        let wv: Vec<f64> = v.iter().map(|&x| x * eigenvalues[col]).collect();
        assert_matrices_close(
            &av,
            &wv,
            tol,
            &format!("{name}: A*v_{col} = w_{col}*v_{col}"),
        );
    }

    // Verify V^T * V = I (orthogonality)
    let vt = transpose(&eigenvectors, n, n);
    let vtv = matmul(&vt, &eigenvectors, n, n, n);
    let mut identity = vec![0.0; n * n];
    for i in 0..n {
        identity[i * n + i] = 1.0;
    }
    assert_matrices_close(&vtv, &identity, tol, &format!("{name}: V^T*V = I"));
}

#[test]
fn eigh_4x4_symmetric() {
    #[rustfmt::skip]
    let a = [
        6.0, 2.0, 1.0, 0.5,
        2.0, 5.0, 1.5, 1.0,
        1.0, 1.5, 4.0, 0.5,
        0.5, 1.0, 0.5, 3.0,
    ];
    verify_eigh(4, &a, 1e-8, "eigh_4x4");
}

#[test]
fn eigh_5x5_symmetric() {
    #[rustfmt::skip]
    let a = [
        10.0,  2.0,  1.0,  0.5,  0.1,
         2.0,  8.0,  1.5,  1.0,  0.5,
         1.0,  1.5,  6.0,  1.0,  0.5,
         0.5,  1.0,  1.0,  5.0,  0.5,
         0.1,  0.5,  0.5,  0.5,  4.0,
    ];
    verify_eigh(5, &a, 1e-8, "eigh_5x5");
}

// ======================== TriangularSolve ========================

#[test]
fn triangular_solve_3x3() {
    // L is lower triangular 3x3
    #[rustfmt::skip]
    let l_data = [
        2.0, 0.0, 0.0,
        3.0, 4.0, 0.0,
        1.0, 2.0, 5.0,
    ];
    // b is 3x1
    let b_data = [10.0, 23.0, 23.0];

    let l = make_f64_matrix(3, 3, &l_data);
    let b = make_f64_matrix(3, 1, &b_data);

    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());

    let result = eval_primitive(Primitive::TriangularSolve, &[l, b], &params).unwrap();
    let x = extract_f64_vec(&result);

    // Verify L * x = b
    let lx = matmul(&l_data, &x, 3, 3, 1);
    assert_matrices_close(&b_data, &lx, 1e-10, "tsolve_3x3: L*x = b");
}

#[test]
fn triangular_solve_3x3_multi_rhs() {
    // L is lower triangular 3x3, B is 3x2
    #[rustfmt::skip]
    let l_data = [
        2.0, 0.0, 0.0,
        3.0, 4.0, 0.0,
        1.0, 2.0, 5.0,
    ];
    #[rustfmt::skip]
    let b_data = [
        10.0, 4.0,
        23.0, 11.0,
        23.0, 14.0,
    ];

    let l = make_f64_matrix(3, 3, &l_data);
    let b = make_f64_matrix(3, 2, &b_data);

    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());

    let result = eval_primitive(Primitive::TriangularSolve, &[l, b], &params).unwrap();
    let x = extract_f64_vec(&result);

    // Verify L * X = B
    let lx = matmul(&l_data, &x, 3, 3, 2);
    assert_matrices_close(&b_data, &lx, 1e-10, "tsolve_3x3_multi: L*X = B");
}

// ======================== FFT ========================

#[test]
fn fft_8_element() {
    use fj_core::Literal::Complex128Bits;

    let input_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0];
    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![8] },
            input_data
                .iter()
                .map(|&v| Literal::from_complex128(v, 0.0))
                .collect(),
        )
        .unwrap(),
    );

    let result =
        eval_primitive(Primitive::Fft, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let fft_result: Vec<(f64, f64)> = result
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
            _ => panic!("expected complex"),
        })
        .collect();

    // Verify Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
    let energy_time: f64 = input_data.iter().map(|&v| v * v).sum();
    let energy_freq: f64 = fft_result.iter().map(|(r, i)| r * r + i * i).sum::<f64>() / 8.0;
    assert!(
        (energy_time - energy_freq).abs() < 1e-10,
        "Parseval's: time energy {energy_time} != freq energy {energy_freq}"
    );

    // For symmetric real input [1,2,3,4,4,3,2,1], X[0] should be sum = 20
    assert!(
        (fft_result[0].0 - 20.0).abs() < 1e-10,
        "DC component should be 20, got {}",
        fft_result[0].0
    );
}

#[test]
fn rfft_8_element() {
    let input_data: Vec<f64> = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
    let x = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![8] },
            input_data.iter().map(|&v| Literal::from_f64(v)).collect(),
        )
        .unwrap(),
    );

    let result =
        eval_primitive(Primitive::Rfft, std::slice::from_ref(&x), &BTreeMap::new()).unwrap();
    let rfft_result = result.as_tensor().unwrap();

    // RFFT of length 8 produces 5 complex values (N/2 + 1)
    assert_eq!(
        rfft_result.elements.len(),
        5,
        "RFFT(8) should produce 5 complex values"
    );

    // DC = sum = 0
    let dc = match rfft_result.elements[0] {
        Literal::Complex128Bits(re, im) => (f64::from_bits(re), f64::from_bits(im)),
        _ => panic!("expected complex"),
    };
    assert!(
        dc.0.abs() < 1e-10 && dc.1.abs() < 1e-10,
        "DC should be 0+0i, got {:?}",
        dc
    );
}
