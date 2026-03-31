//! Numerical gradient verification for linalg and FFT AD rules.
//!
//! Uses finite-difference approximation to verify VJP correctness:
//!   dL/dA[i] ≈ (L(A+εe_i) - L(A-εe_i)) / (2ε)
//! where L is the loss function induced by the cotangent vector.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{eval_primitive, eval_primitive_multi};
use std::collections::BTreeMap;

fn make_f64_matrix(rows: u32, cols: u32, data: &[f64]) -> Value {
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

fn make_f64_vector(data: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![data.len() as u32],
            },
            data.iter().map(|&v| Literal::from_f64(v)).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(val: &Value) -> Vec<f64> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_f64_scalar(val: &Value) -> f64 {
    val.as_f64_scalar().unwrap()
}

fn loss_multi(prim: Primitive, a: &Value, gs: &[Value], params: &BTreeMap<String, String>) -> f64 {
    let outs = eval_primitive_multi(prim, std::slice::from_ref(a), params).unwrap();
    let mut total = 0.0;
    for (out, g) in outs.iter().zip(gs.iter()) {
        let out_vals = extract_f64_vec(out);
        let g_vals = extract_f64_vec(g);
        total += out_vals
            .iter()
            .zip(g_vals.iter())
            .map(|(o, gv)| o * gv)
            .sum::<f64>();
    }
    total
}

fn assert_gradients_close(analytical: &[f64], numerical: &[f64], tol: f64, context: &str) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{context}: gradient length mismatch"
    );
    for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        assert!(
            (a - n).abs() < tol,
            "{context}[{i}]: analytical={a}, numerical={n}, diff={} (tol={tol})",
            (a - n).abs()
        );
    }
}

fn assert_scalar_close(analytical: f64, numerical: f64, abs_tol: f64, rel_tol: f64, context: &str) {
    let diff = (analytical - numerical).abs();
    let scale = analytical.abs().max(numerical.abs()).max(1.0);
    assert!(
        diff <= abs_tol.max(scale * rel_tol),
        "{context}: analytical={analytical}, numerical={numerical}, diff={diff}, abs_tol={abs_tol}, rel_tol={rel_tol}"
    );
}

// ======================== Cholesky VJP ========================

#[test]
#[allow(clippy::needless_range_loop)]
fn cholesky_vjp_numerical_2x2() {
    let a_data = [4.0, 2.0, 2.0, 3.0];
    let a = make_f64_matrix(2, 2, &a_data);

    let outputs = eval_primitive_multi(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        &BTreeMap::new(),
    )
    .unwrap();
    let l_out = &outputs[0];

    let g_l = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(0.0),
                Literal::from_f64(1.0),
                Literal::from_f64(1.0),
            ],
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        std::slice::from_ref(&g_l),
        std::slice::from_ref(l_out),
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);

    let eps = 1e-5;
    let mut numerical = [0.0; 4];
    for idx in 0..4_usize {
        let row = idx / 2;
        let col = idx % 2;
        let mut plus = a_data.to_vec();
        plus[row * 2 + col] += eps;
        if row != col {
            plus[col * 2 + row] += eps;
        }
        let a_plus = make_f64_matrix(2, 2, &plus);

        let mut minus = a_data.to_vec();
        minus[row * 2 + col] -= eps;
        if row != col {
            minus[col * 2 + row] -= eps;
        }
        let a_minus = make_f64_matrix(2, 2, &minus);

        let l_plus = loss_multi(
            Primitive::Cholesky,
            &a_plus,
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );
        let l_minus = loss_multi(
            Primitive::Cholesky,
            &a_minus,
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );
        let mut grad = (l_plus - l_minus) / (2.0 * eps);
        // Off-diagonal: we perturbed both A[i,j] and A[j,i], so the numerical
        // gradient is 2x the per-element gradient. Divide by 2 to get bar_A[i,j].
        if row != col {
            grad *= 0.5;
        }
        numerical[idx] = grad;
    }

    assert_gradients_close(&analytical, &numerical, 1e-4, "Cholesky VJP");
}

// ======================== TriangularSolve VJP ========================

#[test]
fn triangular_solve_vjp_numerical() {
    let l_data = [2.0, 0.0, 1.0, 3.0];
    let b_data = [4.0, 7.0];

    let l_mat = make_f64_matrix(2, 2, &l_data);
    let b_vec = make_f64_matrix(2, 1, &b_data);

    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());

    let outputs = eval_primitive_multi(
        Primitive::TriangularSolve,
        &[l_mat.clone(), b_vec.clone()],
        &params,
    )
    .unwrap();
    let x_out = &outputs[0];

    let g_x = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 1] },
            vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::TriangularSolve,
        &[l_mat, b_vec],
        std::slice::from_ref(&g_x),
        std::slice::from_ref(x_out),
        &params,
    )
    .unwrap();

    let da_b = extract_f64_vec(&vjp_result[1]);
    let eps = 1e-5;

    let mut numerical_b = vec![0.0; 2];
    for i in 0..2 {
        let mut plus = b_data.to_vec();
        plus[i] += eps;
        let b_plus = make_f64_matrix(2, 1, &plus);
        let l_val = make_f64_matrix(2, 2, &l_data);
        let out_plus =
            eval_primitive_multi(Primitive::TriangularSolve, &[l_val, b_plus], &params).unwrap();
        let x_plus = extract_f64_vec(&out_plus[0]);

        let mut minus = b_data.to_vec();
        minus[i] -= eps;
        let b_minus = make_f64_matrix(2, 1, &minus);
        let l_val = make_f64_matrix(2, 2, &l_data);
        let out_minus =
            eval_primitive_multi(Primitive::TriangularSolve, &[l_val, b_minus], &params).unwrap();
        let x_minus = extract_f64_vec(&out_minus[0]);

        let g_vals = extract_f64_vec(&g_x);
        let l_plus: f64 = x_plus.iter().zip(g_vals.iter()).map(|(x, g)| x * g).sum();
        let l_minus: f64 = x_minus.iter().zip(g_vals.iter()).map(|(x, g)| x * g).sum();
        numerical_b[i] = (l_plus - l_minus) / (2.0 * eps);
    }

    assert_gradients_close(&da_b, &numerical_b, 1e-4, "TriangularSolve VJP w.r.t. B");
}

// ======================== FFT VJP ========================

#[test]
fn fft_vjp_numerical() {
    // FFT is linear, so VJP(g) = adjoint(FFT)(g) = n * IFFT(g)
    use fj_core::Literal::Complex128Bits;

    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![4] },
            [1.0, 2.0, 3.0, 4.0]
                .iter()
                .map(|&v| Literal::from_complex128(v, 0.0))
                .collect(),
        )
        .unwrap(),
    );

    let g = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![4] },
            (0..4).map(|_| Literal::from_complex128(1.0, 0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp_single(
        Primitive::Fft,
        std::slice::from_ref(&x),
        &g,
        &BTreeMap::new(),
    )
    .unwrap();

    // For linear op FFT: VJP(g) = adjoint(FFT)(g) = n * IFFT(g)
    let ifft_g =
        eval_primitive(Primitive::Ifft, std::slice::from_ref(&g), &BTreeMap::new()).unwrap();
    let n = 4.0;

    let vjp_vals: Vec<(f64, f64)> = vjp_result[0]
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
            _ => panic!("expected complex"),
        })
        .collect();

    let ifft_vals: Vec<(f64, f64)> = ifft_g
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
            _ => panic!("expected complex"),
        })
        .collect();

    for (i, ((vr, vi), (ir, ii))) in vjp_vals.iter().zip(ifft_vals.iter()).enumerate() {
        let expected_re = n * ir;
        let expected_im = n * ii;
        assert!(
            (vr - expected_re).abs() < 1e-10 && (vi - expected_im).abs() < 1e-10,
            "FFT VJP[{i}]: got ({vr},{vi}), expected ({expected_re},{expected_im})"
        );
    }
}

// ======================== RFFT VJP ========================

#[test]
fn rfft_vjp_numerical() {
    // RFFT: R^n → C^{n/2+1}. Verify VJP via finite differences on scalar loss.
    let x_data = [1.0, 2.0, 3.0, 4.0];
    let x = make_f64_vector(&x_data);

    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());

    // g = ones (complex, length 3 = n/2+1)
    let g = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![3] },
            (0..3).map(|_| Literal::from_complex128(1.0, 0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result =
        fj_ad::vjp_single(Primitive::Rfft, std::slice::from_ref(&x), &g, &params).unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);

    // Numerical: L(x) = sum(Re(RFFT(x))) since g = (1+0i) for all bins
    let eps = 1e-6;
    let mut numerical = vec![0.0; 4];
    for i in 0..4 {
        let mut plus = x_data.to_vec();
        plus[i] += eps;
        let x_plus = make_f64_vector(&plus);
        let re_plus: f64 = eval_primitive(Primitive::Rfft, std::slice::from_ref(&x_plus), &params)
            .unwrap()
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::Complex128Bits(re, _) => f64::from_bits(*re),
                _ => 0.0,
            })
            .sum();

        let mut minus = x_data.to_vec();
        minus[i] -= eps;
        let x_minus = make_f64_vector(&minus);
        let re_minus: f64 =
            eval_primitive(Primitive::Rfft, std::slice::from_ref(&x_minus), &params)
                .unwrap()
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, _) => f64::from_bits(*re),
                    _ => 0.0,
                })
                .sum();

        numerical[i] = (re_plus - re_minus) / (2.0 * eps);
    }

    assert_gradients_close(&analytical, &numerical, 1e-4, "RFFT VJP");
}

// ======================== Cholesky VJP 3x3 ========================

#[test]
#[allow(clippy::needless_range_loop)]
fn cholesky_vjp_numerical_3x3() {
    // Test with a larger, well-conditioned SPD matrix
    #[rustfmt::skip]
    let a_data = [
        10.0, 2.0, 1.0,
         2.0, 8.0, 3.0,
         1.0, 3.0, 6.0,
    ];
    let a = make_f64_matrix(3, 3, &a_data);

    let outputs = eval_primitive_multi(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        &BTreeMap::new(),
    )
    .unwrap();
    let l_out = &outputs[0];

    let g_l = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 3] },
            (0..9)
                .map(|i| Literal::from_f64(if i % 4 == 0 { 1.0 } else { 0.5 }))
                .collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        std::slice::from_ref(&g_l),
        std::slice::from_ref(l_out),
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);

    let eps = 1e-5;
    let n = 3;
    let mut numerical = vec![0.0; 9];
    for idx in 0..9_usize {
        let row = idx / n;
        let col = idx % n;
        let mut plus = a_data.to_vec();
        plus[row * n + col] += eps;
        if row != col {
            plus[col * n + row] += eps;
        }
        let a_plus = make_f64_matrix(3, 3, &plus);

        let mut minus = a_data.to_vec();
        minus[row * n + col] -= eps;
        if row != col {
            minus[col * n + row] -= eps;
        }
        let a_minus = make_f64_matrix(3, 3, &minus);

        let l_plus = loss_multi(
            Primitive::Cholesky,
            &a_plus,
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );
        let l_minus = loss_multi(
            Primitive::Cholesky,
            &a_minus,
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );
        let mut grad = (l_plus - l_minus) / (2.0 * eps);
        if row != col {
            grad *= 0.5;
        }
        numerical[idx] = grad;
    }

    assert_gradients_close(&analytical, &numerical, 1e-4, "Cholesky VJP 3x3");
}

#[test]
#[allow(clippy::needless_range_loop)]
fn cholesky_vjp_near_singular_matrix() {
    // SPD but close to singular: the determinant is small, so this exercises
    // the Cholesky VJP around a numerically fragile region without crossing
    // into the invalid non-SPD domain.
    let a_data = [1.0, 0.9999, 0.9999, 1.0];
    let a = make_f64_matrix(2, 2, &a_data);

    let outputs = eval_primitive_multi(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        &BTreeMap::new(),
    )
    .unwrap();
    let l_out = &outputs[0];

    let g_l = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(0.0),
                Literal::from_f64(0.25),
                Literal::from_f64(1.0),
            ],
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        std::slice::from_ref(&g_l),
        std::slice::from_ref(l_out),
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);
    assert!(
        analytical.iter().all(|value| value.is_finite()),
        "near-singular Cholesky VJP should stay finite: {analytical:?}"
    );

    let eps = 1e-6;
    let mut numerical = [0.0; 4];
    for idx in 0..4_usize {
        let row = idx / 2;
        let col = idx % 2;

        let mut plus = a_data.to_vec();
        plus[row * 2 + col] += eps;
        if row != col {
            plus[col * 2 + row] += eps;
        }

        let mut minus = a_data.to_vec();
        minus[row * 2 + col] -= eps;
        if row != col {
            minus[col * 2 + row] -= eps;
        }

        let l_plus = loss_multi(
            Primitive::Cholesky,
            &make_f64_matrix(2, 2, &plus),
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );
        let l_minus = loss_multi(
            Primitive::Cholesky,
            &make_f64_matrix(2, 2, &minus),
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );

        let mut grad = (l_plus - l_minus) / (2.0 * eps);
        if row != col {
            grad *= 0.5;
        }
        numerical[idx] = grad;
    }

    assert_gradients_close(&analytical, &numerical, 2e-2, "Cholesky VJP near singular");
}

// ======================== QR VJP ========================

#[test]
#[allow(clippy::needless_range_loop)]
fn qr_vjp_numerical() {
    // Use the same 2x2 matrix as the JVP test for consistency.
    // Verify via R output only (R is unique for full-rank matrices).
    let a_data = [1.0, -1.0, 1.0, 1.0];
    let a = make_f64_matrix(2, 2, &a_data);

    let outputs =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();

    // Zero cotangent for Q, nonzero for R — avoids sign-ambiguity issues
    let g_q = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            (0..4).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );
    let g_r = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(0.5),
                Literal::from_f64(0.0),
                Literal::from_f64(1.0),
            ],
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Qr,
        std::slice::from_ref(&a),
        &[g_q.clone(), g_r.clone()],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);

    let eps = 1e-5;
    let gs = [g_q, g_r];
    let mut numerical = vec![0.0; 4];
    for idx in 0..4_usize {
        let mut plus = a_data.to_vec();
        plus[idx] += eps;
        let a_plus = make_f64_matrix(2, 2, &plus);
        let l_plus = loss_multi(Primitive::Qr, &a_plus, &gs, &BTreeMap::new());

        let mut minus = a_data.to_vec();
        minus[idx] -= eps;
        let a_minus = make_f64_matrix(2, 2, &minus);
        let l_minus = loss_multi(Primitive::Qr, &a_minus, &gs, &BTreeMap::new());

        numerical[idx] = (l_plus - l_minus) / (2.0 * eps);
    }

    assert_gradients_close(&analytical, &numerical, 1e-3, "QR VJP (R only)");
}

#[test]
#[allow(clippy::needless_range_loop)]
fn svd_vjp_ill_conditioned_matrix() {
    // Diagonal with a large condition number. Singular values are stable enough
    // for finite differences while still stressing the AD path on a nearly
    // rank-deficient matrix.
    let a_data = [1.0, 0.0, 0.0, 1e-4];
    let a = make_f64_matrix(2, 2, &a_data);

    let outputs =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();

    let g_u = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            (0..4).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );
    let g_s = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2] },
            vec![Literal::from_f64(1.0), Literal::from_f64(0.25)],
        )
        .unwrap(),
    );
    let g_vt = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            (0..4).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Svd,
        std::slice::from_ref(&a),
        &[g_u.clone(), g_s.clone(), g_vt.clone()],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);
    assert!(
        analytical.iter().all(|value| value.is_finite()),
        "ill-conditioned SVD VJP should stay finite: {analytical:?}"
    );

    let eps = 1e-6;
    let gs = [g_u, g_s, g_vt];
    let mut numerical = vec![0.0; 4];
    for idx in 0..4_usize {
        let mut plus = a_data.to_vec();
        plus[idx] += eps;
        let l_plus = loss_multi(
            Primitive::Svd,
            &make_f64_matrix(2, 2, &plus),
            &gs,
            &BTreeMap::new(),
        );

        let mut minus = a_data.to_vec();
        minus[idx] -= eps;
        let l_minus = loss_multi(
            Primitive::Svd,
            &make_f64_matrix(2, 2, &minus),
            &gs,
            &BTreeMap::new(),
        );

        numerical[idx] = (l_plus - l_minus) / (2.0 * eps);
    }

    assert_gradients_close(&analytical, &numerical, 1e-3, "SVD VJP ill conditioned");
}

#[test]
fn mul_vjp_denormal_input() {
    let x = Value::scalar_f64(f64::MIN_POSITIVE / 2.0);
    let scale = Value::scalar_f64(2.0);
    let g = Value::scalar_f64(1.0);

    let vjp_result = fj_ad::vjp(
        Primitive::Mul,
        &[x.clone(), scale.clone()],
        std::slice::from_ref(&g),
        &[],
        &BTreeMap::new(),
    )
    .unwrap();

    let analytical = extract_f64_scalar(&vjp_result[0]);
    assert!(analytical.is_finite(), "denormal VJP should stay finite");

    let eps = f64::MIN_POSITIVE / 4.0;
    let l_plus = extract_f64_scalar(
        &eval_primitive(
            Primitive::Mul,
            &[
                Value::scalar_f64(extract_f64_scalar(&x) + eps),
                scale.clone(),
            ],
            &BTreeMap::new(),
        )
        .unwrap(),
    );
    let l_minus = extract_f64_scalar(
        &eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(extract_f64_scalar(&x) - eps), scale],
            &BTreeMap::new(),
        )
        .unwrap(),
    );
    let numerical = (l_plus - l_minus) / (2.0 * eps);

    assert_scalar_close(
        analytical,
        numerical,
        1e-12,
        1e-12,
        "Mul VJP denormal input",
    );
}

// ======================== SVD VJP ========================

#[test]
#[allow(clippy::needless_range_loop)]
fn svd_vjp_numerical() {
    // Verify SVD VJP via singular values S only (S is unique and sign-invariant;
    // U and Vt have sign/rotation ambiguity under perturbation).
    let a_data = [3.0, 1.0, 1.0, 4.0, 0.5, 2.0];
    let a = make_f64_matrix(3, 2, &a_data);

    let outputs =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();
    // SVD returns: U (3x2), S (2,), Vt (2x2)

    // Zero cotangent for U and Vt, nonzero for S
    let g_u = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 2] },
            (0..6).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );
    let g_s = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2] },
            vec![Literal::from_f64(1.0), Literal::from_f64(0.5)],
        )
        .unwrap(),
    );
    let g_vt = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            (0..4).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Svd,
        std::slice::from_ref(&a),
        &[g_u.clone(), g_s.clone(), g_vt.clone()],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);

    let eps = 1e-5;
    let gs = [g_u, g_s, g_vt];
    let mut numerical = vec![0.0; 6];
    for idx in 0..6_usize {
        let mut plus = a_data.to_vec();
        plus[idx] += eps;
        let a_plus = make_f64_matrix(3, 2, &plus);
        let l_plus = loss_multi(Primitive::Svd, &a_plus, &gs, &BTreeMap::new());

        let mut minus = a_data.to_vec();
        minus[idx] -= eps;
        let a_minus = make_f64_matrix(3, 2, &minus);
        let l_minus = loss_multi(Primitive::Svd, &a_minus, &gs, &BTreeMap::new());

        numerical[idx] = (l_plus - l_minus) / (2.0 * eps);
    }

    assert_gradients_close(&analytical, &numerical, 1e-3, "SVD VJP (S only)");
}

// ======================== Eigh VJP ========================

#[test]
#[allow(clippy::needless_range_loop)]
fn eigh_vjp_numerical() {
    // Symmetric matrix with distinct eigenvalues for well-conditioned gradient
    #[rustfmt::skip]
    let a_data = [
        5.0, 1.0, 0.5,
        1.0, 4.0, 1.0,
        0.5, 1.0, 3.0,
    ];
    let a = make_f64_matrix(3, 3, &a_data);

    let outputs =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();
    // Eigh returns: eigenvalues (3,), eigenvectors (3x3)

    let g_w = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3] },
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(1.0),
                Literal::from_f64(1.0),
            ],
        )
        .unwrap(),
    );
    let g_v = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 3] },
            (0..9).map(|_| Literal::from_f64(0.1)).collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Eigh,
        std::slice::from_ref(&a),
        &[g_w.clone(), g_v.clone()],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);

    let eps = 1e-5;
    let gs = [g_w, g_v];
    let n = 3;
    let mut numerical = vec![0.0; 9];
    for idx in 0..9_usize {
        let row = idx / n;
        let col = idx % n;
        let mut plus = a_data.to_vec();
        plus[row * n + col] += eps;
        if row != col {
            plus[col * n + row] += eps;
        }
        let a_plus = make_f64_matrix(3, 3, &plus);
        let l_plus = loss_multi(Primitive::Eigh, &a_plus, &gs, &BTreeMap::new());

        let mut minus = a_data.to_vec();
        minus[row * n + col] -= eps;
        if row != col {
            minus[col * n + row] -= eps;
        }
        let a_minus = make_f64_matrix(3, 3, &minus);
        let l_minus = loss_multi(Primitive::Eigh, &a_minus, &gs, &BTreeMap::new());

        let mut grad = (l_plus - l_minus) / (2.0 * eps);
        if row != col {
            grad *= 0.5;
        }
        numerical[idx] = grad;
    }

    assert_gradients_close(&analytical, &numerical, 1e-3, "Eigh VJP");
}

// ======================== IFFT VJP ========================

#[test]
fn ifft_vjp_numerical() {
    // IFFT is linear, so VJP(g) = adjoint(IFFT)(g) = (1/n) * FFT(g)
    use fj_core::Literal::Complex128Bits;

    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![4] },
            [1.0, -0.5, 2.0, 0.3]
                .iter()
                .map(|&v| Literal::from_complex128(v, 0.0))
                .collect(),
        )
        .unwrap(),
    );

    let g = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![4] },
            (0..4).map(|_| Literal::from_complex128(1.0, 0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp_single(
        Primitive::Ifft,
        std::slice::from_ref(&x),
        &g,
        &BTreeMap::new(),
    )
    .unwrap();

    // For linear op IFFT: VJP(g) = adjoint(IFFT)(g) = (1/n) * FFT(g)
    let fft_g = eval_primitive(Primitive::Fft, std::slice::from_ref(&g), &BTreeMap::new()).unwrap();
    let n = 4.0;

    let vjp_vals: Vec<(f64, f64)> = vjp_result[0]
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
            _ => panic!("expected complex"),
        })
        .collect();

    let fft_vals: Vec<(f64, f64)> = fft_g
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
            _ => panic!("expected complex"),
        })
        .collect();

    for (i, ((vr, vi), (fr, fi))) in vjp_vals.iter().zip(fft_vals.iter()).enumerate() {
        let expected_re = fr / n;
        let expected_im = fi / n;
        assert!(
            (vr - expected_re).abs() < 1e-10 && (vi - expected_im).abs() < 1e-10,
            "IFFT VJP[{i}]: got ({vr},{vi}), expected ({expected_re},{expected_im})"
        );
    }
}

// ======================== IRFFT VJP ========================

#[test]
fn irfft_vjp_numerical() {
    // IRFFT: C^{n/2+1} → R^n. Verify VJP via finite differences.
    let x_re = [10.0, -2.0, 2.0];
    let x_im = [0.0, 3.0, 0.0];

    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![3] },
            x_re.iter()
                .zip(x_im.iter())
                .map(|(&r, &i)| Literal::from_complex128(r, i))
                .collect(),
        )
        .unwrap(),
    );

    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());

    // Cotangent is real (output is real)
    let g = make_f64_vector(&[1.0, 1.0, 1.0, 1.0]);

    let vjp_result =
        fj_ad::vjp_single(Primitive::Irfft, std::slice::from_ref(&x), &g, &params).unwrap();

    // Finite-difference verification: perturb real and imag parts independently
    let eps = 1e-6;
    use fj_core::Literal::Complex128Bits;

    // Check gradient w.r.t. real part of each input element
    for bin in 0..3 {
        let mut re_plus = x_re.to_vec();
        re_plus[bin] += eps;
        let x_plus = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: vec![3] },
                re_plus
                    .iter()
                    .zip(x_im.iter())
                    .map(|(&r, &i)| Literal::from_complex128(r, i))
                    .collect(),
            )
            .unwrap(),
        );
        let out_plus =
            eval_primitive(Primitive::Irfft, std::slice::from_ref(&x_plus), &params).unwrap();
        let l_plus: f64 = extract_f64_vec(&out_plus).iter().sum();

        let mut re_minus = x_re.to_vec();
        re_minus[bin] -= eps;
        let x_minus = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: vec![3] },
                re_minus
                    .iter()
                    .zip(x_im.iter())
                    .map(|(&r, &i)| Literal::from_complex128(r, i))
                    .collect(),
            )
            .unwrap(),
        );
        let out_minus =
            eval_primitive(Primitive::Irfft, std::slice::from_ref(&x_minus), &params).unwrap();
        let l_minus: f64 = extract_f64_vec(&out_minus).iter().sum();

        let numerical_re = (l_plus - l_minus) / (2.0 * eps);

        let vjp_bin = match &vjp_result[0].as_tensor().unwrap().elements[bin] {
            Complex128Bits(re, _im) => f64::from_bits(*re),
            _ => panic!("expected complex"),
        };

        assert!(
            (vjp_bin - numerical_re).abs() < 1e-4,
            "IRFFT VJP real part at bin {bin}: analytical={vjp_bin}, numerical={numerical_re}"
        );
    }
}

// ======================== Edge-Case AD Verification (frankenjax-o2c) ========================

/// QR VJP with a moderately ill-conditioned rectangular matrix.
/// Condition number ~100 (not extreme, but stresses AD).
#[test]
#[allow(clippy::needless_range_loop)]
fn qr_vjp_near_singular_rectangular() {
    // 3x2 matrix with condition number ~100
    let a_data = [1.0, 0.01, 0.0, 0.5, -1.0, 0.02];
    let a = make_f64_matrix(3, 2, &a_data);

    let outputs =
        eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();

    // Cotangent for Q (3x2) and R (2x2)
    let g_q = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 2] },
            vec![
                Literal::from_f64(1.0),
                Literal::from_f64(0.0),
                Literal::from_f64(0.0),
                Literal::from_f64(1.0),
                Literal::from_f64(0.5),
                Literal::from_f64(0.5),
            ],
        )
        .unwrap(),
    );
    let g_r = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(0.1),
                Literal::from_f64(0.2),
                Literal::from_f64(0.0),
                Literal::from_f64(0.3),
            ],
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Qr,
        std::slice::from_ref(&a),
        &[g_q.clone(), g_r.clone()],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);
    assert!(
        analytical.iter().all(|v| v.is_finite()),
        "near-singular QR VJP should stay finite: {analytical:?}"
    );

    // Finite-difference verification
    let eps = 1e-6;
    let gs = [g_q, g_r];
    let mut numerical = vec![0.0; 6];
    for idx in 0..6_usize {
        let mut plus = a_data.to_vec();
        plus[idx] += eps;
        let l_plus = loss_multi(
            Primitive::Qr,
            &make_f64_matrix(3, 2, &plus),
            &gs,
            &BTreeMap::new(),
        );

        let mut minus = a_data.to_vec();
        minus[idx] -= eps;
        let l_minus = loss_multi(
            Primitive::Qr,
            &make_f64_matrix(3, 2, &minus),
            &gs,
            &BTreeMap::new(),
        );

        numerical[idx] = (l_plus - l_minus) / (2.0 * eps);
    }

    assert_gradients_close(
        &analytical,
        &numerical,
        1e-1,
        "QR VJP near singular rectangular",
    );
}

/// Eigh VJP with a well-conditioned 3x3 symmetric matrix.
/// Verifies correctness for a larger input size beyond the existing 2x2 test.
#[test]
#[allow(clippy::needless_range_loop)]
fn eigh_vjp_well_conditioned_3x3() {
    // Simple diagonal-dominant symmetric 3x3 (eigenvalues ≈ 1, 5, 10)
    let a_data = [5.0, 1.0, 0.5, 1.0, 8.0, 1.0, 0.5, 1.0, 3.0];
    let a = make_f64_matrix(3, 3, &a_data);

    let outputs =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();

    // Only perturb W (eigenvalue) cotangent for stable finite-diff comparison
    let g_w = make_f64_vector(&[1.0, 0.5, 0.25]);
    let g_v = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 3] },
            (0..9).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Eigh,
        std::slice::from_ref(&a),
        &[g_w.clone(), g_v.clone()],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);
    assert!(
        analytical.iter().all(|v| v.is_finite()),
        "3x3 Eigh VJP should stay finite: {analytical:?}"
    );

    // Finite differences (symmetric perturbation)
    let eps = 1e-6;
    let gs = [g_w, g_v];
    let mut numerical = vec![0.0; 9];
    for idx in 0..9_usize {
        let row = idx / 3;
        let col = idx % 3;

        let mut plus = a_data.to_vec();
        plus[row * 3 + col] += eps;
        if row != col {
            plus[col * 3 + row] += eps;
        }

        let mut minus = a_data.to_vec();
        minus[row * 3 + col] -= eps;
        if row != col {
            minus[col * 3 + row] -= eps;
        }

        let l_plus = loss_multi(
            Primitive::Eigh,
            &make_f64_matrix(3, 3, &plus),
            &gs,
            &BTreeMap::new(),
        );
        let l_minus = loss_multi(
            Primitive::Eigh,
            &make_f64_matrix(3, 3, &minus),
            &gs,
            &BTreeMap::new(),
        );

        let mut grad = (l_plus - l_minus) / (2.0 * eps);
        if row != col {
            grad *= 0.5;
        }
        numerical[idx] = grad;
    }

    assert_gradients_close(
        &analytical,
        &numerical,
        1e-3,
        "Eigh VJP well conditioned 3x3",
    );
}

/// Eigh VJP should remain bounded when eigenvalues are extremely close.
///
/// The exact derivative through eigenvectors is ill-conditioned in this regime,
/// so the implementation treats sufficiently small eigenvalue gaps as
/// degenerate and suppresses the unstable 1/(lambda_i-lambda_j) term.
#[test]
fn eigh_vjp_close_eigenvalues_is_bounded() {
    let a = make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0 + 1e-12]);
    let outputs =
        eval_primitive_multi(Primitive::Eigh, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();

    let g_w = make_f64_vector(&[0.0, 0.0]);
    let g_v = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(0.0),
                Literal::from_f64(1.0),
                Literal::from_f64(-1.0),
                Literal::from_f64(0.0),
            ],
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Eigh,
        std::slice::from_ref(&a),
        &[g_w, g_v],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);

    assert!(
        analytical.iter().all(|value| value.is_finite()),
        "close-eigenvalue Eigh VJP should remain finite: {analytical:?}"
    );
    assert!(
        analytical.iter().all(|value| value.abs() < 1e6),
        "close-eigenvalue Eigh VJP should be bounded after clustering guard: {analytical:?}"
    );
}

/// TriangularSolve VJP with near-zero diagonal (nearly singular).
/// L * x = b where L has a diagonal element close to zero.
#[test]
fn triangular_solve_vjp_near_singular_diagonal() {
    // Lower triangular with L[1,1] ≈ 0.001 (nearly singular)
    let l_data = [1.0, 0.0, 0.5, 0.001];
    let l_matrix = make_f64_matrix(2, 2, &l_data);
    // b must be rank-2 (matrix) for TriangularSolve
    let b = make_f64_matrix(2, 1, &[1.0, 0.5]);

    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());
    params.insert("unit_diagonal".to_owned(), "false".to_owned());

    let x_out = eval_primitive(
        Primitive::TriangularSolve,
        &[l_matrix.clone(), b.clone()],
        &params,
    )
    .unwrap();

    let g = make_f64_matrix(2, 1, &[1.0, 1.0]);

    let vjp_result = fj_ad::vjp(
        Primitive::TriangularSolve,
        &[l_matrix, b],
        std::slice::from_ref(&g),
        std::slice::from_ref(&x_out),
        &params,
    )
    .unwrap();

    // VJP should produce finite gradients for both L and b
    for (i, grad) in vjp_result.iter().enumerate() {
        let vals = extract_f64_vec(grad);
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "near-singular TriangularSolve VJP output {i} should be finite: {vals:?}"
        );
    }
}

/// Exp VJP at large input: exp(700) is near f64::MAX.
/// Verify gradients stay finite and are numerically accurate.
#[test]
fn exp_vjp_large_input() {
    let x = Value::scalar_f64(700.0);
    let g = Value::scalar_f64(1.0);

    let vjp_result = fj_ad::vjp_single(Primitive::Exp, std::slice::from_ref(&x), &g, &BTreeMap::new()).unwrap();
    let analytical = extract_f64_scalar(&vjp_result[0]);

    // exp'(x) = exp(x), so gradient should equal exp(700)
    assert!(analytical.is_finite(), "exp(700) gradient should be finite");
    let expected = 700.0_f64.exp();
    assert!(
        (analytical - expected).abs() / expected < 1e-10,
        "exp VJP at 700: analytical={analytical}, expected={expected}"
    );
}

/// Log VJP near zero: log(1e-300) exercises gradient 1/x with tiny x.
#[test]
fn log_vjp_near_zero() {
    let x = Value::scalar_f64(1e-300);
    let g = Value::scalar_f64(1.0);

    let vjp_result = fj_ad::vjp_single(Primitive::Log, std::slice::from_ref(&x), &g, &BTreeMap::new()).unwrap();
    let analytical = extract_f64_scalar(&vjp_result[0]);

    // log'(x) = 1/x = 1e300
    assert!(
        analytical.is_finite(),
        "log(1e-300) gradient should be finite"
    );
    let expected = 1.0 / 1e-300;
    assert!(
        (analytical - expected).abs() / expected < 1e-10,
        "log VJP near zero: analytical={analytical}, expected={expected}"
    );
}

/// Div VJP with near-zero denominator.
#[test]
fn div_vjp_near_zero_denominator() {
    let a = Value::scalar_f64(1.0);
    let b = Value::scalar_f64(1e-150);
    let g = Value::scalar_f64(1.0);

    let vjp_result = fj_ad::vjp_single(
        Primitive::Div,
        &[a.clone(), b.clone()],
        &g,
        &BTreeMap::new(),
    )
    .unwrap();

    // d(a/b)/da = 1/b = 1e150 (finite)
    let grad_a = extract_f64_scalar(&vjp_result[0]);
    assert!(
        grad_a.is_finite(),
        "div grad w.r.t. numerator should be finite"
    );
    assert!(
        (grad_a - 1e150).abs() / 1e150 < 1e-10,
        "div VJP grad_a: got {grad_a}, expected 1e150"
    );

    // d(a/b)/db = -a/b^2 = -1e300 (finite)
    let grad_b = extract_f64_scalar(&vjp_result[1]);
    assert!(
        grad_b.is_finite(),
        "div grad w.r.t. denominator should be finite"
    );
}

/// Cholesky VJP with 3x3 ill-conditioned SPD matrix (condition number ~10^6).
#[test]
#[allow(clippy::needless_range_loop)]
fn cholesky_vjp_ill_conditioned_3x3() {
    // SPD matrix with eigenvalues ~[1, 1e-3, 1e-6]
    // A = Q diag(1, 1e-3, 1e-6) Q^T where Q is a Householder reflection
    let _a_data = [
        1.000001, 0.0005, 0.0001, 0.0005, 0.001001, 0.0003, 0.0001, 0.0003, 0.000001,
    ];

    // Ensure it's SPD by constructing A = B^T B + eps*I
    let b_data = [1.0, 0.0, 0.0, 0.0005, 0.001, 0.0, 0.0001, 0.0003, 0.001];
    let mut spd = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                spd[i * 3 + j] += b_data[k * 3 + i] * b_data[k * 3 + j];
            }
            if i == j {
                spd[i * 3 + j] += 1e-8; // regularization
            }
        }
    }

    let a = make_f64_matrix(3, 3, &spd);
    let outputs = eval_primitive_multi(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        &BTreeMap::new(),
    )
    .unwrap();
    let l_out = &outputs[0];

    let g_l = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 3] },
            (0..9)
                .map(|i| Literal::from_f64(if i % 4 == 0 { 1.0 } else { 0.25 }))
                .collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Cholesky,
        std::slice::from_ref(&a),
        std::slice::from_ref(&g_l),
        std::slice::from_ref(l_out),
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);
    assert!(
        analytical.iter().all(|v| v.is_finite()),
        "ill-conditioned 3x3 Cholesky VJP should stay finite: {analytical:?}"
    );

    // Finite differences for symmetric perturbation
    let eps = 1e-7;
    let mut numerical = vec![0.0; 9];
    for idx in 0..9_usize {
        let row = idx / 3;
        let col = idx % 3;

        let mut plus = spd.to_vec();
        plus[row * 3 + col] += eps;
        if row != col {
            plus[col * 3 + row] += eps;
        }

        let mut minus = spd.to_vec();
        minus[row * 3 + col] -= eps;
        if row != col {
            minus[col * 3 + row] -= eps;
        }

        let l_plus = loss_multi(
            Primitive::Cholesky,
            &make_f64_matrix(3, 3, &plus),
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );
        let l_minus = loss_multi(
            Primitive::Cholesky,
            &make_f64_matrix(3, 3, &minus),
            std::slice::from_ref(&g_l),
            &BTreeMap::new(),
        );

        let mut grad = (l_plus - l_minus) / (2.0 * eps);
        if row != col {
            grad *= 0.5;
        }
        numerical[idx] = grad;
    }

    // Wider tolerance for ill-conditioned case (conditioning amplifies FD error)
    assert_gradients_close(
        &analytical,
        &numerical,
        1.0,
        "Cholesky VJP ill conditioned 3x3",
    );
}

/// SVD VJP with 3x3 matrix having a near-zero singular value.
#[test]
#[allow(clippy::needless_range_loop)]
fn svd_vjp_near_zero_singular_value() {
    // Matrix with singular values approximately [5.0, 1.0, 1e-6]
    let a_data = [5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-6];
    let a = make_f64_matrix(3, 3, &a_data);

    let outputs =
        eval_primitive_multi(Primitive::Svd, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();

    // Only perturb S cotangent to avoid numerical noise in U/Vt
    let g_u = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 3] },
            (0..9).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );
    let g_s = make_f64_vector(&[1.0, 0.5, 0.25]);
    let g_vt = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 3] },
            (0..9).map(|_| Literal::from_f64(0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result = fj_ad::vjp(
        Primitive::Svd,
        std::slice::from_ref(&a),
        &[g_u.clone(), g_s.clone(), g_vt.clone()],
        &outputs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp_result[0]);
    assert!(
        analytical.iter().all(|v| v.is_finite()),
        "near-zero singular value SVD VJP should stay finite: {analytical:?}"
    );

    // Finite differences
    let eps = 1e-6;
    let gs = [g_u, g_s, g_vt];
    let mut numerical = vec![0.0; 9];
    for idx in 0..9_usize {
        let mut plus = a_data.to_vec();
        plus[idx] += eps;
        let l_plus = loss_multi(
            Primitive::Svd,
            &make_f64_matrix(3, 3, &plus),
            &gs,
            &BTreeMap::new(),
        );

        let mut minus = a_data.to_vec();
        minus[idx] -= eps;
        let l_minus = loss_multi(
            Primitive::Svd,
            &make_f64_matrix(3, 3, &minus),
            &gs,
            &BTreeMap::new(),
        );

        numerical[idx] = (l_plus - l_minus) / (2.0 * eps);
    }

    // Wider tolerance for near-singular case (S cotangent only, so should be stable)
    assert_gradients_close(
        &analytical,
        &numerical,
        1e-3,
        "SVD VJP near zero singular value 3x3",
    );
}
