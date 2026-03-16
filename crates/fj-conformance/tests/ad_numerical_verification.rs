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
