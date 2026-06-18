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

fn make_complex128_vector(data: &[(f64, f64)]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: vec![data.len() as u32],
            },
            data.iter()
                .map(|&(re, im)| Literal::from_complex128(re, im))
                .collect(),
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

fn extract_complex_vec(val: &Value) -> Vec<(f64, f64)> {
    val.as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| l.as_complex128().unwrap())
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

fn assert_complex_gradients_close(
    analytical: &[(f64, f64)],
    expected: &[(f64, f64)],
    tol: f64,
    context: &str,
) {
    assert_eq!(
        analytical.len(),
        expected.len(),
        "{context}: gradient length mismatch"
    );
    for (i, ((ar, ai), (er, ei))) in analytical.iter().zip(expected.iter()).enumerate() {
        let re_diff = (ar - er).abs();
        let im_diff = (ai - ei).abs();
        assert!(
            re_diff < tol && im_diff < tol,
            "{context}[{i}]: analytical=({ar},{ai}), expected=({er},{ei}), diff=({re_diff},{im_diff}) (tol={tol})"
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

// ======================== Det / Slogdet / Solve VJP ========================

#[test]
#[allow(clippy::needless_range_loop)]
fn det_vjp_numerical_3x3() {
    // ∂det/∂A = det(A)·inv(A)ᵀ. Well-conditioned non-symmetric 3×3.
    let a_data = [2.0, 0.3, -0.1, 0.4, 3.0, 0.2, -0.2, 0.1, 2.5];
    let a = make_f64_matrix(3, 3, &a_data);
    let det_out =
        eval_primitive(Primitive::Det, std::slice::from_ref(&a), &BTreeMap::new()).unwrap();
    let g = Value::scalar_f64(1.0);
    let vjp = fj_ad::vjp(
        Primitive::Det,
        std::slice::from_ref(&a),
        std::slice::from_ref(&g),
        std::slice::from_ref(&det_out),
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp[0]);

    let eps = 1e-6;
    let mut numerical = [0.0; 9];
    for idx in 0..9 {
        let mut plus = a_data.to_vec();
        plus[idx] += eps;
        let mut minus = a_data.to_vec();
        minus[idx] -= eps;
        let dp = eval_primitive(
            Primitive::Det,
            &[make_f64_matrix(3, 3, &plus)],
            &BTreeMap::new(),
        )
        .unwrap();
        let dm = eval_primitive(
            Primitive::Det,
            &[make_f64_matrix(3, 3, &minus)],
            &BTreeMap::new(),
        )
        .unwrap();
        numerical[idx] = (extract_f64_scalar(&dp) - extract_f64_scalar(&dm)) / (2.0 * eps);
    }
    assert_gradients_close(&analytical, &numerical, 1e-5, "Det VJP");
}

#[test]
#[allow(clippy::needless_range_loop)]
fn slogdet_vjp_numerical_3x3() {
    // ∂logabsdet/∂A = inv(A)ᵀ (the sign output's cotangent contributes 0 for real A).
    let a_data = [2.0, 0.3, -0.1, 0.4, 3.0, 0.2, -0.2, 0.1, 2.5];
    let a = make_f64_matrix(3, 3, &a_data);
    let outs = eval_primitive_multi(
        Primitive::Slogdet,
        std::slice::from_ref(&a),
        &BTreeMap::new(),
    )
    .unwrap();
    // Cotangents: (sign → 0, logabsdet → 1).
    let g = [Value::scalar_f64(0.0), Value::scalar_f64(1.0)];
    let vjp = fj_ad::vjp(
        Primitive::Slogdet,
        std::slice::from_ref(&a),
        &g,
        &outs,
        &BTreeMap::new(),
    )
    .unwrap();
    let analytical = extract_f64_vec(&vjp[0]);

    let eps = 1e-6;
    let mut numerical = [0.0; 9];
    for idx in 0..9 {
        let mut plus = a_data.to_vec();
        plus[idx] += eps;
        let mut minus = a_data.to_vec();
        minus[idx] -= eps;
        let lp = eval_primitive_multi(
            Primitive::Slogdet,
            &[make_f64_matrix(3, 3, &plus)],
            &BTreeMap::new(),
        )
        .unwrap();
        let lm = eval_primitive_multi(
            Primitive::Slogdet,
            &[make_f64_matrix(3, 3, &minus)],
            &BTreeMap::new(),
        )
        .unwrap();
        // loss = 1·logabsdet (output index 1).
        numerical[idx] = (extract_f64_scalar(&lp[1]) - extract_f64_scalar(&lm[1])) / (2.0 * eps);
    }
    assert_gradients_close(&analytical, &numerical, 1e-5, "Slogdet VJP");
}

#[test]
#[allow(clippy::needless_range_loop)]
fn solve_vjp_numerical_3x3() {
    // x = A⁻¹b. ∂L/∂A = −A⁻ᵀ ḡ xᵀ, ∂L/∂b = A⁻ᵀ ḡ. Checked w.r.t. both A and b.
    let a_data = [4.0, 1.0, 0.5, 0.2, 5.0, 1.0, 0.3, 0.4, 6.0];
    let b_data = [1.0, 2.0, 3.0];
    let a = make_f64_matrix(3, 3, &a_data);
    let b = make_f64_vector(&b_data);
    let x = eval_primitive(Primitive::Solve, &[a.clone(), b.clone()], &BTreeMap::new()).unwrap();
    let g_data = [1.0, -0.5, 0.7];
    let g = make_f64_vector(&g_data);
    let vjp = fj_ad::vjp(
        Primitive::Solve,
        &[a.clone(), b.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&x),
        &BTreeMap::new(),
    )
    .unwrap();
    let grad_a = extract_f64_vec(&vjp[0]);
    let grad_b = extract_f64_vec(&vjp[1]);

    let eps = 1e-6;
    let loss = |av: &Value, bv: &Value| -> f64 {
        let xx = eval_primitive(
            Primitive::Solve,
            &[av.clone(), bv.clone()],
            &BTreeMap::new(),
        )
        .unwrap();
        extract_f64_vec(&xx)
            .iter()
            .zip(g_data.iter())
            .map(|(xi, gi)| xi * gi)
            .sum()
    };
    let mut num_a = [0.0; 9];
    for idx in 0..9 {
        let mut p = a_data.to_vec();
        p[idx] += eps;
        let mut m = a_data.to_vec();
        m[idx] -= eps;
        num_a[idx] = (loss(&make_f64_matrix(3, 3, &p), &b) - loss(&make_f64_matrix(3, 3, &m), &b))
            / (2.0 * eps);
    }
    let mut num_b = [0.0; 3];
    for idx in 0..3 {
        let mut p = b_data.to_vec();
        p[idx] += eps;
        let mut m = b_data.to_vec();
        m[idx] -= eps;
        num_b[idx] =
            (loss(&a, &make_f64_vector(&p)) - loss(&a, &make_f64_vector(&m))) / (2.0 * eps);
    }
    assert_gradients_close(&grad_a, &num_a, 1e-5, "Solve VJP w.r.t. A");
    assert_gradients_close(&grad_b, &num_b, 1e-5, "Solve VJP w.r.t. b");
}

// ======================== DotGeneral VJP ========================

#[test]
fn dot_general_vjp_numerical() {
    // DotGeneral's VJP transposes the contraction back onto each operand across
    // batch/contracting/free dims — easy to get subtly wrong, and it was absent
    // from the numerical suites. Verify a BATCHED matmul (lhs=[B,M,K]=[2,2,3],
    // rhs=[B,K,N]=[2,3,2] → out=[B,M,N]=[2,2,2]) against central differences of
    // the cotangent-induced loss w.r.t. BOTH operands.
    let t3 = |dims: Vec<u32>, data: &[f64]| {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    };
    let lhs_data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let rhs_data = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0];
    let lhs = t3(vec![2, 2, 3], &lhs_data);
    let rhs = t3(vec![2, 3, 2], &rhs_data);

    let mut params = BTreeMap::new();
    params.insert("lhs_contracting_dims".to_string(), "2".to_string());
    params.insert("rhs_contracting_dims".to_string(), "1".to_string());
    params.insert("lhs_batch_dims".to_string(), "0".to_string());
    params.insert("rhs_batch_dims".to_string(), "0".to_string());

    let out = eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &params).unwrap();
    let g_data = [1.0, -0.5, 0.7, 0.3, -1.1, 0.9, 0.2, -0.4];
    let g = t3(vec![2, 2, 2], &g_data);

    let vjp = fj_ad::vjp(
        Primitive::DotGeneral,
        &[lhs.clone(), rhs.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad_lhs = extract_f64_vec(&vjp[0]);
    let grad_rhs = extract_f64_vec(&vjp[1]);

    let eps = 1e-6;
    let loss = |lv: &Value, rv: &Value| -> f64 {
        let o = eval_primitive(Primitive::DotGeneral, &[lv.clone(), rv.clone()], &params).unwrap();
        extract_f64_vec(&o)
            .iter()
            .zip(g_data.iter())
            .map(|(oi, gi)| oi * gi)
            .sum()
    };
    let mut num_lhs = vec![0.0; lhs_data.len()];
    for idx in 0..lhs_data.len() {
        let mut p = lhs_data.to_vec();
        p[idx] += eps;
        let mut m = lhs_data.to_vec();
        m[idx] -= eps;
        num_lhs[idx] =
            (loss(&t3(vec![2, 2, 3], &p), &rhs) - loss(&t3(vec![2, 2, 3], &m), &rhs)) / (2.0 * eps);
    }
    let mut num_rhs = vec![0.0; rhs_data.len()];
    for idx in 0..rhs_data.len() {
        let mut p = rhs_data.to_vec();
        p[idx] += eps;
        let mut m = rhs_data.to_vec();
        m[idx] -= eps;
        num_rhs[idx] =
            (loss(&lhs, &t3(vec![2, 3, 2], &p)) - loss(&lhs, &t3(vec![2, 3, 2], &m))) / (2.0 * eps);
    }
    assert_gradients_close(&grad_lhs, &num_lhs, 1e-5, "DotGeneral VJP w.r.t. lhs");
    assert_gradients_close(&grad_rhs, &num_rhs, 1e-5, "DotGeneral VJP w.r.t. rhs");
}

#[test]
fn dot_general_vjp_noncanonical_operand_order() {
    // NON-CANONICAL lhs dim order: lhs=[K,M] has its CONTRACTING dim (K) BEFORE
    // its free dim (M). The VJP computes grad_lhs in [batch,free,contract]=[M,K]
    // layout, which must be transposed back to the operand's order [K,M]. Without
    // the transpose grad_lhs comes out shaped/ordered as [M,K] — a wrong gradient.
    let t = |dims: Vec<u32>, data: &[f64]| {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    };
    let lhs_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [K=3, M=2]
    let rhs_data = [1.0, 0.5, 2.0, -1.0, 0.0, 3.0, 1.0, 1.0, -2.0, 0.0, 0.5, 2.0]; // [K=3, N=4]
    let lhs = t(vec![3, 2], &lhs_data);
    let rhs = t(vec![3, 4], &rhs_data);

    let mut params = BTreeMap::new();
    params.insert("lhs_contracting_dims".to_string(), "0".to_string());
    params.insert("rhs_contracting_dims".to_string(), "0".to_string());
    params.insert("lhs_batch_dims".to_string(), String::new());
    params.insert("rhs_batch_dims".to_string(), String::new());

    let out = eval_primitive(Primitive::DotGeneral, &[lhs.clone(), rhs.clone()], &params).unwrap();
    let g_data = [1.0, -0.5, 0.7, 0.3, -1.1, 0.9, 0.2, -0.4]; // [M=2, N=4]
    let g = t(vec![2, 4], &g_data);

    let vjp = fj_ad::vjp(
        Primitive::DotGeneral,
        &[lhs.clone(), rhs.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad_lhs = extract_f64_vec(&vjp[0]);
    let grad_rhs = extract_f64_vec(&vjp[1]);

    let eps = 1e-6;
    let loss = |lv: &Value, rv: &Value| -> f64 {
        let o = eval_primitive(Primitive::DotGeneral, &[lv.clone(), rv.clone()], &params).unwrap();
        extract_f64_vec(&o)
            .iter()
            .zip(g_data.iter())
            .map(|(o, g)| o * g)
            .sum()
    };
    let mut num_lhs = vec![0.0; lhs_data.len()];
    for idx in 0..lhs_data.len() {
        let (mut p, mut m) = (lhs_data.to_vec(), lhs_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_lhs[idx] =
            (loss(&t(vec![3, 2], &p), &rhs) - loss(&t(vec![3, 2], &m), &rhs)) / (2.0 * eps);
    }
    let mut num_rhs = vec![0.0; rhs_data.len()];
    for idx in 0..rhs_data.len() {
        let (mut p, mut m) = (rhs_data.to_vec(), rhs_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_rhs[idx] =
            (loss(&lhs, &t(vec![3, 4], &p)) - loss(&lhs, &t(vec![3, 4], &m))) / (2.0 * eps);
    }
    assert_gradients_close(
        &grad_lhs,
        &num_lhs,
        1e-5,
        "DotGeneral VJP grad_lhs (non-canonical)",
    );
    assert_gradients_close(
        &grad_rhs,
        &num_rhs,
        1e-5,
        "DotGeneral VJP grad_rhs (non-canonical)",
    );
}

#[test]
fn conv_vjp_strided_numerical() {
    // Conv VJP under STRIDE>1 — the error-prone path (grad_lhs is a strided/
    // transposed conv, grad_rhs sub-samples the input). The existing in-crate
    // conv_vjp test only checks shapes with stride 1, so the strided gradient was
    // never value-verified. lhs=[N,W,Cin]=[1,5,1], rhs=[K,Cin,Cout]=[2,1,1],
    // strides=2, valid → out=[1,2,1]. Compare both grads to central differences.
    let t = |dims: Vec<u32>, data: &[f64]| {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    };
    let lhs_data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let rhs_data = [0.5, -0.25];
    let lhs = t(vec![1, 5, 1], &lhs_data);
    let rhs = t(vec![2, 1, 1], &rhs_data);

    let mut params = BTreeMap::new();
    params.insert("strides".to_string(), "2".to_string());
    params.insert("padding".to_string(), "VALID".to_string());

    let out = eval_primitive(Primitive::Conv, &[lhs.clone(), rhs.clone()], &params).unwrap();
    let g_data = [1.3, -0.7]; // out shape [1,2,1]
    let g = t(vec![1, 2, 1], &g_data);

    let vjp = fj_ad::vjp(
        Primitive::Conv,
        &[lhs.clone(), rhs.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad_lhs = extract_f64_vec(&vjp[0]);
    let grad_rhs = extract_f64_vec(&vjp[1]);

    let eps = 1e-6;
    let loss = |lv: &Value, rv: &Value| -> f64 {
        let o = eval_primitive(Primitive::Conv, &[lv.clone(), rv.clone()], &params).unwrap();
        extract_f64_vec(&o)
            .iter()
            .zip(g_data.iter())
            .map(|(o, g)| o * g)
            .sum()
    };
    let mut num_lhs = vec![0.0; lhs_data.len()];
    for idx in 0..lhs_data.len() {
        let (mut p, mut m) = (lhs_data.to_vec(), lhs_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_lhs[idx] =
            (loss(&t(vec![1, 5, 1], &p), &rhs) - loss(&t(vec![1, 5, 1], &m), &rhs)) / (2.0 * eps);
    }
    let mut num_rhs = vec![0.0; rhs_data.len()];
    for idx in 0..rhs_data.len() {
        let (mut p, mut m) = (rhs_data.to_vec(), rhs_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_rhs[idx] =
            (loss(&lhs, &t(vec![2, 1, 1], &p)) - loss(&lhs, &t(vec![2, 1, 1], &m))) / (2.0 * eps);
    }
    assert_gradients_close(&grad_lhs, &num_lhs, 1e-5, "Conv VJP grad_lhs (stride 2)");
    assert_gradients_close(&grad_rhs, &num_rhs, 1e-5, "Conv VJP grad_rhs (stride 2)");
}

#[test]
fn conv2d_vjp_same_padding_numerical() {
    // 2D conv VJP with SAME padding — the CNN-typical config and the trickiest
    // gradient routing (asymmetric pad must be mirrored into grad_lhs). lhs=
    // [N,H,W,Cin]=[1,3,3,1], rhs=[KH,KW,Cin,Cout]=[2,2,1,1], SAME → out=[1,3,3,1].
    let t = |dims: Vec<u32>, data: &[f64]| {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    };
    let lhs_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let rhs_data = [0.5, -0.25, 0.75, -1.0];
    let lhs = t(vec![1, 3, 3, 1], &lhs_data);
    let rhs = t(vec![2, 2, 1, 1], &rhs_data);

    let mut params = BTreeMap::new();
    params.insert("strides".to_string(), "1".to_string());
    params.insert("padding".to_string(), "SAME".to_string());

    let out = eval_primitive(Primitive::Conv, &[lhs.clone(), rhs.clone()], &params).unwrap();
    let g_data = [1.0, -0.5, 0.7, 0.3, -1.1, 0.9, 0.2, -0.4, 0.6]; // out [1,3,3,1]
    let g = t(vec![1, 3, 3, 1], &g_data);

    let vjp = fj_ad::vjp(
        Primitive::Conv,
        &[lhs.clone(), rhs.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad_lhs = extract_f64_vec(&vjp[0]);
    let grad_rhs = extract_f64_vec(&vjp[1]);

    let eps = 1e-6;
    let loss = |lv: &Value, rv: &Value| -> f64 {
        let o = eval_primitive(Primitive::Conv, &[lv.clone(), rv.clone()], &params).unwrap();
        extract_f64_vec(&o)
            .iter()
            .zip(g_data.iter())
            .map(|(o, g)| o * g)
            .sum()
    };
    let mut num_lhs = vec![0.0; lhs_data.len()];
    for idx in 0..lhs_data.len() {
        let (mut p, mut m) = (lhs_data.to_vec(), lhs_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_lhs[idx] = (loss(&t(vec![1, 3, 3, 1], &p), &rhs)
            - loss(&t(vec![1, 3, 3, 1], &m), &rhs))
            / (2.0 * eps);
    }
    let mut num_rhs = vec![0.0; rhs_data.len()];
    for idx in 0..rhs_data.len() {
        let (mut p, mut m) = (rhs_data.to_vec(), rhs_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_rhs[idx] = (loss(&lhs, &t(vec![2, 2, 1, 1], &p))
            - loss(&lhs, &t(vec![2, 2, 1, 1], &m)))
            / (2.0 * eps);
    }
    assert_gradients_close(&grad_lhs, &num_lhs, 1e-5, "Conv2D VJP grad_lhs (SAME)");
    assert_gradients_close(&grad_rhs, &num_rhs, 1e-5, "Conv2D VJP grad_rhs (SAME)");
}

#[test]
fn pad_vjp_interior_dilation_numerical() {
    // Pad VJP with INTERIOR (dilation) + edge padding — the gradient must
    // strided-slice the operand positions out of the cotangent and sum the rest
    // into the pad-VALUE gradient. operand=[3], pad_value scalar, low=1, high=1,
    // interior=1 → out=[7] laid out [pv, o0, pv, o1, pv, o2, pv]; operand feeds
    // positions 1,3,5. Pad grad was absent from the numerical suites.
    let operand_data = [1.0, 2.0, 3.0];
    let operand = make_f64_vector(&operand_data);
    let pad_value = Value::Scalar(Literal::from_f64(0.5));

    let mut params = BTreeMap::new();
    params.insert("padding_low".to_string(), "1".to_string());
    params.insert("padding_high".to_string(), "1".to_string());
    params.insert("padding_interior".to_string(), "1".to_string());

    let out = eval_primitive(
        Primitive::Pad,
        &[operand.clone(), pad_value.clone()],
        &params,
    )
    .unwrap();
    assert_eq!(extract_f64_vec(&out).len(), 7, "pad output shape");
    let g_data = [1.0, -0.5, 0.7, 0.3, -1.1, 0.9, 0.2];
    let g = make_f64_vector(&g_data);

    let vjp = fj_ad::vjp(
        Primitive::Pad,
        &[operand.clone(), pad_value.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad_operand = extract_f64_vec(&vjp[0]);
    // grad_pad_value is a scalar (pad_value is a scalar input).
    let grad_pad_value = match &vjp[1] {
        Value::Scalar(l) => l.as_f64().unwrap(),
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).sum(),
    };

    let eps = 1e-6;
    let loss = |ov: &Value, pv: &Value| -> f64 {
        let o = eval_primitive(Primitive::Pad, &[ov.clone(), pv.clone()], &params).unwrap();
        extract_f64_vec(&o)
            .iter()
            .zip(g_data.iter())
            .map(|(o, g)| o * g)
            .sum()
    };
    let mut num_operand = vec![0.0; operand_data.len()];
    for idx in 0..operand_data.len() {
        let (mut p, mut m) = (operand_data.to_vec(), operand_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_operand[idx] = (loss(&make_f64_vector(&p), &pad_value)
            - loss(&make_f64_vector(&m), &pad_value))
            / (2.0 * eps);
    }
    let num_pad_value = (loss(&operand, &Value::Scalar(Literal::from_f64(0.5 + eps)))
        - loss(&operand, &Value::Scalar(Literal::from_f64(0.5 - eps))))
        / (2.0 * eps);

    assert_gradients_close(
        &grad_operand,
        &num_operand,
        1e-5,
        "Pad VJP grad_operand (interior)",
    );
    assert_gradients_close(
        &[grad_pad_value],
        &[num_pad_value],
        1e-5,
        "Pad VJP grad_pad_value (interior)",
    );
}

#[test]
fn pad_vjp_negative_crop_numerical() {
    // Pad VJP with NEGATIVE low (crop) — the operand positions that fall outside
    // the cropped output must receive ZERO gradient (the pos<0 / pos>=dim drop).
    // operand=[4], low=-1, high=0, interior=0 → out=[3] = operand[1..4]; operand[0]
    // is cropped away so its grad must be 0.
    let operand_data = [1.0, 2.0, 3.0, 4.0];
    let operand = make_f64_vector(&operand_data);
    let pad_value = Value::Scalar(Literal::from_f64(0.0));

    let mut params = BTreeMap::new();
    params.insert("padding_low".to_string(), "-1".to_string());
    params.insert("padding_high".to_string(), "0".to_string());
    params.insert("padding_interior".to_string(), "0".to_string());

    let out = eval_primitive(
        Primitive::Pad,
        &[operand.clone(), pad_value.clone()],
        &params,
    )
    .unwrap();
    assert_eq!(extract_f64_vec(&out).len(), 3, "cropped output shape");
    let g_data = [1.3, -0.7, 2.1];
    let g = make_f64_vector(&g_data);

    let vjp = fj_ad::vjp(
        Primitive::Pad,
        &[operand.clone(), pad_value.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad_operand = extract_f64_vec(&vjp[0]);

    let eps = 1e-6;
    let loss = |ov: &Value| -> f64 {
        let o = eval_primitive(Primitive::Pad, &[ov.clone(), pad_value.clone()], &params).unwrap();
        extract_f64_vec(&o)
            .iter()
            .zip(g_data.iter())
            .map(|(o, g)| o * g)
            .sum()
    };
    let mut num_operand = vec![0.0; operand_data.len()];
    for idx in 0..operand_data.len() {
        let (mut p, mut m) = (operand_data.to_vec(), operand_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_operand[idx] = (loss(&make_f64_vector(&p)) - loss(&make_f64_vector(&m))) / (2.0 * eps);
    }
    assert_eq!(
        grad_operand[0], 0.0,
        "cropped operand[0] must have zero grad"
    );
    assert_gradients_close(
        &grad_operand,
        &num_operand,
        1e-5,
        "Pad VJP grad_operand (negative crop)",
    );
}

#[test]
fn reduce_max_min_vjp_axis_numerical() {
    // ReduceMax/Min VJP routes the cotangent to the arg-extremum of each
    // reduction window (a select-by-comparison, the same class that bit sort /
    // reduce_window). With DISTINCT values the max/min is smooth a.e., so a
    // finite difference verifies the indicator routing across multiple windows.
    // The existing test is a single full-reduce with no axis/ties/finite-diff.
    let data = [3.0, 1.0, 2.0, 4.0, 6.0, 5.0]; // [2,3]: rows [3,1,2],[4,6,5]
    let input = make_f64_matrix(2, 3, &data);
    let mut params = BTreeMap::new();
    params.insert("axes".to_string(), "1".to_string());

    for prim in [Primitive::ReduceMax, Primitive::ReduceMin] {
        let out = eval_primitive(prim, std::slice::from_ref(&input), &params).unwrap();
        let g_data = [1.3, -0.7]; // out shape [2]
        let g = make_f64_vector(&g_data);
        let vjp = fj_ad::vjp(
            prim,
            &[input.clone()],
            std::slice::from_ref(&g),
            std::slice::from_ref(&out),
            &params,
        )
        .unwrap();
        let grad = extract_f64_vec(&vjp[0]);

        let eps = 1e-6;
        let loss = |xv: &Value| -> f64 {
            let o = eval_primitive(prim, std::slice::from_ref(xv), &params).unwrap();
            extract_f64_vec(&o)
                .iter()
                .zip(g_data.iter())
                .map(|(o, g)| o * g)
                .sum()
        };
        let mut num = vec![0.0; data.len()];
        for idx in 0..data.len() {
            let (mut p, mut m) = (data.to_vec(), data.to_vec());
            p[idx] += eps;
            m[idx] -= eps;
            num[idx] =
                (loss(&make_f64_matrix(2, 3, &p)) - loss(&make_f64_matrix(2, 3, &m))) / (2.0 * eps);
        }
        assert_gradients_close(&grad, &num, 1e-5, &format!("{prim:?} VJP (axis, distinct)"));
    }
}

#[test]
fn reduce_max_vjp_splits_ties_evenly() {
    // At a TIE the chooser cotangent is split EVENLY among the tied maxima
    // (JAX _reduce_chooser). reduce_max([5,5,1]) = 5 with two ties → g=1 routes
    // 0.5 to each tied position, 0 to the non-max. Finite differences are
    // ill-defined at a tie, so this is an exact value check of the tie-split.
    let input = make_f64_vector(&[5.0, 5.0, 1.0]);
    let g = Value::scalar_f64(1.0);
    let grads = fj_ad::vjp(
        Primitive::ReduceMax,
        std::slice::from_ref(&input),
        std::slice::from_ref(&g),
        std::slice::from_ref(
            &eval_primitive(
                Primitive::ReduceMax,
                std::slice::from_ref(&input),
                &BTreeMap::new(),
            )
            .unwrap(),
        ),
        &BTreeMap::new(),
    )
    .unwrap();
    let vals = extract_f64_vec(&grads[0]);
    assert!(
        (vals[0] - 0.5).abs() < 1e-12,
        "tied max[0] must get 0.5, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 0.5).abs() < 1e-12,
        "tied max[1] must get 0.5, got {}",
        vals[1]
    );
    assert!(vals[2].abs() < 1e-12, "non-max must get 0, got {}", vals[2]);
}

#[test]
fn scatter_vjp_both_modes_numerical() {
    // Scatter VJP is MODE-dependent: overwrite → grad_operand is g with the
    // scattered rows zeroed (operand doesn't reach those outputs); add →
    // grad_operand is the full g. grad_updates gathers g at the scattered rows in
    // both modes. Verify BOTH operand and updates grads against finite differences
    // for each mode — the mode split is a classic subtle-bug site and had no
    // finite-diff coverage. operand=[4,2], indices=[0,2], updates=[2,2].
    let operand_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let updates_data = [10.0, 20.0, 30.0, 40.0];
    let operand = make_f64_matrix(4, 2, &operand_data);
    let updates = make_f64_matrix(2, 2, &updates_data);
    let indices = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: vec![2] },
            vec![Literal::I64(0), Literal::I64(2)],
        )
        .unwrap(),
    );

    for mode in ["overwrite", "add"] {
        let mut params = BTreeMap::new();
        params.insert("mode".to_string(), mode.to_string());

        let out = eval_primitive(
            Primitive::Scatter,
            &[operand.clone(), indices.clone(), updates.clone()],
            &params,
        )
        .unwrap();
        let g_data = [1.3, -0.7, 2.1, 0.5, -1.1, 0.9, 0.2, -0.4]; // out shape [4,2]
        let g = make_f64_matrix(4, 2, &g_data);

        let vjp = fj_ad::vjp(
            Primitive::Scatter,
            &[operand.clone(), indices.clone(), updates.clone()],
            std::slice::from_ref(&g),
            std::slice::from_ref(&out),
            &params,
        )
        .unwrap();
        let grad_operand = extract_f64_vec(&vjp[0]);
        let grad_updates = extract_f64_vec(&vjp[2]); // vjp[1] is the (zero) indices grad

        let eps = 1e-6;
        let loss = |ov: &Value, uv: &Value| -> f64 {
            let o = eval_primitive(
                Primitive::Scatter,
                &[ov.clone(), indices.clone(), uv.clone()],
                &params,
            )
            .unwrap();
            extract_f64_vec(&o)
                .iter()
                .zip(g_data.iter())
                .map(|(o, g)| o * g)
                .sum()
        };
        let mut num_operand = vec![0.0; operand_data.len()];
        for idx in 0..operand_data.len() {
            let (mut p, mut m) = (operand_data.to_vec(), operand_data.to_vec());
            p[idx] += eps;
            m[idx] -= eps;
            num_operand[idx] = (loss(&make_f64_matrix(4, 2, &p), &updates)
                - loss(&make_f64_matrix(4, 2, &m), &updates))
                / (2.0 * eps);
        }
        let mut num_updates = vec![0.0; updates_data.len()];
        for idx in 0..updates_data.len() {
            let (mut p, mut m) = (updates_data.to_vec(), updates_data.to_vec());
            p[idx] += eps;
            m[idx] -= eps;
            num_updates[idx] = (loss(&operand, &make_f64_matrix(2, 2, &p))
                - loss(&operand, &make_f64_matrix(2, 2, &m)))
                / (2.0 * eps);
        }
        assert_gradients_close(
            &grad_operand,
            &num_operand,
            1e-5,
            &format!("Scatter VJP grad_operand ({mode})"),
        );
        assert_gradients_close(
            &grad_updates,
            &num_updates,
            1e-5,
            &format!("Scatter VJP grad_updates ({mode})"),
        );
    }
}

#[test]
fn gather_vjp_numerical() {
    // Gather VJP scatter-adds the cotangent back into a zero operand at the
    // gathered rows. Verify against finite differences with a multi-column slice
    // (slice_sizes=[1,2]) so the per-row routing is exercised. operand=[4,2],
    // indices=[2,0,3] → out=[3,2]. (Duplicate-index accumulation + OOB clipping
    // already have value tests; this adds the general finite-diff.)
    let operand_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let operand = make_f64_matrix(4, 2, &operand_data);
    let indices = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: vec![3] },
            vec![Literal::I64(2), Literal::I64(0), Literal::I64(3)],
        )
        .unwrap(),
    );
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_string(), "1,2".to_string());

    let out = eval_primitive(
        Primitive::Gather,
        &[operand.clone(), indices.clone()],
        &params,
    )
    .unwrap();
    let g_data = [1.3, -0.7, 2.1, 0.5, -1.1, 0.9]; // out shape [3,2]
    let g = make_f64_matrix(3, 2, &g_data);

    let vjp = fj_ad::vjp(
        Primitive::Gather,
        &[operand.clone(), indices.clone()],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad_operand = extract_f64_vec(&vjp[0]);

    let eps = 1e-6;
    let loss = |ov: &Value| -> f64 {
        let o = eval_primitive(Primitive::Gather, &[ov.clone(), indices.clone()], &params).unwrap();
        extract_f64_vec(&o)
            .iter()
            .zip(g_data.iter())
            .map(|(o, g)| o * g)
            .sum()
    };
    let mut num_operand = vec![0.0; operand_data.len()];
    for idx in 0..operand_data.len() {
        let (mut p, mut m) = (operand_data.to_vec(), operand_data.to_vec());
        p[idx] += eps;
        m[idx] -= eps;
        num_operand[idx] =
            (loss(&make_f64_matrix(4, 2, &p)) - loss(&make_f64_matrix(4, 2, &m))) / (2.0 * eps);
    }
    assert_gradients_close(&grad_operand, &num_operand, 1e-5, "Gather VJP grad_operand");
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

// ======================== Complex primitive VJP ========================

/// Complex64 scalar Mul VJP (frankenjax-o9xn).
///
/// Closed-form derivatives of `c = a * b` in complex arithmetic are
/// `dc/da = b` and `dc/db = a`. With a Complex64 cotangent `g`, the
/// chain rule yields `g_a = g * b` and `g_b = g * a`. This pins both
/// dtype preservation (gradients stay Complex64) and exact value
/// correctness against complex multiplication.
#[test]
fn mul_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let a = Value::Scalar(Literal::from_complex64(2.0, 3.0)); // 2 + 3i
    let b = Value::Scalar(Literal::from_complex64(-1.0, 4.0)); // -1 + 4i
    let g = Value::Scalar(Literal::from_complex64(0.5, -0.25)); // cotangent

    let grads = fj_ad::vjp_single(
        Primitive::Mul,
        &[a.clone(), b.clone()],
        &g,
        &BTreeMap::new(),
    )
    .expect("mul VJP should accept complex64 scalars");

    assert_eq!(grads.len(), 2, "Mul VJP should return two gradients");

    // dtype preservation
    let extract_complex = |v: &Value| -> (f32, f32) {
        match v {
            Value::Scalar(Complex64Bits(re, im)) => (f32::from_bits(*re), f32::from_bits(*im)),
            other => panic!("expected Complex64 scalar, got {other:?}"),
        }
    };

    let g_a = extract_complex(&grads[0]);
    let g_b = extract_complex(&grads[1]);

    // g * b = (0.5 - 0.25i)(-1 + 4i)
    //       = (0.5*-1 - (-0.25)*4, 0.5*4 + (-0.25)*-1)
    //       = (-0.5 + 1.0, 2.0 + 0.25)
    //       = (0.5, 2.25)
    let close = |x: f32, y: f32| (x - y).abs() < 1e-5;
    assert!(
        close(g_a.0, 0.5) && close(g_a.1, 2.25),
        "g_a = g*b should be (0.5, 2.25); got {g_a:?}"
    );

    // g * a = (0.5 - 0.25i)(2 + 3i)
    //       = (0.5*2 - (-0.25)*3, 0.5*3 + (-0.25)*2)
    //       = (1.0 + 0.75, 1.5 - 0.5)
    //       = (1.75, 1.0)
    assert!(
        close(g_b.0, 1.75) && close(g_b.1, 1.0),
        "g_b = g*a should be (1.75, 1.0); got {g_b:?}"
    );
}

/// F64 scalar Erf VJP value sanity (frankenjax-elpz).
///
/// fj-ad's Erf VJP constants now route through `scalar_constant_matching_dtype`
/// instead of unconditionally F64 `scalar_value()`. The end-to-end F32
/// dtype-preservation path is now also clean — fj-lax's
/// `eval_unary_elementwise` scalar/tensor arms were fixed in
/// frankenjax-e2l3 (scalar) and frankenjax-eldm (tensor) so non-complex
/// F32/BF16/F16 inputs no longer widen to F64 internally. This test pins
/// the F64 value correctness as a smoke check for the constant rewrite.
#[test]
fn erf_vjp_numerical_f64_value_sanity() {
    let x = Value::Scalar(Literal::from_f64(0.5));
    let g = Value::Scalar(Literal::from_f64(1.0));

    let grads = fj_ad::vjp_single(
        Primitive::Erf,
        std::slice::from_ref(&x),
        &g,
        &BTreeMap::new(),
    )
    .expect("erf VJP should accept F64 scalar");
    assert_eq!(grads.len(), 1);

    let actual = grads[0]
        .as_f64_scalar()
        .expect("Erf VJP should produce a real scalar");
    // Expected: 2/√π * exp(-0.25) ≈ 0.878783
    let expected = 2.0 / std::f64::consts::PI.sqrt() * (-0.25_f64).exp();
    assert!(
        (actual - expected).abs() < 1e-10,
        "Erf VJP value: expected {expected}, got {actual}"
    );
}

/// Complex64 Cbrt VJP must FAIL CLOSED (frankenjax-w8u0a).
///
/// JAX's `cbrt` is `standard_unop(_float)` — complex is rejected — so its forward
/// eval fails closed (commit eb5ad225) and therefore its gradient is undefined for
/// complex too: the cbrt VJP rule evaluates the forward cbrt, which surfaces the
/// "operation is not supported for complex operands" error. (Was: asserted a
/// complex VJP value of g/12, stale since the forward fail-close.)
#[test]
fn cbrt_vjp_numerical_complex64() {
    let z = Value::Scalar(Literal::from_complex64(8.0, 0.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let result = fj_ad::vjp_single(
        Primitive::Cbrt,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    );
    assert!(
        result.is_err(),
        "complex cbrt VJP must fail closed (JAX cbrt is float-only); got {result:?}"
    );
}

/// Complex VJP of float-only lgamma/bessel_i0e must FAIL CLOSED (frankenjax-w8u0a).
///
/// Like cbrt, these are `standard_unop(_float)` in JAX (complex rejected). Their VJP
/// rules evaluate a float-only forward op — lgamma's VJP evaluates Digamma, and
/// bessel_i0e's VJP evaluates BesselI0e — both of which fail closed on complex
/// (commit eb5ad225). So grad(lgamma)/grad(bessel_i0e) on complex must surface the
/// unsupported error rather than return a value, keeping the AD side consistent with
/// the forward fail-close.
#[test]
fn float_only_complex_vjp_fails_closed() {
    let z = Value::Scalar(Literal::from_complex64(2.0, 0.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));
    // Lgamma's VJP evaluates Digamma, BesselI0e's evaluates BesselI0e, and ErfInv's
    // evaluates ErfInv — all float-only forward ops that fail closed on complex.
    for prim in [Primitive::Lgamma, Primitive::BesselI0e, Primitive::ErfInv] {
        let result = fj_ad::vjp_single(prim, std::slice::from_ref(&z), &g, &BTreeMap::new());
        assert!(
            result.is_err(),
            "{prim:?} VJP must fail closed on complex (float-only op); got {result:?}"
        );
    }
}

/// Complex64 scalar Square VJP (frankenjax-t8rl).
///
/// `d/dz z² = 2z` → `g_z = 2 * g * z`. Pick `z = 2 + 3i`, `g = 1 + 0i`:
///   2z = 4 + 6i
///   g_z = (1 + 0i)(4 + 6i) = 4 + 6i
#[test]
fn square_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(2.0, 3.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Square,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("square VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - 4.0).abs() < 1e-5 && (im - 6.0).abs() < 1e-5,
                "g_z should be (4.0, 6.0); got ({re}, {im})"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Rsqrt VJP (frankenjax-zvth).
///
/// `d/dz rsqrt(z) = d/dz z^(-1/2) = -0.5 * z^(-3/2)` → `g_z = g * -0.5 * z^(-1.5)`.
/// Pick the Pythagorean triple `z = 3 + 4i` so the powers have clean
/// closed forms via `sqrt(3+4i) = 2+i`:
///   z^(-1.5) = 1 / z^1.5 = 1 / (z * sqrt(z)) = 1 / ((3+4i)(2+i))
///                       = 1 / (6 + 3i + 8i + 4i²)
///                       = 1 / (6 + 11i + (-4))
///                       = 1 / (2 + 11i)
///                       = (2 - 11i) / (4 + 121) = (2 - 11i) / 125
///   g_z = g * -0.5 * z^(-1.5)
///       = (1+0i) * (-0.5) * (2 - 11i)/125
///       = (-1 + 5.5i) / 125
///       = -0.008 + 0.044i
#[test]
fn rsqrt_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(3.0, 4.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Rsqrt,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("rsqrt VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - (-0.008)).abs() < 1e-5 && (im - 0.044).abs() < 1e-5,
                "g_z should be (-0.008, 0.044); got ({re}, {im})"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Sqrt VJP (frankenjax-dgvr).
///
/// `d/dz sqrt(z) = 1/(2*sqrt(z))` → `g_z = g/(2*sqrt(z))`. Pick the
/// Pythagorean triple `z = 3 + 4i` so sqrt has a clean closed form:
///   sqrt(3+4i) = 2+i (verify: (2+i)² = 4 + 4i + i² = 3+4i ✓)
///   1/(2*(2+i)) = (4-2i)/(16+4) = 0.2 - 0.1i
/// With `g = 1+0i`, `g_z = 0.2 - 0.1i`.
#[test]
fn sqrt_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(3.0, 4.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Sqrt,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("sqrt VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - 0.2).abs() < 1e-5 && (im + 0.1).abs() < 1e-5,
                "g_z should be (0.2, -0.1); got ({re}, {im})"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Log VJP (frankenjax-vhdl).
///
/// `d/dz log(z) = 1/z` → `g_z = g/z` (complex division). Pick `z = 1+i`,
/// `g = 1+0i`. Then `1/z = (1-i)/2 = 0.5 - 0.5i`, and
/// `g_z = (1+0i)(0.5 - 0.5i) = 0.5 - 0.5i`.
#[test]
fn log_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(1.0, 1.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Log,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("log VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - 0.5).abs() < 1e-5 && (im + 0.5).abs() < 1e-5,
                "g_z should be (0.5, -0.5); got ({re}, {im})"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Exp VJP (frankenjax-aabz).
///
/// For complex z = a+bi, `exp(z) = e^a * (cos(b) + i*sin(b))`. VJP:
/// `d/dz exp(z) = exp(z)` → `g_z = g * exp(z)` (complex multiplication).
///
/// Pick `z = 0 + 0.5i`, `g = 1 + 0i`. Then `exp(z) = cos(0.5) + i*sin(0.5)`
/// ≈ 0.8775826 + 0.4794255i, and `g_z = g * exp(z) ≈ 0.8775826 + 0.4794255i`.
#[test]
fn exp_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(0.0, 0.5));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Exp,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("exp VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            let expected_re = 0.5_f64.cos() as f32;
            let expected_im = 0.5_f64.sin() as f32;
            assert!(
                (re - expected_re).abs() < 1e-5,
                "g_z real part: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im).abs() < 1e-5,
                "g_z imag part: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Div VJP (frankenjax-ohim).
///
/// `c = a / b` in complex arithmetic; `dc/da = 1/b`, `dc/db = -a/b²`.
/// With cotangent `g`, the chain rule yields `g_a = g/b` and
/// `g_b = -g*a/b²`. Picking `a = 2+3i`, `b = 1+i`, `g = 1+0i`:
///   b² = (1+i)² = 2i
///   g_a = (1+0i)/(1+i) = (1-i)/2 = 0.5 - 0.5i
///   g_b = -(2+3i)/(2i) = -1.5 + 1i
#[test]
fn div_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let a = Value::Scalar(Literal::from_complex64(2.0, 3.0));
    let b = Value::Scalar(Literal::from_complex64(1.0, 1.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(Primitive::Div, &[a, b], &g, &BTreeMap::new())
        .expect("div VJP should accept complex64 scalars");
    assert_eq!(grads.len(), 2);

    let close = |x: f32, y: f32| (x - y).abs() < 1e-5;

    // g_a = g/b = (1+0i)/(1+i) = 0.5 - 0.5i
    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                close(re, 0.5) && close(im, -0.5),
                "g_a should be (0.5, -0.5); got ({re}, {im})"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }

    // g_b = -g*a/b² = -(2+3i)/(2i) = -1.5 + 1i
    match grads[1] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                close(re, -1.5) && close(im, 1.0),
                "g_b should be (-1.5, 1.0); got ({re}, {im})"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Add VJP (frankenjax-6s96).
///
/// Add is linear: `dc/da = 1`, `dc/db = 1`. With cotangent `g`,
/// `g_a = g_b = g`. Pins both dtype preservation and exact pass-through
/// of the cotangent.
#[test]
fn add_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let a = Value::Scalar(Literal::from_complex64(1.5, -2.5));
    let b = Value::Scalar(Literal::from_complex64(-3.0, 4.0));
    let g = Value::Scalar(Literal::from_complex64(0.25, 0.75));

    let grads = fj_ad::vjp_single(Primitive::Add, &[a, b], &g, &BTreeMap::new())
        .expect("add VJP should accept complex64 scalars");
    assert_eq!(grads.len(), 2);
    for grad in &grads {
        assert!(
            matches!(
                grad,
                Value::Scalar(Complex64Bits(re, im))
                    if f32::from_bits(*re) == 0.25 && f32::from_bits(*im) == 0.75
            ),
            "Add VJP: every g_input must equal g; got {grad:?}"
        );
    }
}

/// Complex64 scalar Sub VJP (frankenjax-6s96).
///
/// Sub: `c = a - b`, so `dc/da = 1`, `dc/db = -1`. With cotangent `g`,
/// `g_a = g`, `g_b = -g`.
#[test]
fn sub_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let a = Value::Scalar(Literal::from_complex64(1.5, -2.5));
    let b = Value::Scalar(Literal::from_complex64(-3.0, 4.0));
    let g = Value::Scalar(Literal::from_complex64(0.25, 0.75));

    let grads = fj_ad::vjp_single(Primitive::Sub, &[a, b], &g, &BTreeMap::new())
        .expect("sub VJP should accept complex64 scalars");
    assert_eq!(grads.len(), 2);

    // g_a = g
    assert!(
        matches!(
            grads[0],
            Value::Scalar(Complex64Bits(re, im))
                if f32::from_bits(re) == 0.25 && f32::from_bits(im) == 0.75
        ),
        "Sub VJP g_a must equal g; got {:?}",
        grads[0]
    );

    // g_b = -g (both real and imaginary negated)
    assert!(
        matches!(
            grads[1],
            Value::Scalar(Complex64Bits(re, im))
                if f32::from_bits(re) == -0.25 && f32::from_bits(im) == -0.75
        ),
        "Sub VJP g_b must equal -g; got {:?}",
        grads[1]
    );
}

/// Complex64 scalar Neg VJP (frankenjax-6s96).
///
/// Neg: `c = -a`, so `dc/da = -1`. With cotangent `g`, `g_a = -g`.
#[test]
fn neg_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let a = Value::Scalar(Literal::from_complex64(1.5, -2.5));
    let g = Value::Scalar(Literal::from_complex64(0.25, 0.75));

    let grads = fj_ad::vjp_single(
        Primitive::Neg,
        std::slice::from_ref(&a),
        &g,
        &BTreeMap::new(),
    )
    .expect("neg VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    assert!(
        matches!(
            grads[0],
            Value::Scalar(Complex64Bits(re, im))
                if f32::from_bits(re) == -0.25 && f32::from_bits(im) == -0.75
        ),
        "Neg VJP g_a must equal -g; got {:?}",
        grads[0]
    );
}

/// Complex64 scalar Sin VJP.
///
/// d/dz[sin(z)] = cos(z). For z = 1 + 0.5i and g = 1 + 0i:
/// cos(1 + 0.5i) = cos(1)cosh(0.5) - i*sin(1)sinh(0.5) ≈ 0.609 - 0.4385i
/// VJP: g_z = g * cos(z) ≈ 0.609 - 0.4385i
#[test]
fn sin_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(1.0, 0.5));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Sin,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("sin VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    // Expected: cos(1+0.5i) = cos(1)*cosh(0.5) - i*sin(1)*sinh(0.5)
    let expected_re = 1.0_f64.cos() * 0.5_f64.cosh();
    let expected_im = -(1.0_f64.sin() * 0.5_f64.sinh());

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - expected_re as f32).abs() < 1e-5,
                "sin VJP real: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im as f32).abs() < 1e-5,
                "sin VJP imag: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Cos VJP.
///
/// d/dz[cos(z)] = -sin(z). For z = 1 + 0.5i and g = 1 + 0i:
/// -sin(1 + 0.5i) = -(sin(1)cosh(0.5) + i*cos(1)sinh(0.5)) ≈ -0.949 - 0.2816i
/// VJP: g_z = -g * sin(z)
#[test]
fn cos_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(1.0, 0.5));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Cos,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("cos VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    // Expected: -sin(1+0.5i) = -(sin(1)*cosh(0.5) + i*cos(1)*sinh(0.5))
    let expected_re = -(1.0_f64.sin() * 0.5_f64.cosh());
    let expected_im = -(1.0_f64.cos() * 0.5_f64.sinh());

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - expected_re as f32).abs() < 1e-5,
                "cos VJP real: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im as f32).abs() < 1e-5,
                "cos VJP imag: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Sinh VJP.
///
/// d/dz[sinh(z)] = cosh(z). For z = 0.5 + 1i and g = 1 + 0i:
/// cosh(0.5+i) = cosh(0.5)cos(1) + i*sinh(0.5)sin(1)
#[test]
fn sinh_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(0.5, 1.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Sinh,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("sinh VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    // Expected: cosh(0.5+i) = cosh(0.5)*cos(1) + i*sinh(0.5)*sin(1)
    let expected_re = 0.5_f64.cosh() * 1.0_f64.cos();
    let expected_im = 0.5_f64.sinh() * 1.0_f64.sin();

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - expected_re as f32).abs() < 1e-5,
                "sinh VJP real: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im as f32).abs() < 1e-5,
                "sinh VJP imag: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Cosh VJP.
///
/// d/dz[cosh(z)] = sinh(z). For z = 0.5 + 1i and g = 1 + 0i:
/// sinh(0.5+i) = sinh(0.5)cos(1) + i*cosh(0.5)sin(1)
#[test]
fn cosh_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(0.5, 1.0));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Cosh,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("cosh VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    // Expected: sinh(0.5+i) = sinh(0.5)*cos(1) + i*cosh(0.5)*sin(1)
    let expected_re = 0.5_f64.sinh() * 1.0_f64.cos();
    let expected_im = 0.5_f64.cosh() * 1.0_f64.sin();

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - expected_re as f32).abs() < 1e-5,
                "cosh VJP real: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im as f32).abs() < 1e-5,
                "cosh VJP imag: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Tan VJP.
///
/// d/dz[tan(z)] = sec²(z) = 1/cos²(z). For z = 0.3 + 0.2i:
/// Verify gradient is Complex64 and numerically reasonable.
#[test]
fn tan_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(0.3, 0.2));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Tan,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("tan VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    // Expected: sec²(z) = 1/cos²(z)
    // cos(0.3+0.2i) = cos(0.3)*cosh(0.2) - i*sin(0.3)*sinh(0.2)
    let cos_re = 0.3_f64.cos() * 0.2_f64.cosh();
    let cos_im = -(0.3_f64.sin() * 0.2_f64.sinh());
    // 1/cos²(z) = 1/(cos_re + i*cos_im)² = conj(cos²)/|cos²|²
    let cos2_re = cos_re * cos_re - cos_im * cos_im;
    let cos2_im = 2.0 * cos_re * cos_im;
    let mag2 = cos2_re * cos2_re + cos2_im * cos2_im;
    let expected_re = cos2_re / mag2;
    let expected_im = -cos2_im / mag2;

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - expected_re as f32).abs() < 1e-4,
                "tan VJP real: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im as f32).abs() < 1e-4,
                "tan VJP imag: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Tanh VJP.
///
/// d/dz[tanh(z)] = sech²(z) = 1 - tanh²(z). For z = 0.5 + 0.3i:
/// Verify gradient is Complex64 and equals 1 - tanh²(z).
#[test]
fn tanh_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(0.5, 0.3));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    // Compute tanh(z) first
    let tanh_result = eval_primitive(Primitive::Tanh, std::slice::from_ref(&z), &BTreeMap::new())
        .expect("tanh should accept complex64");

    let (tanh_re, tanh_im) = match tanh_result {
        Value::Scalar(Complex64Bits(re, im)) => {
            (f32::from_bits(re) as f64, f32::from_bits(im) as f64)
        }
        _ => panic!("expected Complex64 scalar"),
    };

    // Expected: 1 - tanh²(z)
    // tanh²(z) = (tanh_re + i*tanh_im)² = tanh_re² - tanh_im² + 2i*tanh_re*tanh_im
    let tanh2_re = tanh_re * tanh_re - tanh_im * tanh_im;
    let tanh2_im = 2.0 * tanh_re * tanh_im;
    let expected_re = 1.0 - tanh2_re;
    let expected_im = -tanh2_im;

    let grads = fj_ad::vjp_single(
        Primitive::Tanh,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("tanh VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - expected_re as f32).abs() < 1e-4,
                "tanh VJP real: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im as f32).abs() < 1e-4,
                "tanh VJP imag: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Asinh VJP.
///
/// d/dz[asinh(z)] = 1/sqrt(z² + 1). Verify dtype preservation and
/// reasonable numerical result for z = 0.5 + 0.3i.
#[test]
fn asinh_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(0.5, 0.3));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Asinh,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("asinh VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    // Verify output is Complex64 and has finite values
    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(re.is_finite(), "asinh VJP real should be finite");
            assert!(im.is_finite(), "asinh VJP imag should be finite");
            // d/dz[asinh(z)] = 1/sqrt(z² + 1) should have magnitude ~0.8-1.0 for small z
            assert!(
                re.abs() < 2.0 && im.abs() < 2.0,
                "asinh VJP should have reasonable magnitude"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Acosh VJP.
///
/// d/dz[acosh(z)] = 1/sqrt(z² - 1). Verify dtype preservation.
/// Use z = 1.5 + 0.3i to avoid branch cut issues.
#[test]
fn acosh_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(1.5, 0.3));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    let grads = fj_ad::vjp_single(
        Primitive::Acosh,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("acosh VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(re.is_finite(), "acosh VJP real should be finite");
            assert!(im.is_finite(), "acosh VJP imag should be finite");
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

/// Complex64 scalar Atanh VJP.
///
/// d/dz[atanh(z)] = 1/(1 - z²). Verify dtype preservation.
/// Use z = 0.3 + 0.2i (inside unit disk).
#[test]
fn atanh_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let z = Value::Scalar(Literal::from_complex64(0.3, 0.2));
    let g = Value::Scalar(Literal::from_complex64(1.0, 0.0));

    // Expected: 1/(1 - z²)
    let z_re = 0.3_f64;
    let z_im = 0.2_f64;
    let z2_re = z_re * z_re - z_im * z_im;
    let z2_im = 2.0 * z_re * z_im;
    let denom_re = 1.0 - z2_re;
    let denom_im = -z2_im;
    let denom_mag2 = denom_re * denom_re + denom_im * denom_im;
    let expected_re = denom_re / denom_mag2;
    let expected_im = -denom_im / denom_mag2;

    let grads = fj_ad::vjp_single(
        Primitive::Atanh,
        std::slice::from_ref(&z),
        &g,
        &BTreeMap::new(),
    )
    .expect("atanh VJP should accept complex64 scalar");
    assert_eq!(grads.len(), 1);

    match grads[0] {
        Value::Scalar(Complex64Bits(re, im)) => {
            let re = f32::from_bits(re);
            let im = f32::from_bits(im);
            assert!(
                (re - expected_re as f32).abs() < 1e-4,
                "atanh VJP real: expected {expected_re}, got {re}"
            );
            assert!(
                (im - expected_im as f32).abs() < 1e-4,
                "atanh VJP imag: expected {expected_im}, got {im}"
            );
        }
        ref other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

#[test]
fn complex_conj_vjp_vector_conjugates_cotangent() {
    let input = make_complex128_vector(&[(1.0, -2.0), (-3.0, 4.0)]);
    let cotangent = make_complex128_vector(&[(5.0, -6.0), (-7.0, 8.0)]);

    let vjp_result = fj_ad::vjp_single(
        Primitive::Conj,
        std::slice::from_ref(&input),
        &cotangent,
        &BTreeMap::new(),
    )
    .unwrap();

    assert_complex_gradients_close(
        &extract_complex_vec(&vjp_result[0]),
        &[(5.0, 6.0), (-7.0, -8.0)],
        1e-10,
        "Conj vector VJP",
    );
}

#[test]
fn complex_projection_vjp_vector_lifts_real_cotangents() {
    let input = make_complex128_vector(&[(2.0, -3.0), (-5.0, 7.0)]);

    let real_cotangent = make_f64_vector(&[1.5, -2.5]);
    let real_vjp = fj_ad::vjp_single(
        Primitive::Real,
        std::slice::from_ref(&input),
        &real_cotangent,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_complex_gradients_close(
        &extract_complex_vec(&real_vjp[0]),
        &[(1.5, 0.0), (-2.5, 0.0)],
        1e-10,
        "Real vector VJP",
    );

    let imag_cotangent = make_f64_vector(&[3.0, -4.5]);
    let imag_vjp = fj_ad::vjp_single(
        Primitive::Imag,
        std::slice::from_ref(&input),
        &imag_cotangent,
        &BTreeMap::new(),
    )
    .unwrap();
    // JAX `imag_p` transpose is complex(0, neg(g)) — `jax.grad(jnp.imag)` returns
    // -1j, the conjugate cotangent convention. So grad = complex(0, -g).
    assert_complex_gradients_close(
        &extract_complex_vec(&imag_vjp[0]),
        &[(0.0, -3.0), (0.0, 4.5)],
        1e-10,
        "Imag vector VJP",
    );
}

#[test]
fn complex_constructor_vjp_vector_splits_cotangent_components() {
    let real = make_f64_vector(&[1.0, -2.0]);
    let imag = make_f64_vector(&[3.0, -4.0]);
    let cotangent = make_complex128_vector(&[(7.0, -11.0), (-13.0, 17.0)]);

    let vjp_result = fj_ad::vjp_single(
        Primitive::Complex,
        &[real, imag],
        &cotangent,
        &BTreeMap::new(),
    )
    .unwrap();

    assert_gradients_close(
        &extract_f64_vec(&vjp_result[0]),
        &[7.0, -13.0],
        1e-10,
        "Complex constructor vector VJP real component",
    );
    // JAX `complex_p` transpose grad_im = imag(neg(g)) = -imag(g) (conjugate
    // cotangent convention). For g=[(7,-11),(-13,17)]: -imag = [11, -17].
    assert_gradients_close(
        &extract_f64_vec(&vjp_result[1]),
        &[11.0, -17.0],
        1e-10,
        "Complex constructor vector VJP imaginary component",
    );
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

/// RFFT VJP regression for F32 input (frankenjax-35ur).
///
/// RFFT VJP previously hard-coded the IFFT scale literal and the final
/// "take real part" output to F64, so an F32 RFFT input received an F64
/// gradient — analogous to the IRFFT widening fixed by frankenjax-4y5q.
/// This test pins the dtype-preservation invariant for the F32 input
/// path AND verifies the gradient values still match a finite-difference
/// reference within f32 tolerance.
#[test]
fn rfft_vjp_numerical_f32() {
    let x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    let make_f32_input = |data: &[f32]| -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        )
    };

    let x = make_f32_input(&x_data);

    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());

    // Complex64 cotangent to match RFFT(F32 input) → Complex64 output.
    let g = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![3] },
            (0..3).map(|_| Literal::from_complex64(1.0, 0.0)).collect(),
        )
        .unwrap(),
    );

    let vjp_result =
        fj_ad::vjp_single(Primitive::Rfft, std::slice::from_ref(&x), &g, &params).unwrap();
    let vjp_tensor = vjp_result[0].as_tensor().unwrap();

    assert_eq!(
        vjp_tensor.dtype,
        DType::F32,
        "RFFT VJP must keep F32 dtype when input is F32"
    );
    vjp_tensor
        .validate_dtype_consistency()
        .expect("RFFT F32 VJP cotangent dtype/element invariant");

    // Finite-difference check on sum(Re(RFFT(x))).
    let analytical: Vec<f64> = vjp_tensor
        .elements
        .iter()
        .map(|l| match l {
            Literal::F32Bits(bits) => f64::from(f32::from_bits(*bits)),
            _ => panic!("expected F32Bits"),
        })
        .collect();

    let eps_f32 = 1e-3_f32;
    let mut numerical = [0.0_f64; 4];
    for i in 0..4 {
        let mut plus = x_data;
        plus[i] += eps_f32;
        let re_plus: f64 = eval_primitive(
            Primitive::Rfft,
            std::slice::from_ref(&make_f32_input(&plus)),
            &params,
        )
        .unwrap()
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Literal::Complex64Bits(re, _) => f64::from(f32::from_bits(*re)),
            _ => 0.0,
        })
        .sum();

        let mut minus = x_data;
        minus[i] -= eps_f32;
        let re_minus: f64 = eval_primitive(
            Primitive::Rfft,
            std::slice::from_ref(&make_f32_input(&minus)),
            &params,
        )
        .unwrap()
        .as_tensor()
        .unwrap()
        .elements
        .iter()
        .map(|l| match l {
            Literal::Complex64Bits(re, _) => f64::from(f32::from_bits(*re)),
            _ => 0.0,
        })
        .sum();

        numerical[i] = (re_plus - re_minus) / (2.0 * f64::from(eps_f32));
    }

    for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        assert!(
            (a - n).abs() < 1e-2,
            "RFFT F32 VJP[{i}]: analytical={a}, numerical={n}"
        );
    }
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

/// FFT VJP regression for Complex64 input (frankenjax-gvkt).
///
/// FFT VJP previously hard-coded the scale literal to F64, so a
/// Complex64 cotangent + Complex64 input produced a Complex128 gradient
/// — same widening pattern as 4y5q / 35ur. This test pins dtype
/// preservation by asserting the VJP output stays Complex64.
#[test]
fn fft_vjp_complex64_preserves_dtype() {
    let x_re: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x_im: [f32; 4] = [0.5, -0.5, 1.5, -1.5];
    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            x_re.iter()
                .zip(x_im.iter())
                .map(|(&r, &i)| Literal::from_complex64(r, i))
                .collect(),
        )
        .unwrap(),
    );

    let g = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            (0..4).map(|_| Literal::from_complex64(1.0, 0.0)).collect(),
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
    let vjp_tensor = vjp_result[0].as_tensor().unwrap();

    assert_eq!(
        vjp_tensor.dtype,
        DType::Complex64,
        "FFT VJP must keep Complex64 dtype for Complex64 input + Complex64 cotangent"
    );
    vjp_tensor
        .validate_dtype_consistency()
        .expect("FFT Complex64 VJP cotangent dtype/element invariant");
}

/// IFFT VJP regression for Complex64 input (frankenjax-gvkt).
#[test]
fn ifft_vjp_complex64_preserves_dtype() {
    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            vec![
                Literal::from_complex64(10.0, 0.0),
                Literal::from_complex64(-2.0, 2.0),
                Literal::from_complex64(-2.0, 0.0),
                Literal::from_complex64(-2.0, -2.0),
            ],
        )
        .unwrap(),
    );

    let g = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            (0..4).map(|_| Literal::from_complex64(1.0, 0.0)).collect(),
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
    let vjp_tensor = vjp_result[0].as_tensor().unwrap();

    assert_eq!(
        vjp_tensor.dtype,
        DType::Complex64,
        "IFFT VJP must keep Complex64 dtype for Complex64 input + Complex64 cotangent"
    );
    vjp_tensor
        .validate_dtype_consistency()
        .expect("IFFT Complex64 VJP cotangent dtype/element invariant");
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

/// IRFFT VJP regression for Complex64 input (frankenjax-7ckd / frankenjax-dums
/// / frankenjax-4y5q).
///
/// The Complex64 silent-zero bug in `scale_hermitian_adjoint` survived
/// because every previous IRFFT VJP test used Complex128. This case
/// exercises the half-precision-complex VJP path through to a real
/// finite-difference comparison and pins the regression: the gradient
/// must be non-zero AND match the numerical derivative within f32-grade
/// tolerance, AND must keep Complex64 dtype (frankenjax-4y5q made the
/// reciprocal scaling dtype-aware so an F32 cotangent no longer widens
/// the FFT(g) product to Complex128).
#[test]
fn irfft_vjp_numerical_complex64() {
    use fj_core::Literal::Complex64Bits;

    let x_re: [f32; 3] = [10.0, -2.0, 2.0];
    let x_im: [f32; 3] = [0.0, 3.0, 0.0];

    let make_input = |re: &[f32], im: &[f32]| -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![3] },
                re.iter()
                    .zip(im.iter())
                    .map(|(&r, &i)| Literal::from_complex64(r, i))
                    .collect(),
            )
            .unwrap(),
        )
    };

    let x = make_input(&x_re, &x_im);

    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "4".to_owned());

    // Real cotangent matching the IRFFT output dtype (F32).
    let g = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape { dims: vec![4] },
            vec![
                Literal::from_f32(1.0),
                Literal::from_f32(1.0),
                Literal::from_f32(1.0),
                Literal::from_f32(1.0),
            ],
        )
        .unwrap(),
    );

    let vjp_result =
        fj_ad::vjp_single(Primitive::Irfft, std::slice::from_ref(&x), &g, &params).unwrap();
    let vjp_tensor = vjp_result[0].as_tensor().unwrap();

    // dtype-preservation regression (frankenjax-4y5q): the IRFFT VJP
    // must keep Complex64 when given Complex64 input + F32 cotangent.
    assert_eq!(
        vjp_tensor.dtype,
        DType::Complex64,
        "IRFFT VJP must keep Complex64 dtype for Complex64 input"
    );

    // Silent-zero regression (frankenjax-7ckd): gradient must NOT be
    // all-zero. The previous bug in scale_hermitian_adjoint produced a
    // degenerate zero tensor here.
    let any_nonzero = vjp_tensor.elements.iter().any(|e| match e {
        Complex64Bits(re, im) => f32::from_bits(*re) != 0.0 || f32::from_bits(*im) != 0.0,
        _ => false,
    });
    assert!(
        any_nonzero,
        "IRFFT Complex64 VJP must not produce all-zero gradient: {:?}",
        vjp_tensor.elements
    );

    let extract_f32_sum = |val: &Value| -> f64 {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(bits) => f64::from(f32::from_bits(*bits)),
                _ => panic!("IRFFT Complex64 output must store F32Bits elements"),
            })
            .sum()
    };

    // Finite-difference verification, looser tolerance for f32 inputs.
    let eps_f32 = 1e-3_f32;

    for bin in 0..3 {
        let mut re_plus = x_re;
        re_plus[bin] += eps_f32;
        let out_plus = eval_primitive(
            Primitive::Irfft,
            std::slice::from_ref(&make_input(&re_plus, &x_im)),
            &params,
        )
        .unwrap();
        let l_plus = extract_f32_sum(&out_plus);

        let mut re_minus = x_re;
        re_minus[bin] -= eps_f32;
        let out_minus = eval_primitive(
            Primitive::Irfft,
            std::slice::from_ref(&make_input(&re_minus, &x_im)),
            &params,
        )
        .unwrap();
        let l_minus = extract_f32_sum(&out_minus);
        let numerical_re = (l_plus - l_minus) / (2.0 * f64::from(eps_f32));

        let vjp_bin = match &vjp_tensor.elements[bin] {
            Complex64Bits(re, _im) => f64::from(f32::from_bits(*re)),
            other => panic!("expected Complex64Bits in VJP, got {other:?}"),
        };

        assert!(
            (vjp_bin - numerical_re).abs() < 1e-2,
            "IRFFT Complex64 VJP real part at bin {bin}: analytical={vjp_bin}, \
             numerical={numerical_re}"
        );
    }
}

/// IRFFT VJP for Complex64 input with odd fft_length — exercises the
/// "no Nyquist bin" branch of `scale_hermitian_adjoint`'s
/// `fft_length.is_multiple_of(2)` guard.
#[test]
fn irfft_vjp_numerical_complex64_odd_fft_length() {
    use fj_core::Literal::Complex64Bits;

    // input length = (fft_length + 1) / 2 = 2 for fft_length=3
    let x_re: [f32; 2] = [4.0, -1.0];
    let x_im: [f32; 2] = [0.0, 2.0];

    let make_input = |re: &[f32], im: &[f32]| -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::Complex64,
                Shape { dims: vec![2] },
                re.iter()
                    .zip(im.iter())
                    .map(|(&r, &i)| Literal::from_complex64(r, i))
                    .collect(),
            )
            .unwrap(),
        )
    };

    let x = make_input(&x_re, &x_im);

    let mut params = BTreeMap::new();
    params.insert("fft_length".to_owned(), "3".to_owned());

    let g = Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape { dims: vec![3] },
            vec![
                Literal::from_f32(1.0),
                Literal::from_f32(1.0),
                Literal::from_f32(1.0),
            ],
        )
        .unwrap(),
    );

    let vjp_result =
        fj_ad::vjp_single(Primitive::Irfft, std::slice::from_ref(&x), &g, &params).unwrap();
    let vjp_tensor = vjp_result[0].as_tensor().unwrap();

    assert_eq!(
        vjp_tensor.dtype,
        DType::Complex64,
        "IRFFT VJP must keep Complex64 dtype for Complex64 input (odd fft_length)"
    );

    // Regression: gradient must NOT be identically zero. This catches
    // any re-introduction of the silent-zero bug from frankenjax-7ckd.
    let any_nonzero = vjp_tensor.elements.iter().any(|e| match e {
        Complex64Bits(re, im) => f32::from_bits(*re) != 0.0 || f32::from_bits(*im) != 0.0,
        _ => false,
    });
    assert!(
        any_nonzero,
        "IRFFT Complex64 VJP (odd fft_length) must not produce all-zero gradient: {:?}",
        vjp_tensor.elements
    );
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

    let vjp_result = fj_ad::vjp_single(
        Primitive::Exp,
        std::slice::from_ref(&x),
        &g,
        &BTreeMap::new(),
    )
    .unwrap();
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

    let vjp_result = fj_ad::vjp_single(
        Primitive::Log,
        std::slice::from_ref(&x),
        &g,
        &BTreeMap::new(),
    )
    .unwrap();
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

// ======================== Elementary Scalar VJP Numerical Tests (frankenjax-suz) ========================

/// Finite-difference VJP verification for unary scalar primitives.
/// For f: R → R, the VJP rule with cotangent g should satisfy:
///   vjp_f(x, g) ≈ g * (f(x+ε) - f(x-ε)) / (2ε)
fn verify_unary_scalar_vjp(prim: Primitive, x: f64, g: f64, tol: f64, label: &str) {
    let x_val = Value::scalar_f64(x);
    let g_val = Value::scalar_f64(g);
    let params = BTreeMap::new();

    // Forward pass
    let out = eval_primitive(prim, std::slice::from_ref(&x_val), &params).unwrap();

    // Analytical VJP
    let vjp_result = fj_ad::vjp(
        prim,
        std::slice::from_ref(&x_val),
        std::slice::from_ref(&g_val),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let analytical = extract_f64_scalar(&vjp_result[0]);

    // Numerical finite-difference
    let eps = 1e-6;
    let f_plus = extract_f64_scalar(
        &eval_primitive(
            prim,
            std::slice::from_ref(&Value::scalar_f64(x + eps)),
            &params,
        )
        .unwrap(),
    );
    let f_minus = extract_f64_scalar(
        &eval_primitive(
            prim,
            std::slice::from_ref(&Value::scalar_f64(x - eps)),
            &params,
        )
        .unwrap(),
    );
    let numerical = g * (f_plus - f_minus) / (2.0 * eps);

    assert_scalar_close(analytical, numerical, tol, 1e-4, label);
}

#[test]
fn sin_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Sin, 1.0, 1.0, 1e-5, "sin VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Sin, 0.0, 1.0, 1e-5, "sin VJP at x=0");
    verify_unary_scalar_vjp(Primitive::Sin, 2.5, 0.7, 1e-5, "sin VJP at x=2.5, g=0.7");
}

#[test]
fn cos_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Cos, 1.0, 1.0, 1e-5, "cos VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Cos, 0.0, 1.0, 1e-5, "cos VJP at x=0");
    verify_unary_scalar_vjp(Primitive::Cos, -0.5, 2.0, 1e-5, "cos VJP at x=-0.5, g=2");
}

#[test]
fn exp_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Exp, 1.0, 1.0, 1e-5, "exp VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Exp, 0.0, 1.0, 1e-5, "exp VJP at x=0");
    verify_unary_scalar_vjp(Primitive::Exp, -2.0, 0.5, 1e-5, "exp VJP at x=-2, g=0.5");
}

#[test]
fn log_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Log, 1.0, 1.0, 1e-5, "log VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Log, 2.0, 1.0, 1e-5, "log VJP at x=2");
    verify_unary_scalar_vjp(Primitive::Log, 0.5, 3.0, 1e-5, "log VJP at x=0.5, g=3");
}

#[test]
fn tanh_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Tanh, 0.5, 1.0, 1e-5, "tanh VJP at x=0.5");
    verify_unary_scalar_vjp(Primitive::Tanh, 0.0, 1.0, 1e-5, "tanh VJP at x=0");
    verify_unary_scalar_vjp(Primitive::Tanh, -1.0, 0.5, 1e-5, "tanh VJP at x=-1, g=0.5");
}

#[test]
fn sqrt_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Sqrt, 4.0, 1.0, 1e-5, "sqrt VJP at x=4");
    verify_unary_scalar_vjp(Primitive::Sqrt, 1.0, 1.0, 1e-5, "sqrt VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Sqrt, 0.25, 2.0, 1e-5, "sqrt VJP at x=0.25, g=2");
}

#[test]
fn neg_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Neg, 3.0, 1.0, 1e-5, "neg VJP at x=3");
    verify_unary_scalar_vjp(Primitive::Neg, -2.0, 0.5, 1e-5, "neg VJP at x=-2, g=0.5");
}

#[test]
fn abs_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Abs, 3.0, 1.0, 1e-5, "abs VJP at x=3");
    verify_unary_scalar_vjp(Primitive::Abs, -3.0, 1.0, 1e-5, "abs VJP at x=-3");
}

#[test]
fn expm1_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Expm1, 0.5, 1.0, 1e-5, "expm1 VJP at x=0.5");
    verify_unary_scalar_vjp(Primitive::Expm1, 0.0, 1.0, 1e-5, "expm1 VJP at x=0");
}

#[test]
fn log1p_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Log1p, 0.5, 1.0, 1e-5, "log1p VJP at x=0.5");
    verify_unary_scalar_vjp(Primitive::Log1p, 1.0, 1.0, 1e-5, "log1p VJP at x=1");
}

#[test]
fn sinh_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Sinh, 1.0, 1.0, 1e-5, "sinh VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Sinh, -0.5, 2.0, 1e-5, "sinh VJP at x=-0.5, g=2");
}

#[test]
fn cosh_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Cosh, 1.0, 1.0, 1e-5, "cosh VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Cosh, 0.0, 1.0, 1e-5, "cosh VJP at x=0");
}

#[test]
fn asinh_vjp_numerical() {
    // asinh'(x) = 1/sqrt(x^2 + 1), defined for all real x
    verify_unary_scalar_vjp(Primitive::Asinh, 0.0, 1.0, 1e-5, "asinh VJP at x=0");
    verify_unary_scalar_vjp(Primitive::Asinh, 1.0, 1.0, 1e-5, "asinh VJP at x=1");
    verify_unary_scalar_vjp(
        Primitive::Asinh,
        -0.5,
        2.0,
        1e-5,
        "asinh VJP at x=-0.5, g=2",
    );
    verify_unary_scalar_vjp(Primitive::Asinh, 2.0, 0.5, 1e-5, "asinh VJP at x=2, g=0.5");
}

#[test]
fn asinh_vjp_overflows_to_zero_for_large_x_matching_jax() {
    // PARITY GUARD. JAX's asinh JVP is `rsqrt(square(x) + 1)` (jax/_src/lax/lax.py): it
    // forms square(x), which overflows to +inf for large |x|, so the gradient is 0.
    // frankenjax computes 1/sqrt(x²+1) the same way, so asinh'(1e200) is 0 — matching
    // JAX — even though the mathematically-exact derivative is ~1e-200. Do NOT "fix" this
    // to a robust form: it would DIVERGE from JAX. (The asinh FORWARD, in contrast, must
    // stay finite ~log(2x), and does — see asinh_oracle.)
    let x = Value::scalar_f64(1e200);
    let g = Value::scalar_f64(1.0);
    let params = BTreeMap::new();
    let out = eval_primitive(Primitive::Asinh, std::slice::from_ref(&x), &params).unwrap();
    assert!(
        extract_f64_scalar(&out).is_finite(),
        "asinh(1e200) forward must be finite"
    );
    let vjp = fj_ad::vjp(
        Primitive::Asinh,
        std::slice::from_ref(&x),
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    assert_eq!(
        extract_f64_scalar(&vjp[0]),
        0.0,
        "asinh'(1e200)=1/sqrt(x²+1) overflows to 0, matching JAX's square(x) JVP"
    );
}

#[test]
fn acosh_vjp_numerical() {
    // acosh'(x) = 1/sqrt(x^2 - 1), x > 1
    verify_unary_scalar_vjp(Primitive::Acosh, 1.5, 1.0, 1e-5, "acosh VJP at x=1.5");
    verify_unary_scalar_vjp(Primitive::Acosh, 2.0, 1.0, 1e-5, "acosh VJP at x=2");
    verify_unary_scalar_vjp(Primitive::Acosh, 3.0, 0.5, 1e-5, "acosh VJP at x=3, g=0.5");
}

#[test]
fn atanh_vjp_numerical() {
    // atanh'(x) = 1/(1 - x^2), |x| < 1
    verify_unary_scalar_vjp(Primitive::Atanh, 0.0, 1.0, 1e-5, "atanh VJP at x=0");
    verify_unary_scalar_vjp(Primitive::Atanh, 0.5, 1.0, 1e-5, "atanh VJP at x=0.5");
    verify_unary_scalar_vjp(
        Primitive::Atanh,
        -0.3,
        2.0,
        1e-5,
        "atanh VJP at x=-0.3, g=2",
    );
    verify_unary_scalar_vjp(
        Primitive::Atanh,
        0.8,
        0.5,
        1e-5,
        "atanh VJP at x=0.8, g=0.5",
    );
}

#[test]
fn tan_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Tan, 0.5, 1.0, 1e-5, "tan VJP at x=0.5");
    verify_unary_scalar_vjp(Primitive::Tan, -0.3, 1.0, 1e-5, "tan VJP at x=-0.3");
}

#[test]
fn asin_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Asin, 0.5, 1.0, 1e-5, "asin VJP at x=0.5");
    verify_unary_scalar_vjp(Primitive::Asin, -0.3, 1.0, 1e-5, "asin VJP at x=-0.3");
}

#[test]
fn acos_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Acos, 0.5, 1.0, 1e-5, "acos VJP at x=0.5");
    verify_unary_scalar_vjp(Primitive::Acos, -0.3, 1.0, 1e-5, "acos VJP at x=-0.3");
}

#[test]
fn atan_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Atan, 1.0, 1.0, 1e-5, "atan VJP at x=1");
    verify_unary_scalar_vjp(Primitive::Atan, -2.0, 0.5, 1e-5, "atan VJP at x=-2, g=0.5");
}

#[test]
fn square_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Square, 3.0, 1.0, 1e-5, "square VJP at x=3");
    verify_unary_scalar_vjp(
        Primitive::Square,
        -2.0,
        0.5,
        1e-5,
        "square VJP at x=-2, g=0.5",
    );
}

#[test]
fn reciprocal_vjp_numerical() {
    verify_unary_scalar_vjp(
        Primitive::Reciprocal,
        2.0,
        1.0,
        1e-5,
        "reciprocal VJP at x=2",
    );
    verify_unary_scalar_vjp(
        Primitive::Reciprocal,
        0.5,
        1.0,
        1e-5,
        "reciprocal VJP at x=0.5",
    );
}

#[test]
fn rsqrt_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Rsqrt, 4.0, 1.0, 1e-5, "rsqrt VJP at x=4");
    verify_unary_scalar_vjp(Primitive::Rsqrt, 1.0, 1.0, 1e-5, "rsqrt VJP at x=1");
}

#[test]
fn cbrt_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Cbrt, 8.0, 1.0, 1e-5, "cbrt VJP at x=8");
    verify_unary_scalar_vjp(Primitive::Cbrt, 1.0, 1.0, 1e-5, "cbrt VJP at x=1");
}

#[test]
fn logistic_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Logistic, 0.0, 1.0, 1e-5, "logistic VJP at x=0");
    verify_unary_scalar_vjp(Primitive::Logistic, 1.0, 1.0, 1e-5, "logistic VJP at x=1");
    verify_unary_scalar_vjp(
        Primitive::Logistic,
        -1.0,
        0.5,
        1e-5,
        "logistic VJP at x=-1, g=0.5",
    );
}

#[test]
fn erf_vjp_numerical() {
    verify_unary_scalar_vjp(Primitive::Erf, 0.5, 1.0, 1e-4, "erf VJP at x=0.5");
    // erf derivative at x=0 is 2/sqrt(pi); finite difference has wider error here
    verify_unary_scalar_vjp(Primitive::Erf, 0.0, 1.0, 2e-3, "erf VJP at x=0");
}

#[test]
fn erfc_vjp_numerical() {
    // d/dx erfc(x) = -2/sqrt(pi) * exp(-x^2). Untested grad rule (erf VJP was the only
    // special-function grad with a numerical check).
    verify_unary_scalar_vjp(Primitive::Erfc, 0.5, 1.0, 1e-4, "erfc VJP at x=0.5");
    verify_unary_scalar_vjp(Primitive::Erfc, -0.7, 0.5, 1e-4, "erfc VJP at x=-0.7");
}

#[test]
fn erf_inv_vjp_numerical() {
    // d/dx erfinv(x) = sqrt(pi)/2 * exp(erfinv(x)^2). Subtle (uses erfinv itself).
    verify_unary_scalar_vjp(Primitive::ErfInv, 0.4, 1.0, 1e-4, "erfinv VJP at x=0.4");
    verify_unary_scalar_vjp(Primitive::ErfInv, -0.6, 1.0, 1e-4, "erfinv VJP at x=-0.6");
}

#[test]
fn lgamma_digamma_vjp_numerical() {
    // d/dx lgamma(x) = digamma(x); d/dx digamma(x) = polygamma(1, x) = trigamma(x).
    verify_unary_scalar_vjp(Primitive::Lgamma, 2.5, 1.0, 1e-4, "lgamma VJP at x=2.5");
    verify_unary_scalar_vjp(Primitive::Lgamma, 0.7, 1.0, 1e-4, "lgamma VJP at x=0.7");
    verify_unary_scalar_vjp(Primitive::Digamma, 2.5, 1.0, 1e-4, "digamma VJP at x=2.5");
}

#[test]
fn bessel_i0e_i1e_vjp_numerical() {
    // i0e/i1e = e^{-|x|}·I0/I1; the grad rule must get the sign right (these are
    // even/odd functions). Test both signs away from the x=0 kink.
    verify_unary_scalar_vjp(
        Primitive::BesselI0e,
        1.5,
        1.0,
        1e-4,
        "bessel_i0e VJP at x=1.5",
    );
    verify_unary_scalar_vjp(
        Primitive::BesselI0e,
        -0.8,
        1.0,
        1e-4,
        "bessel_i0e VJP at x=-0.8",
    );
    verify_unary_scalar_vjp(
        Primitive::BesselI1e,
        1.5,
        1.0,
        1e-4,
        "bessel_i1e VJP at x=1.5",
    );
    verify_unary_scalar_vjp(
        Primitive::BesselI1e,
        -0.8,
        1.0,
        1e-4,
        "bessel_i1e VJP at x=-0.8",
    );
}

#[test]
fn bessel_i1e_vjp_at_zero_is_one_half_matching_jax_guard() {
    // i1e'(x) = i0e(x) - i1e(x)*(sign(x) + 1/x) is 0*inf = NaN at x=0; the limit is 1/2
    // (since I1e(x)/x -> 1/2). JAX's _bessel_i1e_jvp guards this with
    // select(|x| > eps, formula, 0.5), and frankenjax's bessel_i1e_deriv mirrors it
    // exactly. The existing bessel VJP test deliberately stays away from this x=0 kink,
    // so pin the guarded limit directly (FD-free, to avoid kink noise).
    let x = Value::scalar_f64(0.0);
    let g = Value::scalar_f64(1.0);
    let params = BTreeMap::new();
    let out = eval_primitive(Primitive::BesselI1e, std::slice::from_ref(&x), &params).unwrap();
    let vjp = fj_ad::vjp(
        Primitive::BesselI1e,
        std::slice::from_ref(&x),
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    let grad = extract_f64_scalar(&vjp[0]);
    assert!(
        (grad - 0.5).abs() < 1e-12,
        "bessel_i1e'(0) must be 0.5 (JAX select-guard limit), got {grad}"
    );
}

#[test]
fn integer_pow_vjp_zero_exponent_and_general() {
    // IntegerPow's gradient was untested (only its forward oracle existed). x^0 is a
    // constant, so its grad is 0 EVERYWHERE — including x=0, where the generic n*x^(n-1)
    // rule would be 0*x^(-1) = 0*inf = NaN. frankenjax guards n==0 -> 0, matching JAX's
    // _integer_pow_jvp. General: grad = g*n*x^(n-1); negative n supported.
    let vjp_ipow = |x: f64, exp: i64, g: f64| -> f64 {
        let xv = Value::scalar_f64(x);
        let gv = Value::scalar_f64(g);
        let mut params = BTreeMap::new();
        params.insert("exponent".to_string(), exp.to_string());
        let out =
            eval_primitive(Primitive::IntegerPow, std::slice::from_ref(&xv), &params).unwrap();
        let vjp = fj_ad::vjp(
            Primitive::IntegerPow,
            std::slice::from_ref(&xv),
            std::slice::from_ref(&gv),
            std::slice::from_ref(&out),
            &params,
        )
        .unwrap();
        extract_f64_scalar(&vjp[0])
    };
    // n == 0: grad is 0 even at x = 0 (the guard avoids NaN).
    assert_eq!(vjp_ipow(0.0, 0, 1.0), 0.0, "d/dx x^0 = 0 at x=0 (n==0 guard)");
    assert_eq!(vjp_ipow(5.0, 0, 1.0), 0.0, "d/dx x^0 = 0");
    // n == 3: grad = 3*x^2; at x=2 -> 12.
    assert!(
        (vjp_ipow(2.0, 3, 1.0) - 12.0).abs() < 1e-9,
        "d/dx x^3 = 3x^2 = 12 at x=2"
    );
    // n == -1: grad = -x^-2; at x=2 -> -0.25.
    assert!(
        (vjp_ipow(2.0, -1, 1.0) + 0.25).abs() < 1e-9,
        "d/dx x^-1 = -1/x^2 = -0.25 at x=2"
    );
    // cotangent scaling: g=0.5, n=2, x=3 -> 0.5*2*3 = 3.
    assert!(
        (vjp_ipow(3.0, 2, 0.5) - 3.0).abs() < 1e-9,
        "g * d/dx x^2 = 0.5*2*3 = 3 at x=3"
    );
}

#[test]
fn binary_elementwise_vjps_hypot_logaddexp() {
    // VJPs for Hypot / LogAddExp / LogAddExp2 were untested. Formulas (matching JAX):
    //   hypot:      d/dx = x/h,          d/dy = y/h
    //   logaddexp:  d/dx = sigmoid(x-y), d/dy = sigmoid(y-x)   (grads sum to 1)
    //   logaddexp2: d/dx = 2^(x-lae2),   d/dy = 2^(y-lae2)
    let vjp2 = |prim: Primitive, x: f64, y: f64| -> (f64, f64) {
        let inputs = [Value::scalar_f64(x), Value::scalar_f64(y)];
        let g = Value::scalar_f64(1.0);
        let params = BTreeMap::new();
        let out = eval_primitive(prim, &inputs, &params).unwrap();
        let vjp = fj_ad::vjp(
            prim,
            &inputs,
            std::slice::from_ref(&g),
            std::slice::from_ref(&out),
            &params,
        )
        .unwrap();
        (extract_f64_scalar(&vjp[0]), extract_f64_scalar(&vjp[1]))
    };
    // hypot(3,4) = 5 -> grads (3/5, 4/5)
    let (hx, hy) = vjp2(Primitive::Hypot, 3.0, 4.0);
    assert!(
        (hx - 0.6).abs() < 1e-9 && (hy - 0.8).abs() < 1e-9,
        "hypot grad = (3/5, 4/5), got ({hx}, {hy})"
    );
    // logaddexp(1,0) -> grads (sigmoid(1), sigmoid(-1)); they sum to 1
    let (lx, ly) = vjp2(Primitive::LogAddExp, 1.0, 0.0);
    let s = 1.0 / (1.0 + (-1.0_f64).exp());
    assert!(
        (lx - s).abs() < 1e-9 && (ly - (1.0 - s)).abs() < 1e-9,
        "logaddexp grad = (sigmoid(1), sigmoid(-1)), got ({lx}, {ly})"
    );
    assert!((lx + ly - 1.0).abs() < 1e-9, "logaddexp grads must sum to 1");
    // logaddexp2(1,0): 2^x/(2^x+2^y) = 2/3, 1/3
    let (l2x, l2y) = vjp2(Primitive::LogAddExp2, 1.0, 0.0);
    assert!(
        (l2x - 2.0 / 3.0).abs() < 1e-9 && (l2y - 1.0 / 3.0).abs() < 1e-9,
        "logaddexp2 grad = (2/3, 1/3), got ({l2x}, {l2y})"
    );
}

#[test]
fn clamp_vjp_routes_gradient_to_active_branch() {
    // clamp(min, operand, max) gradient (verified to match JAX's defjvp): the cotangent
    // flows entirely to whichever input is the output — operand when min<x<max, min when
    // x<min, max when x>max. The 3-input clamp gradient had no AD coverage.
    let vjp_clamp = |mn: f64, x: f64, mx: f64| -> (f64, f64, f64) {
        let inputs = [
            Value::scalar_f64(mn),
            Value::scalar_f64(x),
            Value::scalar_f64(mx),
        ];
        let g = Value::scalar_f64(1.0);
        let params = BTreeMap::new();
        let out = eval_primitive(Primitive::Clamp, &inputs, &params).unwrap();
        let vjp = fj_ad::vjp(
            Primitive::Clamp,
            &inputs,
            std::slice::from_ref(&g),
            std::slice::from_ref(&out),
            &params,
        )
        .unwrap();
        (
            extract_f64_scalar(&vjp[0]),
            extract_f64_scalar(&vjp[1]),
            extract_f64_scalar(&vjp[2]),
        )
    };
    // min < x < max -> gradient to operand
    assert_eq!(
        vjp_clamp(0.0, 0.5, 1.0),
        (0.0, 1.0, 0.0),
        "in-range: grad flows to operand"
    );
    // x < min -> gradient to min (clamp output is min)
    assert_eq!(
        vjp_clamp(0.0, -0.5, 1.0),
        (1.0, 0.0, 0.0),
        "below min: grad flows to min"
    );
    // x > max -> gradient to max (clamp output is max)
    assert_eq!(
        vjp_clamp(0.0, 1.5, 1.0),
        (0.0, 0.0, 1.0),
        "above max: grad flows to max"
    );
}

#[test]
fn select_vjp_routes_cotangent_per_predicate() {
    // select(cond, on_true, on_false): the cotangent flows to on_true where cond is true
    // and to on_false where it is false; the predicate gets zero gradient. Matches JAX's
    // select transpose. The Select VJP had no AD coverage.
    let params = BTreeMap::new();
    // Build a bool predicate [true, false, true] via a comparison.
    let cond = eval_primitive(
        Primitive::Gt,
        &[
            make_f64_vector(&[3.0, 1.0, 3.0]),
            make_f64_vector(&[2.0, 2.0, 2.0]),
        ],
        &params,
    )
    .unwrap();
    let inputs = [
        cond,
        make_f64_vector(&[1.0, 2.0, 3.0]),
        make_f64_vector(&[10.0, 20.0, 30.0]),
    ];
    let out = eval_primitive(Primitive::Select, &inputs, &params).unwrap();
    let g = make_f64_vector(&[1.0, 1.0, 1.0]);
    let vjp = fj_ad::vjp(
        Primitive::Select,
        &inputs,
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    assert_eq!(
        extract_f64_vec(&vjp[1]),
        vec![1.0, 0.0, 1.0],
        "on_true grad = g where cond is true"
    );
    assert_eq!(
        extract_f64_vec(&vjp[2]),
        vec![0.0, 1.0, 0.0],
        "on_false grad = g where cond is false"
    );
}

#[test]
fn broadcast_in_dim_vjp_sums_over_broadcast_axes() {
    // broadcast_in_dim VJP reduce-sums the cotangent over the broadcast axes, then
    // reshapes to the input shape. input [3] -> output [2,3] with broadcast_dimensions=[1]
    // (input axis 0 -> output axis 1, output axis 0 is broadcast), so grad[j] = sum_i g[i,j].
    let input = make_f64_vector(&[1.0, 2.0, 3.0]);
    let mut params = BTreeMap::new();
    params.insert("shape".to_string(), "2,3".to_string());
    params.insert("broadcast_dimensions".to_string(), "1".to_string());
    let out = eval_primitive(Primitive::BroadcastInDim, std::slice::from_ref(&input), &params).unwrap();
    // g = [[1,2,3],[4,5,6]]
    let g = make_f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let vjp = fj_ad::vjp(
        Primitive::BroadcastInDim,
        std::slice::from_ref(&input),
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    assert_eq!(
        extract_f64_vec(&vjp[0]),
        vec![5.0, 7.0, 9.0],
        "grad sums the cotangent over the broadcast axis: [1+4, 2+5, 3+6]"
    );
}

#[test]
fn rev_vjp_reverses_the_cotangent() {
    // rev is its own inverse, so its VJP reverses the cotangent along the same axes.
    let input = make_f64_vector(&[1.0, 2.0, 3.0]);
    let mut params = BTreeMap::new();
    params.insert("axes".to_string(), "0".to_string());
    let out = eval_primitive(Primitive::Rev, std::slice::from_ref(&input), &params).unwrap();
    let g = make_f64_vector(&[10.0, 20.0, 30.0]);
    let vjp = fj_ad::vjp(
        Primitive::Rev,
        std::slice::from_ref(&input),
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    assert_eq!(
        extract_f64_vec(&vjp[0]),
        vec![30.0, 20.0, 10.0],
        "rev VJP reverses the cotangent"
    );
}

#[test]
fn expand_dims_vjp_squeezes_the_inserted_axis() {
    // expand_dims inserts a size-1 axis; its VJP removes that axis (reshape back to the
    // input shape). input [3] -> output [1,3]; grad of g=[[10,20,30]] is [10,20,30].
    let input = make_f64_vector(&[1.0, 2.0, 3.0]);
    let mut params = BTreeMap::new();
    params.insert("axis".to_string(), "0".to_string());
    let out = eval_primitive(Primitive::ExpandDims, std::slice::from_ref(&input), &params).unwrap();
    let g = make_f64_matrix(1, 3, &[10.0, 20.0, 30.0]);
    let vjp = fj_ad::vjp(
        Primitive::ExpandDims,
        std::slice::from_ref(&input),
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    assert_eq!(
        extract_f64_vec(&vjp[0]),
        vec![10.0, 20.0, 30.0],
        "expand_dims VJP squeezes the inserted axis back to the input shape"
    );
}

#[test]
fn squeeze_vjp_restores_the_removed_axis() {
    // squeeze removes a size-1 axis; its VJP reshapes the cotangent back to the input
    // shape (values unchanged). input [1,3] -> output [3]; grad of g=[10,20,30] is the
    // same values reshaped to [1,3].
    let input = make_f64_matrix(1, 3, &[1.0, 2.0, 3.0]);
    let mut params = BTreeMap::new();
    params.insert("dimensions".to_string(), "0".to_string());
    let out = eval_primitive(Primitive::Squeeze, std::slice::from_ref(&input), &params).unwrap();
    let g = make_f64_vector(&[10.0, 20.0, 30.0]);
    let vjp = fj_ad::vjp(
        Primitive::Squeeze,
        std::slice::from_ref(&input),
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    assert_eq!(
        extract_f64_vec(&vjp[0]),
        vec![10.0, 20.0, 30.0],
        "squeeze VJP restores the removed axis with values unchanged"
    );
}

#[test]
fn igamma_igammac_vjp_numerical() {
    // igamma(a, x): grad w.r.t. a (igamma_grad_a's dedicated series — the most error-prone)
    // AND x (x^(a-1)·e^{-x}/Gamma(a)). igammac = 1 - igamma, so its grads are negated.
    verify_binary_scalar_vjp(
        Primitive::Igamma,
        2.0,
        1.5,
        1.0,
        1e-4,
        "igamma VJP at (2,1.5)",
    );
    verify_binary_scalar_vjp(
        Primitive::Igamma,
        3.5,
        2.0,
        1.0,
        1e-4,
        "igamma VJP at (3.5,2)",
    );
    verify_binary_scalar_vjp(
        Primitive::Igammac,
        2.0,
        1.5,
        1.0,
        1e-4,
        "igammac VJP at (2,1.5)",
    );
}

#[test]
fn betainc_vjp_a_b_grads_nan_not_silent_zero() {
    // JAX raises "Betainc gradient with respect to a and b not supported"; fj-lax's
    // monolithic VJP emits NaN for da/db — VISIBLE, not a silently-wrong 0 — while keeping
    // the JAX-supported x-grad exact: d/dx I_x(a,b) = x^{a-1}(1-x)^{b-1}/B(a,b).
    let a = Value::scalar_f64(2.0);
    let b = Value::scalar_f64(3.0);
    let x = Value::scalar_f64(0.4);
    let g = Value::scalar_f64(1.0);
    let params = BTreeMap::new();
    let out = eval_primitive(
        Primitive::Betainc,
        &[a.clone(), b.clone(), x.clone()],
        &params,
    )
    .unwrap();
    let vjp = fj_ad::vjp(
        Primitive::Betainc,
        &[a, b, x],
        std::slice::from_ref(&g),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();
    assert!(
        extract_f64_scalar(&vjp[0]).is_nan(),
        "betainc d/da must be NaN (JAX raises 'not supported'), got {}",
        extract_f64_scalar(&vjp[0])
    );
    assert!(
        extract_f64_scalar(&vjp[1]).is_nan(),
        "betainc d/db must be NaN (JAX raises 'not supported')"
    );
    // B(2,3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12; dx = 0.4·0.6²/(1/12) = 0.144·12 = 1.728.
    let dx = extract_f64_scalar(&vjp[2]);
    assert!(
        (dx - 1.728).abs() < 1e-6,
        "betainc d/dx = {dx}, expected 1.728"
    );
}

// ======================== Binary Scalar VJP Numerical Tests (frankenjax-2zy) ========================

/// Finite-difference VJP verification for binary scalar primitives.
/// For f(a,b), the VJP with cotangent g gives gradients w.r.t. both inputs:
///   grad_a ≈ g * (f(a+ε,b) - f(a-ε,b)) / (2ε)
///   grad_b ≈ g * (f(a,b+ε) - f(a,b-ε)) / (2ε)
fn verify_binary_scalar_vjp(prim: Primitive, a: f64, b: f64, g: f64, tol: f64, label: &str) {
    let a_val = Value::scalar_f64(a);
    let b_val = Value::scalar_f64(b);
    let g_val = Value::scalar_f64(g);
    let params = BTreeMap::new();

    // Forward pass
    let out = eval_primitive(prim, &[a_val.clone(), b_val.clone()], &params).unwrap();

    // Analytical VJP
    let vjp_result = fj_ad::vjp(
        prim,
        &[a_val, b_val],
        std::slice::from_ref(&g_val),
        std::slice::from_ref(&out),
        &params,
    )
    .unwrap();

    let eps = 1e-6;

    // Verify gradient w.r.t. first input (a)
    let f_a_plus = extract_f64_scalar(
        &eval_primitive(
            prim,
            &[Value::scalar_f64(a + eps), Value::scalar_f64(b)],
            &params,
        )
        .unwrap(),
    );
    let f_a_minus = extract_f64_scalar(
        &eval_primitive(
            prim,
            &[Value::scalar_f64(a - eps), Value::scalar_f64(b)],
            &params,
        )
        .unwrap(),
    );
    let numerical_a = g * (f_a_plus - f_a_minus) / (2.0 * eps);
    let analytical_a = extract_f64_scalar(&vjp_result[0]);
    assert_scalar_close(
        analytical_a,
        numerical_a,
        tol,
        1e-4,
        &format!("{label} grad_a"),
    );

    // Verify gradient w.r.t. second input (b)
    let f_b_plus = extract_f64_scalar(
        &eval_primitive(
            prim,
            &[Value::scalar_f64(a), Value::scalar_f64(b + eps)],
            &params,
        )
        .unwrap(),
    );
    let f_b_minus = extract_f64_scalar(
        &eval_primitive(
            prim,
            &[Value::scalar_f64(a), Value::scalar_f64(b - eps)],
            &params,
        )
        .unwrap(),
    );
    let numerical_b = g * (f_b_plus - f_b_minus) / (2.0 * eps);
    let analytical_b = extract_f64_scalar(&vjp_result[1]);
    assert_scalar_close(
        analytical_b,
        numerical_b,
        tol,
        1e-4,
        &format!("{label} grad_b"),
    );
}

#[test]
fn add_vjp_numerical() {
    // d/da(a+b)=1, d/db(a+b)=1
    verify_binary_scalar_vjp(Primitive::Add, 3.0, 4.0, 1.0, 1e-5, "add VJP");
    verify_binary_scalar_vjp(Primitive::Add, -1.0, 2.0, 0.5, 1e-5, "add VJP negative");
}

#[test]
fn sub_vjp_numerical() {
    // d/da(a-b)=1, d/db(a-b)=-1
    verify_binary_scalar_vjp(Primitive::Sub, 5.0, 2.0, 1.0, 1e-5, "sub VJP");
    verify_binary_scalar_vjp(Primitive::Sub, -3.0, 7.0, 2.0, 1e-5, "sub VJP negative");
}

#[test]
fn mul_vjp_numerical() {
    // d/da(a*b)=b, d/db(a*b)=a (product rule)
    verify_binary_scalar_vjp(Primitive::Mul, 3.0, 4.0, 1.0, 1e-5, "mul VJP");
    verify_binary_scalar_vjp(Primitive::Mul, -2.0, 5.0, 0.7, 1e-5, "mul VJP negative");
    verify_binary_scalar_vjp(Primitive::Mul, 0.0, 3.0, 1.0, 1e-5, "mul VJP zero");
}

#[test]
fn div_vjp_numerical() {
    // d/da(a/b)=1/b, d/db(a/b)=-a/b^2 (quotient rule)
    verify_binary_scalar_vjp(Primitive::Div, 6.0, 2.0, 1.0, 1e-5, "div VJP");
    verify_binary_scalar_vjp(
        Primitive::Div,
        1.0,
        3.0,
        1.0,
        1e-5,
        "div VJP small quotient",
    );
}

#[test]
fn pow_vjp_numerical() {
    // d/da(a^b) = b*a^(b-1), d/db(a^b) = a^b * ln(a)
    verify_binary_scalar_vjp(Primitive::Pow, 2.0, 3.0, 1.0, 1e-4, "pow VJP 2^3");
    verify_binary_scalar_vjp(Primitive::Pow, 3.0, 0.5, 1.0, 1e-4, "pow VJP sqrt(3)");
}

#[test]
fn max_vjp_numerical() {
    // max(a,b): gradient passes through the larger input
    verify_binary_scalar_vjp(Primitive::Max, 3.0, 7.0, 1.0, 1e-5, "max VJP b>a");
    verify_binary_scalar_vjp(Primitive::Max, 7.0, 3.0, 1.0, 1e-5, "max VJP a>b");
}

#[test]
fn min_vjp_numerical() {
    // min(a,b): gradient passes through the smaller input
    verify_binary_scalar_vjp(Primitive::Min, 3.0, 7.0, 1.0, 1e-5, "min VJP b>a");
    verify_binary_scalar_vjp(Primitive::Min, 7.0, 3.0, 1.0, 1e-5, "min VJP a>b");
}

#[test]
fn atan2_vjp_numerical() {
    // d/da(atan2(a,b)) = b/(a^2+b^2), d/db(atan2(a,b)) = -a/(a^2+b^2)
    verify_binary_scalar_vjp(Primitive::Atan2, 1.0, 1.0, 1.0, 1e-5, "atan2 VJP (1,1)");
    verify_binary_scalar_vjp(Primitive::Atan2, 3.0, 4.0, 1.0, 1e-5, "atan2 VJP (3,4)");
}
