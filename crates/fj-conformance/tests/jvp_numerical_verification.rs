//! JVP (forward-mode AD) numerical verification for linalg primitives.
//!
//! Verifies tangent computation via finite differences:
//!   f'(x) · dx ≈ (f(x + ε·dx) - f(x - ε·dx)) / (2ε)

#![allow(clippy::cloned_ref_to_slice_refs)]

use fj_core::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_lax::eval_primitive_multi;
use smallvec::smallvec;
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

fn make_complex64_scalar(re: f32, im: f32) -> Value {
    Value::Scalar(Literal::from_complex64(re, im))
}

fn extract_complex64_scalar(val: &Value) -> (f32, f32) {
    match val {
        Value::Scalar(Literal::Complex64Bits(re, im)) => (f32::from_bits(*re), f32::from_bits(*im)),
        other => panic!("expected Complex64 scalar, got {other:?}"),
    }
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

fn make_single_op_jaxpr(prim: Primitive) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: prim,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

fn make_two_input_jaxpr(prim: Primitive, params: BTreeMap<String, String>) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: prim,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params,
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

/// Perturb a matrix value: a + eps * da (element-wise)
fn perturb(a: &Value, da: &Value, eps: f64) -> Value {
    let a_t = a.as_tensor().unwrap();
    let da_t = da.as_tensor().unwrap();
    let elements: Vec<Literal> = a_t
        .elements
        .iter()
        .zip(da_t.elements.iter())
        .map(|(av, dv)| {
            let a_val = av.as_f64().unwrap();
            let da_val = dv.as_f64().unwrap();
            Literal::from_f64(a_val + eps * da_val)
        })
        .collect();
    Value::Tensor(TensorValue::new(a_t.dtype, a_t.shape.clone(), elements).unwrap())
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{context}[{i}]: got {a}, expected {e}, diff={} (tol={tol})",
            (a - e).abs()
        );
    }
}

fn assert_complex_close(actual: &[(f64, f64)], expected: &[(f64, f64)], tol: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (i, ((ar, ai), (er, ei))) in actual.iter().zip(expected.iter()).enumerate() {
        let re_diff = (ar - er).abs();
        let im_diff = (ai - ei).abs();
        assert!(
            re_diff < tol && im_diff < tol,
            "{context}[{i}]: got ({ar},{ai}), expected ({er},{ei}), diff=({re_diff},{im_diff}) (tol={tol})"
        );
    }
}

fn assert_scalar_close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64, context: &str) {
    let diff = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!(
        diff <= abs_tol.max(scale * rel_tol),
        "{context}: got {actual}, expected {expected}, diff={diff}, abs_tol={abs_tol}, rel_tol={rel_tol}"
    );
}

// ======================== Cholesky JVP ========================

#[test]
fn cholesky_jvp_numerical() {
    // A = [[4, 2], [2, 3]] (SPD), dA = [[0.1, 0.05], [0.05, 0.2]] (symmetric tangent)
    let a = make_f64_matrix(2, 2, &[4.0, 2.0, 2.0, 3.0]);
    let da = make_f64_matrix(2, 2, &[0.1, 0.05, 0.05, 0.2]);

    let jaxpr = make_single_op_jaxpr(Primitive::Cholesky);
    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();
    let analytical_tangent = extract_f64_vec(&jvp_result.tangents[0]);

    // Numerical: (cholesky(A + eps*dA) - cholesky(A - eps*dA)) / (2*eps)
    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);

    let l_plus = eval_primitive_multi(Primitive::Cholesky, &[a_plus], &BTreeMap::new()).unwrap();
    let l_minus = eval_primitive_multi(Primitive::Cholesky, &[a_minus], &BTreeMap::new()).unwrap();

    let vals_plus = extract_f64_vec(&l_plus[0]);
    let vals_minus = extract_f64_vec(&l_minus[0]);

    let numerical: Vec<f64> = vals_plus
        .iter()
        .zip(vals_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(&analytical_tangent, &numerical, 1e-4, "Cholesky JVP");
}

#[test]
fn cholesky_jvp_near_singular_matrix() {
    let a = make_f64_matrix(2, 2, &[1.0, 0.9999, 0.9999, 1.0]);
    let da = make_f64_matrix(2, 2, &[0.05, 0.02, 0.02, 0.04]);

    let jaxpr = make_single_op_jaxpr(Primitive::Cholesky);
    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();
    let analytical_tangent = extract_f64_vec(&jvp_result.tangents[0]);
    assert!(
        analytical_tangent.iter().all(|value| value.is_finite()),
        "near-singular Cholesky JVP should stay finite: {analytical_tangent:?}"
    );

    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);

    let l_plus = eval_primitive_multi(Primitive::Cholesky, &[a_plus], &BTreeMap::new()).unwrap();
    let l_minus = eval_primitive_multi(Primitive::Cholesky, &[a_minus], &BTreeMap::new()).unwrap();

    let vals_plus = extract_f64_vec(&l_plus[0]);
    let vals_minus = extract_f64_vec(&l_minus[0]);

    let numerical: Vec<f64> = vals_plus
        .iter()
        .zip(vals_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(
        &analytical_tangent,
        &numerical,
        2e-2,
        "Cholesky JVP near singular",
    );
}

// ======================== TriangularSolve JVP ========================

#[test]
fn triangular_solve_jvp_numerical() {
    // L = [[2, 0], [1, 3]], B = [[4], [7]]
    // Tangents: dL = small perturbation, dB = small perturbation
    let l_mat = make_f64_matrix(2, 2, &[2.0, 0.0, 1.0, 3.0]);
    let b_vec = make_f64_matrix(2, 1, &[4.0, 7.0]);
    let dl = make_f64_matrix(2, 2, &[0.1, 0.0, 0.05, 0.15]);
    let db = make_f64_matrix(2, 1, &[0.3, 0.2]);

    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());

    let jaxpr = make_two_input_jaxpr(Primitive::TriangularSolve, params.clone());
    let jvp_result = fj_ad::jvp(
        &jaxpr,
        &[l_mat.clone(), b_vec.clone()],
        &[dl.clone(), db.clone()],
    )
    .unwrap();
    let analytical_tangent = extract_f64_vec(&jvp_result.tangents[0]);

    // Numerical: (solve(L+eps*dL, B+eps*dB) - solve(L-eps*dL, B-eps*dB)) / (2*eps)
    let eps = 1e-6;
    let l_plus = perturb(&l_mat, &dl, eps);
    let b_plus = perturb(&b_vec, &db, eps);
    let l_minus = perturb(&l_mat, &dl, -eps);
    let b_minus = perturb(&b_vec, &db, -eps);

    let out_plus =
        eval_primitive_multi(Primitive::TriangularSolve, &[l_plus, b_plus], &params).unwrap();
    let out_minus =
        eval_primitive_multi(Primitive::TriangularSolve, &[l_minus, b_minus], &params).unwrap();

    let vals_plus = extract_f64_vec(&out_plus[0]);
    let vals_minus = extract_f64_vec(&out_minus[0]);

    let numerical: Vec<f64> = vals_plus
        .iter()
        .zip(vals_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(&analytical_tangent, &numerical, 1e-4, "TriangularSolve JVP");
}

// ======================== Conv JVP ========================

#[test]
fn conv_jvp_numerical() {
    // conv(L, R) is bilinear: d conv = conv(dL, R) + conv(L, dR). Verify the JVP
    // against central differences with BOTH tangents nonzero, so a rule that only
    // got one bilinear term (or computed conv(dL, dR)) is caught. The conv JVP was
    // previously absent from this suite, which is how the conv(dL,dR) bug (qipg0)
    // stayed hidden. lhs=[1,3,1] (batch,width,c_in), rhs=[2,1,1] (K,c_in,c_out),
    // valid padding / stride 1 => output [1,2,1].
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
    let lhs = t3(vec![1, 3, 1], &[1.0, 2.0, 3.0]);
    let rhs = t3(vec![2, 1, 1], &[0.5, -0.25]);
    let dlhs = t3(vec![1, 3, 1], &[0.1, 0.2, 0.3]);
    let drhs = t3(vec![2, 1, 1], &[0.05, 0.15]);

    let params = BTreeMap::new();
    let jaxpr = make_two_input_jaxpr(Primitive::Conv, params.clone());
    let jvp_result = fj_ad::jvp(
        &jaxpr,
        &[lhs.clone(), rhs.clone()],
        &[dlhs.clone(), drhs.clone()],
    )
    .unwrap();
    let analytical_tangent = extract_f64_vec(&jvp_result.tangents[0]);

    let eps = 1e-6;
    let l_plus = perturb(&lhs, &dlhs, eps);
    let r_plus = perturb(&rhs, &drhs, eps);
    let l_minus = perturb(&lhs, &dlhs, -eps);
    let r_minus = perturb(&rhs, &drhs, -eps);

    let out_plus = eval_primitive_multi(Primitive::Conv, &[l_plus, r_plus], &params).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Conv, &[l_minus, r_minus], &params).unwrap();
    let vals_plus = extract_f64_vec(&out_plus[0]);
    let vals_minus = extract_f64_vec(&out_minus[0]);

    let numerical: Vec<f64> = vals_plus
        .iter()
        .zip(vals_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(&analytical_tangent, &numerical, 1e-5, "Conv JVP");
}

// ======================== DotGeneral JVP ========================

#[test]
fn dot_general_jvp_numerical() {
    // DotGeneral is BILINEAR like Conv: d dot_general = dot_general(dA, B) +
    // dot_general(A, dB). Exercise the full machinery — a BATCHED matmul with a
    // batch dim, a contracting dim, and free dims — so a JVP that dropped a term
    // or mishandled the dimension_numbers under perturbation is caught. This op
    // was absent from the numerical suites (the same coverage gap that hid the
    // conv(dL,dR) bug). lhs=[B,M,K]=[2,2,3], rhs=[B,K,N]=[2,3,2] → out=[2,2,2].
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
    let lhs = t3(
        vec![2, 2, 3],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let rhs = t3(
        vec![2, 3, 2],
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0],
    );
    let dlhs = t3(
        vec![2, 2, 3],
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    );
    let drhs = t3(
        vec![2, 3, 2],
        &[
            0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ],
    );

    let mut params = BTreeMap::new();
    params.insert("lhs_contracting_dims".to_string(), "2".to_string());
    params.insert("rhs_contracting_dims".to_string(), "1".to_string());
    params.insert("lhs_batch_dims".to_string(), "0".to_string());
    params.insert("rhs_batch_dims".to_string(), "0".to_string());

    let jaxpr = make_two_input_jaxpr(Primitive::DotGeneral, params.clone());
    let jvp_result = fj_ad::jvp(
        &jaxpr,
        &[lhs.clone(), rhs.clone()],
        &[dlhs.clone(), drhs.clone()],
    )
    .unwrap();
    let analytical_tangent = extract_f64_vec(&jvp_result.tangents[0]);

    let eps = 1e-6;
    let l_plus = perturb(&lhs, &dlhs, eps);
    let r_plus = perturb(&rhs, &drhs, eps);
    let l_minus = perturb(&lhs, &dlhs, -eps);
    let r_minus = perturb(&rhs, &drhs, -eps);

    let out_plus = eval_primitive_multi(Primitive::DotGeneral, &[l_plus, r_plus], &params).unwrap();
    let out_minus =
        eval_primitive_multi(Primitive::DotGeneral, &[l_minus, r_minus], &params).unwrap();
    let vals_plus = extract_f64_vec(&out_plus[0]);
    let vals_minus = extract_f64_vec(&out_minus[0]);

    let numerical: Vec<f64> = vals_plus
        .iter()
        .zip(vals_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(&analytical_tangent, &numerical, 1e-5, "DotGeneral JVP");
}

// ======================== Pad JVP (pad_value tangent) ========================

#[test]
fn pad_jvp_pad_value_tangent_numerical() {
    // d pad(operand, pad_value) = pad(d_operand, d_pad_value): the padding region
    // must be filled with the pad_value's TANGENT (a rule that drops it — padding
    // with 0 or the primal pad_value — is a common forward-mode bug). Use a
    // NONZERO d_pad_value so that path is exercised. operand=[3], low=1,high=1,
    // interior=1 → out=[7] laid out [pv,o0,pv,o1,pv,o2,pv].
    let operand = make_f64_vector(&[1.0, 2.0, 3.0]);
    let pad_value = Value::Scalar(Literal::from_f64(0.5));
    let d_operand = make_f64_vector(&[0.1, 0.2, 0.3]);
    let d_pad_value = Value::Scalar(Literal::from_f64(0.7));

    let mut params = BTreeMap::new();
    params.insert("padding_low".to_string(), "1".to_string());
    params.insert("padding_high".to_string(), "1".to_string());
    params.insert("padding_interior".to_string(), "1".to_string());

    let jaxpr = make_two_input_jaxpr(Primitive::Pad, params.clone());
    let jvp = fj_ad::jvp(
        &jaxpr,
        &[operand.clone(), pad_value.clone()],
        &[d_operand.clone(), d_pad_value.clone()],
    )
    .unwrap();
    let analytical = extract_f64_vec(&jvp.tangents[0]);

    let eps = 1e-6;
    let perturb_scalar =
        |base: f64, dv: f64, s: f64| Value::Scalar(Literal::from_f64(base + s * dv));
    let op_plus = perturb(&operand, &d_operand, eps);
    let op_minus = perturb(&operand, &d_operand, -eps);
    let pv_plus = perturb_scalar(0.5, 0.7, eps);
    let pv_minus = perturb_scalar(0.5, 0.7, -eps);
    let out_plus = eval_primitive_multi(Primitive::Pad, &[op_plus, pv_plus], &params).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Pad, &[op_minus, pv_minus], &params).unwrap();
    let numerical: Vec<f64> = extract_f64_vec(&out_plus[0])
        .iter()
        .zip(extract_f64_vec(&out_minus[0]).iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(&analytical, &numerical, 1e-5, "Pad JVP (pad_value tangent)");
}

// ======================== ReduceMax/Min JVP ========================

fn make_single_input_jaxpr(prim: Primitive, params: BTreeMap<String, String>) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: prim,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params,
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

#[test]
fn reduce_max_min_jvp_axis_numerical() {
    // ReduceMax/Min JVP gathers the tangent at each window's arg-extremum (a
    // selection — same class as sort/reduce_window). With DISTINCT values the
    // chooser is the true derivative, so a finite difference verifies the
    // forward-mode routing. The forward direction had NO numerical coverage.
    let data = [3.0, 1.0, 2.0, 4.0, 6.0, 5.0]; // [2,3]: rows [3,1,2],[4,6,5]
    let input = make_f64_matrix(2, 3, &data);
    let d_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let d_input = make_f64_matrix(2, 3, &d_data);
    let mut params = BTreeMap::new();
    params.insert("axes".to_string(), "1".to_string());

    for prim in [Primitive::ReduceMax, Primitive::ReduceMin] {
        let jaxpr = make_single_input_jaxpr(prim, params.clone());
        let jvp = fj_ad::jvp(&jaxpr, &[input.clone()], &[d_input.clone()]).unwrap();
        let analytical = extract_f64_vec(&jvp.tangents[0]);

        let eps = 1e-6;
        let plus = perturb(&input, &d_input, eps);
        let minus = perturb(&input, &d_input, -eps);
        let out_plus = eval_primitive_multi(prim, std::slice::from_ref(&plus), &params).unwrap();
        let out_minus = eval_primitive_multi(prim, std::slice::from_ref(&minus), &params).unwrap();
        let numerical: Vec<f64> = extract_f64_vec(&out_plus[0])
            .iter()
            .zip(extract_f64_vec(&out_minus[0]).iter())
            .map(|(p, m)| (p - m) / (2.0 * eps))
            .collect();
        assert_close(
            &analytical,
            &numerical,
            1e-5,
            &format!("{prim:?} JVP (axis)"),
        );
    }
}

// ======================== Eigh JVP ========================

#[test]
fn eigh_jvp_numerical() {
    // A = [[4, 2], [2, 3]] (symmetric), dA = [[0.1, 0.05], [0.05, 0.2]]
    let a = make_f64_matrix(2, 2, &[4.0, 2.0, 2.0, 3.0]);
    let da = make_f64_matrix(2, 2, &[0.1, 0.05, 0.05, 0.2]);

    // Eigh is multi-output: (eigenvalues, eigenvectors)
    // Build a Jaxpr with 2 outputs
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3)],
        vec![Equation {
            primitive: Primitive::Eigh,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2), VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();

    // Tangent of eigenvalues
    let dw_analytical = extract_f64_vec(&jvp_result.tangents[0]);

    // Numerical: (eigh(A+eps*dA).eigenvalues - eigh(A-eps*dA).eigenvalues) / (2*eps)
    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);

    let out_plus = eval_primitive_multi(Primitive::Eigh, &[a_plus], &BTreeMap::new()).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Eigh, &[a_minus], &BTreeMap::new()).unwrap();

    let w_plus = extract_f64_vec(&out_plus[0]);
    let w_minus = extract_f64_vec(&out_minus[0]);

    let dw_numerical: Vec<f64> = w_plus
        .iter()
        .zip(w_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(&dw_analytical, &dw_numerical, 1e-4, "Eigh JVP eigenvalues");
}

#[test]
fn eig_jvp_self_consistency_3x3() {
    // Forward-mode through non-symmetric eig. Differentiating A·V = V·diag(λ) gives
    //   dA·V + A·dV = dV·diag(λ) + V·diag(dλ),
    // which the JVP (dλ, dV) must satisfy exactly — a convention-/order-free check.
    // A has a complex-conjugate eigenvalue pair {1±i} plus a real eigenvalue {3}.
    let a_data = [1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 3.0];
    let da_data = [0.1, 0.02, -0.03, 0.04, 0.15, 0.01, 0.02, -0.01, 0.2];
    let a = make_f64_matrix(3, 3, &a_data);
    let da = make_f64_matrix(3, 3, &da_data);

    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3)],
        vec![Equation {
            primitive: Primitive::Eig,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2), VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let res = fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();
    let lam = extract_complex_vec(&res.primals[0]);
    let v = extract_complex_vec(&res.primals[1]);
    let dlam = extract_complex_vec(&res.tangents[0]);
    let dv = extract_complex_vec(&res.tangents[1]);

    let n = 3usize;
    let cm = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let ca = |a: (f64, f64), b: (f64, f64)| (a.0 + b.0, a.1 + b.1);
    for i in 0..n {
        for j in 0..n {
            let mut lhs = (0.0_f64, 0.0_f64);
            for k in 0..n {
                lhs = ca(lhs, cm((da_data[i * n + k], 0.0), v[k * n + j]));
                lhs = ca(lhs, cm((a_data[i * n + k], 0.0), dv[k * n + j]));
            }
            let rhs = ca(cm(dv[i * n + j], lam[j]), cm(v[i * n + j], dlam[j]));
            let (dr, di) = (lhs.0 - rhs.0, lhs.1 - rhs.1);
            assert!(
                dr.abs() < 1e-7 && di.abs() < 1e-7,
                "Eig JVP self-consistency ({i},{j}): lhs={lhs:?} rhs={rhs:?}"
            );
        }
    }
}

// ======================== QR JVP ========================

#[test]
fn qr_jvp_numerical() {
    // A = [[1, -1], [1, 1]], dA = [[0.1, 0.05], [0.03, 0.15]]
    let a = make_f64_matrix(2, 2, &[1.0, -1.0, 1.0, 1.0]);
    let da = make_f64_matrix(2, 2, &[0.1, 0.05, 0.03, 0.15]);

    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3)],
        vec![Equation {
            primitive: Primitive::Qr,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2), VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();

    // Check tangent of R (output[1]) — more numerically stable than Q
    let dr_analytical = extract_f64_vec(&jvp_result.tangents[1]);

    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);

    let out_plus = eval_primitive_multi(Primitive::Qr, &[a_plus], &BTreeMap::new()).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Qr, &[a_minus], &BTreeMap::new()).unwrap();

    let r_plus = extract_f64_vec(&out_plus[1]);
    let r_minus = extract_f64_vec(&out_minus[1]);

    let dr_numerical: Vec<f64> = r_plus
        .iter()
        .zip(r_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(&dr_analytical, &dr_numerical, 1e-3, "QR JVP (R tangent)");
}

#[test]
fn slogdet_jvp_numerical() {
    // slogdet(A) = (sign, logabsdet); d logabsdet = tr(A⁻¹ dA), d sign = 0 (the sign is
    // locally constant for an invertible A). The slogdet JVP is implemented in
    // jvp_rule_multi but was previously UNTESTED — the same "looks done but untested"
    // pattern that hid the solve-vector-RHS bug. Verify both output tangents against
    // central differences.
    let a = make_f64_matrix(3, 3, &[2.0, 0.5, -1.0, 0.3, 1.7, 0.2, -0.4, 0.1, 1.9]);
    let da = make_f64_matrix(
        3,
        3,
        &[0.1, 0.05, 0.03, 0.02, 0.12, -0.04, 0.06, -0.01, 0.09],
    );
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3)],
        vec![Equation {
            primitive: Primitive::Slogdet,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2), VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let jvp = fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();
    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);
    let out_plus = eval_primitive_multi(Primitive::Slogdet, &[a_plus], &BTreeMap::new()).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Slogdet, &[a_minus], &BTreeMap::new()).unwrap();
    for i in 0..2 {
        let an = extract_f64_scalar(&jvp.tangents[i]);
        let fd =
            (extract_f64_scalar(&out_plus[i]) - extract_f64_scalar(&out_minus[i])) / (2.0 * eps);
        assert!(
            (an - fd).abs() < 1e-4,
            "Slogdet JVP output[{i}]: analytical {an}, numerical {fd}"
        );
    }
}

#[test]
fn lu_jvp_numerical() {
    // LU returns (lu, pivots, permutation); only `lu` (output[0]) is differentiable
    // (pivots/perm are integer). The LU JVP is implemented in jvp_rule_multi but was
    // previously UNTESTED. A strongly diagonally-dominant A keeps partial pivoting on the
    // diagonal, so the permutation is stable under the perturbation and the central
    // difference of `lu` is well defined.
    let a = make_f64_matrix(3, 3, &[5.0, 0.3, -0.4, 0.2, 6.0, 0.1, -0.3, 0.2, 7.0]);
    let da = make_f64_matrix(
        3,
        3,
        &[0.1, 0.05, 0.03, 0.02, 0.12, -0.04, 0.06, -0.01, 0.09],
    );
    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3), VarId(4)],
        vec![Equation {
            primitive: Primitive::Lu,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2), VarId(3), VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let jvp = fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();
    let dlu_analytical = extract_f64_vec(&jvp.tangents[0]);

    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);
    let out_plus = eval_primitive_multi(Primitive::Lu, &[a_plus], &BTreeMap::new()).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Lu, &[a_minus], &BTreeMap::new()).unwrap();
    let lu_plus = extract_f64_vec(&out_plus[0]);
    let lu_minus = extract_f64_vec(&out_minus[0]);
    let dlu_numerical: Vec<f64> = lu_plus
        .iter()
        .zip(lu_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();
    assert_close(&dlu_analytical, &dlu_numerical, 1e-4, "LU JVP (lu tangent)");
}

#[test]
fn betainc_jvp_raises_on_a_b_differentiation() {
    // JAX's defjvp registers betainc_grad_not_implemented for a and b (RAISES) and
    // betainc_gradx for x. Forward mode sees which inputs carry a tangent, so fj-lax
    // matches JAX exactly: the x-tangent is supported; differentiating w.r.t. a or b raises.
    let jaxpr = Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::Betainc,
            inputs: smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );
    let primals = [
        Value::scalar_f64(2.0),
        Value::scalar_f64(3.0),
        Value::scalar_f64(0.4),
    ];
    // x-tangent only (a, b constant) is JAX-supported: tangent = d/dx I_x(2,3) at 0.4 =
    // 0.4·0.6²/B(2,3) = 1.728.
    let t_x = [
        Value::scalar_f64(0.0),
        Value::scalar_f64(0.0),
        Value::scalar_f64(1.0),
    ];
    let jvp_x = fj_ad::jvp(&jaxpr, &primals, &t_x).unwrap();
    let dx = extract_f64_scalar(&jvp_x.tangents[0]);
    assert!(
        (dx - 1.728).abs() < 1e-6,
        "betainc d/dx JVP = {dx}, expected 1.728"
    );
    // Differentiating w.r.t. a or b must RAISE (JAX: not supported).
    let t_a = [
        Value::scalar_f64(1.0),
        Value::scalar_f64(0.0),
        Value::scalar_f64(0.0),
    ];
    assert!(
        fj_ad::jvp(&jaxpr, &primals, &t_a).is_err(),
        "betainc JVP w.r.t. a must raise (JAX: not supported)"
    );
    let t_b = [
        Value::scalar_f64(0.0),
        Value::scalar_f64(1.0),
        Value::scalar_f64(0.0),
    ];
    assert!(
        fj_ad::jvp(&jaxpr, &primals, &t_b).is_err(),
        "betainc JVP w.r.t. b must raise (JAX: not supported)"
    );
}

#[test]
fn svd_jvp_ill_conditioned_matrix() {
    let a = make_f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1e-4]);
    let da = make_f64_matrix(2, 2, &[0.08, 0.0, 0.0, 0.02]);

    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3), VarId(4)],
        vec![Equation {
            primitive: Primitive::Svd,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2), VarId(3), VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();
    let ds_analytical = extract_f64_vec(&jvp_result.tangents[1]);
    assert!(
        ds_analytical.iter().all(|value| value.is_finite()),
        "ill-conditioned SVD JVP should stay finite: {ds_analytical:?}"
    );

    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);

    let out_plus = eval_primitive_multi(Primitive::Svd, &[a_plus], &BTreeMap::new()).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Svd, &[a_minus], &BTreeMap::new()).unwrap();

    let s_plus = extract_f64_vec(&out_plus[1]);
    let s_minus = extract_f64_vec(&out_minus[1]);

    let ds_numerical: Vec<f64> = s_plus
        .iter()
        .zip(s_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(
        &ds_analytical,
        &ds_numerical,
        1e-3,
        "SVD JVP ill conditioned",
    );
}

// ======================== SVD JVP ========================

#[test]
fn svd_jvp_numerical() {
    // A = [[3, 1], [1, 2]], dA = [[0.1, 0.05], [0.05, 0.15]]
    let a = make_f64_matrix(2, 2, &[3.0, 1.0, 1.0, 2.0]);
    let da = make_f64_matrix(2, 2, &[0.1, 0.05, 0.05, 0.15]);

    let jaxpr = Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2), VarId(3), VarId(4)],
        vec![Equation {
            primitive: Primitive::Svd,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2), VarId(3), VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    );

    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();

    // Check tangent of singular values (output[1]) — most numerically stable
    let ds_analytical = extract_f64_vec(&jvp_result.tangents[1]);

    let eps = 1e-6;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);

    let out_plus = eval_primitive_multi(Primitive::Svd, &[a_plus], &BTreeMap::new()).unwrap();
    let out_minus = eval_primitive_multi(Primitive::Svd, &[a_minus], &BTreeMap::new()).unwrap();

    let s_plus = extract_f64_vec(&out_plus[1]);
    let s_minus = extract_f64_vec(&out_minus[1]);

    let ds_numerical: Vec<f64> = s_plus
        .iter()
        .zip(s_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();

    assert_close(
        &ds_analytical,
        &ds_numerical,
        1e-3,
        "SVD JVP (singular value tangents)",
    );
}

#[test]
fn mul_jvp_denormal_input() {
    let x = Value::scalar_f64(f64::MIN_POSITIVE / 2.0);
    let scale = Value::scalar_f64(2.0);
    let dx = Value::scalar_f64(1.0);
    let dscale = Value::scalar_f64(0.0);

    let jaxpr = make_two_input_jaxpr(Primitive::Mul, BTreeMap::new());
    let jvp_result = fj_ad::jvp(
        &jaxpr,
        &[x.clone(), scale.clone()],
        &[dx.clone(), dscale.clone()],
    )
    .unwrap();
    let analytical = extract_f64_scalar(&jvp_result.tangents[0]);
    assert!(analytical.is_finite(), "denormal JVP should stay finite");

    let eps = f64::MIN_POSITIVE / 4.0;
    let plus = eval_primitive_multi(
        Primitive::Mul,
        &[
            Value::scalar_f64(extract_f64_scalar(&x) + eps * extract_f64_scalar(&dx)),
            scale.clone(),
        ],
        &BTreeMap::new(),
    )
    .unwrap();
    let minus = eval_primitive_multi(
        Primitive::Mul,
        &[
            Value::scalar_f64(extract_f64_scalar(&x) - eps * extract_f64_scalar(&dx)),
            scale,
        ],
        &BTreeMap::new(),
    )
    .unwrap();
    let numerical = (extract_f64_scalar(&plus[0]) - extract_f64_scalar(&minus[0])) / (2.0 * eps);

    assert_scalar_close(
        analytical,
        numerical,
        1e-12,
        1e-12,
        "Mul JVP denormal input",
    );
}

// ============================================================================
// Additional JVP edge-case tests (frankenjax-gl1)
// ============================================================================

fn make_multi_output_jaxpr(prim: Primitive, num_outputs: usize) -> Jaxpr {
    let outvars: Vec<VarId> = (0..num_outputs).map(|i| VarId(2 + i as u32)).collect();
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        outvars.clone(),
        vec![Equation {
            primitive: prim,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: outvars.into_iter().collect(),
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

/// QR JVP on a 3x2 moderately ill-conditioned rectangular matrix.
#[test]
fn qr_jvp_rectangular_ill_conditioned() {
    let a_data = [1.0, 0.01, 0.0, 0.5, -1.0, 0.02];
    let a = make_f64_matrix(3, 2, &a_data);
    // Random tangent direction
    let da_data = [0.1, -0.05, 0.2, 0.15, -0.1, 0.08];
    let da = make_f64_matrix(3, 2, &da_data);

    let jaxpr = make_multi_output_jaxpr(Primitive::Qr, 2);
    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();

    // Verify tangents are finite
    for (i, t) in jvp_result.tangents.iter().enumerate() {
        let vals = extract_f64_vec(t);
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "QR JVP rectangular tangent {i} should be finite: {vals:?}"
        );
    }

    // Finite-difference verification (sum of all outputs)
    let eps = 1e-7;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);
    let outs_plus = eval_primitive_multi(Primitive::Qr, &[a_plus], &BTreeMap::new()).unwrap();
    let outs_minus = eval_primitive_multi(Primitive::Qr, &[a_minus], &BTreeMap::new()).unwrap();

    for (idx, ((tangent, plus), minus)) in jvp_result
        .tangents
        .iter()
        .zip(outs_plus.iter())
        .zip(outs_minus.iter())
        .enumerate()
    {
        let t_vals = extract_f64_vec(tangent);
        let p_vals = extract_f64_vec(plus);
        let m_vals = extract_f64_vec(minus);
        let numerical: Vec<f64> = p_vals
            .iter()
            .zip(m_vals.iter())
            .map(|(p, m)| (p - m) / (2.0 * eps))
            .collect();
        assert_close(
            &t_vals,
            &numerical,
            5e-2,
            &format!("QR JVP rectangular output {idx}"),
        );
    }
}

/// SVD JVP on a 3x3 diagonal matrix with a near-zero singular value.
#[test]
fn svd_jvp_near_zero_singular_value_3x3() {
    let a_data = [5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-6];
    let a = make_f64_matrix(3, 3, &a_data);
    let da_data = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01];
    let da = make_f64_matrix(3, 3, &da_data);

    let jaxpr = make_multi_output_jaxpr(Primitive::Svd, 3);
    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();

    // Verify all tangents are finite
    for (i, t) in jvp_result.tangents.iter().enumerate() {
        let vals = extract_f64_vec(t);
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "SVD JVP near-zero SV tangent {i} should be finite: {vals:?}"
        );
    }

    // Verify S tangent (most stable for finite-diff comparison)
    let eps = 1e-7;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);
    let outs_plus = eval_primitive_multi(Primitive::Svd, &[a_plus], &BTreeMap::new()).unwrap();
    let outs_minus = eval_primitive_multi(Primitive::Svd, &[a_minus], &BTreeMap::new()).unwrap();

    // Index 1 is S (singular values)
    let s_tangent = extract_f64_vec(&jvp_result.tangents[1]);
    let s_plus = extract_f64_vec(&outs_plus[1]);
    let s_minus = extract_f64_vec(&outs_minus[1]);
    let s_numerical: Vec<f64> = s_plus
        .iter()
        .zip(s_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();
    assert_close(
        &s_tangent,
        &s_numerical,
        1e-3,
        "SVD JVP near-zero SV S tangent",
    );
}

/// Eigh JVP on a well-conditioned 3x3 symmetric matrix.
#[test]
fn eigh_jvp_well_conditioned_3x3() {
    let a_data = [5.0, 1.0, 0.5, 1.0, 8.0, 1.0, 0.5, 1.0, 3.0];
    let a = make_f64_matrix(3, 3, &a_data);
    // Symmetric tangent
    let da_data = [0.1, 0.02, 0.01, 0.02, 0.15, 0.03, 0.01, 0.03, 0.08];
    let da = make_f64_matrix(3, 3, &da_data);

    let jaxpr = make_multi_output_jaxpr(Primitive::Eigh, 2);
    let jvp_result =
        fj_ad::jvp(&jaxpr, std::slice::from_ref(&a), std::slice::from_ref(&da)).unwrap();

    for (i, t) in jvp_result.tangents.iter().enumerate() {
        let vals = extract_f64_vec(t);
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "Eigh JVP 3x3 tangent {i} should be finite: {vals:?}"
        );
    }

    // Verify W (eigenvalue) tangent
    let eps = 1e-7;
    let a_plus = perturb(&a, &da, eps);
    let a_minus = perturb(&a, &da, -eps);
    let outs_plus = eval_primitive_multi(Primitive::Eigh, &[a_plus], &BTreeMap::new()).unwrap();
    let outs_minus = eval_primitive_multi(Primitive::Eigh, &[a_minus], &BTreeMap::new()).unwrap();

    // Index 0 is W (eigenvalues)
    let w_tangent = extract_f64_vec(&jvp_result.tangents[0]);
    let w_plus = extract_f64_vec(&outs_plus[0]);
    let w_minus = extract_f64_vec(&outs_minus[0]);
    let w_numerical: Vec<f64> = w_plus
        .iter()
        .zip(w_minus.iter())
        .map(|(p, m)| (p - m) / (2.0 * eps))
        .collect();
    assert_close(&w_tangent, &w_numerical, 1e-3, "Eigh JVP 3x3 W tangent");
}

/// TriangularSolve JVP with near-zero diagonal.
#[test]
fn triangular_solve_jvp_near_singular_diagonal() {
    let l_data = [1.0, 0.0, 0.5, 0.001];
    let l_matrix = make_f64_matrix(2, 2, &l_data);
    let b = make_f64_matrix(2, 1, &[1.0, 0.5]);
    // Tangent for L and b
    let dl = make_f64_matrix(2, 2, &[0.01, 0.0, 0.02, 0.001]);
    let db = make_f64_matrix(2, 1, &[0.1, 0.05]);

    let mut params = BTreeMap::new();
    params.insert("lower".to_owned(), "true".to_owned());
    params.insert("unit_diagonal".to_owned(), "false".to_owned());

    let jaxpr = make_two_input_jaxpr(Primitive::TriangularSolve, params.clone());
    let jvp_result = fj_ad::jvp(
        &jaxpr,
        &[l_matrix.clone(), b.clone()],
        &[dl.clone(), db.clone()],
    )
    .unwrap();

    let t_vals = extract_f64_vec(&jvp_result.tangents[0]);
    assert!(
        t_vals.iter().all(|v| v.is_finite()),
        "TriangularSolve JVP near-singular tangent should be finite: {t_vals:?}"
    );
}

/// Exp JVP at large input (near overflow).
#[test]
fn exp_jvp_large_input() {
    let x = Value::scalar_f64(700.0);
    let dx = Value::scalar_f64(1.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Exp);
    let jvp_result = fj_ad::jvp(&jaxpr, &[x], &[dx]).unwrap();
    let tangent = extract_f64_scalar(&jvp_result.tangents[0]);

    assert!(tangent.is_finite(), "exp(700) JVP tangent should be finite");
    let expected = 700.0_f64.exp();
    assert!(
        (tangent - expected).abs() / expected < 1e-10,
        "exp JVP at 700: tangent={tangent}, expected={expected}"
    );
}

/// Log JVP near zero: gradient 1/x with tiny x.
#[test]
fn log_jvp_near_zero() {
    let x = Value::scalar_f64(1e-300);
    let dx = Value::scalar_f64(1.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Log);
    let jvp_result = fj_ad::jvp(&jaxpr, &[x], &[dx]).unwrap();
    let tangent = extract_f64_scalar(&jvp_result.tangents[0]);

    assert!(
        tangent.is_finite(),
        "log(1e-300) JVP tangent should be finite"
    );
    let expected = 1.0 / 1e-300;
    assert!(
        (tangent - expected).abs() / expected < 1e-10,
        "log JVP near zero: tangent={tangent}, expected={expected}"
    );
}

// ======================== Elementary Scalar JVP Numerical Tests (frankenjax-5uy) ========================

/// Finite-difference JVP verification for unary scalar primitives.
/// For f: R → R, the JVP should satisfy:
///   f'(x) · dx ≈ (f(x + ε·dx) - f(x - ε·dx)) / (2ε)
fn verify_unary_scalar_jvp(prim: Primitive, x: f64, dx: f64, tol: f64, label: &str) {
    let jaxpr = make_single_op_jaxpr(prim);
    let jvp_result = fj_ad::jvp(&jaxpr, &[Value::scalar_f64(x)], &[Value::scalar_f64(dx)]).unwrap();
    let analytical = extract_f64_scalar(&jvp_result.tangents[0]);

    let eps = 1e-6;
    let f_plus = extract_f64_scalar(
        &fj_lax::eval_primitive(prim, &[Value::scalar_f64(x + eps * dx)], &BTreeMap::new())
            .unwrap(),
    );
    let f_minus = extract_f64_scalar(
        &fj_lax::eval_primitive(prim, &[Value::scalar_f64(x - eps * dx)], &BTreeMap::new())
            .unwrap(),
    );
    let numerical = (f_plus - f_minus) / (2.0 * eps);

    assert_scalar_close(analytical, numerical, tol, 1e-4, label);
}

/// Finite-difference JVP verification for binary scalar primitives.
fn verify_binary_scalar_jvp(
    prim: Primitive,
    a: f64,
    b: f64,
    da: f64,
    db: f64,
    tol: f64,
    label: &str,
) {
    let jaxpr = make_two_input_jaxpr(prim, BTreeMap::new());
    let jvp_result = fj_ad::jvp(
        &jaxpr,
        &[Value::scalar_f64(a), Value::scalar_f64(b)],
        &[Value::scalar_f64(da), Value::scalar_f64(db)],
    )
    .unwrap();
    let analytical = extract_f64_scalar(&jvp_result.tangents[0]);

    let eps = 1e-6;
    let f_plus = extract_f64_scalar(
        &fj_lax::eval_primitive(
            prim,
            &[
                Value::scalar_f64(a + eps * da),
                Value::scalar_f64(b + eps * db),
            ],
            &BTreeMap::new(),
        )
        .unwrap(),
    );
    let f_minus = extract_f64_scalar(
        &fj_lax::eval_primitive(
            prim,
            &[
                Value::scalar_f64(a - eps * da),
                Value::scalar_f64(b - eps * db),
            ],
            &BTreeMap::new(),
        )
        .unwrap(),
    );
    let numerical = (f_plus - f_minus) / (2.0 * eps);

    assert_scalar_close(analytical, numerical, tol, 1e-4, label);
}

// ── Complex primitive JVP tests ──

#[test]
fn complex_projection_jvp_vector_projects_complex_tangents() {
    let primals = make_complex128_vector(&[(2.0, -3.0), (-5.0, 7.0)]);
    let tangents = make_complex128_vector(&[(1.5, -2.5), (3.25, -4.75)]);

    let real_jaxpr = make_single_op_jaxpr(Primitive::Real);
    let real_jvp = fj_ad::jvp(
        &real_jaxpr,
        std::slice::from_ref(&primals),
        std::slice::from_ref(&tangents),
    )
    .unwrap();
    assert_close(
        &extract_f64_vec(&real_jvp.tangents[0]),
        &[1.5, 3.25],
        1e-10,
        "Real vector JVP",
    );

    let imag_jaxpr = make_single_op_jaxpr(Primitive::Imag);
    let imag_jvp = fj_ad::jvp(
        &imag_jaxpr,
        std::slice::from_ref(&primals),
        std::slice::from_ref(&tangents),
    )
    .unwrap();
    assert_close(
        &extract_f64_vec(&imag_jvp.tangents[0]),
        &[-2.5, -4.75],
        1e-10,
        "Imag vector JVP",
    );
}

#[test]
fn complex_conj_jvp_vector_conjugates_complex_tangent() {
    let primals = make_complex128_vector(&[(1.0, -2.0), (-3.0, 4.0)]);
    let tangents = make_complex128_vector(&[(5.0, -6.0), (-7.0, 8.0)]);

    let jaxpr = make_single_op_jaxpr(Primitive::Conj);
    let jvp_result = fj_ad::jvp(&jaxpr, &[primals], &[tangents]).unwrap();

    assert_complex_close(
        &extract_complex_vec(&jvp_result.tangents[0]),
        &[(5.0, 6.0), (-7.0, -8.0)],
        1e-10,
        "Conj vector JVP",
    );
}

#[test]
fn complex_constructor_jvp_vector_builds_complex_tangent() {
    let real = make_f64_vector(&[1.0, -2.0]);
    let imag = make_f64_vector(&[3.0, -4.0]);
    let real_tangent = make_f64_vector(&[7.0, -11.0]);
    let imag_tangent = make_f64_vector(&[-13.0, 17.0]);

    let jaxpr = make_two_input_jaxpr(Primitive::Complex, BTreeMap::new());
    let jvp_result = fj_ad::jvp(&jaxpr, &[real, imag], &[real_tangent, imag_tangent]).unwrap();

    assert_complex_close(
        &extract_complex_vec(&jvp_result.tangents[0]),
        &[(7.0, -13.0), (-11.0, 17.0)],
        1e-10,
        "Complex constructor vector JVP",
    );
}

// ── Unary JVP tests ──

#[test]
fn sin_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Sin, 1.0, 1.0, 1e-5, "sin JVP at x=1");
    verify_unary_scalar_jvp(Primitive::Sin, 0.0, 1.0, 1e-5, "sin JVP at x=0");
}

#[test]
fn cos_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Cos, 1.0, 1.0, 1e-5, "cos JVP at x=1");
    verify_unary_scalar_jvp(Primitive::Cos, 0.0, 1.0, 1e-5, "cos JVP at x=0");
}

#[test]
fn exp_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Exp, 1.0, 1.0, 1e-5, "exp JVP at x=1");
    verify_unary_scalar_jvp(Primitive::Exp, 0.0, 1.0, 1e-5, "exp JVP at x=0");
}

#[test]
fn log_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Log, 1.0, 1.0, 1e-5, "log JVP at x=1");
    verify_unary_scalar_jvp(Primitive::Log, 2.0, 1.0, 1e-5, "log JVP at x=2");
}

#[test]
fn tanh_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Tanh, 0.5, 1.0, 1e-5, "tanh JVP at x=0.5");
    verify_unary_scalar_jvp(Primitive::Tanh, 0.0, 1.0, 1e-5, "tanh JVP at x=0");
}

#[test]
fn sqrt_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Sqrt, 4.0, 1.0, 1e-5, "sqrt JVP at x=4");
    verify_unary_scalar_jvp(Primitive::Sqrt, 1.0, 1.0, 1e-5, "sqrt JVP at x=1");
}

#[test]
fn neg_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Neg, 3.0, 1.0, 1e-5, "neg JVP at x=3");
}

#[test]
fn abs_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Abs, 3.0, 1.0, 1e-5, "abs JVP at x=3");
    verify_unary_scalar_jvp(Primitive::Abs, -3.0, 1.0, 1e-5, "abs JVP at x=-3");
}

#[test]
fn expm1_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Expm1, 0.5, 1.0, 1e-5, "expm1 JVP at x=0.5");
}

#[test]
fn log1p_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Log1p, 0.5, 1.0, 1e-5, "log1p JVP at x=0.5");
}

#[test]
fn sinh_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Sinh, 1.0, 1.0, 1e-5, "sinh JVP at x=1");
}

#[test]
fn cosh_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Cosh, 1.0, 1.0, 1e-5, "cosh JVP at x=1");
}

#[test]
fn tan_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Tan, 0.5, 1.0, 1e-5, "tan JVP at x=0.5");
}

#[test]
fn asin_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Asin, 0.5, 1.0, 1e-5, "asin JVP at x=0.5");
}

#[test]
fn acos_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Acos, 0.5, 1.0, 1e-5, "acos JVP at x=0.5");
}

#[test]
fn atan_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Atan, 1.0, 1.0, 1e-5, "atan JVP at x=1");
}

#[test]
fn square_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Square, 3.0, 1.0, 1e-5, "square JVP at x=3");
}

#[test]
fn reciprocal_jvp_numerical() {
    verify_unary_scalar_jvp(
        Primitive::Reciprocal,
        2.0,
        1.0,
        1e-5,
        "reciprocal JVP at x=2",
    );
}

#[test]
fn rsqrt_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Rsqrt, 4.0, 1.0, 1e-5, "rsqrt JVP at x=4");
}

#[test]
fn cbrt_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Cbrt, 8.0, 1.0, 1e-5, "cbrt JVP at x=8");
}

#[test]
fn logistic_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Logistic, 0.0, 1.0, 1e-5, "logistic JVP at x=0");
    verify_unary_scalar_jvp(Primitive::Logistic, 1.0, 1.0, 1e-5, "logistic JVP at x=1");
}

#[test]
fn erf_jvp_numerical() {
    verify_unary_scalar_jvp(Primitive::Erf, 0.5, 1.0, 1e-4, "erf JVP at x=0.5");
}

// ── Binary JVP tests ──

#[test]
fn add_jvp_numerical() {
    verify_binary_scalar_jvp(Primitive::Add, 3.0, 4.0, 1.0, 1.0, 1e-5, "add JVP");
}

#[test]
fn sub_jvp_numerical() {
    verify_binary_scalar_jvp(Primitive::Sub, 5.0, 2.0, 1.0, 1.0, 1e-5, "sub JVP");
}

#[test]
fn mul_jvp_numerical_basic() {
    verify_binary_scalar_jvp(Primitive::Mul, 3.0, 4.0, 1.0, 1.0, 1e-5, "mul JVP");
    verify_binary_scalar_jvp(Primitive::Mul, 3.0, 4.0, 1.0, 0.0, 1e-5, "mul JVP da only");
    verify_binary_scalar_jvp(Primitive::Mul, 3.0, 4.0, 0.0, 1.0, 1e-5, "mul JVP db only");
}

#[test]
fn div_jvp_numerical() {
    verify_binary_scalar_jvp(Primitive::Div, 6.0, 2.0, 1.0, 1.0, 1e-5, "div JVP");
    verify_binary_scalar_jvp(
        Primitive::Div,
        1.0,
        3.0,
        0.5,
        0.5,
        1e-5,
        "div JVP fractional",
    );
}

#[test]
fn pow_jvp_numerical() {
    verify_binary_scalar_jvp(Primitive::Pow, 2.0, 3.0, 1.0, 0.0, 1e-4, "pow JVP da only");
    verify_binary_scalar_jvp(Primitive::Pow, 2.0, 3.0, 0.0, 1.0, 1e-4, "pow JVP db only");
    verify_binary_scalar_jvp(Primitive::Pow, 2.0, 3.0, 1.0, 1.0, 1e-4, "pow JVP both");
}

#[test]
fn atan2_jvp_numerical() {
    verify_binary_scalar_jvp(
        Primitive::Atan2,
        1.0,
        1.0,
        1.0,
        1.0,
        1e-5,
        "atan2 JVP (1,1)",
    );
    verify_binary_scalar_jvp(
        Primitive::Atan2,
        3.0,
        4.0,
        1.0,
        0.0,
        1e-5,
        "atan2 JVP da only",
    );
}

#[test]
fn max_jvp_numerical() {
    verify_binary_scalar_jvp(Primitive::Max, 3.0, 7.0, 1.0, 1.0, 1e-5, "max JVP b>a");
    verify_binary_scalar_jvp(Primitive::Max, 7.0, 3.0, 1.0, 1.0, 1e-5, "max JVP a>b");
}

#[test]
fn min_jvp_numerical() {
    verify_binary_scalar_jvp(Primitive::Min, 3.0, 7.0, 1.0, 1.0, 1e-5, "min JVP b>a");
    verify_binary_scalar_jvp(Primitive::Min, 7.0, 3.0, 1.0, 1.0, 1e-5, "min JVP a>b");
}

// ── Complex64 JVP numerical verification (parity with VJP tests) ──
//
// These tests verify that JVP for complex primitives produces correct tangent
// values by checking against analytically derived expected results.

/// Helper: compute complex multiplication (a_re + a_im*i) * (b_re + b_im*i)
fn complex_mul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> (f32, f32) {
    (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)
}

/// Helper: compute complex division (a_re + a_im*i) / (b_re + b_im*i)
fn complex_div(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> (f32, f32) {
    let denom = b_re * b_re + b_im * b_im;
    (
        (a_re * b_re + a_im * b_im) / denom,
        (a_im * b_re - a_re * b_im) / denom,
    )
}

/// Helper: compute complex negation
fn complex_neg(re: f32, im: f32) -> (f32, f32) {
    (-re, -im)
}

/// Helper: compute complex exp
fn complex_exp(re: f32, im: f32) -> (f32, f32) {
    let r = re.exp();
    (r * im.cos(), r * im.sin())
}

fn assert_complex64_close(actual: (f32, f32), expected: (f32, f32), tol: f32, context: &str) {
    let (ar, ai) = actual;
    let (er, ei) = expected;
    let re_diff = (ar - er).abs();
    let im_diff = (ai - ei).abs();
    assert!(
        re_diff < tol && im_diff < tol,
        "{context}: got ({ar}, {ai}), expected ({er}, {ei}), diff=({re_diff}, {im_diff}) (tol={tol})"
    );
}

/// Complex64 Exp JVP: d/dz exp(z) = exp(z), so tangent = exp(z) * dz
#[test]
fn exp_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.0, 0.5);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Exp);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    let exp_z = complex_exp(0.0, 0.5);
    let expected = complex_mul(exp_z.0, exp_z.1, 1.0, 0.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-5, "exp JVP at z=0.5i");
}

/// Complex64 Log JVP: d/dz log(z) = 1/z, so tangent = dz/z
#[test]
fn log_jvp_numerical_complex64() {
    let z = make_complex64_scalar(1.0, 1.0);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Log);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // 1/z = 1/(1+i) = (1-i)/2 = 0.5 - 0.5i
    let expected = complex_div(1.0, 0.0, 1.0, 1.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-5, "log JVP at z=1+i");
}

/// Complex64 Sin JVP: d/dz sin(z) = cos(z), so tangent = cos(z) * dz
#[test]
fn sin_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.5, 0.3);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Sin);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // cos(z) for z = a+bi: cos(a)*cosh(b) - i*sin(a)*sinh(b)
    let (a, b) = (0.5_f32, 0.3_f32);
    let cos_z = (a.cos() * b.cosh(), -a.sin() * b.sinh());
    let expected = complex_mul(cos_z.0, cos_z.1, 1.0, 0.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "sin JVP at z=0.5+0.3i");
}

/// Complex64 Cbrt JVP must FAIL CLOSED (frankenjax-w8u0a, JVP-side).
///
/// JAX's `cbrt` is `standard_unop(_float)` — complex rejected (commit eb5ad225) —
/// and the cbrt JVP rule evaluates the forward Cbrt, so jvp of a cbrt jaxpr on
/// complex primals must surface the unsupported error rather than a tangent. This
/// is the forward-mode analog of cbrt_vjp_numerical_complex64.
#[test]
fn cbrt_jvp_numerical_complex64_fails_closed() {
    let z = make_complex64_scalar(2.0, 0.0);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Cbrt);
    let result = fj_ad::jvp(&jaxpr, &[z], &[dz]);
    assert!(
        result.is_err(),
        "complex cbrt JVP must fail closed (JAX cbrt is float-only); got {result:?}"
    );
}

/// Complex64 Cos JVP: d/dz cos(z) = -sin(z), so tangent = -sin(z) * dz
#[test]
fn cos_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.5, 0.3);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Cos);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // sin(z) for z = a+bi: sin(a)*cosh(b) + i*cos(a)*sinh(b)
    let (a, b) = (0.5_f32, 0.3_f32);
    let sin_z = (a.sin() * b.cosh(), a.cos() * b.sinh());
    let neg_sin_z = complex_neg(sin_z.0, sin_z.1);
    let expected = complex_mul(neg_sin_z.0, neg_sin_z.1, 1.0, 0.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "cos JVP at z=0.5+0.3i");
}

/// Complex64 Sinh JVP: d/dz sinh(z) = cosh(z)
#[test]
fn sinh_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.5, 0.3);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Sinh);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // cosh(z) for z = a+bi: cosh(a)*cos(b) + i*sinh(a)*sin(b)
    let (a, b) = (0.5_f32, 0.3_f32);
    let cosh_z = (a.cosh() * b.cos(), a.sinh() * b.sin());
    let expected = complex_mul(cosh_z.0, cosh_z.1, 1.0, 0.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "sinh JVP at z=0.5+0.3i");
}

/// Complex64 Cosh JVP: d/dz cosh(z) = sinh(z)
#[test]
fn cosh_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.5, 0.3);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Cosh);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // sinh(z) for z = a+bi: sinh(a)*cos(b) + i*cosh(a)*sin(b)
    let (a, b) = (0.5_f32, 0.3_f32);
    let sinh_z = (a.sinh() * b.cos(), a.cosh() * b.sin());
    let expected = complex_mul(sinh_z.0, sinh_z.1, 1.0, 0.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "cosh JVP at z=0.5+0.3i");
}

/// Complex64 Tan JVP: d/dz tan(z) = sec²(z) = 1/cos²(z)
#[test]
fn tan_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.3, 0.2);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Tan);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // cos(z), then 1/cos²(z)
    let (a, b) = (0.3_f32, 0.2_f32);
    let cos_z = (a.cos() * b.cosh(), -a.sin() * b.sinh());
    let cos_sq = complex_mul(cos_z.0, cos_z.1, cos_z.0, cos_z.1);
    let expected = complex_div(1.0, 0.0, cos_sq.0, cos_sq.1);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "tan JVP at z=0.3+0.2i");
}

/// Complex64 Tanh JVP: d/dz tanh(z) = sech²(z) = 1/cosh²(z)
#[test]
fn tanh_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.3, 0.2);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Tanh);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // cosh(z), then 1/cosh²(z)
    let (a, b) = (0.3_f32, 0.2_f32);
    let cosh_z = (a.cosh() * b.cos(), a.sinh() * b.sin());
    let cosh_sq = complex_mul(cosh_z.0, cosh_z.1, cosh_z.0, cosh_z.1);
    let expected = complex_div(1.0, 0.0, cosh_sq.0, cosh_sq.1);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "tanh JVP at z=0.3+0.2i");
}

/// Complex64 Neg JVP: d/dz (-z) = -1, so tangent = -dz
#[test]
fn neg_jvp_numerical_complex64() {
    let z = make_complex64_scalar(2.0, 3.0);
    let dz = make_complex64_scalar(1.5, -0.5);

    let jaxpr = make_single_op_jaxpr(Primitive::Neg);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    let expected = complex_neg(1.5, -0.5);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-6, "neg JVP");
}

/// Complex64 Sqrt JVP: d/dz sqrt(z) = 1/(2*sqrt(z))
#[test]
fn sqrt_jvp_numerical_complex64() {
    let z = make_complex64_scalar(3.0, 4.0);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Sqrt);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // sqrt(3+4i) = 2+i (verify: (2+i)² = 4 + 4i - 1 = 3+4i)
    // derivative = 1/(2*sqrt(z)) = 1/(2*(2+i)) = 1/(4+2i) = (4-2i)/20 = 0.2 - 0.1i
    let expected = complex_div(1.0, 0.0, 4.0, 2.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-5, "sqrt JVP at z=3+4i");
}

/// Complex64 Square JVP: d/dz z² = 2z
#[test]
fn square_jvp_numerical_complex64() {
    let z = make_complex64_scalar(2.0, 3.0);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Square);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // 2*z = 2*(2+3i) = 4+6i
    let expected = (4.0, 6.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-6, "square JVP at z=2+3i");
}

/// Complex64 Add JVP: d/dz (a+b) w.r.t a is 1, w.r.t b is 1
#[test]
fn add_jvp_numerical_complex64() {
    let a = make_complex64_scalar(2.0, 3.0);
    let b = make_complex64_scalar(-1.0, 4.0);
    let da = make_complex64_scalar(1.0, 0.5);
    let db = make_complex64_scalar(-0.5, 1.0);

    let jaxpr = make_two_input_jaxpr(Primitive::Add, BTreeMap::new());
    let jvp_result = fj_ad::jvp(&jaxpr, &[a, b], &[da, db]).unwrap();

    // tangent = da + db = (1+0.5i) + (-0.5+i) = 0.5 + 1.5i
    let expected = (0.5, 1.5);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-6, "add JVP complex64");
}

/// Complex64 Sub JVP: d/dz (a-b) w.r.t a is 1, w.r.t b is -1
#[test]
fn sub_jvp_numerical_complex64() {
    let a = make_complex64_scalar(2.0, 3.0);
    let b = make_complex64_scalar(-1.0, 4.0);
    let da = make_complex64_scalar(1.0, 0.5);
    let db = make_complex64_scalar(-0.5, 1.0);

    let jaxpr = make_two_input_jaxpr(Primitive::Sub, BTreeMap::new());
    let jvp_result = fj_ad::jvp(&jaxpr, &[a, b], &[da, db]).unwrap();

    // tangent = da - db = (1+0.5i) - (-0.5+i) = 1.5 - 0.5i
    let expected = (1.5, -0.5);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-6, "sub JVP complex64");
}

/// Complex64 Mul JVP: d/dz (a*b) = da*b + a*db
#[test]
fn mul_jvp_numerical_complex64() {
    let a = make_complex64_scalar(2.0, 3.0);
    let b = make_complex64_scalar(-1.0, 4.0);
    let da = make_complex64_scalar(1.0, 0.0);
    let db = make_complex64_scalar(0.0, 1.0);

    let jaxpr = make_two_input_jaxpr(Primitive::Mul, BTreeMap::new());
    let jvp_result = fj_ad::jvp(&jaxpr, &[a, b], &[da, db]).unwrap();

    // tangent = da*b + a*db = (1+0i)*(-1+4i) + (2+3i)*(0+i)
    //         = (-1+4i) + (2i + 3i²) = (-1+4i) + (-3+2i) = -4+6i
    let da_b = complex_mul(1.0, 0.0, -1.0, 4.0);
    let a_db = complex_mul(2.0, 3.0, 0.0, 1.0);
    let expected = (da_b.0 + a_db.0, da_b.1 + a_db.1);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-5, "mul JVP complex64");
}

/// Complex64 Div JVP: d/dz (a/b) = (da*b - a*db)/b²
#[test]
fn div_jvp_numerical_complex64() {
    let a = make_complex64_scalar(2.0, 3.0);
    let b = make_complex64_scalar(1.0, 1.0);
    let da = make_complex64_scalar(1.0, 0.0);
    let db = make_complex64_scalar(0.0, 0.0);

    let jaxpr = make_two_input_jaxpr(Primitive::Div, BTreeMap::new());
    let jvp_result = fj_ad::jvp(&jaxpr, &[a, b], &[da, db]).unwrap();

    // tangent with db=0: da/b = (1+0i)/(1+i) = (1-i)/2 = 0.5 - 0.5i
    let expected = complex_div(1.0, 0.0, 1.0, 1.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-5, "div JVP complex64 da only");
}

/// Complex64 Rsqrt JVP: d/dz rsqrt(z) = -1/(2*z^(3/2))
#[test]
fn rsqrt_jvp_numerical_complex64() {
    let z = make_complex64_scalar(3.0, 4.0);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Rsqrt);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // sqrt(3+4i) = 2+i, rsqrt = 1/(2+i) = (2-i)/5 = 0.4 - 0.2i
    // d/dz rsqrt(z) = -1/(2*z*sqrt(z)) = -rsqrt/(2z)
    // = -(0.4-0.2i)/(2*(3+4i)) = -(0.4-0.2i)/(6+8i)
    let rsqrt = complex_div(1.0, 0.0, 2.0, 1.0);
    let two_z = (6.0, 8.0);
    let neg_rsqrt = complex_neg(rsqrt.0, rsqrt.1);
    let expected = complex_div(neg_rsqrt.0, neg_rsqrt.1, two_z.0, two_z.1);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-5, "rsqrt JVP at z=3+4i");
}

/// Complex64 Cbrt JVP: d/dz cbrt(z) = 1/(3*cbrt(z)²)
#[test]
fn cbrt_jvp_numerical_complex64() {
    let z = make_complex64_scalar(8.0, 0.0);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Cbrt);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // cbrt(8) = 2, d/dz cbrt(z) = 1/(3*z^(2/3)) = 1/(3*4) = 1/12 ≈ 0.0833
    let expected = (1.0 / 12.0, 0.0);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "cbrt JVP at z=8");
}

/// Complex64 Asinh JVP: d/dz asinh(z) = 1/sqrt(1+z²)
#[test]
fn asinh_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.5, 0.3);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Asinh);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // z² = (0.5+0.3i)² = 0.25 - 0.09 + 0.3i = 0.16 + 0.3i
    // 1+z² = 1.16 + 0.3i
    // sqrt(1.16+0.3i) ≈ compute numerically
    let z_sq = complex_mul(0.5, 0.3, 0.5, 0.3);
    let one_plus_z_sq = (1.0 + z_sq.0, z_sq.1);
    // Use numerical approximation for sqrt
    let r = (one_plus_z_sq.0 * one_plus_z_sq.0 + one_plus_z_sq.1 * one_plus_z_sq.1).sqrt();
    let sqrt_re = ((r + one_plus_z_sq.0) / 2.0).sqrt();
    let sqrt_im = one_plus_z_sq.1.signum() * ((r - one_plus_z_sq.0) / 2.0).sqrt();
    let expected = complex_div(1.0, 0.0, sqrt_re, sqrt_im);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "asinh JVP at z=0.5+0.3i");
}

/// Complex64 Acosh JVP: d/dz acosh(z) = 1/sqrt(z²-1)
#[test]
fn acosh_jvp_numerical_complex64() {
    let z = make_complex64_scalar(2.0, 0.5);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Acosh);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // z² = (2+0.5i)² = 4 - 0.25 + 2i = 3.75 + 2i
    // z²-1 = 2.75 + 2i
    let z_sq = complex_mul(2.0, 0.5, 2.0, 0.5);
    let z_sq_minus_1 = (z_sq.0 - 1.0, z_sq.1);
    // Numerical sqrt
    let r = (z_sq_minus_1.0 * z_sq_minus_1.0 + z_sq_minus_1.1 * z_sq_minus_1.1).sqrt();
    let sqrt_re = ((r + z_sq_minus_1.0) / 2.0).sqrt();
    let sqrt_im = z_sq_minus_1.1.signum() * ((r - z_sq_minus_1.0) / 2.0).sqrt();
    let expected = complex_div(1.0, 0.0, sqrt_re, sqrt_im);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "acosh JVP at z=2+0.5i");
}

/// Complex64 Atanh JVP: d/dz atanh(z) = 1/(1-z²)
#[test]
fn atanh_jvp_numerical_complex64() {
    let z = make_complex64_scalar(0.3, 0.2);
    let dz = make_complex64_scalar(1.0, 0.0);

    let jaxpr = make_single_op_jaxpr(Primitive::Atanh);
    let jvp_result = fj_ad::jvp(&jaxpr, &[z], &[dz]).unwrap();

    // z² = (0.3+0.2i)² = 0.09 - 0.04 + 0.12i = 0.05 + 0.12i
    // 1-z² = 0.95 - 0.12i
    let z_sq = complex_mul(0.3, 0.2, 0.3, 0.2);
    let one_minus_z_sq = (1.0 - z_sq.0, -z_sq.1);
    let expected = complex_div(1.0, 0.0, one_minus_z_sq.0, one_minus_z_sq.1);
    let actual = extract_complex64_scalar(&jvp_result.tangents[0]);
    assert_complex64_close(actual, expected, 1e-4, "atanh JVP at z=0.3+0.2i");
}

// ── FFT JVP tests ──
//
// FFT and IFFT are linear operations, so their JVP is simply the
// operation applied to the tangent: JVP of FFT(x) with tangent dx is FFT(dx).

/// FFT JVP: since FFT is linear, JVP tangent = FFT(tangent input)
#[test]
fn fft_jvp_numerical() {
    let x = make_complex128_vector(&[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
    let dx = make_complex128_vector(&[(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0)]);

    let jaxpr = make_single_op_jaxpr(Primitive::Fft);
    let jvp_result = fj_ad::jvp(&jaxpr, &[x], &[dx.clone()]).unwrap();

    // For linear FFT: tangent = FFT(dx)
    let expected_tangent = eval_primitive_multi(Primitive::Fft, &[dx], &BTreeMap::new()).unwrap();

    let actual = extract_complex_vec(&jvp_result.tangents[0]);
    let expected = extract_complex_vec(&expected_tangent[0]);

    assert_complex_close(&actual, &expected, 1e-10, "FFT JVP tangent = FFT(dx)");
}

/// IFFT JVP: since IFFT is linear, JVP tangent = IFFT(tangent input)
#[test]
fn ifft_jvp_numerical() {
    let x = make_complex128_vector(&[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)]);
    let dx = make_complex128_vector(&[(1.0, 0.0), (0.5, 0.5), (0.0, 1.0), (0.5, -0.5)]);

    let jaxpr = make_single_op_jaxpr(Primitive::Ifft);
    let jvp_result = fj_ad::jvp(&jaxpr, &[x], &[dx.clone()]).unwrap();

    // For linear IFFT: tangent = IFFT(dx)
    let expected_tangent = eval_primitive_multi(Primitive::Ifft, &[dx], &BTreeMap::new()).unwrap();

    let actual = extract_complex_vec(&jvp_result.tangents[0]);
    let expected = extract_complex_vec(&expected_tangent[0]);

    assert_complex_close(&actual, &expected, 1e-10, "IFFT JVP tangent = IFFT(dx)");
}

/// FFT JVP preserves Complex64 dtype
#[test]
fn fft_jvp_complex64_preserves_dtype() {
    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            (0..4)
                .map(|i| Literal::from_complex64(i as f32, 0.0))
                .collect(),
        )
        .unwrap(),
    );
    let dx = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            (0..4)
                .map(|i| Literal::from_complex64(0.1 * (i + 1) as f32, 0.0))
                .collect(),
        )
        .unwrap(),
    );

    let jaxpr = make_single_op_jaxpr(Primitive::Fft);
    let jvp_result = fj_ad::jvp(&jaxpr, &[x], &[dx]).unwrap();

    assert_eq!(
        jvp_result.tangents[0].dtype(),
        DType::Complex64,
        "FFT JVP should preserve Complex64 dtype"
    );
}

/// IFFT JVP preserves Complex64 dtype
#[test]
fn ifft_jvp_complex64_preserves_dtype() {
    let x = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            (0..4)
                .map(|i| Literal::from_complex64(i as f32 + 1.0, 0.0))
                .collect(),
        )
        .unwrap(),
    );
    let dx = Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape { dims: vec![4] },
            (0..4)
                .map(|i| Literal::from_complex64(0.1 * (i + 1) as f32, 0.0))
                .collect(),
        )
        .unwrap(),
    );

    let jaxpr = make_single_op_jaxpr(Primitive::Ifft);
    let jvp_result = fj_ad::jvp(&jaxpr, &[x], &[dx]).unwrap();

    assert_eq!(
        jvp_result.tangents[0].dtype(),
        DType::Complex64,
        "IFFT JVP should preserve Complex64 dtype"
    );
}

#[test]
fn hypot_logaddexp_jvp_numerical() {
    // Forward-mode duals of the (previously untested) Hypot/LogAddExp/LogAddExp2 VJPs.
    verify_binary_scalar_jvp(Primitive::Hypot, 3.0, 4.0, 1.0, 0.0, 1e-4, "hypot JVP d/dx @ (3,4)");
    verify_binary_scalar_jvp(Primitive::Hypot, 3.0, 4.0, 0.0, 1.0, 1e-4, "hypot JVP d/dy @ (3,4)");
    verify_binary_scalar_jvp(Primitive::LogAddExp, 1.0, 0.0, 1.0, 0.0, 1e-4, "logaddexp JVP d/dx");
    verify_binary_scalar_jvp(Primitive::LogAddExp, 1.0, 0.0, 0.0, 1.0, 1e-4, "logaddexp JVP d/dy");
    verify_binary_scalar_jvp(
        Primitive::LogAddExp2,
        1.0,
        0.0,
        1.0,
        0.0,
        1e-4,
        "logaddexp2 JVP d/dx",
    );
}

#[test]
fn integer_pow_jvp_zero_exponent_and_general() {
    // Dual of the IntegerPow VJP test (the JVP rule had no forward-mode coverage).
    // x^0 is constant -> tangent 0 even at x=0 (the n==0 guard avoids 0*inf=NaN, matching
    // JAX's _integer_pow_jvp); general tangent = n*x^(n-1)*dx.
    let jvp_ip = |x: f64, exp: i64, dx: f64| -> f64 {
        let mut params = BTreeMap::new();
        params.insert("exponent".to_string(), exp.to_string());
        let jaxpr = make_single_input_jaxpr(Primitive::IntegerPow, params);
        let r = fj_ad::jvp(&jaxpr, &[Value::scalar_f64(x)], &[Value::scalar_f64(dx)]).unwrap();
        extract_f64_scalar(&r.tangents[0])
    };
    assert_eq!(jvp_ip(0.0, 0, 1.0), 0.0, "d(x^0) = 0 at x=0 (n==0 guard)");
    assert!(
        (jvp_ip(2.0, 3, 1.0) - 12.0).abs() < 1e-9,
        "d(x^3) = 3x^2 = 12 at x=2"
    );
    assert!(
        (jvp_ip(2.0, 3, 0.5) - 6.0).abs() < 1e-9,
        "tangent scaling: 12 * 0.5 = 6"
    );
    assert!(
        (jvp_ip(2.0, -1, 1.0) + 0.25).abs() < 1e-9,
        "d(x^-1) = -1/x^2 = -0.25 at x=2"
    );
}
