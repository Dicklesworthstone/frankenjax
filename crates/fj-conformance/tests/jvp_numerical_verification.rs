//! JVP (forward-mode AD) numerical verification for linalg primitives.
//!
//! Verifies tangent computation via finite differences:
//!   f'(x) · dx ≈ (f(x + ε·dx) - f(x - ε·dx)) / (2ε)

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
