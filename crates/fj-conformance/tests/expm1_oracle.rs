//! Oracle tests for Expm1 primitive.
//!
//! expm1(x) = e^x - 1
//!
//! This function is numerically stable for small x, avoiding catastrophic
//! cancellation when computing e^x - 1 directly.
//!
//! Tests:
//! - Zero: expm1(0) = 0
//! - Positive values
//! - Negative values → approaches -1
//! - Small values: expm1(x) ≈ x for small x
//! - Infinity: expm1(+inf) = +inf, expm1(-inf) = -1
//! - NaN propagation
//! - Identity: expm1(x) = e^x - 1

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn make_f64_tensor(shape: &[u32], data: Vec<f64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff={}",
        msg,
        expected,
        actual,
        (actual - expected).abs()
    );
}

// ======================== Zero ========================

#[test]
fn oracle_expm1_zero() {
    // expm1(0) = e^0 - 1 = 0
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "expm1(0) = 0");
}

#[test]
fn oracle_expm1_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "expm1(-0.0) = 0");
}

// ======================== Positive Values ========================

#[test]
fn oracle_expm1_one() {
    // expm1(1) = e - 1 ≈ 1.71828
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::E - 1.0,
        1e-14,
        "expm1(1)",
    );
}

#[test]
fn oracle_expm1_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        2.0_f64.exp() - 1.0,
        1e-14,
        "expm1(2)",
    );
}

#[test]
fn oracle_expm1_ten() {
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        10.0_f64.exp() - 1.0,
        1e-6,
        "expm1(10)",
    );
}

#[test]
fn oracle_expm1_ln2() {
    // expm1(ln(2)) = 2 - 1 = 1
    let input = make_f64_tensor(&[], vec![std::f64::consts::LN_2]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "expm1(ln(2))");
}

// ======================== Negative Values ========================

#[test]
fn oracle_expm1_neg_one() {
    // expm1(-1) = e^(-1) - 1 = 1/e - 1 ≈ -0.6321
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-1.0_f64).exp() - 1.0,
        1e-14,
        "expm1(-1)",
    );
}

#[test]
fn oracle_expm1_neg_two() {
    let input = make_f64_tensor(&[], vec![-2.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        (-2.0_f64).exp() - 1.0,
        1e-14,
        "expm1(-2)",
    );
}

#[test]
fn oracle_expm1_neg_ten() {
    // expm1(-10) ≈ -1 (approaches -1 as x → -inf)
    let input = make_f64_tensor(&[], vec![-10.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, (-10.0_f64).exp() - 1.0, 1e-14, "expm1(-10)");
    assert!(val > -1.0, "expm1(-10) should be > -1");
}

#[test]
fn oracle_expm1_large_negative() {
    // expm1(-100) ≈ -1 (very close to -1)
    let input = make_f64_tensor(&[], vec![-100.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, -1.0, 1e-14, "expm1(-100) ≈ -1");
}

// ======================== Small Values (Numerical Stability) ========================

#[test]
fn oracle_expm1_small() {
    // For small x, expm1(x) ≈ x (first-order Taylor expansion)
    let input = make_f64_tensor(&[], vec![1e-10]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1e-10, 1e-20, "expm1(1e-10) ≈ 1e-10");
}

#[test]
fn oracle_expm1_very_small() {
    let input = make_f64_tensor(&[], vec![1e-15]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1e-15, 1e-29, "expm1(1e-15) ≈ 1e-15");
}

#[test]
fn oracle_expm1_neg_small() {
    let input = make_f64_tensor(&[], vec![-1e-10]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, -1e-10, 1e-20, "expm1(-1e-10) ≈ -1e-10");
}

// ======================== Infinity ========================

#[test]
fn oracle_expm1_pos_infinity() {
    // expm1(+inf) = +inf
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "expm1(+inf) = +inf");
}

#[test]
fn oracle_expm1_neg_infinity() {
    // expm1(-inf) = -1 (e^(-inf) = 0, so 0 - 1 = -1)
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "expm1(-inf) = -1");
}

// ======================== NaN ========================

#[test]
fn oracle_expm1_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "expm1(NaN) = NaN");
}

// ======================== Bounds: expm1(x) >= -1 for all finite x ========================

#[test]
fn oracle_expm1_bounds() {
    for x in [-10.0, -5.0, -2.0, -1.0, -0.5] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        assert!(val > -1.0, "expm1({}) should be > -1", x);
    }
}

// ======================== Sign preservation ========================

#[test]
fn oracle_expm1_sign() {
    // expm1(x) has the same sign as x
    for x in [0.1, 0.5, 1.0, 2.0] {
        let input_pos = make_f64_tensor(&[], vec![x]);
        let input_neg = make_f64_tensor(&[], vec![-x]);

        let result_pos = eval_primitive(Primitive::Expm1, &[input_pos], &no_params()).unwrap();
        let result_neg = eval_primitive(Primitive::Expm1, &[input_neg], &no_params()).unwrap();

        assert!(
            extract_f64_scalar(&result_pos) > 0.0,
            "expm1({}) should be positive",
            x
        );
        assert!(
            extract_f64_scalar(&result_neg) < 0.0,
            "expm1(-{}) should be negative",
            x
        );
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_expm1_1d() {
    let input = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-2.0_f64).exp() - 1.0, 1e-14, "expm1(-2)");
    assert_close(vals[1], (-1.0_f64).exp() - 1.0, 1e-14, "expm1(-1)");
    assert_eq!(vals[2], 0.0, "expm1(0)");
    assert_close(vals[3], 1.0_f64.exp() - 1.0, 1e-14, "expm1(1)");
    assert_close(vals[4], 2.0_f64.exp() - 1.0, 1e-14, "expm1(2)");
}

#[test]
fn oracle_expm1_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "expm1(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "expm1(+inf)");
    assert_eq!(vals[2], -1.0, "expm1(-inf)");
    assert!(vals[3].is_nan(), "expm1(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_expm1_2d() {
    let input = make_f64_tensor(&[2, 3], vec![-1.0, 0.0, 1.0, -0.5, 0.5, 2.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], (-1.0_f64).exp() - 1.0, 1e-14, "expm1(-1)");
    assert_eq!(vals[1], 0.0, "expm1(0)");
    assert_close(vals[2], 1.0_f64.exp() - 1.0, 1e-14, "expm1(1)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_expm1_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[3], 0.0, "expm1(0)");
    assert_close(vals[7], 2.0_f64.exp() - 1.0, 1e-14, "expm1(2)");
}

// ======================== Identity: expm1(x) = e^x - 1 ========================

#[test]
fn oracle_expm1_identity() {
    for x in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = x.exp() - 1.0;
        assert_close(val, expected, 1e-13, &format!("expm1({}) = e^{} - 1", x, x));
    }
}

// ======================== Monotonicity ========================

#[test]
fn oracle_expm1_monotonic() {
    let inputs = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let input = make_f64_tensor(&[7], inputs);
    let result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "expm1 should be monotonically increasing"
        );
    }
}

// ======================== Inverse relationship with log1p ========================

#[test]
fn oracle_expm1_log1p_inverse() {
    // For x > -1: log1p(expm1(x)) = x
    // For y > 0: expm1(log1p(y)) ≈ y (when y not too large)
    for x in [-0.5, 0.0, 0.5, 1.0, 2.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let expm1_result = eval_primitive(Primitive::Expm1, &[input1], &no_params()).unwrap();
        let expm1_val = extract_f64_scalar(&expm1_result);

        let input2 = make_f64_tensor(&[], vec![expm1_val]);
        let log1p_result = eval_primitive(Primitive::Log1p, &[input2], &no_params()).unwrap();
        let roundtrip = extract_f64_scalar(&log1p_result);

        assert_close(roundtrip, x, 1e-14, &format!("log1p(expm1({})) = {}", x, x));
    }
}

// ======================== METAMORPHIC: expm1(log1p(x)) = x ========================

#[test]
fn metamorphic_expm1_log1p_identity() {
    // expm1(log1p(x)) = x for x > -1
    // This tests the OTHER direction of the inverse relationship
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let log1p_result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
        let expm1_log1p = eval_primitive(Primitive::Expm1, &[log1p_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&expm1_log1p),
            x,
            1e-12,
            &format!("expm1(log1p({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: log1p(expm1(x)) = x (relabeled for consistency) ========================

#[test]
fn metamorphic_log1p_expm1_identity() {
    // log1p(expm1(x)) = x for all finite x
    for x in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let expm1_result = eval_primitive(Primitive::Expm1, &[input], &no_params()).unwrap();
        let log1p_expm1 = eval_primitive(Primitive::Log1p, &[expm1_result], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&log1p_expm1),
            x,
            1e-12,
            &format!("log1p(expm1({})) = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_expm1_log1p_tensor_roundtrip() {
    // For a tensor: expm1(log1p(x)) = x for x > -1
    let input = make_f64_tensor(&[6], vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0]);
    let log1p_result = eval_primitive(Primitive::Log1p, &[input.clone()], &no_params()).unwrap();
    let roundtrip = eval_primitive(Primitive::Expm1, &[log1p_result], &no_params()).unwrap();

    let original = extract_f64_vec(&input);
    let recovered = extract_f64_vec(&roundtrip);

    for (orig, rec) in original.iter().zip(recovered.iter()) {
        assert_close(*rec, *orig, 1e-12, &format!("expm1(log1p({})) = {}", orig, orig));
    }
}
