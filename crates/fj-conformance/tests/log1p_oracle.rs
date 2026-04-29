//! Oracle tests for Log1p primitive.
//!
//! log1p(x) = log(1 + x)
//!
//! This function is numerically stable for small x, avoiding loss of precision
//! when computing log(1 + x) directly for small x.
//!
//! Tests:
//! - Zero: log1p(0) = 0
//! - Positive values
//! - Negative values (x > -1)
//! - Domain: log1p(x) is defined for x > -1
//! - log1p(-1) = -infinity
//! - log1p(x < -1) = NaN
//! - Infinity: log1p(+inf) = +inf
//! - NaN propagation
//! - Inverse relationship with expm1

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
fn oracle_log1p_zero() {
    // log1p(0) = log(1) = 0
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "log1p(0) = 0");
}

#[test]
fn oracle_log1p_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "log1p(-0.0) = 0");
}

// ======================== Positive Values ========================

#[test]
fn oracle_log1p_one() {
    // log1p(1) = log(2) ≈ 0.693147
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::LN_2,
        1e-14,
        "log1p(1)",
    );
}

#[test]
fn oracle_log1p_e_minus_one() {
    // log1p(e-1) = log(e) = 1
    let input = make_f64_tensor(&[], vec![std::f64::consts::E - 1.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "log1p(e-1)");
}

#[test]
fn oracle_log1p_two() {
    // log1p(2) = log(3)
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 3.0_f64.ln(), 1e-14, "log1p(2)");
}

#[test]
fn oracle_log1p_ten() {
    // log1p(10) = log(11)
    let input = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        11.0_f64.ln(),
        1e-14,
        "log1p(10)",
    );
}

#[test]
fn oracle_log1p_hundred() {
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        101.0_f64.ln(),
        1e-14,
        "log1p(100)",
    );
}

// ======================== Negative Values (valid domain: x > -1) ========================

#[test]
fn oracle_log1p_neg_half() {
    // log1p(-0.5) = log(0.5) = -ln(2)
    let input = make_f64_tensor(&[], vec![-0.5]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        -std::f64::consts::LN_2,
        1e-14,
        "log1p(-0.5)",
    );
}

#[test]
fn oracle_log1p_neg_small() {
    // log1p(-0.1) = log(0.9)
    let input = make_f64_tensor(&[], vec![-0.1]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.9_f64.ln(),
        1e-14,
        "log1p(-0.1)",
    );
}

#[test]
fn oracle_log1p_neg_point_nine() {
    // log1p(-0.9) = log(0.1) = -ln(10)
    let input = make_f64_tensor(&[], vec![-0.9]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        0.1_f64.ln(),
        1e-12,
        "log1p(-0.9)",
    );
}

// ======================== Small Values (Numerical Stability) ========================

#[test]
fn oracle_log1p_small() {
    // For small x, log1p(x) ≈ x (first-order Taylor expansion)
    let input = make_f64_tensor(&[], vec![1e-10]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1e-10, 1e-20, "log1p(1e-10) ≈ 1e-10");
}

#[test]
fn oracle_log1p_very_small() {
    let input = make_f64_tensor(&[], vec![1e-15]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1e-15, 1e-29, "log1p(1e-15) ≈ 1e-15");
}

#[test]
fn oracle_log1p_neg_small_value() {
    let input = make_f64_tensor(&[], vec![-1e-10]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, -1e-10, 1e-20, "log1p(-1e-10) ≈ -1e-10");
}

// ======================== Boundary: x = -1 ========================

#[test]
fn oracle_log1p_negative_one() {
    // log1p(-1) = log(0) = -infinity
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val < 0.0, "log1p(-1) = -inf");
}

// ======================== Out of Domain: x < -1 ========================

#[test]
fn oracle_log1p_out_of_domain() {
    // log1p(x) = NaN for x < -1
    for x in [-1.1, -2.0, -10.0, -100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
        assert!(
            extract_f64_scalar(&result).is_nan(),
            "log1p({}) should be NaN",
            x
        );
    }
}

// ======================== Infinity ========================

#[test]
fn oracle_log1p_pos_infinity() {
    // log1p(+inf) = +inf
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val.is_infinite() && val > 0.0, "log1p(+inf) = +inf");
}

#[test]
fn oracle_log1p_neg_infinity() {
    // log1p(-inf) = NaN (out of domain)
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "log1p(-inf) = NaN"
    );
}

// ======================== NaN ========================

#[test]
fn oracle_log1p_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "log1p(NaN) = NaN");
}

// ======================== Sign of output ========================

#[test]
fn oracle_log1p_sign() {
    // log1p(x) has the same sign as x for x in (-1, inf)
    for x in [0.1, 0.5, 1.0, 2.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
        assert!(
            extract_f64_scalar(&result) > 0.0,
            "log1p({}) should be positive",
            x
        );
    }

    for x in [-0.1, -0.5, -0.9] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
        assert!(
            extract_f64_scalar(&result) < 0.0,
            "log1p({}) should be negative",
            x
        );
    }
}

// ======================== 1D Tensor ========================

#[test]
fn oracle_log1p_1d() {
    let input = make_f64_tensor(&[5], vec![-0.5, 0.0, 0.5, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], 0.5_f64.ln(), 1e-14, "log1p(-0.5)");
    assert_eq!(vals[1], 0.0, "log1p(0)");
    assert_close(vals[2], 1.5_f64.ln(), 1e-14, "log1p(0.5)");
    assert_close(vals[3], std::f64::consts::LN_2, 1e-14, "log1p(1)");
    assert_close(vals[4], 3.0_f64.ln(), 1e-14, "log1p(2)");
}

#[test]
fn oracle_log1p_1d_special() {
    let input = make_f64_tensor(&[4], vec![0.0, f64::INFINITY, -1.0, f64::NAN]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[0], 0.0, "log1p(0)");
    assert!(vals[1].is_infinite() && vals[1] > 0.0, "log1p(+inf)");
    assert!(vals[2].is_infinite() && vals[2] < 0.0, "log1p(-1)");
    assert!(vals[3].is_nan(), "log1p(NaN)");
}

// ======================== 2D Tensor ========================

#[test]
fn oracle_log1p_2d() {
    let input = make_f64_tensor(&[2, 3], vec![-0.5, 0.0, 1.0, 2.0, 4.0, 9.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);

    assert_close(vals[0], 0.5_f64.ln(), 1e-14, "log1p(-0.5)");
    assert_eq!(vals[1], 0.0, "log1p(0)");
    assert_close(vals[5], 10.0_f64.ln(), 1e-14, "log1p(9)");
}

// ======================== 3D Tensor ========================

#[test]
fn oracle_log1p_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 9.0]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);

    assert_eq!(vals[1], 0.0, "log1p(0)");
    assert_close(vals[7], 10.0_f64.ln(), 1e-14, "log1p(9)");
}

// ======================== Identity: log1p(x) = log(1+x) ========================

#[test]
fn oracle_log1p_identity() {
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
        let val = extract_f64_scalar(&result);
        let expected = (1.0 + x).ln();
        assert_close(val, expected, 1e-14, &format!("log1p({}) = log(1+{})", x, x));
    }
}

// ======================== Monotonicity ========================

#[test]
fn oracle_log1p_monotonic() {
    let inputs = vec![-0.9, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
    let input = make_f64_tensor(&[7], inputs);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "log1p should be monotonically increasing"
        );
    }
}

// ======================== Inverse relationship with expm1 ========================

#[test]
fn oracle_log1p_expm1_inverse() {
    // For x > -1: expm1(log1p(x)) = x
    for x in [-0.5, 0.0, 0.5, 1.0, 2.0, 10.0] {
        let input1 = make_f64_tensor(&[], vec![x]);
        let log1p_result = eval_primitive(Primitive::Log1p, &[input1], &no_params()).unwrap();
        let log1p_val = extract_f64_scalar(&log1p_result);

        let input2 = make_f64_tensor(&[], vec![log1p_val]);
        let expm1_result = eval_primitive(Primitive::Expm1, &[input2], &no_params()).unwrap();
        let roundtrip = extract_f64_scalar(&expm1_result);

        assert_close(roundtrip, x, 1e-14, &format!("expm1(log1p({})) = {}", x, x));
    }
}

// ======================== Very Large Values ========================

#[test]
fn oracle_log1p_very_large() {
    // For very large x, log1p(x) ≈ log(x)
    let input = make_f64_tensor(&[], vec![1e100]);
    let result = eval_primitive(Primitive::Log1p, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert_close(val, 1e100_f64.ln(), 1e-10, "log1p(1e100) ≈ log(1e100)");
}
