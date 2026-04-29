//! Oracle tests for Exp primitive.
//!
//! exp(x) = e^x (exponential function)
//!
//! Properties:
//! - exp(0) = 1
//! - exp(1) = e ≈ 2.718281828
//! - exp(x + y) = exp(x) * exp(y)
//! - exp(-x) = 1 / exp(x)
//! - exp(ln(x)) = x for x > 0
//! - d/dx exp(x) = exp(x)
//!
//! Tests:
//! - Special values (0, 1, -1)
//! - Large/small values
//! - Negative values
//! - Mathematical properties
//! - Complex numbers
//! - Tensor shapes

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

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_complex_scalar(v: &Value) -> (f64, f64) {
    match v {
        Value::Tensor(t) => {
            assert!(t.shape.dims.is_empty(), "expected scalar");
            match &t.elements[0] {
                Literal::Complex128Bits(re, im) => (f64::from_bits(*re), f64::from_bits(*im)),
                _ => unreachable!("expected complex128"),
            }
        }
        Value::Scalar(Literal::Complex128Bits(re, im)) => {
            (f64::from_bits(*re), f64::from_bits(*im))
        }
        _ => unreachable!("expected complex128"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
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

fn assert_close_rel(actual: f64, expected: f64, rel_tol: f64, msg: &str) {
    let diff = (actual - expected).abs();
    let rel = diff / expected.abs().max(1e-100);
    assert!(
        rel < rel_tol,
        "{}: expected {}, got {}, rel_diff={}",
        msg,
        expected,
        actual,
        rel
    );
}

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_exp_zero() {
    // exp(0) = 1
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "exp(0) = 1");
}

#[test]
fn oracle_exp_one() {
    // exp(1) = e
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::E,
        1e-14,
        "exp(1) = e",
    );
}

#[test]
fn oracle_exp_neg_one() {
    // exp(-1) = 1/e
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1.0 / std::f64::consts::E,
        1e-14,
        "exp(-1) = 1/e",
    );
}

#[test]
fn oracle_exp_ln_2() {
    // exp(ln(2)) = 2
    let input = make_f64_tensor(&[], vec![std::f64::consts::LN_2]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 2.0, 1e-14, "exp(ln(2)) = 2");
}

#[test]
fn oracle_exp_ln_10() {
    // exp(ln(10)) = 10
    let input = make_f64_tensor(&[], vec![std::f64::consts::LN_10]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 10.0, 1e-13, "exp(ln(10)) = 10");
}

// ====================== LARGE/SMALL VALUES ======================

#[test]
fn oracle_exp_large_positive() {
    // exp(100) is very large
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val > 1e40, "exp(100) should be very large");
    assert!(val.is_finite(), "exp(100) should be finite");
}

#[test]
fn oracle_exp_large_negative() {
    // exp(-100) is very small but positive
    let input = make_f64_tensor(&[], vec![-100.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val > 0.0, "exp(-100) should be positive");
    assert!(val < 1e-40, "exp(-100) should be very small");
}

#[test]
fn oracle_exp_overflow() {
    // exp(1000) overflows to infinity
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "exp(1000) = inf"
    );
}

#[test]
fn oracle_exp_underflow() {
    // exp(-1000) underflows to 0
    let input = make_f64_tensor(&[], vec![-1000.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "exp(-1000) = 0");
}

// ====================== SPECIAL FLOAT VALUES ======================

#[test]
fn oracle_exp_pos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), f64::INFINITY, "exp(inf) = inf");
}

#[test]
fn oracle_exp_neg_infinity() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "exp(-inf) = 0");
}

#[test]
fn oracle_exp_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "exp(NaN) = NaN");
}

// ====================== ADDITIVITY PROPERTY ======================

#[test]
fn oracle_exp_additive() {
    // exp(x + y) = exp(x) * exp(y)
    let test_pairs = [(1.0, 2.0), (0.5, 0.5), (-1.0, 2.0), (0.0, 1.0)];
    for (x, y) in test_pairs {
        let sum = x + y;
        let exp_sum = extract_f64_scalar(
            &eval_primitive(
                Primitive::Exp,
                &[make_f64_tensor(&[], vec![sum])],
                &no_params(),
            )
            .unwrap(),
        );
        let exp_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Exp,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let exp_y = extract_f64_scalar(
            &eval_primitive(
                Primitive::Exp,
                &[make_f64_tensor(&[], vec![y])],
                &no_params(),
            )
            .unwrap(),
        );

        assert_close_rel(
            exp_sum,
            exp_x * exp_y,
            1e-14,
            &format!("exp({} + {}) = exp({}) * exp({})", x, y, x, y),
        );
    }
}

// ====================== INVERSE PROPERTY ======================

#[test]
fn oracle_exp_inverse() {
    // exp(-x) = 1 / exp(x)
    for x in [0.5, 1.0, 2.0, 3.0] {
        let exp_neg_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Exp,
                &[make_f64_tensor(&[], vec![-x])],
                &no_params(),
            )
            .unwrap(),
        );
        let exp_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Exp,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close_rel(
            exp_neg_x,
            1.0 / exp_x,
            1e-14,
            &format!("exp(-{}) = 1/exp({})", x, x),
        );
    }
}

// ====================== EXP-LOG RELATIONSHIP ======================

#[test]
fn oracle_exp_log_inverse() {
    // exp(log(x)) = x for x > 0
    for x in [0.5, 1.0, 2.0, 10.0, 100.0] {
        let log_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Log,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let exp_log_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Exp,
                &[make_f64_tensor(&[], vec![log_x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close_rel(exp_log_x, x, 1e-14, &format!("exp(log({})) = {}", x, x));
    }
}

// ====================== STRICTLY POSITIVE ======================

#[test]
fn oracle_exp_always_positive() {
    // exp(x) > 0 for all finite x
    for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let result = extract_f64_scalar(
            &eval_primitive(
                Primitive::Exp,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert!(
            result >= 0.0,
            "exp({}) = {} should be non-negative",
            x,
            result
        );
    }
}

// ====================== MONOTONICITY ======================

#[test]
fn oracle_exp_monotonic() {
    // exp is strictly increasing
    let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let input = make_f64_tensor(&[values.len() as u32], values);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "exp should be strictly increasing: exp[{}] = {} > exp[{}] = {}",
            i,
            vals[i],
            i - 1,
            vals[i - 1]
        );
    }
}

// ====================== COMPLEX NUMBERS ======================

#[test]
fn oracle_exp_complex_pure_imag() {
    // exp(i*theta) = cos(theta) + i*sin(theta) (Euler's formula)
    let theta = std::f64::consts::FRAC_PI_4; // π/4
    let input = make_complex128_tensor(&[], vec![(0.0, theta)]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, theta.cos(), 1e-14, "exp(i*π/4) real part");
    assert_close(im, theta.sin(), 1e-14, "exp(i*π/4) imag part");
}

#[test]
fn oracle_exp_eulers_identity() {
    // exp(i*π) = -1 (Euler's identity)
    let input = make_complex128_tensor(&[], vec![(0.0, std::f64::consts::PI)]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    assert_close(re, -1.0, 1e-14, "exp(i*π) = -1 (real part)");
    assert_close(im, 0.0, 1e-14, "exp(i*π) = -1 (imag part)");
}

#[test]
fn oracle_exp_complex_general() {
    // exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
    let a = 1.0;
    let b = std::f64::consts::FRAC_PI_3; // π/3
    let input = make_complex128_tensor(&[], vec![(a, b)]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    let (re, im) = extract_complex_scalar(&result);
    let exp_a = std::f64::consts::E;
    assert_close(re, exp_a * b.cos(), 1e-14, "exp(1 + i*π/3) real part");
    assert_close(im, exp_a * b.sin(), 1e-14, "exp(1 + i*π/3) imag part");
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_exp_1d() {
    let input = make_f64_tensor(&[5], vec![-1.0, 0.0, 1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    let e = std::f64::consts::E;
    assert_close(vals[0], 1.0 / e, 1e-14, "exp(-1)");
    assert_close(vals[1], 1.0, 1e-14, "exp(0)");
    assert_close(vals[2], e, 1e-14, "exp(1)");
    assert_close(vals[3], e * e, 1e-14, "exp(2)");
    assert_close(vals[4], e * e * e, 1e-14, "exp(3)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_exp_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 2.0]);
    let result = eval_primitive(Primitive::Exp, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    let e = std::f64::consts::E;
    assert_close(vals[0], 1.0, 1e-14, "");
    assert_close(vals[1], e, 1e-14, "");
    assert_close(vals[2], 1.0 / e, 1e-14, "");
    assert_close(vals[3], e * e, 1e-14, "");
}
