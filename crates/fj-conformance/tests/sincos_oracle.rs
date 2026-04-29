//! Oracle tests for Sin and Cos primitives.
//!
//! sin(x), cos(x) - Trigonometric functions
//!
//! Properties:
//! - sin(0) = 0, cos(0) = 1
//! - sin(π/2) = 1, cos(π/2) = 0
//! - sin(π) = 0, cos(π) = -1
//! - sin^2(x) + cos^2(x) = 1 (Pythagorean identity)
//! - sin(-x) = -sin(x) (odd), cos(-x) = cos(x) (even)
//! - sin(x + 2π) = sin(x) (period 2π)
//!
//! Tests:
//! - Special values at multiples of π/6
//! - Symmetry properties
//! - Pythagorean identity
//! - Periodicity
//! - NaN/infinity propagation
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
        _ => panic!("expected tensor"),
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

// ====================== SIN SPECIAL VALUES ======================

#[test]
fn oracle_sin_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "sin(0) = 0");
}

#[test]
fn oracle_sin_pi_over_6() {
    // sin(π/6) = 0.5
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_6]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "sin(π/6) = 0.5");
}

#[test]
fn oracle_sin_pi_over_4() {
    // sin(π/4) = sqrt(2)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_4]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_1_SQRT_2,
        1e-14,
        "sin(π/4)",
    );
}

#[test]
fn oracle_sin_pi_over_3() {
    // sin(π/3) = sqrt(3)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_3]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0_f64.sqrt() / 2.0,
        1e-14,
        "sin(π/3)",
    );
}

#[test]
fn oracle_sin_pi_over_2() {
    // sin(π/2) = 1
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_2]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-14, "sin(π/2) = 1");
}

#[test]
fn oracle_sin_pi() {
    // sin(π) = 0
    let input = make_f64_tensor(&[], vec![std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-14, "sin(π) = 0");
}

#[test]
fn oracle_sin_3pi_over_2() {
    // sin(3π/2) = -1
    let input = make_f64_tensor(&[], vec![3.0 * std::f64::consts::FRAC_PI_2]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -1.0, 1e-14, "sin(3π/2) = -1");
}

#[test]
fn oracle_sin_2pi() {
    // sin(2π) = 0
    let input = make_f64_tensor(&[], vec![2.0 * std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-13, "sin(2π) = 0");
}

// ====================== COS SPECIAL VALUES ======================

#[test]
fn oracle_cos_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "cos(0) = 1");
}

#[test]
fn oracle_cos_pi_over_6() {
    // cos(π/6) = sqrt(3)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_6]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0_f64.sqrt() / 2.0,
        1e-14,
        "cos(π/6)",
    );
}

#[test]
fn oracle_cos_pi_over_4() {
    // cos(π/4) = sqrt(2)/2
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_4]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::FRAC_1_SQRT_2,
        1e-14,
        "cos(π/4)",
    );
}

#[test]
fn oracle_cos_pi_over_3() {
    // cos(π/3) = 0.5
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_3]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.5, 1e-14, "cos(π/3) = 0.5");
}

#[test]
fn oracle_cos_pi_over_2() {
    // cos(π/2) = 0
    let input = make_f64_tensor(&[], vec![std::f64::consts::FRAC_PI_2]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 0.0, 1e-14, "cos(π/2) = 0");
}

#[test]
fn oracle_cos_pi() {
    // cos(π) = -1
    let input = make_f64_tensor(&[], vec![std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), -1.0, 1e-14, "cos(π) = -1");
}

#[test]
fn oracle_cos_2pi() {
    // cos(2π) = 1
    let input = make_f64_tensor(&[], vec![2.0 * std::f64::consts::PI]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_close(extract_f64_scalar(&result), 1.0, 1e-13, "cos(2π) = 1");
}

// ====================== SIN ODD FUNCTION ======================

#[test]
fn oracle_sin_odd() {
    // sin(-x) = -sin(x)
    for x in [0.1, 0.5, 1.0, std::f64::consts::FRAC_PI_4, 2.0] {
        let pos = extract_f64_scalar(
            &eval_primitive(Primitive::Sin, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let neg = extract_f64_scalar(
            &eval_primitive(Primitive::Sin, &[make_f64_tensor(&[], vec![-x])], &no_params()).unwrap(),
        );
        assert_close(neg, -pos, 1e-14, &format!("sin(-{}) = -sin({})", x, x));
    }
}

// ====================== COS EVEN FUNCTION ======================

#[test]
fn oracle_cos_even() {
    // cos(-x) = cos(x)
    for x in [0.1, 0.5, 1.0, std::f64::consts::FRAC_PI_4, 2.0] {
        let pos = extract_f64_scalar(
            &eval_primitive(Primitive::Cos, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let neg = extract_f64_scalar(
            &eval_primitive(Primitive::Cos, &[make_f64_tensor(&[], vec![-x])], &no_params()).unwrap(),
        );
        assert_close(neg, pos, 1e-14, &format!("cos(-{}) = cos({})", x, x));
    }
}

// ====================== PYTHAGOREAN IDENTITY ======================

#[test]
fn oracle_sin_cos_pythagorean() {
    // sin^2(x) + cos^2(x) = 1
    for x in [0.0, 0.5, 1.0, std::f64::consts::PI, 2.5, -1.0] {
        let sin_x = extract_f64_scalar(
            &eval_primitive(Primitive::Sin, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let cos_x = extract_f64_scalar(
            &eval_primitive(Primitive::Cos, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let sum = sin_x * sin_x + cos_x * cos_x;
        assert_close(sum, 1.0, 1e-14, &format!("sin^2({}) + cos^2({}) = 1", x, x));
    }
}

// ====================== PERIODICITY ======================

#[test]
fn oracle_sin_periodic() {
    // sin(x + 2π) = sin(x)
    for x in [0.0, 0.5, 1.0, 2.0] {
        let sin_x = extract_f64_scalar(
            &eval_primitive(Primitive::Sin, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let sin_x_2pi = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x + 2.0 * std::f64::consts::PI])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(sin_x_2pi, sin_x, 1e-13, &format!("sin({} + 2π) = sin({})", x, x));
    }
}

#[test]
fn oracle_cos_periodic() {
    // cos(x + 2π) = cos(x)
    for x in [0.0, 0.5, 1.0, 2.0] {
        let cos_x = extract_f64_scalar(
            &eval_primitive(Primitive::Cos, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let cos_x_2pi = extract_f64_scalar(
            &eval_primitive(
                Primitive::Cos,
                &[make_f64_tensor(&[], vec![x + 2.0 * std::f64::consts::PI])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(cos_x_2pi, cos_x, 1e-13, &format!("cos({} + 2π) = cos({})", x, x));
    }
}

// ====================== PHASE SHIFT ======================

#[test]
fn oracle_cos_sin_phase_shift() {
    // cos(x) = sin(x + π/2)
    for x in [0.0, 0.5, 1.0, -0.5] {
        let cos_x = extract_f64_scalar(
            &eval_primitive(Primitive::Cos, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        let sin_x_plus = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sin,
                &[make_f64_tensor(&[], vec![x + std::f64::consts::FRAC_PI_2])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(cos_x, sin_x_plus, 1e-14, &format!("cos({}) = sin({} + π/2)", x, x));
    }
}

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_sin_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sin(NaN) = NaN");
}

#[test]
fn oracle_cos_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "cos(NaN) = NaN");
}

#[test]
fn oracle_sin_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sin(inf) = NaN");
}

#[test]
fn oracle_cos_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "cos(inf) = NaN");
}

// ====================== RANGE VERIFICATION ======================

#[test]
fn oracle_sin_range() {
    // sin(x) is always in [-1, 1]
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, -1.0, -5.0] {
        let val = extract_f64_scalar(
            &eval_primitive(Primitive::Sin, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        assert!(
            val >= -1.0 && val <= 1.0,
            "sin({}) = {} should be in [-1, 1]",
            x,
            val
        );
    }
}

#[test]
fn oracle_cos_range() {
    // cos(x) is always in [-1, 1]
    for x in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, -1.0, -5.0] {
        let val = extract_f64_scalar(
            &eval_primitive(Primitive::Cos, &[make_f64_tensor(&[], vec![x])], &no_params()).unwrap(),
        );
        assert!(
            val >= -1.0 && val <= 1.0,
            "cos({}) = {} should be in [-1, 1]",
            x,
            val
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_sin_1d() {
    let pi = std::f64::consts::PI;
    let input = make_f64_tensor(&[4], vec![0.0, pi / 6.0, pi / 2.0, pi]);
    let result = eval_primitive(Primitive::Sin, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 0.0, 1e-14, "sin(0)");
    assert_close(vals[1], 0.5, 1e-14, "sin(π/6)");
    assert_close(vals[2], 1.0, 1e-14, "sin(π/2)");
    assert_close(vals[3], 0.0, 1e-14, "sin(π)");
}

#[test]
fn oracle_cos_1d() {
    let pi = std::f64::consts::PI;
    let input = make_f64_tensor(&[4], vec![0.0, pi / 3.0, pi / 2.0, pi]);
    let result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert_close(vals[0], 1.0, 1e-14, "cos(0)");
    assert_close(vals[1], 0.5, 1e-14, "cos(π/3)");
    assert_close(vals[2], 0.0, 1e-14, "cos(π/2)");
    assert_close(vals[3], -1.0, 1e-14, "cos(π)");
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_sincos_2d() {
    let pi = std::f64::consts::PI;
    let input = make_f64_tensor(&[2, 2], vec![0.0, pi / 2.0, pi, 3.0 * pi / 2.0]);
    let sin_result = eval_primitive(Primitive::Sin, &[input.clone()], &no_params()).unwrap();
    let cos_result = eval_primitive(Primitive::Cos, &[input], &no_params()).unwrap();

    assert_eq!(extract_shape(&sin_result), vec![2, 2]);
    let sin_vals = extract_f64_vec(&sin_result);
    let cos_vals = extract_f64_vec(&cos_result);

    assert_close(sin_vals[0], 0.0, 1e-14, "sin(0)");
    assert_close(sin_vals[1], 1.0, 1e-14, "sin(π/2)");
    assert_close(sin_vals[2], 0.0, 1e-14, "sin(π)");
    assert_close(sin_vals[3], -1.0, 1e-14, "sin(3π/2)");

    assert_close(cos_vals[0], 1.0, 1e-14, "cos(0)");
    assert_close(cos_vals[1], 0.0, 1e-14, "cos(π/2)");
    assert_close(cos_vals[2], -1.0, 1e-14, "cos(π)");
    assert_close(cos_vals[3], 0.0, 1e-14, "cos(3π/2)");
}
