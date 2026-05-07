//! Oracle tests for Sqrt primitive.
//!
//! sqrt(x) = √x (square root)
//!
//! Domain: [0, inf) for real numbers
//! Range: [0, inf)
//!
//! Properties:
//! - sqrt(0) = 0
//! - sqrt(1) = 1
//! - sqrt(x)^2 = x for x >= 0
//! - sqrt(x * y) = sqrt(x) * sqrt(y) for x, y >= 0
//! - sqrt(x^2) = |x|
//!
//! Tests:
//! - Perfect squares
//! - Special values
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

// ====================== PERFECT SQUARES ======================

#[test]
fn oracle_sqrt_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "sqrt(0) = 0");
}

#[test]
fn oracle_sqrt_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "sqrt(1) = 1");
}

#[test]
fn oracle_sqrt_four() {
    let input = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 2.0, "sqrt(4) = 2");
}

#[test]
fn oracle_sqrt_nine() {
    let input = make_f64_tensor(&[], vec![9.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 3.0, "sqrt(9) = 3");
}

#[test]
fn oracle_sqrt_sixteen() {
    let input = make_f64_tensor(&[], vec![16.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 4.0, "sqrt(16) = 4");
}

#[test]
fn oracle_sqrt_hundred() {
    let input = make_f64_tensor(&[], vec![100.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 10.0, "sqrt(100) = 10");
}

// ====================== IRRATIONAL VALUES ======================

#[test]
fn oracle_sqrt_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        std::f64::consts::SQRT_2,
        1e-14,
        "sqrt(2)",
    );
}

#[test]
fn oracle_sqrt_three() {
    let input = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        3.0_f64.sqrt(),
        1e-14,
        "sqrt(3)",
    );
}

// ====================== SPECIAL VALUES ======================

#[test]
fn oracle_sqrt_infinity() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "sqrt(inf) = inf"
    );
}

#[test]
fn oracle_sqrt_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sqrt(NaN) = NaN");
}

#[test]
fn oracle_sqrt_negative() {
    // sqrt(-1) = NaN for real numbers
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sqrt(-1) = NaN");
}

// ====================== SQRT SQUARED IDENTITY ======================

#[test]
fn oracle_sqrt_squared() {
    // sqrt(x)^2 = x for x >= 0
    for x in [0.0, 0.25, 1.0, 2.0, 10.0, 100.0] {
        let sqrt_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sqrt,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let squared = sqrt_x * sqrt_x;
        assert_close(squared, x, 1e-14, &format!("sqrt({})^2 = {}", x, x));
    }
}

// ====================== MULTIPLICATIVITY ======================

#[test]
fn oracle_sqrt_multiplicative() {
    // sqrt(x * y) = sqrt(x) * sqrt(y) for x, y >= 0
    let test_pairs = [(4.0, 9.0), (2.0, 8.0), (1.0, 16.0), (0.25, 4.0)];
    for (x, y) in test_pairs {
        let sqrt_xy = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sqrt,
                &[make_f64_tensor(&[], vec![x * y])],
                &no_params(),
            )
            .unwrap(),
        );
        let sqrt_x = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sqrt,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let sqrt_y = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sqrt,
                &[make_f64_tensor(&[], vec![y])],
                &no_params(),
            )
            .unwrap(),
        );
        assert_close(
            sqrt_xy,
            sqrt_x * sqrt_y,
            1e-14,
            &format!("sqrt({} * {}) = sqrt({}) * sqrt({})", x, y, x, y),
        );
    }
}

// ====================== NON-NEGATIVITY ======================

#[test]
fn oracle_sqrt_non_negative() {
    // sqrt(x) >= 0 for all x >= 0
    for x in [0.0, 0.001, 1.0, 10.0, 100.0] {
        let result = extract_f64_scalar(
            &eval_primitive(
                Primitive::Sqrt,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert!(
            result >= 0.0,
            "sqrt({}) = {} should be non-negative",
            x,
            result
        );
    }
}

// ====================== MONOTONICITY ======================

#[test]
fn oracle_sqrt_monotonic() {
    // sqrt is strictly increasing on [0, inf)
    let values = vec![0.0, 0.1, 1.0, 4.0, 9.0, 100.0];
    let input = make_f64_tensor(&[values.len() as u32], values);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);

    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "sqrt should be strictly increasing: sqrt[{}] = {} > sqrt[{}] = {}",
            i,
            vals[i],
            i - 1,
            vals[i - 1]
        );
    }
}

// ====================== 1D TENSOR ======================

#[test]
fn oracle_sqrt_1d() {
    let input = make_f64_tensor(&[5], vec![0.0, 1.0, 4.0, 9.0, 16.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

// ====================== 2D TENSOR ======================

#[test]
fn oracle_sqrt_2d() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 4.0, 9.0, 16.0]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

// ====================== FRACTIONAL VALUES ======================

#[test]
fn oracle_sqrt_quarter() {
    let input = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5, "sqrt(0.25) = 0.5");
}

#[test]
fn oracle_sqrt_small() {
    // sqrt of very small positive number
    let x = 1e-100;
    let input = make_f64_tensor(&[], vec![x]);
    let result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
    assert_close(
        extract_f64_scalar(&result),
        1e-50,
        1e-60,
        "sqrt(1e-100) = 1e-50",
    );
}

// ======================== METAMORPHIC: sqrt(x)^2 = x ========================

#[test]
fn metamorphic_sqrt_mul_identity() {
    // sqrt(x)^2 = x for x >= 0, using Mul primitive for squaring
    for x in [0.0, 0.25, 1.0, 2.0, 4.0, 10.0, 100.0, 1000.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let sqrt_result = eval_primitive(Primitive::Sqrt, &[input], &no_params()).unwrap();
        let squared = eval_primitive(
            Primitive::Mul,
            &[sqrt_result.clone(), sqrt_result],
            &no_params(),
        )
        .unwrap();

        assert_close(
            extract_f64_scalar(&squared),
            x,
            1e-12,
            &format!("sqrt({})^2 = {}", x, x),
        );
    }
}

// ======================== METAMORPHIC: sqrt(x^2) = |x| ========================

#[test]
fn metamorphic_square_sqrt_abs() {
    // sqrt(x^2) = |x| for all real x
    for x in [-100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0] {
        let input = make_f64_tensor(&[], vec![x]);
        let squared = eval_primitive(Primitive::Mul, &[input.clone(), input], &no_params()).unwrap();
        let sqrt_squared = eval_primitive(Primitive::Sqrt, &[squared], &no_params()).unwrap();

        assert_close(
            extract_f64_scalar(&sqrt_squared),
            x.abs(),
            1e-12,
            &format!("sqrt({}^2) = |{}| = {}", x, x, x.abs()),
        );
    }
}

// ======================== METAMORPHIC: tensor round-trip ========================

#[test]
fn metamorphic_sqrt_tensor_roundtrip() {
    // For a tensor of non-negative values: sqrt(x)^2 = x
    let input = make_f64_tensor(&[6], vec![0.0, 0.25, 1.0, 4.0, 9.0, 100.0]);
    let sqrt_result = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&input), &no_params()).unwrap();
    let squared = eval_primitive(
        Primitive::Mul,
        &[sqrt_result.clone(), sqrt_result],
        &no_params(),
    )
    .unwrap();

    let original = extract_f64_vec(&input);
    let round_trip = extract_f64_vec(&squared);

    for (orig, rt) in original.iter().zip(round_trip.iter()) {
        assert_close(*rt, *orig, 1e-12, &format!("sqrt({})^2 = {}", orig, orig));
    }
}
