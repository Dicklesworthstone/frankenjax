//! Oracle tests for Zeta primitive (Hurwitz zeta).
//!
//! zeta(x, q) = Hurwitz zeta function ζ(x, q) = Σ_{n=0}^∞ 1/(n+q)^x
//!
//! When q=1, this equals the Riemann zeta: ζ(x, 1) = ζ(x)
//!
//! Key properties:
//! - zeta(2, 1) = π²/6 ≈ 1.6449 (Riemann zeta(2))
//! - zeta(3, 1) ≈ 1.2020569 (Apéry's constant)
//! - zeta(2, 0.5) = 4 * zeta(2, 1) - zeta(2, 1) = 3 * π²/6
//!
//! Tests:
//! - Hurwitz zeta with q=1 (Riemann zeta)
//! - Hurwitz zeta with various q values
//! - Tensor shapes and broadcasting

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::f64::consts::PI;

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

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
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

// ======================== Hurwitz Zeta with q=1 (Riemann Zeta) ========================

#[test]
fn oracle_zeta_2_q1() {
    // ζ(2, 1) = π²/6
    let x = make_f64_tensor(&[], vec![2.0]);
    let q = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = PI * PI / 6.0;
    assert!(
        (actual - expected).abs() < 1e-4,
        "zeta(2, 1) = π²/6 ≈ {}, got {}",
        expected, actual
    );
}

#[test]
fn oracle_zeta_4_q1() {
    // ζ(4, 1) = π⁴/90
    let x = make_f64_tensor(&[], vec![4.0]);
    let q = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = PI.powi(4) / 90.0;
    assert!(
        (actual - expected).abs() < 1e-4,
        "zeta(4, 1) = π⁴/90 ≈ {}, got {}",
        expected, actual
    );
}

#[test]
fn oracle_zeta_3_q1() {
    // ζ(3, 1) ≈ 1.2020569... (Apéry's constant)
    let x = make_f64_tensor(&[], vec![3.0]);
    let q = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 1.2020569031595942;
    assert!(
        (actual - expected).abs() < 1e-4,
        "zeta(3, 1) ≈ {}, got {}",
        expected, actual
    );
}

// ======================== Hurwitz Zeta with various q ========================

#[test]
fn oracle_zeta_2_q2() {
    // ζ(2, 2) = ζ(2) - 1 = π²/6 - 1
    let x = make_f64_tensor(&[], vec![2.0]);
    let q = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = PI * PI / 6.0 - 1.0;
    assert!(
        (actual - expected).abs() < 1e-4,
        "zeta(2, 2) = π²/6 - 1 ≈ {}, got {}",
        expected, actual
    );
}

#[test]
fn oracle_zeta_2_q_half() {
    // ζ(2, 0.5) = Σ 1/(n+0.5)² = 4*(π²/6 - 1) + 4 = (4π²/6) - 4 + 4 = (2/3)π² ... actually π²/2
    // More precisely: ζ(2, 0.5) = 4 * Σ 1/(2n+1)² = 4 * (π²/8) = π²/2
    let x = make_f64_tensor(&[], vec![2.0]);
    let q = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = PI * PI / 2.0;
    assert!(
        (actual - expected).abs() < 1e-3,
        "zeta(2, 0.5) ≈ π²/2 = {}, got {}",
        expected, actual
    );
}

// ======================== Large x (rapid convergence) ========================

#[test]
fn oracle_zeta_large_x() {
    // For large x, ζ(x, q) ≈ 1/q^x
    let x = make_f64_tensor(&[], vec![10.0]);
    let q = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // ζ(10, 1) ≈ 1.000994575...
    assert!(
        (actual - 1.0).abs() < 0.01,
        "zeta(10, 1) ≈ 1, got {}",
        actual
    );
}

// ======================== Edge cases ========================

#[test]
fn oracle_zeta_q_nan() {
    let x = make_f64_tensor(&[], vec![2.0]);
    let q = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.is_nan(), "zeta(2, NaN) should be NaN");
}

#[test]
fn oracle_zeta_x_nan() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let q = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.is_nan(), "zeta(NaN, 1) should be NaN");
}

#[test]
fn oracle_zeta_q_zero() {
    // q <= 0 is invalid for Hurwitz zeta
    let x = make_f64_tensor(&[], vec![2.0]);
    let q = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.is_nan(), "zeta(2, 0) should be NaN (q must be positive)");
}

#[test]
fn oracle_zeta_q_negative() {
    let x = make_f64_tensor(&[], vec![2.0]);
    let q = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.is_nan(), "zeta(2, -1) should be NaN");
}

#[test]
fn oracle_zeta_x_one_pole() {
    // x=1 is a pole
    let x = make_f64_tensor(&[], vec![1.0]);
    let q = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.is_infinite() && actual > 0.0, "zeta(1, q) should be +inf");
}

// ======================== Tensor shapes ========================

#[test]
fn oracle_zeta_vector() {
    let x = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let q = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - PI * PI / 6.0).abs() < 1e-4);
    assert!((vals[1] - 1.2020569031595942).abs() < 1e-4);
    assert!((vals[2] - PI.powi(4) / 90.0).abs() < 1e-4);
}

#[test]
fn oracle_zeta_2d() {
    let x = make_f64_tensor(&[2, 2], vec![2.0, 3.0, 4.0, 2.0]);
    let q = make_f64_tensor(&[2, 2], vec![1.0, 1.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - PI * PI / 6.0).abs() < 1e-4);
    assert!((vals[1] - 1.2020569031595942).abs() < 1e-4);
    assert!((vals[2] - PI.powi(4) / 90.0).abs() < 1e-4);
    assert!((vals[3] - (PI * PI / 6.0 - 1.0)).abs() < 1e-4);
}

// ======================== Broadcast tests ========================

#[test]
fn oracle_zeta_scalar_tensor_broadcast() {
    // scalar x, tensor q
    let x = scalar_f64(2.0);
    let q = make_f64_tensor(&[3], vec![1.0, 2.0, 0.5]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - PI * PI / 6.0).abs() < 1e-4);
    assert!((vals[1] - (PI * PI / 6.0 - 1.0)).abs() < 1e-4);
    assert!((vals[2] - PI * PI / 2.0).abs() < 1e-3);
}

#[test]
fn oracle_zeta_tensor_scalar_broadcast() {
    // tensor x, scalar q
    let x = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let q = scalar_f64(1.0);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - PI * PI / 6.0).abs() < 1e-4);
    assert!((vals[1] - 1.2020569031595942).abs() < 1e-4);
    assert!((vals[2] - PI.powi(4) / 90.0).abs() < 1e-4);
}

#[test]
fn oracle_zeta_row_vector_broadcast() {
    // [1, 3] x [2, 3]
    let x = make_f64_tensor(&[1, 3], vec![2.0, 3.0, 4.0]);
    let q = make_f64_tensor(&[2, 3], vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    let result = eval_primitive(Primitive::Zeta, &[x, q], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

// ======================== Arity error ========================

#[test]
fn oracle_zeta_arity_error() {
    // Unary call should fail (JAX zeta is binary)
    let x = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Zeta, &[x], &no_params());
    assert!(result.is_err(), "Unary zeta should error - JAX zeta requires two arguments");
}
