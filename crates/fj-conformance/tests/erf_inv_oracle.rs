//! Oracle tests for ErfInv primitive.
//!
//! Tests against expected behavior matching JAX/scipy.special.erfinv:
//! - erfinv is the inverse of erf
//! - erfinv(erf(x)) = x for x in reasonable range
//! - erfinv(0) = 0
//! - erfinv(-1) = -inf, erfinv(1) = +inf

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
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
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

// Helper: compute erf using Horner approximation (matches fj-lax)
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_erf_inv_zero() {
    // erfinv(0) = 0
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-10);
}

#[test]
fn oracle_erf_inv_small_positive() {
    // erfinv(0.5) ≈ 0.4769
    let input = Value::Scalar(Literal::from_f64(0.5));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.4769).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_small_negative() {
    // erfinv(-0.5) ≈ -0.4769
    let input = Value::Scalar(Literal::from_f64(-0.5));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-0.4769)).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_close_to_one() {
    // erfinv(0.9) ≈ 1.1631
    let input = Value::Scalar(Literal::from_f64(0.9));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1631).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_close_to_neg_one() {
    // erfinv(-0.9) ≈ -1.1631
    let input = Value::Scalar(Literal::from_f64(-0.9));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - (-1.1631)).abs() < 0.01);
}

// ======================== Inverse Property Tests ========================

#[test]
fn oracle_erf_inv_inverse_property() {
    // erfinv(erf(x)) ≈ x for small x
    let x_values = vec![0.0, 0.1, 0.5, 1.0, -0.1, -0.5, -1.0];
    let erf_values: Vec<f64> = x_values.iter().map(|&x| erf_approx(x)).collect();
    let input = make_f64_tensor(&[7], erf_values);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    for (v, &x) in vals.iter().zip(x_values.iter()) {
        assert!((v - x).abs() < 0.05, "erfinv(erf({x})) = {v}, expected {x}");
    }
}

// ======================== 1D Tests ========================

#[test]
fn oracle_erf_inv_1d() {
    let input = make_f64_tensor(&[5], vec![-0.5, -0.25, 0.0, 0.25, 0.5]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let vals = extract_f64_vec(&result);
    // erfinv is odd: erfinv(-x) = -erfinv(x)
    assert!((vals[0] + vals[4]).abs() < 0.01);
    assert!((vals[1] + vals[3]).abs() < 0.01);
    assert!(vals[2].abs() < 1e-10);
}

#[test]
fn oracle_erf_inv_1d_symmetric() {
    // Test symmetry: erfinv(-x) = -erfinv(x)
    let input = make_f64_tensor(&[4], vec![0.3, -0.3, 0.7, -0.7]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] + vals[1]).abs() < 1e-10);
    assert!((vals[2] + vals[3]).abs() < 1e-10);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_erf_inv_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 0.5, -0.5, 0.8]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-10); // erfinv(0) = 0
    assert!((vals[1] + vals[2]).abs() < 1e-10); // symmetry
}

// ======================== Edge Cases ========================

#[test]
fn oracle_erf_inv_at_one() {
    // erfinv(1) = +inf
    let input = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
}

#[test]
fn oracle_erf_inv_at_neg_one() {
    // erfinv(-1) = -inf
    let input = Value::Scalar(Literal::from_f64(-1.0));
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] < 0.0);
}

#[test]
fn oracle_erf_inv_single_element() {
    let input = make_f64_tensor(&[1], vec![0.5]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.4769).abs() < 0.01);
}

#[test]
fn oracle_erf_inv_known_values() {
    // Known erfinv values (approximate)
    // erfinv(0.1) ≈ 0.0889
    // erfinv(0.2) ≈ 0.1791
    // erfinv(0.3) ≈ 0.2725
    let input = make_f64_tensor(&[3], vec![0.1, 0.2, 0.3]);
    let result = eval_primitive(Primitive::ErfInv, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0889).abs() < 0.01);
    assert!((vals[1] - 0.1791).abs() < 0.01);
    assert!((vals[2] - 0.2725).abs() < 0.01);
}
