//! Oracle tests for Clamp primitive.
//!
//! Tests against expected behavior matching JAX/lax.clamp:
//! - clamp(min, x, max) returns x clamped to [min, max]
//! - If x < min, returns min; if x > max, returns max; else returns x

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn make_i64_tensor(shape: &[u32], data: Vec<i64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: shape.to_vec() },
            data.into_iter().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

fn make_f64_tensor(shape: &[u32], data: Vec<f64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: shape.to_vec() },
            data.into_iter().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => panic!("expected tensor"),
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => panic!("expected tensor"),
    }
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_clamp_scalar_in_range() {
    // JAX: lax.clamp(0, 5, 10) => 5
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(5);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 5),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_below_min() {
    // JAX: lax.clamp(0, -5, 10) => 0
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(-5);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 0),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_above_max() {
    // JAX: lax.clamp(0, 15, 10) => 10
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(15);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 10),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_at_min() {
    // JAX: lax.clamp(0, 0, 10) => 0
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(0);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 0),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_scalar_at_max() {
    // JAX: lax.clamp(0, 10, 10) => 10
    let lo = Value::scalar_i64(0);
    let x = Value::scalar_i64(10);
    let hi = Value::scalar_i64(10);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 10),
        _ => panic!("expected scalar"),
    }
}

// ======================== 1D Tensor Tests ========================

#[test]
fn oracle_clamp_1d_i64() {
    // JAX: lax.clamp(jnp.array([0,0,0,0,0]), jnp.array([-2, 3, 7, 15, -5]), jnp.array([10,10,10,10,10]))
    // => [0, 3, 7, 10, 0]
    let lo = make_i64_tensor(&[5], vec![0, 0, 0, 0, 0]);
    let x = make_i64_tensor(&[5], vec![-2, 3, 7, 15, -5]);
    let hi = make_i64_tensor(&[5], vec![10, 10, 10, 10, 10]);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 3, 7, 10, 0]);
}

#[test]
fn oracle_clamp_1d_f64() {
    // JAX: lax.clamp(0.0, jnp.array([-1.5, 0.5, 1.5]), 1.0)
    // => [0.0, 0.5, 1.0]
    let lo = make_f64_tensor(&[3], vec![0.0, 0.0, 0.0]);
    let x = make_f64_tensor(&[3], vec![-1.5, 0.5, 1.5]);
    let hi = make_f64_tensor(&[3], vec![1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-10);
    assert!((vals[1] - 0.5).abs() < 1e-10);
    assert!((vals[2] - 1.0).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_clamp_negative_range() {
    // Clamp to negative range
    let lo = Value::scalar_i64(-10);
    let x = Value::scalar_i64(5);
    let hi = Value::scalar_i64(-5);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), -5),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_single_point() {
    // min == max case
    let lo = Value::scalar_i64(5);
    let x = Value::scalar_i64(10);
    let hi = Value::scalar_i64(5);
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 5),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn oracle_clamp_f64_scalar() {
    let lo = Value::Scalar(Literal::from_f64(0.0));
    let x = Value::Scalar(Literal::from_f64(0.7));
    let hi = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Clamp, &[lo, x, hi], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert!((lit.as_f64().unwrap() - 0.7).abs() < 1e-10),
        _ => panic!("expected scalar"),
    }
}
