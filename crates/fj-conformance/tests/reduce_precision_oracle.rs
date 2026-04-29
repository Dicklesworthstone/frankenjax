//! Oracle tests for ReducePrecision primitive.
//!
//! Tests against expected behavior for precision reduction:
//! - Reduces floating-point precision by limiting exponent and mantissa bits
//! - Used to simulate lower precision hardware

#![allow(clippy::approx_constant)]

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

fn make_f32_tensor(shape: &[u32], data: Vec<f32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|x| Literal::F32Bits(x.to_bits()))
                .collect(),
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

fn extract_f32_vec(v: &Value) -> Vec<f32> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(bits) => f32::from_bits(*bits),
                _ => l.as_f64().unwrap() as f32,
            })
            .collect(),
        Value::Scalar(Literal::F32Bits(bits)) => vec![f32::from_bits(*bits)],
        _ => unreachable!("expected f32"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn precision_params(exponent_bits: u32, mantissa_bits: u32) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("exponent_bits".to_string(), exponent_bits.to_string());
    p.insert("mantissa_bits".to_string(), mantissa_bits.to_string());
    p
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Basic Tests ========================

#[test]
fn oracle_reduce_precision_f64_no_change() {
    // Full f64 precision (11 exponent, 52 mantissa) should not change value
    let input = Value::Scalar(Literal::from_f64(3.14159265358979));
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(11, 52),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.14159265358979).abs() < 1e-14);
}

#[test]
fn oracle_reduce_precision_f64_zero() {
    // Zero should remain zero at any precision
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(5, 10),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].abs() < 1e-15);
}

#[test]
fn oracle_reduce_precision_f64_one() {
    // 1.0 should remain 1.0 at any reasonable precision
    let input = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(5, 10),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_reduce_precision_f64_powers_of_two() {
    // Powers of 2 should be exact at any mantissa precision
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 4.0, 8.0]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(5, 5),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 2.0).abs() < 1e-10);
    assert!((vals[2] - 4.0).abs() < 1e-10);
    assert!((vals[3] - 8.0).abs() < 1e-10);
}

// ======================== Precision Reduction Tests ========================

#[test]
fn oracle_reduce_precision_f64_low_mantissa() {
    // With low mantissa bits, small differences get lost
    let input = make_f64_tensor(&[2], vec![1.0, 1.001]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(8, 3),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    // With only 3 mantissa bits, 1.001 might round to 1.0
    assert!((vals[0] - 1.0).abs() < 0.1);
    assert!((vals[1] - 1.0).abs() < 0.1);
}

#[test]
fn oracle_reduce_precision_f64_idempotent() {
    // Applying reduce_precision twice should give same result
    let input = Value::Scalar(Literal::from_f64(3.14159));
    let result1 = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(5, 10),
    )
    .unwrap();
    let result2 = eval_primitive(
        Primitive::ReducePrecision,
        std::slice::from_ref(&result1),
        &precision_params(5, 10),
    )
    .unwrap();
    let vals1 = extract_f64_vec(&result1);
    let vals2 = extract_f64_vec(&result2);
    assert!((vals1[0] - vals2[0]).abs() < 1e-15);
}

// ======================== f32 Tests ========================

#[test]
fn oracle_reduce_precision_f32_no_change() {
    // Full f32 precision (8 exponent, 23 mantissa)
    let input = make_f32_tensor(&[1], vec![3.14159_f32]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(8, 23),
    )
    .unwrap();
    let vals = extract_f32_vec(&result);
    assert!((vals[0] - 3.14159_f32).abs() < 1e-5);
}

#[test]
fn oracle_reduce_precision_f32_zero() {
    let input = make_f32_tensor(&[1], vec![0.0_f32]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(5, 10),
    )
    .unwrap();
    let vals = extract_f32_vec(&result);
    assert!(vals[0].abs() < 1e-10);
}

#[test]
fn oracle_reduce_precision_f32_powers() {
    let input = make_f32_tensor(&[3], vec![1.0_f32, 2.0_f32, 4.0_f32]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(5, 5),
    )
    .unwrap();
    let vals = extract_f32_vec(&result);
    assert!((vals[0] - 1.0_f32).abs() < 1e-5);
    assert!((vals[1] - 2.0_f32).abs() < 1e-5);
    assert!((vals[2] - 4.0_f32).abs() < 1e-5);
}

// ======================== Shape Tests ========================

#[test]
fn oracle_reduce_precision_1d() {
    let input = make_f64_tensor(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(8, 10),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
}

#[test]
fn oracle_reduce_precision_2d() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(8, 10),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

// ======================== Default Parameters ========================

#[test]
fn oracle_reduce_precision_default_params() {
    // Without params, should use dtype's default bits (no change)
    let input = Value::Scalar(Literal::from_f64(3.14159));
    let result = eval_primitive(Primitive::ReducePrecision, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.14159).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_reduce_precision_negative() {
    let input = make_f64_tensor(&[3], vec![-1.0, -2.5, -100.0]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(8, 10),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] < 0.0);
    assert!(vals[1] < 0.0);
    assert!(vals[2] < 0.0);
}

#[test]
fn oracle_reduce_precision_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(
        Primitive::ReducePrecision,
        &[input],
        &precision_params(8, 10),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
}
