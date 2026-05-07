//! Oracle tests for Copy primitive.
//!
//! Tests against expected behavior matching JAX/lax.copy:
//! - Returns an identical copy of the input
//! - Preserves dtype, shape, and all elements

#![allow(clippy::approx_constant)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn make_i64_tensor(shape: &[u32], data: Vec<i64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

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

fn make_bool_tensor(shape: &[u32], data: Vec<bool>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Bool,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::Bool).collect(),
        )
        .unwrap(),
    )
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
    }
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => unreachable!("expected bool"),
            })
            .collect(),
        Value::Scalar(Literal::Bool(b)) => vec![*b],
        _ => unreachable!("expected bool"),
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

// ======================== Scalar Tests ========================

#[test]
fn oracle_copy_scalar_i64() {
    let input = Value::scalar_i64(42);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_copy_scalar_f64() {
    let input = Value::Scalar(Literal::from_f64(3.17));
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.17).abs() < 1e-10);
}

#[test]
fn oracle_copy_scalar_bool() {
    let input = Value::Scalar(Literal::Bool(true));
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_copy_1d_i64() {
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_copy_1d_f64() {
    let input = make_f64_tensor(&[3], vec![1.1, 2.2, 3.3]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[2] - 3.3).abs() < 1e-10);
}

#[test]
fn oracle_copy_1d_bool() {
    let input = make_bool_tensor(&[4], vec![true, false, true, false]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, true, false]);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_copy_2d() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_copy_2d_f64() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[3] - 4.0).abs() < 1e-10);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_copy_3d() {
    let input = make_i64_tensor(&[2, 2, 2], (1..=8).collect());
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), (1..=8).collect::<Vec<_>>());
}

// ======================== Edge Cases ========================

#[test]
fn oracle_copy_empty() {
    let input = make_i64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_i64_vec(&result), vec![] as Vec<i64>);
}

#[test]
fn oracle_copy_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_copy_with_negatives() {
    let input = make_i64_tensor(&[4], vec![-100, -1, 0, 100]);
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-100, -1, 0, 100]);
}

#[test]
fn oracle_copy_large() {
    let input = make_i64_tensor(&[100], (1..=100).collect());
    let result = eval_primitive(Primitive::Copy, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![100]);
    assert_eq!(extract_i64_vec(&result), (1..=100).collect::<Vec<_>>());
}
