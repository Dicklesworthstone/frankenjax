//! Oracle tests for ExpandDims primitive.
//!
//! Tests against expected behavior matching JAX/lax.expand_dims:
//! - axis: position to insert new dimension of size 1
//! - Preserves data, only changes shape

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

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn expand_params(axis: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_expand_dims_scalar() {
    // JAX: lax.expand_dims(42, dimensions=(0,)) => shape [1]
    let input = Value::scalar_i64(42);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_expand_dims_scalar_f64() {
    let input = Value::Scalar(Literal::from_f64(3.17));
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.17).abs() < 1e-10);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_expand_dims_1d_axis0() {
    // JAX: lax.expand_dims(jnp.array([1, 2, 3]), dimensions=(0,)) => shape [1, 3]
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_expand_dims_1d_axis1() {
    // JAX: lax.expand_dims(jnp.array([1, 2, 3]), dimensions=(1,)) => shape [3, 1]
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_expand_dims_1d_single_element() {
    // [1] -> [1, 1] at axis 0
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_expand_dims_1d_large() {
    let input = make_i64_tensor(&[10], (1..=10).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 10]);
    assert_eq!(extract_i64_vec(&result), (1..=10).collect::<Vec<_>>());
}

// ======================== 2D Tests ========================

#[test]
fn oracle_expand_dims_2d_axis0() {
    // [2, 3] -> [1, 2, 3]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_expand_dims_2d_axis1() {
    // [2, 3] -> [2, 1, 3]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_expand_dims_2d_axis2() {
    // [2, 3] -> [2, 3, 1]
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_expand_dims_2d_square() {
    // [3, 3] -> [1, 3, 3]
    let input = make_i64_tensor(&[3, 3], (1..=9).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 3]);
}

#[test]
fn oracle_expand_dims_2d_single_row() {
    // [1, 5] -> [1, 1, 5]
    let input = make_i64_tensor(&[1, 5], (1..=5).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 5]);
}

#[test]
fn oracle_expand_dims_2d_single_col() {
    // [5, 1] -> [5, 1, 1]
    let input = make_i64_tensor(&[5, 1], (1..=5).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 1, 1]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_expand_dims_3d_axis0() {
    // [2, 3, 4] -> [1, 2, 3, 4]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 3, 4]);
    assert_eq!(extract_i64_vec(&result).len(), 24);
}

#[test]
fn oracle_expand_dims_3d_axis1() {
    // [2, 3, 4] -> [2, 1, 3, 4]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 3, 4]);
}

#[test]
fn oracle_expand_dims_3d_axis2() {
    // [2, 3, 4] -> [2, 3, 1, 4]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 1, 4]);
}

#[test]
fn oracle_expand_dims_3d_axis3() {
    // [2, 3, 4] -> [2, 3, 4, 1]
    let input = make_i64_tensor(&[2, 3, 4], (1..=24).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4, 1]);
}

// ======================== Float Tests ========================

#[test]
fn oracle_expand_dims_f64_1d() {
    let input = make_f64_tensor(&[3], vec![1.1, 2.2, 3.3]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[2] - 3.3).abs() < 1e-10);
}

#[test]
fn oracle_expand_dims_f64_2d() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[3] - 4.0).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_expand_dims_with_negatives() {
    let input = make_i64_tensor(&[3], vec![-5, 0, 5]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-5, 0, 5]);
}

#[test]
fn oracle_expand_dims_preserves_data_order() {
    let input = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}

#[test]
fn oracle_expand_dims_unit_tensor() {
    // [1, 1] -> [1, 1, 1]
    let input = make_i64_tensor(&[1, 1], vec![42]);
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_expand_dims_4d() {
    // [2, 2, 2, 2] -> [1, 2, 2, 2, 2]
    let input = make_i64_tensor(&[2, 2, 2, 2], (1..=16).collect());
    let result = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2, 2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), (1..=16).collect::<Vec<_>>());
}

// ======================== Compose with Squeeze ========================

#[test]
fn oracle_expand_dims_squeeze_identity() {
    // ExpandDims then Squeeze should give back original
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let expanded = eval_primitive(Primitive::ExpandDims, &[input], &expand_params(0)).unwrap();
    assert_eq!(extract_shape(&expanded), vec![1, 2, 3]);

    let mut squeeze_params = BTreeMap::new();
    squeeze_params.insert("dimensions".to_string(), "0".to_string());
    let squeezed = eval_primitive(Primitive::Squeeze, &[expanded], &squeeze_params).unwrap();
    assert_eq!(extract_shape(&squeezed), vec![2, 3]);
    assert_eq!(extract_i64_vec(&squeezed), vec![1, 2, 3, 4, 5, 6]);
}
