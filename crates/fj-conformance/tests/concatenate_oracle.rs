//! Oracle tests for Concatenate primitive.
//!
//! Tests against expected behavior matching JAX/lax.concatenate:
//! - dimension: axis to concatenate along (default 0)
//! - All inputs must have same rank and matching dims on non-concat axes

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
        _ => panic!("expected tensor"),
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
        _ => panic!("expected tensor"),
    }
}

fn concat_params(dimension: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("dimension".to_string(), dimension.to_string());
    p
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== 1D Tests ========================

#[test]
fn oracle_concat_1d_two_arrays() {
    // JAX: lax.concatenate([[1,2,3], [4,5,6]], dimension=0) => [1,2,3,4,5,6]
    let a = make_i64_tensor(&[3], vec![1, 2, 3]);
    let b = make_i64_tensor(&[3], vec![4, 5, 6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_concat_1d_three_arrays() {
    let a = make_i64_tensor(&[2], vec![1, 2]);
    let b = make_i64_tensor(&[3], vec![3, 4, 5]);
    let c = make_i64_tensor(&[1], vec![6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_concat_1d_different_sizes() {
    let a = make_i64_tensor(&[1], vec![1]);
    let b = make_i64_tensor(&[4], vec![2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_concat_1d_single_array() {
    let a = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Concatenate, &[a], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

// ======================== 2D Tests - Axis 0 ========================

#[test]
fn oracle_concat_2d_axis0() {
    // JAX: lax.concatenate([[[1,2],[3,4]], [[5,6]]], dimension=0)
    // => [[1,2],[3,4],[5,6]]
    let a = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[1, 2], vec![5, 6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_concat_2d_axis0_same_size() {
    let a = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let b = make_i64_tensor(&[2, 3], vec![7, 8, 9, 10, 11, 12]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 3]);
    assert_eq!(
        extract_i64_vec(&result),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    );
}

#[test]
fn oracle_concat_2d_axis0_default() {
    // Default dimension is 0
    let a = make_i64_tensor(&[1, 3], vec![1, 2, 3]);
    let b = make_i64_tensor(&[2, 3], vec![4, 5, 6, 7, 8, 9]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

// ======================== 2D Tests - Axis 1 ========================

#[test]
fn oracle_concat_2d_axis1() {
    // JAX: lax.concatenate([[[1,2],[3,4]], [[5],[6]]], dimension=1)
    // => [[1,2,5],[3,4,6]]
    let a = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[2, 1], vec![5, 6]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 5, 3, 4, 6]);
}

#[test]
fn oracle_concat_2d_axis1_same_size() {
    let a = make_i64_tensor(&[2, 2], vec![1, 2, 3, 4]);
    let b = make_i64_tensor(&[2, 2], vec![5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 5, 6, 3, 4, 7, 8]);
}

#[test]
fn oracle_concat_2d_axis1_three_arrays() {
    let a = make_i64_tensor(&[2, 1], vec![1, 4]);
    let b = make_i64_tensor(&[2, 2], vec![2, 3, 5, 6]);
    let c = make_i64_tensor(&[2, 1], vec![7, 8]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b, c], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 7, 4, 5, 6, 8]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_concat_3d_axis0() {
    // Shape [1,2,3] + [2,2,3] -> [3,2,3]
    let a = make_i64_tensor(&[1, 2, 3], vec![1, 2, 3, 4, 5, 6]);
    let b = make_i64_tensor(&[2, 2, 3], (7..=18).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2, 3]);
    assert_eq!(extract_i64_vec(&result).len(), 18);
}

#[test]
fn oracle_concat_3d_axis1() {
    // Shape [2,1,3] + [2,2,3] -> [2,3,3]
    let a = make_i64_tensor(&[2, 1, 3], vec![1, 2, 3, 4, 5, 6]);
    let b = make_i64_tensor(&[2, 2, 3], (7..=18).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 3]);
}

#[test]
fn oracle_concat_3d_axis2() {
    // Shape [2,2,2] + [2,2,3] -> [2,2,5]
    let a = make_i64_tensor(&[2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let b = make_i64_tensor(&[2, 2, 3], (9..=20).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 5]);
    assert_eq!(extract_i64_vec(&result).len(), 20);
}

// ======================== Float Tests ========================

#[test]
fn oracle_concat_f64_1d() {
    let a = make_f64_tensor(&[2], vec![1.1, 2.2]);
    let b = make_f64_tensor(&[2], vec![3.3, 4.4]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![4]);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[3] - 4.4).abs() < 1e-10);
}

#[test]
fn oracle_concat_f64_2d() {
    let a = make_f64_tensor(&[1, 3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[1, 3], vec![4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_concat_with_negatives() {
    let a = make_i64_tensor(&[2], vec![-3, -1]);
    let b = make_i64_tensor(&[2], vec![0, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-3, -1, 0, 5]);
}

#[test]
fn oracle_concat_empty_and_nonempty() {
    let a = make_i64_tensor(&[0], vec![]);
    let b = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_concat_large() {
    let a = make_i64_tensor(&[10], (1..=10).collect());
    let b = make_i64_tensor(&[10], (11..=20).collect());
    let c = make_i64_tensor(&[10], (21..=30).collect());
    let result = eval_primitive(Primitive::Concatenate, &[a, b, c], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![30]);
    assert_eq!(extract_i64_vec(&result), (1..=30).collect::<Vec<_>>());
}

#[test]
fn oracle_concat_2d_single_row() {
    // Single row matrices
    let a = make_i64_tensor(&[1, 2], vec![1, 2]);
    let b = make_i64_tensor(&[1, 3], vec![3, 4, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_concat_2d_single_col() {
    // Single column matrices
    let a = make_i64_tensor(&[2, 1], vec![1, 2]);
    let b = make_i64_tensor(&[3, 1], vec![3, 4, 5]);
    let result = eval_primitive(Primitive::Concatenate, &[a, b], &concat_params(0)).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}
