//! Oracle tests for BroadcastInDim primitive.
//!
//! Tests against expected behavior matching JAX/lax.broadcast_in_dim:
//! - shape: target output shape
//! - broadcast_dimensions: mapping of input axes to output axes
//! - Input dims of 1 are broadcast; others must match target

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

fn broadcast_params(shape: &[i64], broadcast_dims: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "shape".to_string(),
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p.insert(
        "broadcast_dimensions".to_string(),
        broadcast_dims
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

fn broadcast_params_shape_only(shape: &[i64]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "shape".to_string(),
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

// ======================== Scalar Broadcast Tests ========================

#[test]
fn oracle_broadcast_scalar_to_1d() {
    // JAX: lax.broadcast_in_dim(5, (3,), ()) => [5, 5, 5]
    let input = Value::scalar_i64(5);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params_shape_only(&[3]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![5, 5, 5]);
}

#[test]
fn oracle_broadcast_scalar_to_2d() {
    // JAX: lax.broadcast_in_dim(7, (2, 3), ()) => [[7,7,7],[7,7,7]]
    let input = Value::scalar_i64(7);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params_shape_only(&[2, 3]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![7, 7, 7, 7, 7, 7]);
}

#[test]
fn oracle_broadcast_scalar_to_3d() {
    let input = Value::scalar_i64(1);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params_shape_only(&[2, 2, 2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 1, 1, 1, 1, 1]);
}

// ======================== 1D to 2D Broadcast Tests ========================

#[test]
fn oracle_broadcast_1d_to_2d_trailing() {
    // JAX: lax.broadcast_in_dim([1,2,3], (2,3), (1,)) => [[1,2,3],[1,2,3]]
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[1]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 1, 2, 3]);
}

#[test]
fn oracle_broadcast_1d_to_2d_leading() {
    // JAX: lax.broadcast_in_dim([1,2], (2,3), (0,)) => [[1,1,1],[2,2,2]]
    let input = make_i64_tensor(&[2], vec![1, 2]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[0]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 2, 2, 2]);
}

#[test]
fn oracle_broadcast_1d_to_2d_default() {
    // Default broadcast_dimensions maps to trailing axes
    let input = make_i64_tensor(&[3], vec![1, 2, 3]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params_shape_only(&[2, 3]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 1, 2, 3]);
}

// ======================== Broadcasting Size-1 Dimensions ========================

#[test]
fn oracle_broadcast_size1_to_larger() {
    // JAX: lax.broadcast_in_dim([[1],[2]], (2,3), (0,1))
    // Input [2,1] -> Output [2,3]: broadcast along axis 1
    let input = make_i64_tensor(&[2, 1], vec![1, 2]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[0, 1]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 1, 2, 2, 2]);
}

#[test]
fn oracle_broadcast_size1_column() {
    // Input [1,3] -> Output [2,3]: broadcast along axis 0
    let input = make_i64_tensor(&[1, 3], vec![1, 2, 3]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[0, 1]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 1, 2, 3]);
}

#[test]
fn oracle_broadcast_size1_both() {
    // Input [1,1] -> Output [2,3]: broadcast both axes
    let input = make_i64_tensor(&[1, 1], vec![42]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[0, 1]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![42, 42, 42, 42, 42, 42]);
}

// ======================== 1D to 3D Broadcast Tests ========================

#[test]
fn oracle_broadcast_1d_to_3d_last() {
    // Input [4] -> Output [2,3,4] with broadcast_dimensions=(2,)
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3, 4], &[2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // Each [1,2,3,4] repeated 6 times
    assert_eq!(vals.len(), 24);
    assert_eq!(&vals[0..4], &[1, 2, 3, 4]);
    assert_eq!(&vals[20..24], &[1, 2, 3, 4]);
}

#[test]
fn oracle_broadcast_1d_to_3d_first() {
    // Input [2] -> Output [2,3,4] with broadcast_dimensions=(0,)
    let input = make_i64_tensor(&[2], vec![10, 20]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3, 4], &[0]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // First half all 10s, second half all 20s
    assert!(vals[0..12].iter().all(|&v| v == 10));
    assert!(vals[12..24].iter().all(|&v| v == 20));
}

// ======================== 2D to 3D Broadcast Tests ========================

#[test]
fn oracle_broadcast_2d_to_3d_trailing() {
    // Input [3,4] -> Output [2,3,4] with default (trailing)
    let input = make_i64_tensor(&[3, 4], (1..=12).collect());
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params_shape_only(&[2, 3, 4]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // Input repeated twice
    assert_eq!(&vals[0..12], &(1..=12).collect::<Vec<_>>()[..]);
    assert_eq!(&vals[12..24], &(1..=12).collect::<Vec<_>>()[..]);
}

#[test]
fn oracle_broadcast_2d_to_3d_explicit() {
    // Input [2,4] -> Output [2,3,4] with broadcast_dimensions=(0,2)
    let input = make_i64_tensor(&[2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3, 4], &[0, 2]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
}

// ======================== Identity Broadcast Tests ========================

#[test]
fn oracle_broadcast_identity_1d() {
    // Same shape broadcast is identity
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[5], &[0]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_broadcast_identity_2d() {
    let input = make_i64_tensor(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[0, 1]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

// ======================== Float Tests ========================

#[test]
fn oracle_broadcast_f64_scalar() {
    let input = Value::Scalar(Literal::from_f64(7.5));
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params_shape_only(&[2, 2]),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert!(vals.iter().all(|&v| (v - 7.5).abs() < 1e-10));
}

#[test]
fn oracle_broadcast_f64_1d_to_2d() {
    let input = make_f64_tensor(&[3], vec![1.1, 2.2, 3.3]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[1]),
    )
    .unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[3] - 1.1).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_broadcast_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[5], &[0]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![42, 42, 42, 42, 42]);
}

#[test]
fn oracle_broadcast_with_negatives() {
    let input = make_i64_tensor(&[2], vec![-3, 5]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 3], &[0]),
    )
    .unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-3, -3, -3, 5, 5, 5]);
}

#[test]
fn oracle_broadcast_large_expansion() {
    // Small input to larger output
    let input = make_i64_tensor(&[2], vec![1, 2]);
    let result = eval_primitive(
        Primitive::BroadcastInDim,
        &[input],
        &broadcast_params(&[2, 5, 5], &[0]),
    )
    .unwrap();
    assert_eq!(extract_shape(&result), vec![2, 5, 5]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals.len(), 50);
    assert!(vals[0..25].iter().all(|&v| v == 1));
    assert!(vals[25..50].iter().all(|&v| v == 2));
}
