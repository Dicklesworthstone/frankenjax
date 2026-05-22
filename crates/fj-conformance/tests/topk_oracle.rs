//! Oracle tests for TopK primitive.
//!
//! top_k(x, k) returns the k largest elements and their indices
//!
//! Tests:
//! - Basic: top_k([3, 1, 4, 1, 5], 2) = ([5, 4], [4, 2])
//! - Full k: top_k(x, len(x)) returns sorted x
//! - k=1: returns max
//! - Negative values
//! - Tensor shapes (batched)

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::{eval_primitive, eval_primitive_multi};
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

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    v.as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|l| l.as_i64().unwrap())
        .collect()
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => Vec::new(),
    }
}

fn extract_f64_scalar(v: &Value) -> Option<f64> {
    match v {
        Value::Scalar(literal) => literal.as_f64(),
        Value::Tensor(tensor) if tensor.shape.dims.is_empty() => {
            tensor.elements.first().and_then(|literal| literal.as_f64())
        }
        _ => None,
    }
}

fn extract_i64_scalar(v: &Value) -> Option<i64> {
    match v {
        Value::Scalar(literal) => literal.as_i64(),
        Value::Tensor(tensor) if tensor.shape.dims.is_empty() => {
            tensor.elements.first().and_then(|literal| literal.as_i64())
        }
        _ => None,
    }
}

fn topk_params(k: usize) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("k".to_string(), k.to_string());
    params
}

// ======================== Basic Cases ========================

#[test]
fn oracle_topk_basic() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_f64_vec(&result);
    assert!(vals.contains(&5.0) && vals.contains(&4.0));
}

#[test]
fn oracle_topk_k1() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(1)).unwrap();

    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_f64_vec(&result), vec![5.0]);
}

#[test]
fn oracle_topk_full() {
    let input = make_f64_tensor(&[4], vec![3.0, 1.0, 4.0, 2.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(4)).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 2.0, 1.0]);
}

// ======================== Negative Values ========================

#[test]
fn oracle_topk_negative() {
    let input = make_f64_tensor(&[5], vec![-3.0, -1.0, -4.0, -1.0, -5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    let vals = extract_f64_vec(&result);
    assert!(vals.iter().all(|&v| v >= -3.0));
}

// ======================== Integer Types ========================

#[test]
fn oracle_topk_i64() {
    let input = make_i64_tensor(&[5], vec![30, 10, 40, 10, 50]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(extract_shape(&result), vec![2]);
    let vals = extract_i64_vec(&result);
    assert!(vals.contains(&50) && vals.contains(&40));
}

// ======================== 2D (Batched) ========================

#[test]
fn oracle_topk_2d() {
    // [[3, 1, 4], [1, 5, 9]] -> top 2 per row
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![4.0, 3.0, 9.0, 5.0]);
}

// ======================== Multi-output Values and Indices ========================

#[test]
fn oracle_topk_multi_output_values_and_indices_1d() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(extract_shape(&result[0]), vec![2]);
    assert_eq!(extract_shape(&result[1]), vec![2]);
    assert_eq!(extract_f64_vec(&result[0]), vec![5.0, 4.0]);
    assert_eq!(extract_i64_vec(&result[1]), vec![4, 2]);
}

#[test]
fn oracle_topk_multi_output_indices_are_per_last_axis_slice() {
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(extract_shape(&result[0]), vec![2, 2]);
    assert_eq!(extract_shape(&result[1]), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result[0]), vec![4.0, 3.0, 9.0, 5.0]);
    assert_eq!(extract_i64_vec(&result[1]), vec![2, 0, 2, 1]);
}

#[test]
fn oracle_topk_multi_output_scalar_returns_zero_index() {
    let result =
        eval_primitive_multi(Primitive::TopK, &[Value::scalar_f64(7.0)], &topk_params(1)).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(extract_f64_scalar(&result[0]), Some(7.0));
    assert_eq!(extract_i64_scalar(&result[1]), Some(0));
}

#[test]
fn oracle_topk_multi_output_allows_zero_k() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(0)).unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(extract_shape(&result[0]), vec![0]);
    assert_eq!(extract_shape(&result[1]), vec![0]);
    assert!(extract_f64_vec(&result[0]).is_empty());
    assert!(extract_i64_vec(&result[1]).is_empty());
}

#[test]
fn oracle_topk_multi_output_rejects_k_larger_than_axis() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let err = eval_primitive_multi(Primitive::TopK, &[input], &topk_params(4))
        .expect_err("k larger than axis size should fail");

    assert!(
        err.to_string().contains("exceeds axis size"),
        "unexpected oversized-k error: {err}",
    );
}
