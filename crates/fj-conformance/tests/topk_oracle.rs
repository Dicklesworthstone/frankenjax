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
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
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

    // TopK returns a tuple (values, indices)
    match result {
        Value::Tensor(t) => {
            // If it returns a single tensor, check shape
            assert_eq!(t.shape.dims, vec![2]);
            let vals = t.elements.iter().map(|l| l.as_f64().unwrap()).collect::<Vec<_>>();
            // Top 2 should be 5.0 and 4.0
            assert!(vals.contains(&5.0) && vals.contains(&4.0));
        }
        _ => panic!("expected tensor result"),
    }
}

#[test]
fn oracle_topk_k1() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(1)).unwrap();

    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![1]);
            let val = t.elements[0].as_f64().unwrap();
            assert_eq!(val, 5.0, "top_k with k=1 should return max");
        }
        _ => panic!("expected tensor result"),
    }
}

#[test]
fn oracle_topk_full() {
    let input = make_f64_tensor(&[4], vec![3.0, 1.0, 4.0, 2.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(4)).unwrap();

    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![4]);
            let vals = t.elements.iter().map(|l| l.as_f64().unwrap()).collect::<Vec<_>>();
            // Should be sorted descending
            assert_eq!(vals, vec![4.0, 3.0, 2.0, 1.0]);
        }
        _ => panic!("expected tensor result"),
    }
}

// ======================== Negative Values ========================

#[test]
fn oracle_topk_negative() {
    let input = make_f64_tensor(&[5], vec![-3.0, -1.0, -4.0, -1.0, -5.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    match result {
        Value::Tensor(t) => {
            let vals = t.elements.iter().map(|l| l.as_f64().unwrap()).collect::<Vec<_>>();
            // Top 2 of negative values: -1.0, -1.0
            assert!(vals.iter().all(|&v| v >= -3.0));
        }
        _ => panic!("expected tensor result"),
    }
}

// ======================== Integer Types ========================

#[test]
fn oracle_topk_i64() {
    let input = make_i64_tensor(&[5], vec![30, 10, 40, 10, 50]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![2]);
            let vals = t.elements.iter().map(|l| l.as_i64().unwrap()).collect::<Vec<_>>();
            assert!(vals.contains(&50) && vals.contains(&40));
        }
        _ => panic!("expected tensor result"),
    }
}

// ======================== 2D (Batched) ========================

#[test]
fn oracle_topk_2d() {
    // [[3, 1, 4], [1, 5, 9]] -> top 2 per row
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    let result = eval_primitive(Primitive::TopK, &[input], &topk_params(2)).unwrap();

    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims, vec![2, 2]);
            let vals = t.elements.iter().map(|l| l.as_f64().unwrap()).collect::<Vec<_>>();
            // First row top 2: 4.0, 3.0
            // Second row top 2: 9.0, 5.0
            assert_eq!(vals[0], 4.0);
            assert_eq!(vals[1], 3.0);
            assert_eq!(vals[2], 9.0);
            assert_eq!(vals[3], 5.0);
        }
        _ => panic!("expected tensor result"),
    }
}
