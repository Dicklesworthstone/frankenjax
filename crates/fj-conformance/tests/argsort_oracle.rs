//! Oracle tests for Argsort primitive.
//!
//! argsort(x) returns the indices that would sort the array
//!
//! Tests:
//! - Basic: argsort([3, 1, 2]) = [1, 2, 0]
//! - Already sorted
//! - Reverse sorted
//! - With duplicates
//! - Negative values
//! - 2D (per-row)

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

fn argsort_params(axis: i64, descending: bool) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("axis".to_string(), axis.to_string());
    params.insert("descending".to_string(), descending.to_string());
    params
}

// ======================== Basic Cases ========================

#[test]
fn oracle_argsort_basic() {
    let input = make_f64_tensor(&[5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    let indices = extract_i64_vec(&result);
    // Sorted: 1.0, 1.0, 3.0, 4.0, 5.0 -> indices: 1 or 3, 3 or 1, 0, 2, 4
    // Check that applying these indices sorts the array
    let original = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let sorted: Vec<f64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert!(sorted.windows(2).all(|w| w[0] <= w[1]), "result should be sorted");
}

#[test]
fn oracle_argsort_already_sorted() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    assert_eq!(indices, vec![0, 1, 2, 3]);
}

#[test]
fn oracle_argsort_reverse_sorted() {
    let input = make_f64_tensor(&[4], vec![4.0, 3.0, 2.0, 1.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    assert_eq!(indices, vec![3, 2, 1, 0]);
}

#[test]
fn oracle_argsort_descending() {
    let input = make_f64_tensor(&[4], vec![1.0, 4.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, true)).unwrap();
    let indices = extract_i64_vec(&result);
    // Descending: 4, 3, 2, 1 -> indices: 1, 3, 2, 0
    assert_eq!(indices, vec![1, 3, 2, 0]);
}

// ======================== Negative Values ========================

#[test]
fn oracle_argsort_negative() {
    let input = make_f64_tensor(&[4], vec![-3.0, -1.0, -4.0, -2.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: -4, -3, -2, -1 -> indices: 2, 0, 3, 1
    assert_eq!(indices, vec![2, 0, 3, 1]);
}

// ======================== Integer Types ========================

#[test]
fn oracle_argsort_i64() {
    let input = make_i64_tensor(&[4], vec![30, 10, 40, 20]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: 10, 20, 30, 40 -> indices: 1, 3, 0, 2
    assert_eq!(indices, vec![1, 3, 0, 2]);
}

// ======================== 2D (Per-Row) ========================

#[test]
fn oracle_argsort_2d() {
    // [[3, 1, 2], [6, 4, 5]]
    let input = make_f64_tensor(&[2, 3], vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let indices = extract_i64_vec(&result);
    // Row 0: sorted [1, 2, 3] -> indices [1, 2, 0]
    // Row 1: sorted [4, 5, 6] -> indices [1, 2, 0]
    assert_eq!(indices, vec![1, 2, 0, 1, 2, 0]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_argsort_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_argsort_2d_axis0() {
    // Sort along axis 0 (columns)
    // [[3, 1], [1, 3]] -> sorted by columns: [[1, 1], [3, 3]]
    // Column 0: [3, 1] -> indices [1, 0]
    // Column 1: [1, 3] -> indices [0, 1]
    let input = make_f64_tensor(&[2, 2], vec![3.0, 1.0, 1.0, 3.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(0, false)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let indices = extract_i64_vec(&result);
    assert_eq!(indices, vec![1, 0, 0, 1]);
}

#[test]
fn oracle_argsort_preserves_index_dtype() {
    let input = make_f64_tensor(&[3], vec![3.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    match &result {
        Value::Tensor(t) => {
            // Argsort output should be integer type
            assert!(
                matches!(t.dtype, DType::I32 | DType::I64),
                "argsort should return integer indices"
            );
        }
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_argsort_with_zeros() {
    let input = make_f64_tensor(&[5], vec![0.0, -1.0, 1.0, 0.0, -2.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: -2, -1, 0, 0, 1 -> indices: 4, 1, 0 or 3, 3 or 0, 2
    let original = vec![0.0, -1.0, 1.0, 0.0, -2.0];
    let sorted: Vec<f64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert!(sorted.windows(2).all(|w| w[0] <= w[1]), "result should be sorted");
}

#[test]
fn oracle_argsort_all_equal() {
    let input = make_f64_tensor(&[4], vec![5.0, 5.0, 5.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // All equal: stable sort should preserve original order
    assert_eq!(indices, vec![0, 1, 2, 3]);
}

#[test]
fn oracle_argsort_large_values() {
    let input = make_f64_tensor(&[4], vec![1e10, -1e10, 1e-10, -1e-10]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, false)).unwrap();
    let indices = extract_i64_vec(&result);
    // Sorted: -1e10, -1e-10, 1e-10, 1e10
    assert_eq!(indices, vec![1, 3, 2, 0]);
}

#[test]
fn oracle_argsort_descending_3d() {
    // 3D tensor, sort along last axis descending
    let input = make_f64_tensor(&[2, 1, 3], vec![1.0, 3.0, 2.0, 6.0, 4.0, 5.0]);
    let result = eval_primitive(Primitive::Argsort, &[input], &argsort_params(-1, true)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1, 3]);
    let indices = extract_i64_vec(&result);
    // [1, 3, 2] desc -> [3, 2, 1] -> indices [1, 2, 0]
    // [6, 4, 5] desc -> [6, 5, 4] -> indices [0, 2, 1]
    assert_eq!(indices, vec![1, 2, 0, 0, 2, 1]);
}
