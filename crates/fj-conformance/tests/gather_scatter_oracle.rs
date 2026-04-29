//! Oracle tests for Gather and Scatter primitives.
//!
//! Tests against JAX lax.gather and lax.scatter semantics:
//! - Gather: extracts slices from operand at indices
//! - Scatter: updates operand at indices with values
//!
//! Shape rules:
//! - Gather slice_sizes must have length == operand.rank()
//! - Scatter updates shape = index_shape ++ operand.shape[1..]

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
        Value::Scalar(Literal::I64(v)) => vec![*v],
        _ => panic!("expected i64"),
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

fn gather_params(slice_sizes: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "slice_sizes".to_string(),
        slice_sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

fn scatter_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Gather 1D Tests ========================

#[test]
fn oracle_gather_1d_single_index() {
    // Gather single element from 1D array
    // operand=[10, 20, 30, 40] (rank 1), indices=[2] -> output shape [1]
    let operand = make_i64_tensor(&[4], vec![10, 20, 30, 40]);
    let indices = make_i64_tensor(&[1], vec![2]);
    // slice_sizes must match operand rank (1 element)
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    // Output shape = indices.shape ++ slice_sizes[1..] = [1] ++ [] = [1]
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![30]);
}

#[test]
fn oracle_gather_1d_multiple_indices() {
    // Gather multiple elements from 1D array
    // operand=[10, 20, 30, 40], indices=[0, 2, 3] -> [10, 30, 40]
    let operand = make_i64_tensor(&[4], vec![10, 20, 30, 40]);
    let indices = make_i64_tensor(&[3], vec![0, 2, 3]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![10, 30, 40]);
}

#[test]
fn oracle_gather_1d_first_element() {
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let indices = make_i64_tensor(&[1], vec![0]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1]);
}

#[test]
fn oracle_gather_1d_last_element() {
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let indices = make_i64_tensor(&[1], vec![4]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5]);
}

// ======================== Gather 2D Tests ========================

#[test]
fn oracle_gather_2d_row_select() {
    // Gather rows from 2D array
    // operand=[[1,2,3],[4,5,6],[7,8,9]] shape [3, 3]
    // indices=[0, 2] -> select rows 0 and 2
    let operand = make_i64_tensor(&[3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let indices = make_i64_tensor(&[2], vec![0, 2]);
    // slice_sizes = [1, 3] (select 1 row at a time, keep all 3 columns)
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1, 3])).unwrap();
    // Output shape = [2] ++ [3] = [2, 3]
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 7, 8, 9]);
}

#[test]
fn oracle_gather_2d_single_row() {
    let operand = make_i64_tensor(&[3, 2], vec![1, 2, 3, 4, 5, 6]);
    let indices = make_i64_tensor(&[1], vec![1]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 2]);
    assert_eq!(extract_i64_vec(&result), vec![3, 4]);
}

// ======================== Gather F64 and Edge Cases ========================

#[test]
fn oracle_gather_f64_values() {
    let operand = make_f64_tensor(&[4], vec![1.1, 2.2, 3.3, 4.4]);
    let indices = make_i64_tensor(&[2], vec![1, 3]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 2.2).abs() < 1e-10);
    assert!((vals[1] - 4.4).abs() < 1e-10);
}

#[test]
fn oracle_gather_negative_values() {
    let operand = make_i64_tensor(&[4], vec![-10, -20, 30, 40]);
    let indices = make_i64_tensor(&[2], vec![0, 1]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-10, -20]);
}

#[test]
fn oracle_gather_duplicate_indices() {
    let operand = make_i64_tensor(&[4], vec![10, 20, 30, 40]);
    let indices = make_i64_tensor(&[3], vec![1, 1, 1]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![20, 20, 20]);
}

#[test]
fn oracle_gather_single_element_operand() {
    let operand = make_i64_tensor(&[1], vec![42]);
    let indices = make_i64_tensor(&[1], vec![0]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_gather_zeros() {
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[2], vec![0, 3]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0]);
}

#[test]
fn oracle_gather_large_values() {
    let operand = make_i64_tensor(&[3], vec![i64::MAX, i64::MIN, 0]);
    let indices = make_i64_tensor(&[2], vec![0, 1]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![i64::MAX, i64::MIN]);
}

#[test]
fn oracle_gather_preserves_dtype() {
    let operand = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let indices = make_i64_tensor(&[2], vec![0, 1]);
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1])).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

// ======================== Gather Error Cases ========================

#[test]
fn oracle_gather_out_of_bounds() {
    let operand = make_i64_tensor(&[4], vec![10, 20, 30, 40]);
    let indices = make_i64_tensor(&[1], vec![5]); // out of bounds
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1]));
    assert!(result.is_err());
}

#[test]
fn oracle_gather_slice_sizes_must_match_rank() {
    let operand = make_i64_tensor(&[4], vec![10, 20, 30, 40]);
    let indices = make_i64_tensor(&[1], vec![0]);
    // slice_sizes has 2 elements but operand rank is 1
    let result = eval_primitive(Primitive::Gather, &[operand, indices], &gather_params(&[1, 1]));
    assert!(result.is_err());
}

// ======================== Scatter 1D Tests ========================

#[test]
fn oracle_scatter_1d_single_update() {
    // Scatter single value into 1D array
    // operand=[0,0,0,0], indices=[2], updates=[99] -> [0,0,99,0]
    // For 1D operand: updates shape = index_shape ++ operand.shape[1..] = [1] ++ [] = [1]
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[1], vec![2]);
    let updates = make_i64_tensor(&[1], vec![99]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 99, 0]);
}

#[test]
fn oracle_scatter_1d_multiple_updates() {
    // operand=[0,0,0,0], indices=[0,2,3], updates=[10,20,30] -> [10,0,20,30]
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[3], vec![0, 2, 3]);
    let updates = make_i64_tensor(&[3], vec![10, 20, 30]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![10, 0, 20, 30]);
}

#[test]
fn oracle_scatter_1d_first_position() {
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let indices = make_i64_tensor(&[1], vec![0]);
    let updates = make_i64_tensor(&[1], vec![99]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![99, 2, 3, 4, 5]);
}

#[test]
fn oracle_scatter_1d_last_position() {
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let indices = make_i64_tensor(&[1], vec![4]);
    let updates = make_i64_tensor(&[1], vec![99]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 99]);
}

#[test]
fn oracle_scatter_all_positions() {
    let operand = make_i64_tensor(&[3], vec![0, 0, 0]);
    let indices = make_i64_tensor(&[3], vec![0, 1, 2]);
    let updates = make_i64_tensor(&[3], vec![10, 20, 30]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![10, 20, 30]);
}

// ======================== Scatter 2D Tests ========================

#[test]
fn oracle_scatter_2d_row_update() {
    // Update rows in 2D array
    // operand=[[0,0],[0,0],[0,0]] shape [3, 2]
    // indices=[1] -> update row 1
    // updates shape = [1] ++ [2] = [1, 2]
    let operand = make_i64_tensor(&[3, 2], vec![0, 0, 0, 0, 0, 0]);
    let indices = make_i64_tensor(&[1], vec![1]);
    let updates = make_i64_tensor(&[1, 2], vec![99, 88]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 99, 88, 0, 0]);
}

#[test]
fn oracle_scatter_2d_multiple_rows() {
    let operand = make_i64_tensor(&[3, 2], vec![0, 0, 0, 0, 0, 0]);
    let indices = make_i64_tensor(&[2], vec![0, 2]);
    let updates = make_i64_tensor(&[2, 2], vec![1, 2, 5, 6]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 0, 0, 5, 6]);
}

// ======================== Scatter F64 and Edge Cases ========================

#[test]
fn oracle_scatter_f64_values() {
    let operand = make_f64_tensor(&[4], vec![0.0, 0.0, 0.0, 0.0]);
    let indices = make_i64_tensor(&[2], vec![1, 3]);
    let updates = make_f64_tensor(&[2], vec![1.5, 2.5]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 0.0).abs() < 1e-10);
    assert!((vals[1] - 1.5).abs() < 1e-10);
    assert!((vals[2] - 0.0).abs() < 1e-10);
    assert!((vals[3] - 2.5).abs() < 1e-10);
}

#[test]
fn oracle_scatter_negative_values() {
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[2], vec![0, 2]);
    let updates = make_i64_tensor(&[2], vec![-10, -20]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-10, 0, -20, 0]);
}

#[test]
fn oracle_scatter_preserves_untouched() {
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let indices = make_i64_tensor(&[1], vec![2]);
    let updates = make_i64_tensor(&[1], vec![99]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 99, 4, 5]);
}

#[test]
fn oracle_scatter_single_element_operand() {
    let operand = make_i64_tensor(&[1], vec![0]);
    let indices = make_i64_tensor(&[1], vec![0]);
    let updates = make_i64_tensor(&[1], vec![99]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![99]);
}

#[test]
fn oracle_scatter_to_zeros() {
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[2], vec![1, 2]);
    let updates = make_i64_tensor(&[2], vec![0, 0]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0]);
}

#[test]
fn oracle_scatter_large_values() {
    let operand = make_i64_tensor(&[3], vec![0, 0, 0]);
    let indices = make_i64_tensor(&[2], vec![0, 2]);
    let updates = make_i64_tensor(&[2], vec![i64::MAX, i64::MIN]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![i64::MAX, 0, i64::MIN]);
}

#[test]
fn oracle_scatter_preserves_dtype() {
    let operand = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let indices = make_i64_tensor(&[1], vec![0]);
    let updates = make_f64_tensor(&[1], vec![99.0]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_scatter_output_matches_operand_shape() {
    let operand = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let indices = make_i64_tensor(&[2], vec![0, 2]);
    let updates = make_i64_tensor(&[2], vec![10, 30]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
}

// ======================== Scatter Error Cases ========================

#[test]
fn oracle_scatter_out_of_bounds() {
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[1], vec![5]); // out of bounds
    let updates = make_i64_tensor(&[1], vec![99]);
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params());
    assert!(result.is_err());
}

#[test]
fn oracle_scatter_dtype_mismatch() {
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[1], vec![0]);
    let updates = make_f64_tensor(&[1], vec![1.5]); // wrong dtype
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params());
    assert!(result.is_err());
}

#[test]
fn oracle_scatter_wrong_update_shape() {
    let operand = make_i64_tensor(&[4], vec![0, 0, 0, 0]);
    let indices = make_i64_tensor(&[2], vec![0, 1]);
    let updates = make_i64_tensor(&[3], vec![1, 2, 3]); // should be [2] to match indices
    let result = eval_primitive(Primitive::Scatter, &[operand, indices, updates], &scatter_params());
    assert!(result.is_err());
}
