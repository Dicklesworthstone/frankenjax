//! Oracle tests for Tile primitive.
//!
//! tile(x, reps) repeats the input array according to reps
//!
//! For example:
//! - tile([1, 2], [2]) = [1, 2, 1, 2]
//! - tile([[1, 2]], [2, 3]) = [[1, 2, 1, 2, 1, 2], [1, 2, 1, 2, 1, 2]]
//!
//! Tests:
//! - 1D tiling
//! - 2D tiling
//! - Identity (reps = [1, 1, ...])
//! - Expanding dimensions

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

fn tile_params(reps: &[i64]) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    if !reps.is_empty() {
        params.insert(
            "reps".to_string(),
            reps.iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    } else {
        params.insert("reps".to_string(), "1".to_string());
    }
    params
}

// ======================== 1D Tiling ========================

#[test]
fn oracle_tile_1d_repeat_2() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn oracle_tile_1d_repeat_3() {
    let input = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn oracle_tile_1d_identity() {
    let input = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0]);
}

// ======================== 2D Tiling ========================

#[test]
fn oracle_tile_2d_row_only() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 2]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn oracle_tile_2d_col_only() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
    );
}

#[test]
fn oracle_tile_2d_both() {
    let input = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 4]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals.len(), 16);
    // First row of first block
    assert_eq!(vals[0..4], [1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn oracle_tile_2d_identity() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[1, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ======================== Integer Types ========================

#[test]
fn oracle_tile_i64() {
    let input = make_i64_tensor(&[2], vec![10, 20]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(extract_i64_vec(&result), vec![10, 20, 10, 20, 10, 20]);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_tile_single_element() {
    let input = make_f64_tensor(&[1], vec![42.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[5])).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_f64_vec(&result), vec![42.0, 42.0, 42.0, 42.0, 42.0]);
}

#[test]
fn oracle_tile_rejects_reps_rank_mismatch() {
    // Current impl requires reps length to match tensor rank
    let input = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[3, 2]));
    assert!(result.is_err(), "reps length > rank should error");
}

#[test]
fn oracle_tile_preserves_dtype() {
    let input = make_i64_tensor(&[2], vec![1, 2]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::I64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_tile_large_reps() {
    let input = make_f64_tensor(&[1], vec![7.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[100])).unwrap();
    assert_eq!(extract_shape(&result), vec![100]);
    let vals = extract_f64_vec(&result);
    assert!(vals.iter().all(|&v| (v - 7.0).abs() < 1e-10));
}

#[test]
fn oracle_tile_3d() {
    let input = make_f64_tensor(&[1, 1, 2], vec![1.0, 2.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2, 2, 1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    // All 4 2-element blocks should be [1, 2]
    assert_eq!(vals, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn oracle_tile_negative_values() {
    let input = make_f64_tensor(&[3], vec![-1.0, -2.0, -3.0]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
    assert_eq!(extract_shape(&result), vec![6]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![-1.0, -2.0, -3.0, -1.0, -2.0, -3.0]
    );
}

#[test]
fn oracle_tile_metamorphic_product() {
    // tile(x, [a]) then tile(result, [b]) should equal tile(x, [a*b])
    let input = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let once = eval_primitive(Primitive::Tile, &[input.clone()], &tile_params(&[6])).unwrap();
    let twice = {
        let step1 = eval_primitive(Primitive::Tile, &[input], &tile_params(&[2])).unwrap();
        eval_primitive(Primitive::Tile, &[step1], &tile_params(&[3])).unwrap()
    };
    assert_eq!(extract_shape(&once), extract_shape(&twice));
    assert_eq!(extract_f64_vec(&once), extract_f64_vec(&twice));
}

#[test]
fn oracle_tile_empty_input() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Tile, &[input], &tile_params(&[5])).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_f64_vec(&result), vec![] as Vec<f64>);
}
