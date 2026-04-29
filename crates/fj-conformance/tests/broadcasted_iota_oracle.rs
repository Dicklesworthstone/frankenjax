//! Oracle tests for BroadcastedIota primitive.
//!
//! Tests against expected behavior matching JAX/lax.broadcasted_iota:
//! - shape: output shape
//! - dimension: axis along which to broadcast indices
//! - No inputs, output is indices broadcasted across shape

use fj_core::{Primitive, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

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

fn iota_params(shape: &[usize], dimension: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "shape".to_string(),
        shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p.insert("dimension".to_string(), dimension.to_string());
    p
}

// ======================== 1D Tests ========================

#[test]
fn oracle_broadcasted_iota_1d() {
    // shape [5], dimension 0 -> [0, 1, 2, 3, 4]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[5], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_broadcasted_iota_1d_single() {
    // shape [1], dimension 0 -> [0]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[1], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_broadcasted_iota_2d_dim0() {
    // shape [3, 4], dimension 0 -> row indices broadcast across columns
    // [[0,0,0,0], [1,1,1,1], [2,2,2,2]]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 4], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 4]);
    let vals = extract_i64_vec(&result);
    // Row 0: all 0s
    assert_eq!(&vals[0..4], &[0, 0, 0, 0]);
    // Row 1: all 1s
    assert_eq!(&vals[4..8], &[1, 1, 1, 1]);
    // Row 2: all 2s
    assert_eq!(&vals[8..12], &[2, 2, 2, 2]);
}

#[test]
fn oracle_broadcasted_iota_2d_dim1() {
    // shape [3, 4], dimension 1 -> column indices broadcast across rows
    // [[0,1,2,3], [0,1,2,3], [0,1,2,3]]
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 4], 1)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 4]);
    let vals = extract_i64_vec(&result);
    // Each row has [0, 1, 2, 3]
    assert_eq!(&vals[0..4], &[0, 1, 2, 3]);
    assert_eq!(&vals[4..8], &[0, 1, 2, 3]);
    assert_eq!(&vals[8..12], &[0, 1, 2, 3]);
}

#[test]
fn oracle_broadcasted_iota_2d_square() {
    // shape [3, 3], dimension 0
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 3], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    let vals = extract_i64_vec(&result);
    assert_eq!(vals, vec![0, 0, 0, 1, 1, 1, 2, 2, 2]);
}

#[test]
fn oracle_broadcasted_iota_2d_square_dim1() {
    // shape [3, 3], dimension 1
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[3, 3], 1)).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_broadcasted_iota_3d_dim0() {
    // shape [2, 3, 4], dimension 0 -> indices [0, 0, ..., 1, 1, ...]
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[2, 3, 4], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // First half (12 elements): all 0s
    assert!(vals[0..12].iter().all(|&x| x == 0));
    // Second half (12 elements): all 1s
    assert!(vals[12..24].iter().all(|&x| x == 1));
}

#[test]
fn oracle_broadcasted_iota_3d_dim1() {
    // shape [2, 3, 4], dimension 1 -> row indices within each slice
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[2, 3, 4], 1)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // Within each 2D slice (12 elements), we have rows 0, 1, 2
    // First slice
    assert_eq!(&vals[0..4], &[0, 0, 0, 0]); // row 0
    assert_eq!(&vals[4..8], &[1, 1, 1, 1]); // row 1
    assert_eq!(&vals[8..12], &[2, 2, 2, 2]); // row 2
}

#[test]
fn oracle_broadcasted_iota_3d_dim2() {
    // shape [2, 3, 4], dimension 2 -> column indices
    let result =
        eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[2, 3, 4], 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3, 4]);
    let vals = extract_i64_vec(&result);
    // Each row has [0, 1, 2, 3]
    for i in 0..6 {
        assert_eq!(&vals[i * 4..(i + 1) * 4], &[0, 1, 2, 3]);
    }
}

// ======================== Edge Cases ========================

#[test]
fn oracle_broadcasted_iota_unit_dims() {
    // shape [1, 5], dimension 1
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[1, 5], 1)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_broadcasted_iota_unit_dim_0() {
    // shape [1, 5], dimension 0 -> all zeros
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[1, 5], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 5]);
    assert_eq!(extract_i64_vec(&result), vec![0, 0, 0, 0, 0]);
}

#[test]
fn oracle_broadcasted_iota_tall() {
    // shape [5, 1], dimension 0
    let result = eval_primitive(Primitive::BroadcastedIota, &[], &iota_params(&[5, 1], 0)).unwrap();
    assert_eq!(extract_shape(&result), vec![5, 1]);
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}
