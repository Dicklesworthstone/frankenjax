//! Oracle tests for Split primitive.
//!
//! Tests against expected behavior for tensor splitting:
//! - axis: dimension to split along (default 0)
//! - num_sections: number of equal sections to split into
//! - sizes: explicit sizes for unequal splits
//! - Result has a new leading dimension for sections

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

fn split_params(axis: usize, num_sections: usize) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p.insert("num_sections".to_string(), num_sections.to_string());
    p
}

fn split_params_sizes(axis: usize, sizes: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p.insert(
        "sizes".to_string(),
        sizes
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    p
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== 1D Tests - Equal Splits ========================

#[test]
fn oracle_split_1d_two_sections() {
    // [6] -> [2, 3] (2 sections of 3 elements)
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_split_1d_three_sections() {
    // [6] -> [3, 2] (3 sections of 2 elements)
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_split_1d_six_sections() {
    // [6] -> [6, 1] (6 sections of 1 element)
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 6)).unwrap();
    assert_eq!(extract_shape(&result), vec![6, 1]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn oracle_split_1d_default_axis() {
    // Default axis is 0
    let input = make_i64_tensor(&[6], vec![1, 2, 3, 4, 5, 6]);
    let mut params = BTreeMap::new();
    params.insert("num_sections".to_string(), "3".to_string());
    let result = eval_primitive(Primitive::Split, &[input], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 2]);
}

#[test]
fn oracle_split_1d_no_params() {
    // Default: single section
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Split, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}

// ======================== 2D Tests - Axis 0 ========================

#[test]
fn oracle_split_2d_axis0_two_sections() {
    // [4, 3] -> [2, 2, 3] (split along rows)
    let input = make_i64_tensor(&[4, 3], (1..=12).collect());
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 3]);
    assert_eq!(extract_i64_vec(&result), (1..=12).collect::<Vec<_>>());
}

#[test]
fn oracle_split_2d_axis0_four_sections() {
    // [4, 3] -> [4, 1, 3] (4 sections of 1 row each)
    let input = make_i64_tensor(&[4, 3], (1..=12).collect());
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 4)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 1, 3]);
}

// ======================== 2D Tests - Axis 1 ========================

#[test]
fn oracle_split_2d_axis1_two_sections() {
    // [2, 4] -> [2, 2, 2] (split along columns)
    let input = make_i64_tensor(&[2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(1, 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
}

#[test]
fn oracle_split_2d_axis1_four_sections() {
    // [2, 4] -> [2, 4, 1] (4 sections of 1 column each, sections dim inserted at axis)
    let input = make_i64_tensor(&[2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(1, 4)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 4, 1]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_split_3d_axis0() {
    // [4, 2, 3] -> [2, 2, 2, 3]
    let input = make_i64_tensor(&[4, 2, 3], (1..=24).collect());
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 3]);
}

#[test]
fn oracle_split_3d_axis1() {
    // [2, 4, 3] -> [2, 2, 2, 3]
    let input = make_i64_tensor(&[2, 4, 3], (1..=24).collect());
    let result = eval_primitive(Primitive::Split, &[input], &split_params(1, 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 3]);
}

#[test]
fn oracle_split_3d_axis2() {
    // [2, 2, 6] -> [2, 2, 2, 3]
    let input = make_i64_tensor(&[2, 2, 6], (1..=24).collect());
    let result = eval_primitive(Primitive::Split, &[input], &split_params(2, 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2, 3]);
}

// ======================== Unequal Splits (sizes param) ========================

#[test]
fn oracle_split_unequal_two_three() {
    // Split [5] with sizes [2, 3] - returns first section
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params_sizes(0, &[2, 3])).unwrap();
    assert_eq!(extract_shape(&result), vec![2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2]);
}

#[test]
fn oracle_split_unequal_three_two() {
    // Split [5] with sizes [3, 2] - returns first section
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params_sizes(0, &[3, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

// ======================== Float Tests ========================

#[test]
fn oracle_split_f64() {
    let input = make_f64_tensor(&[6], vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 2)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[5] - 6.6).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_split_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 1)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 1]);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_split_with_negatives() {
    let input = make_i64_tensor(&[4], vec![-2, -1, 1, 2]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 2)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-2, -1, 1, 2]);
}

#[test]
fn oracle_split_preserves_data() {
    // Verify data is preserved exactly
    let input = make_i64_tensor(&[8], (1..=8).collect());
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 4)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 2]);
    assert_eq!(extract_i64_vec(&result), (1..=8).collect::<Vec<_>>());
}

// ======================== Error Cases ========================

#[test]
fn oracle_split_rejects_zero_num_sections() {
    // num_sections=0 must be rejected, not cause divide-by-zero panic
    let input = make_i64_tensor(&[4], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 0));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("positive"),
        "error should mention positive: {err}"
    );
}

#[test]
fn oracle_split_rejects_zero_num_sections_empty_axis() {
    // Even for empty axis, num_sections=0 must be rejected
    let input = make_i64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 0));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("positive"),
        "error should mention positive: {err}"
    );
}

#[test]
fn oracle_split_rejects_indivisible() {
    // axis_size=5 is not divisible by num_sections=2
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Split, &[input], &split_params(0, 2));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("divisible"));
}
