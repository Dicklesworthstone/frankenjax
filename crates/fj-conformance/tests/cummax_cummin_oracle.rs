//! Oracle tests for Cummax and Cummin primitives.
//!
//! Tests against expected behavior matching JAX:
//! - jax.lax.cummax: cumulative maximum along axis
//! - jax.lax.cummin: cumulative minimum along axis

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("axis".to_string(), axis.to_string());
    p
}

fn reverse_axis_params(axis: i64) -> BTreeMap<String, String> {
    let mut p = axis_params(axis);
    p.insert("reverse".to_owned(), "true".to_owned());
    p
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
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_i64().unwrap())
        .collect()
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    let tensor = v.as_tensor().expect("expected tensor");
    tensor
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap())
        .collect()
}

fn extract_i64_scalar(v: &Value) -> i64 {
    match v {
        Value::Scalar(l) => l.as_i64().unwrap(),
        Value::Tensor(t) if t.shape.dims.is_empty() => t.elements[0].as_i64().unwrap(),
        _ => panic!("expected scalar or 0-d tensor"),
    }
}

// ======================== Cummax Oracle Tests ========================

#[test]
fn oracle_cummax_1d_i64() {
    // jax.lax.cummax([1, 3, 2, 5, 4]) => [1, 3, 3, 5, 5]
    let input = make_i64_tensor(&[5], vec![1, 3, 2, 5, 4]);
    let result = eval_primitive(Primitive::Cummax, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 3, 5, 5]);
}

#[test]
fn oracle_cummax_1d_f64() {
    // jax.lax.cummax([1.0, 3.0, 2.0]) => [1.0, 3.0, 3.0]
    let input = make_f64_tensor(&[3], vec![1.0, 3.0, 2.0]);
    let result = eval_primitive(Primitive::Cummax, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 3.0).abs() < 1e-10);
}

#[test]
fn oracle_cummax_2d_last_axis() {
    // jax.lax.cummax([[1,3,2],[4,2,6]], axis=-1)
    // => [[1, 3, 3], [4, 4, 6]]
    let input = make_i64_tensor(&[2, 3], vec![1, 3, 2, 4, 2, 6]);
    let result = eval_primitive(Primitive::Cummax, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 3, 4, 4, 6]);
}

#[test]
fn oracle_cummax_2d_first_axis() {
    // jax.lax.cummax([[1,3,2],[4,2,6]], axis=0)
    // => [[1, 3, 2], [4, 3, 6]]
    let input = make_i64_tensor(&[2, 3], vec![1, 3, 2, 4, 2, 6]);
    let result = eval_primitive(Primitive::Cummax, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 2, 4, 3, 6]);
}

#[test]
fn oracle_cummax_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Cummax, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_cummax_with_negatives() {
    // jax.lax.cummax([-5, -3, -4, -1]) => [-5, -3, -3, -1]
    let input = make_i64_tensor(&[4], vec![-5, -3, -4, -1]);
    let result = eval_primitive(Primitive::Cummax, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-5, -3, -3, -1]);
}

#[test]
fn oracle_cummax_reverse() {
    // cummax([1, 3, 2, 5], reverse=True) => [5, 5, 5, 5]
    let input = make_i64_tensor(&[4], vec![1, 3, 2, 5]);
    let result = eval_primitive(Primitive::Cummax, &[input], &reverse_axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 5, 5, 5]);
}

#[test]
fn oracle_cummax_already_ascending() {
    // jax.lax.cummax([1, 2, 3, 4, 5]) => [1, 2, 3, 4, 5]
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Cummax, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

// ======================== Cummin Oracle Tests ========================

#[test]
fn oracle_cummin_1d_i64() {
    // jax.lax.cummin([5, 3, 4, 1, 2]) => [5, 3, 3, 1, 1]
    let input = make_i64_tensor(&[5], vec![5, 3, 4, 1, 2]);
    let result = eval_primitive(Primitive::Cummin, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 3, 3, 1, 1]);
}

#[test]
fn oracle_cummin_1d_f64() {
    // jax.lax.cummin([3.0, 1.0, 2.0]) => [3.0, 1.0, 1.0]
    let input = make_f64_tensor(&[3], vec![3.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Cummin, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 3.0).abs() < 1e-10);
    assert!((vals[1] - 1.0).abs() < 1e-10);
    assert!((vals[2] - 1.0).abs() < 1e-10);
}

#[test]
fn oracle_cummin_2d_last_axis() {
    // jax.lax.cummin([[3,1,2],[6,4,5]], axis=-1)
    // => [[3, 1, 1], [6, 4, 4]]
    let input = make_i64_tensor(&[2, 3], vec![3, 1, 2, 6, 4, 5]);
    let result = eval_primitive(Primitive::Cummin, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![3, 1, 1, 6, 4, 4]);
}

#[test]
fn oracle_cummin_2d_first_axis() {
    // jax.lax.cummin([[3,1,2],[6,4,5]], axis=0)
    // => [[3, 1, 2], [3, 1, 2]]
    let input = make_i64_tensor(&[2, 3], vec![3, 1, 2, 6, 4, 5]);
    let result = eval_primitive(Primitive::Cummin, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![3, 1, 2, 3, 1, 2]);
}

#[test]
fn oracle_cummin_single_element() {
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Cummin, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_cummin_with_negatives() {
    // jax.lax.cummin([-1, -3, -2, -5]) => [-1, -3, -3, -5]
    let input = make_i64_tensor(&[4], vec![-1, -3, -2, -5]);
    let result = eval_primitive(Primitive::Cummin, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-1, -3, -3, -5]);
}

#[test]
fn oracle_cummin_reverse() {
    // cummin([5, 1, 3, 2], reverse=True) => [1, 1, 2, 2]
    let input = make_i64_tensor(&[4], vec![5, 1, 3, 2]);
    let result = eval_primitive(Primitive::Cummin, &[input], &reverse_axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 2, 2]);
}

#[test]
fn oracle_cummin_already_descending() {
    // jax.lax.cummin([5, 4, 3, 2, 1]) => [5, 4, 3, 2, 1]
    let input = make_i64_tensor(&[5], vec![5, 4, 3, 2, 1]);
    let result = eval_primitive(Primitive::Cummin, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 4, 3, 2, 1]);
}

// ======================== Metamorphic Tests ========================

#[test]
fn metamorphic_cummax_last_equals_reduce_max() {
    let input = make_i64_tensor(&[5], vec![3, 1, 4, 1, 5]);
    let cummax_result =
        eval_primitive(Primitive::Cummax, std::slice::from_ref(&input), &no_params()).unwrap();
    let cummax_vals = extract_i64_vec(&cummax_result);

    let max_result = eval_primitive(Primitive::ReduceMax, &[input], &axis_params(0)).unwrap();
    let max_val = extract_i64_scalar(&max_result);

    assert_eq!(
        cummax_vals.last().copied(),
        Some(max_val),
        "last(cummax(x)) should equal reduce_max(x)"
    );
}

#[test]
fn metamorphic_cummin_last_equals_reduce_min() {
    let input = make_i64_tensor(&[5], vec![3, 1, 4, 1, 5]);
    let cummin_result =
        eval_primitive(Primitive::Cummin, std::slice::from_ref(&input), &no_params()).unwrap();
    let cummin_vals = extract_i64_vec(&cummin_result);

    let min_result = eval_primitive(Primitive::ReduceMin, &[input], &axis_params(0)).unwrap();
    let min_val = extract_i64_scalar(&min_result);

    assert_eq!(
        cummin_vals.last().copied(),
        Some(min_val),
        "last(cummin(x)) should equal reduce_min(x)"
    );
}

#[test]
fn metamorphic_cummax_first_element_identity() {
    let input = make_i64_tensor(&[4], vec![7, 2, 9, 3]);
    let result = eval_primitive(Primitive::Cummax, &[input], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 7, "cummax(x)[0] should equal x[0]");
}

#[test]
fn metamorphic_cummin_first_element_identity() {
    let input = make_i64_tensor(&[4], vec![7, 2, 9, 3]);
    let result = eval_primitive(Primitive::Cummin, &[input], &no_params()).unwrap();
    let vals = extract_i64_vec(&result);
    assert_eq!(vals[0], 7, "cummin(x)[0] should equal x[0]");
}
