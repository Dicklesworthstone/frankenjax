//! Oracle tests for Sort and Argsort primitives.
//!
//! Tests against expected behavior matching JAX/NumPy:
//! - jax.lax.sort: stable ascending sort by default
//! - jax.lax.sort with is_stable=True, dimension=-1
//! - jnp.argsort: returns indices that would sort the array

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

fn descending_params() -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("descending".to_string(), "true".to_string());
    p
}

fn make_i64_tensor(shape: &[u32], data: Vec<i64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape { dims: shape.to_vec() },
            data.into_iter().map(Literal::I64).collect(),
        )
        .unwrap(),
    )
}

fn make_f64_tensor(shape: &[u32], data: Vec<f64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: shape.to_vec() },
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

// ======================== Sort Oracle Tests ========================

#[test]
fn oracle_sort_1d_i64_ascending() {
    // JAX: jax.lax.sort(jnp.array([3, 1, 4, 1, 5])) => [1, 1, 3, 4, 5]
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 1, 3, 4, 5]);
}

#[test]
fn oracle_sort_1d_f64_ascending() {
    // JAX: jax.lax.sort(jnp.array([3.5, 1.2, 4.8, 1.1])) => [1.1, 1.2, 3.5, 4.8]
    let input = make_f64_tensor(&[4u32], vec![3.5, 1.2, 4.8, 1.1]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[1] - 1.2).abs() < 1e-10);
    assert!((vals[2] - 3.5).abs() < 1e-10);
    assert!((vals[3] - 4.8).abs() < 1e-10);
}

#[test]
fn oracle_sort_1d_descending() {
    // JAX: jax.lax.sort(x, is_ascending=False) => [5, 4, 3, 1, 1]
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Sort, &[input], &descending_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![5, 4, 3, 1, 1]);
}

#[test]
fn oracle_sort_2d_last_axis() {
    // JAX: jax.lax.sort(jnp.array([[3,1],[4,2]]), dimension=-1) => [[1,3],[2,4]]
    let input = make_i64_tensor(&[2u32, 2], vec![3, 1, 4, 2]);
    let result = eval_primitive(Primitive::Sort, &[input], &axis_params(-1)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 3, 2, 4]);
}

#[test]
fn oracle_sort_2d_first_axis() {
    // JAX: jax.lax.sort(jnp.array([[3,1],[4,2]]), dimension=0) => [[3,1],[4,2]]
    // Actually sorts along axis 0: [[3,1],[4,2]] => for col 0: [3,4] sorted is [3,4]
    // for col 1: [1,2] sorted is [1,2] => result [[3,1],[4,2]]
    let input = make_i64_tensor(&[2u32, 2], vec![3, 1, 4, 2]);
    let result = eval_primitive(Primitive::Sort, &[input], &axis_params(0)).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![3, 1, 4, 2]);
}

#[test]
fn oracle_sort_empty_array() {
    let input = make_i64_tensor(&[0u32], vec![]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), Vec::<i64>::new());
}

#[test]
fn oracle_sort_single_element() {
    let input = make_i64_tensor(&[1u32], vec![42]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_sort_already_sorted() {
    let input = make_i64_tensor(&[5u32], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_sort_reverse_sorted() {
    let input = make_i64_tensor(&[5u32], vec![5, 4, 3, 2, 1]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4, 5]);
}

#[test]
fn oracle_sort_with_negatives() {
    let input = make_i64_tensor(&[5u32], vec![-3, 0, -1, 2, -5]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-5, -3, -1, 0, 2]);
}

#[test]
fn oracle_sort_f64_with_special_values() {
    // NaN handling: JAX sorts NaN to the end
    let input = make_f64_tensor(&[4u32], vec![f64::NAN, 1.0, f64::NEG_INFINITY, 2.0]);
    let result = eval_primitive(Primitive::Sort, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], f64::NEG_INFINITY);
    assert!((vals[1] - 1.0).abs() < 1e-10);
    assert!((vals[2] - 2.0).abs() < 1e-10);
    assert!(vals[3].is_nan());
}

// ======================== Argsort Oracle Tests ========================

#[test]
fn oracle_argsort_1d_i64() {
    // JAX: jnp.argsort(jnp.array([3, 1, 4, 1, 5])) => [1, 3, 0, 2, 4]
    // indices that would sort: positions of [1,1,3,4,5] in original
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    let indices = extract_i64_vec(&result);

    // Verify: applying indices to original yields sorted array
    let original = vec![3i64, 1, 4, 1, 5];
    let sorted_via_indices: Vec<i64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert_eq!(sorted_via_indices, vec![1, 1, 3, 4, 5]);
}

#[test]
fn oracle_argsort_1d_f64() {
    let input = make_f64_tensor(&[4u32], vec![3.5, 1.2, 4.8, 1.1]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    let indices = extract_i64_vec(&result);

    let original = vec![3.5, 1.2, 4.8, 1.1];
    let sorted_via_indices: Vec<f64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert!((sorted_via_indices[0] - 1.1).abs() < 1e-10);
    assert!((sorted_via_indices[1] - 1.2).abs() < 1e-10);
    assert!((sorted_via_indices[2] - 3.5).abs() < 1e-10);
    assert!((sorted_via_indices[3] - 4.8).abs() < 1e-10);
}

#[test]
fn oracle_argsort_descending() {
    let input = make_i64_tensor(&[5u32], vec![3, 1, 4, 1, 5]);
    let result = eval_primitive(Primitive::Argsort, &[input], &descending_params()).unwrap();
    let indices = extract_i64_vec(&result);

    let original = vec![3i64, 1, 4, 1, 5];
    let sorted_via_indices: Vec<i64> = indices.iter().map(|&i| original[i as usize]).collect();
    assert_eq!(sorted_via_indices, vec![5, 4, 3, 1, 1]);
}

#[test]
fn oracle_argsort_2d_last_axis() {
    // JAX: jnp.argsort(jnp.array([[3,1],[4,2]]), axis=-1) => [[1,0],[1,0]]
    let input = make_i64_tensor(&[2u32, 2], vec![3, 1, 4, 2]);
    let result = eval_primitive(Primitive::Argsort, &[input], &axis_params(-1)).unwrap();
    let indices = extract_i64_vec(&result);

    // Row 0: [3,1] -> argsort -> [1,0] (index 1 has smaller value)
    // Row 1: [4,2] -> argsort -> [1,0]
    assert_eq!(indices, vec![1, 0, 1, 0]);
}

#[test]
fn oracle_argsort_empty() {
    let input = make_i64_tensor(&[0u32], vec![]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), Vec::<i64>::new());
}

#[test]
fn oracle_argsort_single_element() {
    let input = make_i64_tensor(&[1u32], vec![42]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0]);
}

#[test]
fn oracle_argsort_already_sorted() {
    let input = make_i64_tensor(&[5u32], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2, 3, 4]);
}

#[test]
fn oracle_argsort_stability_check() {
    // When elements are equal, stable sort should preserve original order
    // JAX: jnp.argsort(jnp.array([1, 1, 1])) => [0, 1, 2]
    let input = make_i64_tensor(&[3u32], vec![1, 1, 1]);
    let result = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![0, 1, 2]);
}

// ======================== Sort+Argsort Consistency ========================

#[test]
fn oracle_sort_argsort_consistency() {
    // Sort and Argsort should be consistent:
    // sort(x) == x[argsort(x)]
    let data = vec![7i64, 2, 9, 1, 5, 3, 8, 4, 6];
    let input = make_i64_tensor(&[9u32], data.clone());

    let sorted = eval_primitive(Primitive::Sort, &[input.clone()], &no_params()).unwrap();
    let indices = eval_primitive(Primitive::Argsort, &[input], &no_params()).unwrap();

    let sorted_vals = extract_i64_vec(&sorted);
    let idx_vals = extract_i64_vec(&indices);

    let reconstructed: Vec<i64> = idx_vals.iter().map(|&i| data[i as usize]).collect();
    assert_eq!(sorted_vals, reconstructed);
}
