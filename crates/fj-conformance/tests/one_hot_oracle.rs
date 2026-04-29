//! Oracle tests for OneHot primitive.
//!
//! Tests against expected behavior matching JAX/jax.nn.one_hot:
//! - num_classes: number of classes (output dimension)
//! - on_value: value for the active index (default 1.0)
//! - off_value: value for inactive indices (default 0.0)
//! - dtype: output dtype (default F64)
//! - Output shape is input_shape + [num_classes]

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

fn one_hot_params(num_classes: u32) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("num_classes".to_string(), num_classes.to_string());
    p
}

fn one_hot_params_full(
    num_classes: u32,
    on_value: f64,
    off_value: f64,
    dtype: &str,
) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert("num_classes".to_string(), num_classes.to_string());
    p.insert("on_value".to_string(), on_value.to_string());
    p.insert("off_value".to_string(), off_value.to_string());
    p.insert("dtype".to_string(), dtype.to_string());
    p
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_one_hot_scalar_0() {
    // JAX: jax.nn.one_hot(0, num_classes=3) => [1, 0, 0]
    let input = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, 0.0, 0.0]);
}

#[test]
fn oracle_one_hot_scalar_1() {
    // JAX: jax.nn.one_hot(1, num_classes=3) => [0, 1, 0]
    let input = Value::scalar_i64(1);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 1.0, 0.0]);
}

#[test]
fn oracle_one_hot_scalar_2() {
    // JAX: jax.nn.one_hot(2, num_classes=3) => [0, 0, 1]
    let input = Value::scalar_i64(2);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 1.0]);
}

#[test]
fn oracle_one_hot_scalar_out_of_range() {
    // JAX: jax.nn.one_hot(5, num_classes=3) => [0, 0, 0] (all zeros)
    let input = Value::scalar_i64(5);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 0.0]);
}

#[test]
fn oracle_one_hot_scalar_negative() {
    // JAX: jax.nn.one_hot(-1, num_classes=3) => [0, 0, 0] (all zeros)
    let input = Value::scalar_i64(-1);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 0.0]);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_one_hot_1d_basic() {
    // JAX: jax.nn.one_hot([0, 1, 2], num_classes=3)
    // => [[1,0,0], [0,1,0], [0,0,1]]
    let input = make_i64_tensor(&[3], vec![0, 1, 2]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn oracle_one_hot_1d_repeated() {
    // [0, 0, 0] => all same one-hot encoding
    let input = make_i64_tensor(&[3], vec![0, 0, 0]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn oracle_one_hot_1d_single() {
    let input = make_i64_tensor(&[1], vec![2]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(4)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn oracle_one_hot_1d_with_out_of_range() {
    // [0, 5, 1] where 5 is out of range
    let input = make_i64_tensor(&[3], vec![0, 5, 1]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn oracle_one_hot_1d_binary() {
    // Binary one-hot: num_classes=2
    let input = make_i64_tensor(&[4], vec![0, 1, 0, 1]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(2)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 2]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn oracle_one_hot_1d_many_classes() {
    let input = make_i64_tensor(&[2], vec![0, 9]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(10)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 10]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 1.0); // First row: index 0
    assert_eq!(vals[1..10], vec![0.0; 9]);
    assert_eq!(vals[10..19], vec![0.0; 9]); // Second row: all zeros except index 9
    assert_eq!(vals[19], 1.0);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_one_hot_2d() {
    // [2, 2] -> [2, 2, 3]
    let input = make_i64_tensor(&[2, 2], vec![0, 1, 2, 0]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 3]);
    // Flattened: [1,0,0, 0,1,0, 0,0,1, 1,0,0]
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn oracle_one_hot_2d_single_row() {
    let input = make_i64_tensor(&[1, 3], vec![0, 1, 2]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![1, 3, 3]);
}

#[test]
fn oracle_one_hot_2d_single_col() {
    let input = make_i64_tensor(&[3, 1], vec![0, 1, 2]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(3)).unwrap();
    assert_eq!(extract_shape(&result), vec![3, 1, 3]);
}

// ======================== Custom Values ========================

#[test]
fn oracle_one_hot_custom_on_off() {
    // Custom on_value and off_value
    let input = make_i64_tensor(&[3], vec![0, 1, 2]);
    let params = one_hot_params_full(3, 5.0, -1.0, "F64");
    let result = eval_primitive(Primitive::OneHot, &[input], &params).unwrap();
    assert_eq!(
        extract_f64_vec(&result),
        vec![5.0, -1.0, -1.0, -1.0, 5.0, -1.0, -1.0, -1.0, 5.0]
    );
}

#[test]
fn oracle_one_hot_custom_on_value() {
    let input = Value::scalar_i64(1);
    let params = one_hot_params_full(3, 100.0, 0.0, "F64");
    let result = eval_primitive(Primitive::OneHot, &[input], &params).unwrap();
    assert_eq!(extract_f64_vec(&result), vec![0.0, 100.0, 0.0]);
}

#[test]
fn oracle_one_hot_integer_dtype() {
    let input = make_i64_tensor(&[2], vec![0, 1]);
    let params = one_hot_params_full(2, 1.0, 0.0, "I64");
    let result = eval_primitive(Primitive::OneHot, &[input], &params).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 0, 0, 1]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_one_hot_single_class() {
    // num_classes=1: always [1] for index 0
    let input = Value::scalar_i64(0);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(1)).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_f64_vec(&result), vec![1.0]);
}

#[test]
fn oracle_one_hot_large_num_classes() {
    let input = Value::scalar_i64(50);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(100)).unwrap();
    assert_eq!(extract_shape(&result), vec![100]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[50], 1.0);
    assert_eq!(vals.iter().filter(|&&x| x == 1.0).count(), 1);
    assert_eq!(vals.iter().filter(|&&x| x == 0.0).count(), 99);
}

#[test]
fn oracle_one_hot_all_same_index() {
    let input = make_i64_tensor(&[4], vec![2, 2, 2, 2]);
    let result = eval_primitive(Primitive::OneHot, &[input], &one_hot_params(5)).unwrap();
    assert_eq!(extract_shape(&result), vec![4, 5]);
    let vals = extract_f64_vec(&result);
    // All rows should have 1.0 at position 2
    for i in 0..4 {
        assert_eq!(vals[i * 5 + 2], 1.0);
    }
}
