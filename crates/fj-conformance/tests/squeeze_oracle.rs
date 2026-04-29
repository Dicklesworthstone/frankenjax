//! Oracle tests for Squeeze primitive.
//!
//! Tests against expected behavior matching JAX/lax.squeeze:
//! - dimensions: specific axes to squeeze (must be size 1)
//! - If no dimensions specified, removes all size-1 dims

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
        Value::Scalar(lit) => vec![lit.as_i64().unwrap()],
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

fn squeeze_params(dimensions: &[usize]) -> BTreeMap<String, String> {
    let mut p = BTreeMap::new();
    p.insert(
        "dimensions".to_string(),
        dimensions
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

// ======================== Automatic Squeeze (no params) ========================

#[test]
fn oracle_squeeze_auto_2d() {
    // JAX: lax.squeeze(jnp.array([[1, 2, 3]])) => [1, 2, 3]
    // Shape [1, 3] -> [3]
    let input = make_i64_tensor(&[1, 3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3]);
}

#[test]
fn oracle_squeeze_auto_3d_first() {
    // Shape [1, 2, 3] -> [2, 3]
    let input = make_i64_tensor(&[1, 2, 3], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), (1..=6).collect::<Vec<_>>());
}

#[test]
fn oracle_squeeze_auto_3d_last() {
    // Shape [2, 3, 1] -> [2, 3]
    let input = make_i64_tensor(&[2, 3, 1], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

#[test]
fn oracle_squeeze_auto_3d_middle() {
    // Shape [2, 1, 3] -> [2, 3]
    let input = make_i64_tensor(&[2, 1, 3], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

#[test]
fn oracle_squeeze_auto_multiple() {
    // Shape [1, 2, 1, 3, 1] -> [2, 3]
    let input = make_i64_tensor(&[1, 2, 1, 3, 1], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

#[test]
fn oracle_squeeze_auto_all_ones() {
    // Shape [1, 1, 1] -> scalar
    let input = make_i64_tensor(&[1, 1, 1], vec![42]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![] as Vec<u32>);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

#[test]
fn oracle_squeeze_auto_no_ones() {
    // Shape [2, 3] has no size-1 dims, stays unchanged
    let input = make_i64_tensor(&[2, 3], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

// ======================== Explicit Dimensions ========================

#[test]
fn oracle_squeeze_explicit_first() {
    // Squeeze only axis 0
    let input = make_i64_tensor(&[1, 3], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &squeeze_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
}

#[test]
fn oracle_squeeze_explicit_last() {
    // Squeeze only last axis
    let input = make_i64_tensor(&[3, 1], vec![1, 2, 3]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &squeeze_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
}

#[test]
fn oracle_squeeze_explicit_middle() {
    // Shape [2, 1, 3], squeeze axis 1
    let input = make_i64_tensor(&[2, 1, 3], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &squeeze_params(&[1])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

#[test]
fn oracle_squeeze_explicit_multiple() {
    // Shape [1, 2, 1, 3], squeeze axes 0 and 2
    let input = make_i64_tensor(&[1, 2, 1, 3], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &squeeze_params(&[0, 2])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
}

#[test]
fn oracle_squeeze_explicit_partial() {
    // Shape [1, 2, 1], squeeze only axis 0 (leave axis 2)
    let input = make_i64_tensor(&[1, 2, 1], vec![1, 2]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &squeeze_params(&[0])).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 1]);
}

// ======================== Scalar and 1D Tests ========================

#[test]
fn oracle_squeeze_scalar_passthrough() {
    let input = Value::scalar_i64(42);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    match result {
        Value::Scalar(lit) => assert_eq!(lit.as_i64().unwrap(), 42),
        _ => panic!("expected scalar"),
    }
}

#[test]
fn oracle_squeeze_1d_no_change() {
    // 1D tensor with no size-1 dims
    let input = make_i64_tensor(&[5], vec![1, 2, 3, 4, 5]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
}

#[test]
fn oracle_squeeze_1d_single_to_scalar() {
    // [1] -> scalar
    let input = make_i64_tensor(&[1], vec![42]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![] as Vec<u32>);
    assert_eq!(extract_i64_vec(&result), vec![42]);
}

// ======================== Float Tests ========================

#[test]
fn oracle_squeeze_f64() {
    let input = make_f64_tensor(&[1, 3], vec![1.1, 2.2, 3.3]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![3]);
    assert!((vals[0] - 1.1).abs() < 1e-10);
    assert!((vals[2] - 3.3).abs() < 1e-10);
}

#[test]
fn oracle_squeeze_f64_to_scalar() {
    let input = make_f64_tensor(&[1, 1], vec![99.5]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert_eq!(extract_shape(&result), vec![] as Vec<u32>);
    assert!((vals[0] - 99.5).abs() < 1e-10);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_squeeze_with_negatives() {
    let input = make_i64_tensor(&[1, 3], vec![-5, 0, 5]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_i64_vec(&result), vec![-5, 0, 5]);
}

#[test]
fn oracle_squeeze_4d() {
    // Shape [1, 2, 1, 3] -> [2, 3]
    let input = make_i64_tensor(&[1, 2, 1, 3], (1..=6).collect());
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_i64_vec(&result), (1..=6).collect::<Vec<_>>());
}

#[test]
fn oracle_squeeze_preserves_data() {
    // Verify data is preserved through squeeze
    let input = make_i64_tensor(&[1, 2, 1, 2], vec![1, 2, 3, 4]);
    let result = eval_primitive(Primitive::Squeeze, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_i64_vec(&result), vec![1, 2, 3, 4]);
}
