//! Oracle tests for Heaviside primitive.
//!
//! heaviside(x, h0) returns:
//! - 0 if x < 0
//! - h0 if x == 0
//! - 1 if x > 0
//!
//! Tests:
//! - Basic: negative, zero, positive
//! - Different h0 values
//! - Special values: infinity, NaN
//! - Tensor shapes

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

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Basic Cases ========================

#[test]
fn oracle_heaviside_positive() {
    let x = make_f64_tensor(&[], vec![1.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        1.0,
        "heaviside(1, h0) = 1"
    );
}

#[test]
fn oracle_heaviside_negative() {
    let x = make_f64_tensor(&[], vec![-1.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.0,
        "heaviside(-1, h0) = 0"
    );
}

#[test]
fn oracle_heaviside_zero_half() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.5,
        "heaviside(0, 0.5) = 0.5"
    );
}

#[test]
fn oracle_heaviside_zero_one() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        1.0,
        "heaviside(0, 1) = 1"
    );
}

#[test]
fn oracle_heaviside_zero_zero() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.0,
        "heaviside(0, 0) = 0"
    );
}

// ======================== Small Values ========================

#[test]
fn oracle_heaviside_small_positive() {
    let x = make_f64_tensor(&[], vec![1e-10]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        1.0,
        "heaviside(small positive, h0) = 1"
    );
}

#[test]
fn oracle_heaviside_small_negative() {
    let x = make_f64_tensor(&[], vec![-1e-10]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.0,
        "heaviside(small negative, h0) = 0"
    );
}

// ======================== Special Values ========================

#[test]
fn oracle_heaviside_positive_inf() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        1.0,
        "heaviside(inf, h0) = 1"
    );
}

#[test]
fn oracle_heaviside_negative_inf() {
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.0,
        "heaviside(-inf, h0) = 0"
    );
}

#[test]
fn oracle_heaviside_nan() {
    // NaN compares neither less-than nor greater-than zero, matching JAX's
    // fallback to h0.
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        0.5,
        "heaviside(NaN, 0.5) = 0.5"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_heaviside_vector() {
    let x = make_f64_tensor(&[5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let h0 = make_f64_tensor(&[5], vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![5]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 0.5, 1.0, 1.0]);
}

#[test]
fn oracle_heaviside_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![-1.0, 0.0, 0.0, 1.0]);
    let h0 = make_f64_tensor(&[2, 2], vec![0.5, 0.5, 1.0, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.5, 1.0, 1.0]);
}
