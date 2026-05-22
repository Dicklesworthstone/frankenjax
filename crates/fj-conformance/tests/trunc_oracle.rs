//! Oracle tests for Trunc primitive.
//!
//! trunc(x) = round towards zero (truncate fractional part)
//!
//! Tests:
//! - Integers: trunc(n) = n
//! - Positive fractional: trunc(1.9) = 1
//! - Negative fractional: trunc(-1.9) = -1 (different from floor!)
//! - Zero: trunc(0) = 0, trunc(-0) = -0
//! - Infinity: trunc(±inf) = ±inf
//! - NaN propagation
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

// ======================== Integers ========================

#[test]
fn oracle_trunc_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "trunc(0) = +0");
}

#[test]
fn oracle_trunc_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), (-0.0_f64).to_bits(), "trunc(-0.0) = -0");
}

#[test]
fn oracle_trunc_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1) = 1");
}

#[test]
fn oracle_trunc_neg_one() {
    let input = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1) = -1");
}

#[test]
fn oracle_trunc_large_integer() {
    let input = make_f64_tensor(&[], vec![1000.0]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1000.0, "trunc(1000) = 1000");
}

// ======================== Positive Fractional ========================

#[test]
fn oracle_trunc_one_point_one() {
    let input = make_f64_tensor(&[], vec![1.1]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1.1) = 1");
}

#[test]
fn oracle_trunc_one_point_five() {
    let input = make_f64_tensor(&[], vec![1.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1.5) = 1");
}

#[test]
fn oracle_trunc_one_point_nine() {
    let input = make_f64_tensor(&[], vec![1.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "trunc(1.9) = 1");
}

// ======================== Negative Fractional ========================

#[test]
fn oracle_trunc_neg_one_point_one() {
    let input = make_f64_tensor(&[], vec![-1.1]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1.1) = -1");
}

#[test]
fn oracle_trunc_neg_one_point_five() {
    let input = make_f64_tensor(&[], vec![-1.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1.5) = -1");
}

#[test]
fn oracle_trunc_neg_one_point_nine() {
    let input = make_f64_tensor(&[], vec![-1.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -1.0, "trunc(-1.9) = -1");
}

// ======================== Special Values ========================

#[test]
fn oracle_trunc_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "trunc(+inf) = +inf"
    );
}

#[test]
fn oracle_trunc_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::NEG_INFINITY,
        "trunc(-inf) = -inf"
    );
}

#[test]
fn oracle_trunc_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "trunc(NaN) = NaN");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_trunc_vector() {
    let input = make_f64_tensor(&[4], vec![1.1, -1.1, 2.9, -2.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![1.0, -1.0, 2.0, -2.0]);
}

#[test]
fn oracle_trunc_matrix() {
    let input = make_f64_tensor(&[2, 3], vec![0.1, 0.5, 0.9, -0.1, -0.5, -0.9]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn oracle_trunc_3d_tensor() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 4.5, -4.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    );
}

// ======================== Comparison with Floor ========================

#[test]
fn oracle_trunc_vs_floor_negative() {
    let input = make_f64_tensor(&[3], vec![-0.1, -0.5, -0.9]);
    let trunc_result = eval_primitive(Primitive::Trunc, &[input.clone()], &no_params()).unwrap();
    let floor_result = eval_primitive(Primitive::Floor, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_vec(&trunc_result), vec![0.0, 0.0, 0.0]);
    assert_eq!(extract_f64_vec(&floor_result), vec![-1.0, -1.0, -1.0]);
}

// ======================== Specific Edge Cases ========================

#[test]
fn oracle_trunc_very_small_positive() {
    let input = make_f64_tensor(&[], vec![0.0000001]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "trunc(tiny) = 0");
}

#[test]
fn oracle_trunc_very_small_negative() {
    let input = make_f64_tensor(&[], vec![-0.0000001]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "trunc(-tiny) = 0");
}

#[test]
fn oracle_trunc_large_magnitude() {
    let input = make_f64_tensor(&[], vec![1e15 + 0.5]);
    let result = eval_primitive(Primitive::Trunc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1e15, "trunc(1e15 + 0.5)");
}
