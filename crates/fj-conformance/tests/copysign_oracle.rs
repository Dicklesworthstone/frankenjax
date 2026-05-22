//! Oracle tests for CopySign primitive.
//!
//! copysign(x, y) = |x| with the sign of y
//!
//! Tests:
//! - Basic: copysign(1, -1) = -1, copysign(-1, 1) = 1
//! - Zero: copysign with exact +0/-0 bit patterns
//! - Same signs: no change
//! - Special values: infinity, NaN
//! - Broadcast-compatible sign inputs
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
fn oracle_copysign_positive_negative() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let y = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0, "copysign(5, -1) = -5");
}

#[test]
fn oracle_copysign_negative_positive() {
    let x = make_f64_tensor(&[], vec![-5.0]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "copysign(-5, 1) = 5");
}

#[test]
fn oracle_copysign_positive_positive() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 5.0, "copysign(5, 1) = 5");
}

#[test]
fn oracle_copysign_negative_negative() {
    let x = make_f64_tensor(&[], vec![-5.0]);
    let y = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0, "copysign(-5, -1) = -5");
}

// ======================== Zero Cases ========================

#[test]
fn oracle_copysign_zero_positive() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "copysign(0, 1) = +0");
}

#[test]
fn oracle_copysign_zero_negative() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let y = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(
        actual.to_bits(),
        (-0.0_f64).to_bits(),
        "copysign(0, -1) = -0"
    );
}

#[test]
fn oracle_copysign_with_neg_zero() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let y = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -5.0, "copysign(5, -0) = -5");
}

// ======================== Special Values ========================

#[test]
fn oracle_copysign_inf_positive() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "copysign(inf, 1) = inf"
    );
}

#[test]
fn oracle_copysign_inf_negative() {
    let x = make_f64_tensor(&[], vec![f64::INFINITY]);
    let y = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::NEG_INFINITY,
        "copysign(inf, -1) = -inf"
    );
}

#[test]
fn oracle_copysign_neg_inf_positive() {
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(
        extract_f64_scalar(&result),
        f64::INFINITY,
        "copysign(-inf, 1) = inf"
    );
}

#[test]
fn oracle_copysign_nan_positive() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let y = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.is_nan() && actual.is_sign_positive(),
        "copysign(NaN, 1) = +NaN"
    );
}

#[test]
fn oracle_copysign_nan_negative() {
    let x = make_f64_tensor(&[], vec![f64::NAN]);
    let y = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.is_nan() && actual.is_sign_negative(),
        "copysign(NaN, -1) = -NaN"
    );
}

#[test]
fn oracle_copysign_value_nan() {
    let x = make_f64_tensor(&[], vec![5.0]);
    let y = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // NaN is typically positive, so copysign should use that sign
    assert!(actual.abs() == 5.0, "copysign(5, NaN) magnitude = 5");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_copysign_vector() {
    let x = make_f64_tensor(&[4], vec![1.0, -2.0, 3.0, -4.0]);
    let y = make_f64_tensor(&[4], vec![-1.0, 1.0, -1.0, 1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn oracle_copysign_matrix() {
    let x = make_f64_tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let y = make_f64_tensor(&[2, 2], vec![-1.0, -1.0, 1.0, 1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_f64_vec(&result), vec![-1.0, -2.0, 3.0, 4.0]);
}

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_copysign_scalar_x_tensor_y_broadcast() {
    // scalar magnitude with tensor sign
    let x = scalar_f64(5.0);
    let y = make_f64_tensor(&[4], vec![1.0, -1.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![5.0, -5.0, 5.0, -5.0]);
}

#[test]
fn oracle_copysign_tensor_x_scalar_y_broadcast() {
    // tensor magnitude with scalar sign
    let x = make_f64_tensor(&[4], vec![1.0, -2.0, 3.0, -4.0]);
    let y = scalar_f64(-1.0);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn oracle_copysign_singleton_x_vector_y_broadcast() {
    // [1] magnitude with [3] sign -> [3]
    let x = make_f64_tensor(&[1], vec![7.0]);
    let y = make_f64_tensor(&[3], vec![1.0, -1.0, 1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![7.0, -7.0, 7.0]);
}

#[test]
fn oracle_copysign_vector_x_singleton_y_broadcast() {
    // [3] magnitude with [1] sign -> [3]
    let x = make_f64_tensor(&[3], vec![1.0, -2.0, 3.0]);
    let y = make_f64_tensor(&[1], vec![-1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![-1.0, -2.0, -3.0]);
}

#[test]
fn oracle_copysign_column_x_matrix_y_broadcast() {
    // [2, 1] magnitude with [2, 3] sign -> [2, 3]
    let x = make_f64_tensor(&[2, 1], vec![3.0, 5.0]);
    let y = make_f64_tensor(&[2, 3], vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: copysign(3, 1)=3, copysign(3, -1)=-3, copysign(3, 1)=3
    assert_eq!(vals[0], 3.0);
    assert_eq!(vals[1], -3.0);
    assert_eq!(vals[2], 3.0);
    // Row 1: copysign(5, -1)=-5, copysign(5, 1)=5, copysign(5, -1)=-5
    assert_eq!(vals[3], -5.0);
    assert_eq!(vals[4], 5.0);
    assert_eq!(vals[5], -5.0);
}

#[test]
fn oracle_copysign_different_ranks_broadcast() {
    // [3] magnitude with [2, 3] sign -> [2, 3]
    let x = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let y = make_f64_tensor(&[2, 3], vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals, vec![1.0, -2.0, 3.0, -1.0, 2.0, -3.0]);
}

#[test]
fn oracle_copysign_all_scalars_broadcast() {
    // scalar copysign scalar -> scalar
    let x = scalar_f64(7.0);
    let y = scalar_f64(-1.0);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), -7.0);
}

#[test]
fn oracle_copysign_incompatible_shapes_error() {
    // [2] copysign [3] should error
    let x = make_f64_tensor(&[2], vec![1.0, 2.0]);
    let y = make_f64_tensor(&[3], vec![1.0, -1.0, 1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_copysign_matrix_scalar_sign_broadcast() {
    let x = make_f64_tensor(&[2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    let y = make_f64_tensor(&[], vec![-1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    );
}

#[test]
fn oracle_copysign_matrix_row_sign_broadcast() {
    let x = make_f64_tensor(&[2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    let y = make_f64_tensor(&[3], vec![-1.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::CopySign, &[x, y], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![-1.0, 2.0, -3.0, -4.0, 5.0, -6.0]
    );
}
