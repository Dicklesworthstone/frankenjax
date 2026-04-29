//! Oracle tests for IsFinite primitive.
//!
//! Tests against expected behavior matching JAX/jnp.isfinite:
//! - Returns True for finite numbers
//! - Returns False for NaN, +inf, -inf
//! - Works on scalars and tensors

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

fn extract_bool_vec(v: &Value) -> Vec<bool> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => panic!("expected bool"),
            })
            .collect(),
        Value::Scalar(Literal::Bool(b)) => vec![*b],
        _ => panic!("expected bool"),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        Value::Scalar(_) => vec![],
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== Scalar Tests ========================

#[test]
fn oracle_is_finite_scalar_zero() {
    let input = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

#[test]
fn oracle_is_finite_scalar_positive() {
    let input = Value::Scalar(Literal::from_f64(42.5));
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

#[test]
fn oracle_is_finite_scalar_negative() {
    let input = Value::Scalar(Literal::from_f64(-123.456));
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

#[test]
fn oracle_is_finite_scalar_nan() {
    let input = Value::Scalar(Literal::from_f64(f64::NAN));
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_is_finite_scalar_pos_inf() {
    let input = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_is_finite_scalar_neg_inf() {
    let input = Value::Scalar(Literal::from_f64(f64::NEG_INFINITY));
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_is_finite_scalar_i64() {
    // Integers are always finite
    let input = Value::scalar_i64(999999999);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_is_finite_1d_all_finite() {
    let input = make_f64_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_bool_vec(&result), vec![true, true, true, true]);
}

#[test]
fn oracle_is_finite_1d_mixed() {
    let input = make_f64_tensor(&[5], vec![1.0, f64::NAN, 2.0, f64::INFINITY, 3.0]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true, false, true, false, true]);
}

#[test]
fn oracle_is_finite_1d_all_special() {
    let input = make_f64_tensor(&[3], vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![false, false, false]);
}

#[test]
fn oracle_is_finite_1d_zeros() {
    let input = make_f64_tensor(&[3], vec![0.0, -0.0, 0.0]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true, true, true]);
}

#[test]
fn oracle_is_finite_1d_i64() {
    // All integers are finite
    let input = make_i64_tensor(&[4], vec![-100, 0, 100, i64::MAX]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true, true, true, true]);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_is_finite_2d_all_finite() {
    let input = make_f64_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 3]);
    assert_eq!(extract_bool_vec(&result), vec![true; 6]);
}

#[test]
fn oracle_is_finite_2d_mixed() {
    let input = make_f64_tensor(
        &[2, 2],
        vec![1.0, f64::NAN, f64::INFINITY, -10.0],
    );
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, false, true]);
}

#[test]
fn oracle_is_finite_2d_row_with_inf() {
    let input = make_f64_tensor(
        &[2, 3],
        vec![1.0, 2.0, 3.0, f64::INFINITY, f64::INFINITY, f64::INFINITY],
    );
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true, true, true, false, false, false]);
}

// ======================== 3D Tests ========================

#[test]
fn oracle_is_finite_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, f64::INFINITY, 8.0]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    assert_eq!(extract_bool_vec(&result), vec![true, false, true, true, true, true, false, true]);
}

// ======================== Edge Cases ========================

#[test]
fn oracle_is_finite_very_small() {
    // Subnormal numbers are still finite
    let input = make_f64_tensor(&[3], vec![f64::MIN_POSITIVE, 1e-300, -1e-300]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true, true, true]);
}

#[test]
fn oracle_is_finite_very_large() {
    // Large but finite numbers
    let input = make_f64_tensor(&[3], vec![f64::MAX, -f64::MAX, 1e308]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true, true, true]);
}

#[test]
fn oracle_is_finite_single_element() {
    let input = make_f64_tensor(&[1], vec![f64::NAN]);
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![1]);
    assert_eq!(extract_bool_vec(&result), vec![false]);
}

#[test]
fn oracle_is_finite_negative_zero() {
    // -0.0 is finite
    let input = Value::Scalar(Literal::from_f64(-0.0));
    let result = eval_primitive(Primitive::IsFinite, &[input], &no_params()).unwrap();
    assert_eq!(extract_bool_vec(&result), vec![true]);
}
