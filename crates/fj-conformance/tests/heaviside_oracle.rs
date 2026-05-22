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
//! - Broadcast-compatible h0 inputs
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
    assert_eq!(extract_f64_scalar(&result), 1.0, "heaviside(1, h0) = 1");
}

#[test]
fn oracle_heaviside_negative() {
    let x = make_f64_tensor(&[], vec![-1.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "heaviside(-1, h0) = 0");
}

#[test]
fn oracle_heaviside_zero_half() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5, "heaviside(0, 0.5) = 0.5");
}

#[test]
fn oracle_heaviside_zero_one() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "heaviside(0, 1) = 1");
}

#[test]
fn oracle_heaviside_zero_zero() {
    let x = make_f64_tensor(&[], vec![0.0]);
    let h0 = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "heaviside(0, 0) = 0");
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
    assert_eq!(extract_f64_scalar(&result), 1.0, "heaviside(inf, h0) = 1");
}

#[test]
fn oracle_heaviside_negative_inf() {
    let x = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "heaviside(-inf, h0) = 0");
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

// ======================== Broadcasting ========================

fn scalar_f64(v: f64) -> Value {
    Value::Scalar(Literal::from_f64(v))
}

#[test]
fn oracle_heaviside_scalar_x_tensor_h0_broadcast() {
    // scalar x with tensor h0 (zero x, so h0 values are used)
    let x = scalar_f64(0.0);
    let h0 = make_f64_tensor(&[4], vec![0.1, 0.5, 0.9, 1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.1, 0.5, 0.9, 1.0]);
}

#[test]
fn oracle_heaviside_tensor_x_scalar_h0_broadcast() {
    // tensor x with scalar h0
    let x = make_f64_tensor(&[4], vec![-1.0, 0.0, 0.0, 1.0]);
    let h0 = scalar_f64(0.5);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![4]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.5, 0.5, 1.0]);
}

#[test]
fn oracle_heaviside_singleton_x_vector_h0_broadcast() {
    // [1] x with [3] h0 -> [3] (x=0, so h0 values are used)
    let x = make_f64_tensor(&[1], vec![0.0]);
    let h0 = make_f64_tensor(&[3], vec![0.25, 0.5, 0.75]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![0.25, 0.5, 0.75]);
}

#[test]
fn oracle_heaviside_vector_x_singleton_h0_broadcast() {
    // [3] x with [1] h0 -> [3]
    let x = make_f64_tensor(&[3], vec![-1.0, 0.0, 1.0]);
    let h0 = make_f64_tensor(&[1], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3]);
    assert_eq!(extract_f64_vec(&result), vec![0.0, 0.5, 1.0]);
}

#[test]
fn oracle_heaviside_column_x_matrix_h0_broadcast() {
    // [2, 1] x with [2, 3] h0 -> [2, 3]
    let x = make_f64_tensor(&[2, 1], vec![0.0, 1.0]);
    let h0 = make_f64_tensor(&[2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: x=0, so use h0 values: 0.1, 0.2, 0.3
    assert_eq!(vals[0], 0.1);
    assert_eq!(vals[1], 0.2);
    assert_eq!(vals[2], 0.3);
    // Row 1: x=1 (positive), so result is 1.0 for all
    assert_eq!(vals[3], 1.0);
    assert_eq!(vals[4], 1.0);
    assert_eq!(vals[5], 1.0);
}

#[test]
fn oracle_heaviside_different_ranks_broadcast() {
    // [3] x with [2, 3] h0 -> [2, 3]
    let x = make_f64_tensor(&[3], vec![-1.0, 0.0, 1.0]);
    let h0 = make_f64_tensor(&[2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![2, 3]);
    let vals = extract_f64_vec(&result);
    // Row 0: heaviside(-1, 0.1)=0, heaviside(0, 0.2)=0.2, heaviside(1, 0.3)=1
    assert_eq!(vals[0], 0.0);
    assert_eq!(vals[1], 0.2);
    assert_eq!(vals[2], 1.0);
    // Row 1: heaviside(-1, 0.4)=0, heaviside(0, 0.5)=0.5, heaviside(1, 0.6)=1
    assert_eq!(vals[3], 0.0);
    assert_eq!(vals[4], 0.5);
    assert_eq!(vals[5], 1.0);
}

#[test]
fn oracle_heaviside_all_scalars_broadcast() {
    // scalar heaviside scalar -> scalar
    let x = scalar_f64(0.0);
    let h0 = scalar_f64(0.5);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.5);
}

#[test]
fn oracle_heaviside_incompatible_shapes_error() {
    // [2] heaviside [3] should error
    let x = make_f64_tensor(&[2], vec![0.0, 1.0]);
    let h0 = make_f64_tensor(&[3], vec![0.5, 0.5, 0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params());
    assert!(result.is_err(), "incompatible shapes should error");
}

#[test]
fn oracle_heaviside_matrix_scalar_h0_broadcast() {
    let x = make_f64_tensor(
        &[3, 3],
        vec![-2.0, 0.0, 3.0, 5.0, -1.0, 0.0, 0.0, 7.0, -3.0],
    );
    let h0 = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![0.0, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0]
    );
}

#[test]
fn oracle_heaviside_matrix_row_h0_broadcast() {
    let x = make_f64_tensor(
        &[3, 3],
        vec![-2.0, 0.0, 3.0, 5.0, -1.0, 0.0, 0.0, 7.0, -3.0],
    );
    let h0 = make_f64_tensor(&[3], vec![2.0, 0.5, 1.0]);
    let result = eval_primitive(Primitive::Heaviside, &[x, h0], &no_params()).unwrap();

    assert_eq!(extract_shape(&result), vec![3, 3]);
    assert_eq!(
        extract_f64_vec(&result),
        vec![0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0]
    );
}
