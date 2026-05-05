//! Oracle tests for Nextafter primitive.
//!
//! Tests against expected behavior matching C's nextafter:
//! - Returns the next representable floating-point value after x towards y
//! - If x == y, returns y
//! - If x or y is NaN, returns NaN

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

fn make_f32_tensor(shape: &[u32], data: Vec<f32>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F32,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::from_f32).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        Value::Scalar(lit) => vec![lit.as_f64().unwrap()],
    }
}

fn extract_f32_vec(v: &Value) -> Vec<f32> {
    match v {
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|l| match l {
                Literal::F32Bits(bits) => f32::from_bits(*bits),
                other => panic!("expected F32Bits, got {other:?}"),
            })
            .collect(),
        Value::Scalar(Literal::F32Bits(bits)) => vec![f32::from_bits(*bits)],
        other => panic!("expected F32 tensor or scalar, got {other:?}"),
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
fn oracle_nextafter_same() {
    // nextafter(x, x) = x
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-15);
}

#[test]
fn oracle_nextafter_towards_positive() {
    // nextafter(1.0, 2.0) > 1.0
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(2.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0);
    assert!(vals[0] < 1.0 + 1e-10);
}

#[test]
fn oracle_nextafter_towards_negative() {
    // nextafter(1.0, 0.0) < 1.0
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] < 1.0);
    assert!(vals[0] > 1.0 - 1e-10);
}

#[test]
fn oracle_nextafter_zero_to_positive() {
    // nextafter(0.0, 1.0) > 0.0 (smallest positive subnormal)
    let a = Value::Scalar(Literal::from_f64(0.0));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0);
    assert!(vals[0] < 1e-300);
}

#[test]
fn oracle_nextafter_zero_to_negative() {
    // nextafter(0.0, -1.0) < 0.0 (smallest negative subnormal)
    let a = Value::Scalar(Literal::from_f64(0.0));
    let b = Value::Scalar(Literal::from_f64(-1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] < 0.0);
    assert!(vals[0] > -1e-300);
}

#[test]
fn oracle_nextafter_nan_x() {
    // nextafter(NaN, y) = NaN
    let a = Value::Scalar(Literal::from_f64(f64::NAN));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
}

#[test]
fn oracle_nextafter_nan_y() {
    // nextafter(x, NaN) = NaN
    let a = Value::Scalar(Literal::from_f64(1.0));
    let b = Value::Scalar(Literal::from_f64(f64::NAN));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_nan());
}

#[test]
fn oracle_nextafter_f32_scalar_preserves_dtype() {
    let a = Value::Scalar(Literal::from_f32(1.0));
    let b = Value::Scalar(Literal::from_f32(2.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f32_vec(&result);
    assert!(vals[0] > 1.0);
    assert!(vals[0] < 1.0 + 1e-5);
}

// ======================== 1D Tests ========================

#[test]
fn oracle_nextafter_1d() {
    let a = make_f64_tensor(&[3], vec![1.0, 2.0, 3.0]);
    let b = make_f64_tensor(&[3], vec![2.0, 1.0, 3.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 1.0); // towards 2
    assert!(vals[1] < 2.0); // towards 1
    assert!((vals[2] - 3.0).abs() < 1e-15); // same
}

#[test]
fn oracle_nextafter_1d_zeros() {
    let a = make_f64_tensor(&[2], vec![0.0, 0.0]);
    let b = make_f64_tensor(&[2], vec![1.0, -1.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0); // towards positive
    assert!(vals[1] < 0.0); // towards negative
}

#[test]
fn oracle_nextafter_f32_tensor_preserves_dtype() {
    let a = make_f32_tensor(&[3], vec![1.0, 2.0, 0.0]);
    let b = make_f32_tensor(&[3], vec![2.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F32),
        other => panic!("expected F32 tensor, got {other:?}"),
    }
    let vals = extract_f32_vec(&result);
    assert!(vals[0] > 1.0);
    assert!(vals[1] < 2.0);
    assert!(vals[2] < 0.0);
}

// ======================== 2D Tests ========================

#[test]
fn oracle_nextafter_2d() {
    let a = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 2.0]);
    let b = make_f64_tensor(&[2, 2], vec![1.0, 0.0, 0.0, 3.0]);
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0); // 0 towards 1
    assert!(vals[1] < 1.0); // 1 towards 0
    assert!(vals[2] > -1.0); // -1 towards 0
    assert!(vals[3] > 2.0); // 2 towards 3
}

// ======================== Edge Cases ========================

#[test]
fn oracle_nextafter_inf_towards_finite() {
    // nextafter(inf, 0) < inf
    let a = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let b = Value::Scalar(Literal::from_f64(0.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] < f64::INFINITY);
    assert!(vals[0] > 1e308);
}

#[test]
fn oracle_nextafter_finite_towards_inf() {
    // nextafter(MAX, inf) = inf
    let a = Value::Scalar(Literal::from_f64(f64::MAX));
    let b = Value::Scalar(Literal::from_f64(f64::INFINITY));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0].is_infinite() && vals[0] > 0.0);
}

#[test]
fn oracle_nextafter_negative_zero() {
    // nextafter(-0.0, 1.0) = smallest positive
    let a = Value::Scalar(Literal::from_f64(-0.0));
    let b = Value::Scalar(Literal::from_f64(1.0));
    let result = eval_primitive(Primitive::Nextafter, &[a, b], &no_params()).unwrap();
    let vals = extract_f64_vec(&result);
    assert!(vals[0] > 0.0);
}
