//! Oracle tests for Sinc primitive.
//!
//! sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1 (the normalized sinc)
//!
//! Tests:
//! - Zero: sinc(0) = 1
//! - Known values: sinc(1) = 0, sinc(0.5) = 2/pi
//! - Symmetry: sinc(-x) = sinc(x)
//! - Special values: infinity, NaN
//! - Tensor shapes

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::f64::consts::PI;

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

// ======================== Zero Case ========================

#[test]
fn oracle_sinc_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "sinc(0) = 1");
}

#[test]
fn oracle_sinc_neg_zero() {
    let input = make_f64_tensor(&[], vec![-0.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "sinc(-0) = 1");
}

// ======================== Known Values (normalized sinc) ========================

#[test]
fn oracle_sinc_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.abs() < 1e-15,
        "sinc(1) = sin(pi)/pi ~ 0 (got {})",
        actual
    );
}

#[test]
fn oracle_sinc_two() {
    let input = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.abs() < 1e-15,
        "sinc(2) = sin(2pi)/(2pi) ~ 0 (got {})",
        actual
    );
}

#[test]
fn oracle_sinc_half() {
    let input = make_f64_tensor(&[], vec![0.5]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 2.0 / PI;
    assert!(
        (actual - expected).abs() < 1e-15,
        "sinc(0.5) = sin(pi/2)/(pi/2) = 2/pi ~ 0.6366"
    );
}

#[test]
fn oracle_sinc_quarter() {
    let input = make_f64_tensor(&[], vec![0.25]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = (PI / 4.0).sin() / (PI / 4.0);
    assert!(
        (actual - expected).abs() < 1e-15,
        "sinc(0.25) = sin(pi/4)/(pi/4)"
    );
}

// ======================== Symmetry ========================

#[test]
fn oracle_sinc_symmetry() {
    for x in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::Sinc, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::Sinc, &[neg_input], &no_params()).unwrap();
        let pos_val = extract_f64_scalar(&pos_result);
        let neg_val = extract_f64_scalar(&neg_result);
        assert!(
            (pos_val - neg_val).abs() < 1e-15,
            "sinc({}) = sinc(-{}) = {} vs {}",
            x,
            x,
            pos_val,
            neg_val
        );
    }
}

// ======================== Small Values (near limit) ========================

#[test]
fn oracle_sinc_small() {
    let input = make_f64_tensor(&[], vec![1e-10]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!((actual - 1.0).abs() < 1e-10, "sinc(tiny) ~ 1");
}

// ======================== Special Values ========================

#[test]
fn oracle_sinc_positive_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(
        actual.is_nan(),
        "sinc(inf) = NaN (sin(inf)/inf is indeterminate)"
    );
}

#[test]
fn oracle_sinc_negative_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert!(actual.is_nan(), "sinc(-inf) = NaN");
}

#[test]
fn oracle_sinc_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "sinc(NaN) = NaN");
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_sinc_vector() {
    let input = make_f64_tensor(&[4], vec![0.0, 0.5, 1.0, 2.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![4]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-15);
    assert!((vals[1] - 2.0 / PI).abs() < 1e-15);
    assert!(vals[2].abs() < 1e-15);
    assert!(vals[3].abs() < 1e-15);
}

#[test]
fn oracle_sinc_matrix() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 0.5, -0.5, 1.0]);
    let result = eval_primitive(Primitive::Sinc, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - 1.0).abs() < 1e-15);
    assert!((vals[1] - 2.0 / PI).abs() < 1e-15);
    assert!((vals[2] - 2.0 / PI).abs() < 1e-15);
    assert!(vals[3].abs() < 1e-15);
}
