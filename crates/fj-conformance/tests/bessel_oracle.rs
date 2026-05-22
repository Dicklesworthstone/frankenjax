//! Oracle tests for BesselI0e and BesselI1e primitives.
//!
//! BesselI0e(x) = I0(x) * exp(-|x|) (exponentially scaled modified Bessel I0)
//! BesselI1e(x) = I1(x) * exp(-|x|) (exponentially scaled modified Bessel I1)
//!
//! Key properties:
//! - I0e(0) = 1
//! - I0e is symmetric: I0e(-x) = I0e(x)
//! - I1e(0) = 0
//! - I1e is odd: I1e(-x) = -I1e(x)
//! - For large |x|, both approach 1/sqrt(2*pi*|x|)
//!
//! Tests:
//! - Zero values
//! - Symmetry properties
//! - Known values
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

// ======================== BesselI0e Cases ========================

#[test]
fn oracle_bessel_i0e_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "I0e(0) = 1");
}

#[test]
fn oracle_bessel_i0e_symmetry() {
    for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::BesselI0e, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::BesselI0e, &[neg_input], &no_params()).unwrap();
        let pos_val = extract_f64_scalar(&pos_result);
        let neg_val = extract_f64_scalar(&neg_result);
        assert!(
            (pos_val - neg_val).abs() < 1e-14,
            "I0e({}) = I0e(-{}) = {} vs {}",
            x,
            x,
            pos_val,
            neg_val
        );
    }
}

#[test]
fn oracle_bessel_i0e_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // I0e(1) ~ 0.46575959...
    assert!(
        (actual - 0.46575959666109185).abs() < 1e-10,
        "I0e(1) ~ 0.4658, got {}",
        actual
    );
}

#[test]
fn oracle_bessel_i0e_large() {
    // For large x, I0e(x) ~ 1/sqrt(2*pi*x)
    let x = 100.0;
    let input = make_f64_tensor(&[], vec![x]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let asymptotic = 1.0 / (2.0 * PI * x).sqrt();
    assert!(
        (actual - asymptotic).abs() < 0.01,
        "I0e(100) ~ 1/sqrt(200*pi), got {} vs {}",
        actual,
        asymptotic
    );
}

#[test]
fn oracle_bessel_i0e_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual, 0.0, "I0e(inf) = 0 (approaches 0 asymptotically)");
}

#[test]
fn oracle_bessel_i0e_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "I0e(NaN) = NaN"
    );
}

// ======================== BesselI1e Cases ========================

#[test]
fn oracle_bessel_i1e_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "I1e(0) = 0");
}

#[test]
fn oracle_bessel_i1e_odd() {
    for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::BesselI1e, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::BesselI1e, &[neg_input], &no_params()).unwrap();
        let pos_val = extract_f64_scalar(&pos_result);
        let neg_val = extract_f64_scalar(&neg_result);
        assert!(
            (pos_val + neg_val).abs() < 1e-14,
            "I1e({}) = -I1e(-{}) = {} vs {}",
            x,
            x,
            pos_val,
            neg_val
        );
    }
}

#[test]
fn oracle_bessel_i1e_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // I1e(1) ~ 0.20791041...
    assert!(
        (actual - 0.207910412991402).abs() < 1e-10,
        "I1e(1) ~ 0.2079, got {}",
        actual
    );
}

#[test]
fn oracle_bessel_i1e_large() {
    // For large x, I1e(x) ~ 1/sqrt(2*pi*x)
    let x = 100.0;
    let input = make_f64_tensor(&[], vec![x]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let asymptotic = 1.0 / (2.0 * PI * x).sqrt();
    assert!(
        (actual - asymptotic).abs() < 0.01,
        "I1e(100) ~ 1/sqrt(200*pi), got {} vs {}",
        actual,
        asymptotic
    );
}

#[test]
fn oracle_bessel_i1e_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual, 0.0, "I1e(inf) = 0 (approaches 0 asymptotically)");
}

#[test]
fn oracle_bessel_i1e_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    assert!(
        extract_f64_scalar(&result).is_nan(),
        "I1e(NaN) = NaN"
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_bessel_i0e_vector() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 1.0);
    assert!((vals[1] - vals[2]).abs() < 1e-15, "I0e symmetric");
}

#[test]
fn oracle_bessel_i1e_vector() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] + vals[2]).abs() < 1e-15, "I1e odd function");
}
